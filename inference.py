#!/usr/bin/env python3
"""
ST-GCN Inference (Val + Test) with Sliding Window Averaging

What this script does:
- Loads a trained model from `model_dir` (prefers *_last_weights.pt).
- Rebuilds the same splitter/dataset config used at training time.
- Instantiates inference datasets for VAL and TEST (one sample = one full clip).
- Slides a fixed-length window (T from config) across each long clip with stride=STRIDE,
  runs the model on each window, and averages per-frame SOFTMAX probabilities
  across all overlapping windows.
- Final per-frame labels are argmax over the averaged probabilities.
- Metrics (accuracy, balanced accuracy) are computed after masking out
  frames with true label == -1 (outside Interesting Moment).
- Saves per-clip outputs (meta, df_index, targets, preds, probs, confusion matrix).

Shapes:
- Dataset provides: data: (C, T_long, V, 1). We `.squeeze(-1)` → (C, T_long, V)
- Model expects: (N, C, T_win, V)  and returns logits shaped like (N, C_class, T_win)
"""

import os
import json
import argparse
import glob
from typing import Dict, Any, Tuple, List
import cv2
import signal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.multiprocessing.set_start_method("spawn", force=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

# --- project imports (match your training script layout) ---
from torch_datasets import *             # must include TennisSTGCNInferenceDataset
from torch_models import *
from training import *                   # if you have shared utilities here
from classes_collections import COLLECTION
from pose_toolkit.pt_poc_viewers.offline import *


class TimeoutException(Exception):
    pass
def handler(signum, frame):
    raise TimeoutException()

# set signal handler
signal.signal(signal.SIGALRM, handler)
# =========================
# Redefine plotting
# =========================
class MyVideoMultiPoseDS_Viewer(VideoMultiPoseDS_Viewer):
    def __init__(self, true_labels, predicted_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call original init
        self.true_labels = true_labels               
        self.predicted_labels = predicted_labels             
    
    def _prepare_browsers(self):
            if not self._pose_drawer_class_2d or not self._pose_drawer_class_3d:
                first_valid_index = self._df['pose'].first_valid_index() if 'pose' in self._df else None
                if first_valid_index is not None:
                    first_found_pose = self._df.loc[first_valid_index, 'pose']
                    if not self._pose_drawer_class_2d:
                        self._pose_drawer_class_2d = pose_drawer_class_getter.get_class(first_found_pose, '2d')
                    if not self._pose_drawer_class_3d and self._show_3d:
                        self._pose_drawer_class_3d = pose_drawer_class_getter.get_class(first_found_pose, '3d')
                else:
                    if not self._pose_drawer_class_2d:
                        self._pose_drawer_class_2d = J13PoseDrawer2d
                    if not self._pose_drawer_class_3d:
                        self._pose_drawer_class_3d = J13PoseDrawer3d

            if self._x0_3d is None:
                self._x0_3d = -self._box_side_length_3d / 2
            if self._y0_3d is None:
                self._y0_3d = -self._box_side_length_3d / 2
            if self._z0_3d is None:
                self._z0_3d = -self._box_side_length_3d / 2

            self._xlim_3d = (self._x0_3d, self._x0_3d + self._box_side_length_3d)
            self._ylim_3d = (self._y0_3d, self._y0_3d + self._box_side_length_3d)
            self._zlim_3d = (self._z0_3d, self._z0_3d + self._box_side_length_3d)

            self._init_ts = self._df.index[len(self._df) // 2]
            self._browsers.b0 = TightAxesBrowser('b0',
                                                [
                                                    # --- timeline lane with per-class channels for true/pred labels ---
                                                    MultiplePlotsLaneWCursor(
                                                        'mplwc',
                                                        [
                                                            PlotParams('p0',
                                                                        color=self._p0_plot_color,
                                                                        markersize=self._p0_plot_markersize,
                                                                        marker=self._p0_plot_marker,
                                                                        linestyle='None'),
                                                            # true_labels (row y=1)
                                                            PlotParams('l1_m1', color='gray',   marker='s', markersize=8, linestyle='None'),
                                                            PlotParams('l1_0',  color='yellowgreen', marker='s', markersize=8, linestyle='None'),
                                                            PlotParams('l1_1',  color='indigo',   marker='s', markersize=8, linestyle='None'),
                                                            PlotParams('l1_2',  color='hotpink',   marker='s', markersize=8, linestyle='None'),
                                                            PlotParams('l1_3',  color='dodgerblue',  marker='s', markersize=8, linestyle='None'),

                                                            # predicted_labels (row y=2)
                                                            PlotParams('l2_m1', color='gray',   marker='s', markersize=8, linestyle='None'),
                                                            PlotParams('l2_0',  color='yellowgreen', marker='s', markersize=8, linestyle='None'),
                                                            PlotParams('l2_1',  color='indigo',   marker='s', markersize=8, linestyle='None'),
                                                            PlotParams('l2_2',  color='hotpink',   marker='s', markersize=8, linestyle='None'),
                                                            PlotParams('l2_3',  color='dodgerblue',  marker='s', markersize=8, linestyle='None'),
                                                        ],
                                                        cursor_x=self._init_ts,
                                                        cursor_set_handler=self.cursor_set_handler_wrapper,
                                                        control_w_keyboard=True,
                                                        xlim=(self._df.index[0], self._df.index[-1]),
                                                        ylim=self._plot_ylim,                 # we'll set this in _set_initial_data()
                                                        snap_to_plot_key='p0',              # any l1_* is fine; choose one you expect to exist
                                                        x_write_prec=2,
                                                        y_write_plot_key='p0',
                                                        subplot2grid=((24, 1), (1, 0), 3, 1),
                                                    ),
                                                    self._image_axes_class('p2dia',
                                                                            self._genarate_pose_params_list_2d(),
                                                                            self._frames_bank,
                                                                            subplot2grid=((24, 1), (5, 0), 18, 1),
                                                                            **self._image_axes_kwargs)
                                                ])
            self._browsers.b1 = Axes3dBrowser('b1',
                                            Pose3dAxes('p3da',
                                                        self._genarate_pose_params_list_3d(),
                                                        xlim=self._xlim_3d,
                                                        ylim=self._ylim_3d,
                                                        zlim=self._zlim_3d),
                                            auto_azim=('circular', 360, 6), shape=(2, 3))

    @can_reimplement
    def _set_initial_data(self):
        import numpy as np

        # --- Shared time axis ---
        ts = self._df.index.to_numpy()

        # --- Keep p0 as the anchor (unchanged) ---
        # If your anchor is not 'fps', replace 'fps' below with your signal column.
        self._data[self._browsers.b0.mplwc.plot_x_key('p0')] = ts
        self._data[self._browsers.b0.mplwc.plot_y_key('p0')] = self._df['fps']

        # --- Inputs for labeled rows (values in {-1,0,1,2,3}) ---
        true_labels = np.asarray(self.true_labels)
        predicted_labels = np.asarray(self.predicted_labels)

        # --- Fixed y-rows: true at 1, pred at 2 ---
        y_true = np.full_like(true_labels, 10, dtype=float)
        y_pred = np.full_like(predicted_labels, 20, dtype=float)

        # --- Expand existing ylim to at least (-1, 3), without shrinking broader ranges ---
        self._plot_ylim = (0,30)

        # --- Helper to push class-split scatters into the timeline ---
        def push(group_prefix: str, values: np.ndarray, yvals: np.ndarray):
            """
            group_prefix: 'l1' for true_labels, 'l2' for predicted_labels
            values:      array of class ids in {-1,0,1,2,3}
            yvals:       array of same length with constant y (1 or 2)
            """
            for v, suffix in [(-1, 'm1'), (0, '0'), (1, '1'), (2, '2'), (3, '3')]:
                mask = (values == v)
                chan = f'{group_prefix}_{suffix}'
                self._data[self._browsers.b0.mplwc.plot_x_key(chan)] = ts[mask]
                self._data[self._browsers.b0.mplwc.plot_y_key(chan)] = yvals[mask]

        # --- Fill both rows (true → y=1, pred → y=2) ---
        push('l1', true_labels, y_true)
        push('l2', predicted_labels, y_pred)

        # --- (Optional) Label the rows on the y-axis if the widget exposes axes ---
        try:
            ax = getattr(self._browsers.b0.mplwc, 'axes', None)
            if ax is not None:
                ax.set_yticks([1, 2])
                ax.set_yticklabels(['true', 'pred'])
        except Exception:
            pass  # Safe no-op if the timeline manages ticks internally
# =========================
# Utilities
# =========================
def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def load_config(model_dir: str) -> Dict[str, Any]:
    """Load the JSON config saved next to checkpoints."""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(cfg_path, "r") as f:
        return json.load(f)

def pick_checkpoint(model_dir: str) -> str:
    """
    Pick a checkpoint that will work:
      1) *_last_weights.pt
      2) *_last.pt
      3) *_best_weights.pt
      4) *_best.pt
      5) *_stopped_weights.pt
      6) *_stopped.pt
      7) *.pt (fallback)
    If multiple files match, pick the lexicographically last (usually highest epoch).
    """
    priorities = [
        "*_last_weights.pt",
        "*_last.pt",
        "*_best_weights.pt",
        "*_best.pt",
        "*_stopped_weights.pt",
        "*_stopped.pt",
        "*.pt",
    ]
    for pat in priorities:
        matches = sorted(glob.glob(os.path.join(model_dir, pat)))
        if matches:
            return matches[-1]
    raise FileNotFoundError(f"No checkpoint *.pt found in {model_dir}")

def build_objects_from_config(cfg: Dict[str, Any]):
    """
    Recreate the runtime objects needed for inference from the saved config:
    - splitter → to get val/test paths
    - pose normalizer config
    - dataset cfgclass (we'll pass it to TennisSTGCNInferenceDataset)
    - model + model_cfg, with correct num_point/joint_names/bones
    """
    # --- Splitter (to retrieve val/test clip directories) ---
    splitter_section  = cfg["splitter"]
    splitter_cfgclass = COLLECTION["datasets"][splitter_section["name"]]["cfgclass"]
    splitter_class    = COLLECTION["datasets"][splitter_section["name"]]["class"]
    splitter_cfg      = splitter_cfgclass(**splitter_section["hyperparams"])
    splitter          = splitter_class(splitter_cfg)
    # We only need val/test paths for inference
    _, val_paths, test_paths = splitter.train, splitter.val, splitter.test

    # --- Pose normalizer config (same as training) ---
    pose_norm_section  = cfg["pose_norm"]
    pose_norm_cfgclass = COLLECTION["datasets"][pose_norm_section["name"]]["cfgclass"]
    pose_norm_cfg      = pose_norm_cfgclass(**pose_norm_section["hyperparams"])

    # --- Dataset base cfgclass (we reuse the same dataclass for inference dataset) ---
    dataset_section  = cfg["dataset"]
    dataset_cfgclass = COLLECTION["datasets"][dataset_section["name"]]["cfgclass"]
    # dataset_class  = COLLECTION["datasets"][dataset_section["name"]]["class"]  # not needed for inference

    # Common kwargs used to instantiate dataset configs
    common_kwargs = dict(
        **dataset_section["hyperparams"],
        pose_norm_config=pose_norm_cfg,
        split_type=splitter_section["name"],
        random_seed=splitter.config.random_seed,
    )

    # We need joint_names & bones for the model config
    # Build a temporary dataset cfg to access canonical joint/bone definitions from defaults
    tmp_cfg      = dataset_cfgclass(**common_kwargs, path_list=val_paths, caching_path="source/cache/tmp")
    num_point    = len(tmp_cfg.all_joint_names_ordered)
    joint_names  = tmp_cfg.all_joint_names_ordered
    bones        = tmp_cfg.all_bones_names

    # --- Model & model config (same class as training) ---
    model_section  = cfg["model"]
    model_cfgclass = COLLECTION["models"][model_section["name"]]["cfgclass"]
    model_class    = COLLECTION["models"][model_section["name"]]["class"]

    model_cfg = model_cfgclass(
        **model_section["hyperparams"],
        num_point=num_point,
        joint_names=joint_names,
        bones=bones,
    )
    model = model_class(model_cfg)

    return dataset_cfgclass, common_kwargs, val_paths, test_paths, model, model_cfg, joint_names

def build_inference_loader(dataset_cfgclass, common_kwargs, paths: List[str], batch_size: int, num_workers: int):
    """
    Instantiate TennisSTGCNInferenceDataset for given paths and wrap it in a DataLoader.

    Notes:
    - batch_size must be 1 because each clip is variable-length (T_long differs).
    - DataLoader will collate metadata items (e.g., DataFrame) into lists automatically.
    """
    ds_cfg = dataset_cfgclass(**common_kwargs, path_list=paths, caching_path="source/cache/ignored")
    inf_ds = TennisSTGCNInferenceDataset(ds_cfg)

    loader = DataLoader(
        inf_ds,
        batch_size=batch_size,   # keep 1 unless you implement a custom collate_fn
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return inf_ds, loader

def save_confusion_png(cm: np.ndarray, acc: float, bacc: float, out_png: str) -> None:
    """
    Save a confusion matrix image with numbers in each cell and a title showing accuracy & balanced accuracy.
    
    Args:
        cm (np.ndarray): Confusion matrix (2D array).
        acc (float): Accuracy score.
        bacc (float): Balanced accuracy score.
        out_png (str): Output PNG file path.
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    
    # Display confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap="viridis")
    
    # Title and axis labels
    ax.set_title(f"Confusion Matrix\n{os.path.basename(os.path.dirname(out_png))}\nacc={acc:.4f} | bacc={bacc:.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    
    # Annotate each cell with numeric value
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] < cm.max() / 2 else "black",
                fontsize=8
            )
    
    # Colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def find_smallest_peak(hmap):
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(hmap, prominence=0.03)
    if peaks.size > 0:
        smallest_peak_idx = peaks[hmap[peaks].argmin()]
        smallest_peak_val = np.mean(hmap[peaks])*0.5
        # smallest_peak_val = hmap[smallest_peak_idx]
    else:
        smallest_peak_val = 0.05
    return smallest_peak_val

def run_split_inference(
    split_name: str,
    loader: DataLoader,
    dset: TennisSTGCNInferenceDataset,
    model: torch.nn.Module,
    device: torch.device,
    T_win: int,
    stride: int,
    out_root: str,
    generate_video:bool = False
) -> None:
    """
    Run inference over an entire split (val/test).

    Windowing/Stride:
    - We slide a fixed-size window of length T_win across the long clip with step=stride.
    - We ALWAYS add a final window anchored to end (start = max(T_long - T_win, 0))
      to guarantee coverage of the tail region.
    - For windows shorter than T_win (e.g., short clips or tail-end), we pad symmetrically
      by repeating edge frames (consistent with your training padding strategy).
    - For each window, we forward the model, softmax to probabilities, then map those per-window
      probabilities back onto the original timeline and accumulate:
         probs_sum[t, :] += prob_win[t_local, :]
         counts[t]       += 1
    - After processing all windows, averaged_probs = probs_sum / counts[:, None]
      and final labels = argmax(averaged_probs, axis=1).
    - For metrics, we mask out time steps with true label == -1.
    """
    out_dir = os.path.join(out_root, split_name)
    ensure_dir(out_dir)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=loader.__len__(), desc='Inference')
        all_cms, all_peaks = [], []
        all_acc, all_bacc = [], []
        for idx, sample in pbar:
            try:
                signal.alarm(10*60)
                pbar.set_description("Inference")
                # ---- Unpack batch (batch_size=1) ----
                # data: (1, C, T_long, V, 1) → squeeze(-1) → (1, C, T_long, V) → [0] → (C, T_long, V)
                data   = sample["data"][0].to(device).squeeze(-1)
                labels = sample["labels"][0].cpu().numpy()  # (T_long,)
                idx_fromdset = sample["id"]

                fullitem = dset.__getitem_full__(idx_fromdset)
                meta = fullitem['meta']
                labels_gaus = fullitem['labels_gaus']
                labels_phase = fullitem['labels_phase']
                sample['meta'] = meta

                # Long clip length (frames/timesteps)
                T_long = int(data.shape[1])

                # ---- Determine sliding window starts ----
                starts = list(range(0, max(T_long - T_win + 1, 1), stride))
                tail_start = max(T_long - T_win, 0)
                if tail_start not in starts:
                    starts.append(tail_start)
                starts = sorted(set(starts))

                # ---- Dry run to get num_classes (logit shape) ----
                s0 = starts[0]
                e0 = min(s0 + T_win, T_long)
                win0 = data[:, s0:e0, :]  # (C, t, V)
                if win0.shape[1] < T_win:
                    # symmetric padding by repeating edges
                    deficit = T_win - win0.shape[1]
                    left  = deficit // 2 + (deficit % 2)
                    right = deficit // 2
                    pad_left  = win0[:, :1, :].repeat(1, left, 1)
                    pad_right = win0[:, -1:, :].repeat(1, right, 1)
                    win0 = torch.cat([pad_left, win0, pad_right], dim=1)  # (C, T_win, V)
                out0 = model(win0.unsqueeze(0))  # model expects (N, C, T, V)
                if isinstance(out0, (list, tuple)):
                    out0 = out0[0]
                while out0.dim() > 3:
                    out0 = out0.squeeze(-1)      # squeeze trailing singleton dims if present
                # out0: (1, C_class, T_win)
                num_classes = int(out0.shape[1])

                # ---- read labels_format from dataset config ----
                labels_format = getattr(dset.config, "labels_format", "sequence_phases")

                # ---- Accumulators for probability averaging ----
                # For sequence_phases: per-class probs; for gaussian_heatmaps: transition probs (4 maps)
                probs_sum = np.zeros((T_long, num_classes), dtype=np.float64)
                counts    = np.zeros((T_long,), dtype=np.int64)

                # ---- Slide windows across the clip ----
                for s in starts:
                    e = min(s + T_win, T_long)
                    win = data[:, s:e, :]  # (C, t, V)

                    # pad to T_win if needed (tail/short clips)
                    if win.shape[1] < T_win:
                        deficit = T_win - win.shape[1]
                        left  = deficit // 2 + (deficit % 2)
                        right = deficit // 2
                        pad_left  = win[:, :1, :].repeat(1, left, 1)
                        pad_right = win[:, -1:, :].repeat(1, right, 1)
                        win = torch.cat([pad_left, win, pad_right], dim=1)  # (C, T_win, V)

                    out = model(win.unsqueeze(0))  # (1, C_class, T_win)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    while out.dim() > 3:
                        out = out.squeeze(-1)

                    # ---- NEW: convert logits to probabilities depending on labels_format ----
                    if labels_format == "sequence_phases":
                        prob = F.softmax(out, dim=1)          # (1, C_class, T_win)
                    elif labels_format == "gaussian_heatmaps":
                        prob = torch.sigmoid(out)             # (1, 4, T_win)
                        # probs_avg: (T, 4)
                        # print("max per channel of a window:", prob.max(axis=2))  # expect something like [0.7, 0.8, 0.6, 0.5]

                    else:
                        raise NotImplementedError(f"Unknown labels_format: {labels_format}")
                    prob_np = prob[0].detach().cpu().numpy().transpose(1, 0)  # (T_win, C_class)

                    # Remove padding before accumulation (we centered padding symmetrically)
                    orig_len = e - s
                    if orig_len < T_win:
                        deficit = T_win - orig_len
                        left  = deficit // 2 + (deficit % 2)
                        right = deficit // 2
                        prob_np = prob_np[left:T_win - right, :]

                    # Accumulate probabilities & counts for frames [s:e)
                    probs_sum[s:e, :] += prob_np
                    counts[s:e]       += 1

                # ---- Average per-frame probabilities & argmax ----
                counts_safe = np.maximum(counts, 1)[:, None]  # avoid division by zero
                probs_avg = probs_sum / counts_safe
                # probs_avg: (T, 4)
                # print("max peaks:", probs_avg.max(axis=0))  # expect something like [0.7, 0.8, 0.6, 0.5]
                # ---- Turn averaged probs into per-frame predictions ----
                if labels_format == "sequence_phases":
                    preds = np.argmax(probs_avg, axis=1)    # (T_long,)
                elif labels_format == "gaussian_heatmaps":
                    # probs_avg is (T, 4) → transpose to (4, T) for decoder
                    heat_avg = probs_avg.T
                    thr = [find_smallest_peak(hmap) for hmap in heat_avg]
                    # print("thresholds selected:", thr)
                    preds = dset._decode_transition_heatmaps_to_phase_sequence(
                        heat_avg, 
                        threshold=thr, 
                        nms_radius=30, # i assume tht 2 strikes can happen only 1.5s or 45frames apart, not clother
                        default_phase=0,
                        fallback_if_phase_not_found=15
                    )  # -> (T_long,)

                    #extra visualizations
                    phase_colors = {
                                        -1: "gray",
                                        0: "yellowgreen",
                                        1: "indigo",
                                        2: "hotpink",
                                        3: "dodgerblue"
                                    }
                    peaks_figure, axes = plt.subplots(4, 1, figsize=(12,8))
                    for a, ax in enumerate(axes):
                        hmap = heat_avg[a]
                        T = len(preds)
                        # plot the heatmap curve
                        ax.plot(hmap, label=f"Transition {a}->{a+1 if a+1<4 else 0}", c='black')
                        ax.axhline(y=thr[a], 
                                color="red", 
                                linestyle="--", 
                                linewidth=1,
                                label=f"threshold = {np.round(thr[a],3)}")

                        for t in range(T):
                            ph = int(preds[t])
                            ax.scatter(t, -0.1, color=phase_colors.get(ph, "black"), marker="s", s=10)
                        for t in range(T):
                            gt = int(labels_phase.cpu().numpy()[t])
                            ax.scatter(t, -0.3, color=phase_colors.get(gt, "black"), marker="s", s=10)
                        ax.set_ylim(-.5, 1.1 * hmap.max())
                        ax.legend(loc="upper right")
                    plt.tight_layout()
                else:
                    raise NotImplementedError(f"Unknown labels_format: {labels_format}")

                # ---- Prepare ground-truth labels for metrics/viz ----
                labels_np = labels_phase.cpu().numpy()
                
                # ---- Metrics (ignore label -1) ----
                pbar.set_description("Confusion Matrix")
                mask = labels_np != -1
                
                if np.any(mask):
                    y_true = labels_np[mask]
                    y_pred = preds[mask]
                    acc  = float(accuracy_score(y_true, y_pred))
                    bacc = float(balanced_accuracy_score(y_true, y_pred))
                    cm   = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
                    all_acc.append(acc)
                    all_bacc.append(bacc)
                else:
                    acc = float("nan")
                    bacc = float("nan")
                    cm = np.zeros((num_classes, num_classes), dtype=int)

                # ---- Per-sample output directory: "{idx:05d}_{clip_name}" ----
                # meta fields are lists after default collate; unwrap them
                clip_name = meta["clip_name"][0] if isinstance(meta["clip_name"], list) else meta["clip_name"]
                # sanitize clip_name for filesystem safety
                clip_sanitized = str(clip_name).replace(os.sep, "_")
                sample_dir = os.path.join(out_dir, f"{idx:05d}_{clip_sanitized}")
                ensure_dir(sample_dir)

                # ---- Viz ----
                pbar.set_description("Video generation")
                if generate_video: 
                    viewer = MyVideoMultiPoseDS_Viewer(ds_path=meta['clip_dir'], true_labels=labels_np, predicted_labels=preds)
                    viewer.prepare().generate_video(os.path.join(sample_dir, "video.mp4"))

                # ---- Save meta.json (basic info; df_index saved separately) ----
                pbar.set_description("Saving")
                clip_dir   = meta["clip_dir"][0]   if isinstance(meta["clip_dir"], list)   else meta["clip_dir"]
                person_dir = meta["person_dir"][0] if isinstance(meta["person_dir"], list) else meta["person_dir"]
                used_vis   = meta["used_visibility"][0] if isinstance(meta["used_visibility"], list) else meta["used_visibility"]
                joint_names = meta["joint_names"][0] if isinstance(meta["joint_names"], list) and len(meta["joint_names"]) == 1 else meta["joint_names"]
                
                peaks_figure.suptitle(f"{idx:05d}_{clip_sanitized}")
                peaks_figure.savefig(os.path.join(sample_dir,"peaks_and_phases.png"), 
                                    dpi=300, 
                                    bbox_inches="tight")
                all_peaks.append(plt.imread(os.path.join(sample_dir,"peaks_and_phases.png")))
                meta_json = {
                    "clip_name": clip_name,
                    "clip_dir": str(clip_dir),
                    "person_dir": str(person_dir),
                    "joint_names": joint_names,
                    "used_visibility": bool(used_vis),
                    "T_long": int(T_long),
                    "T_win": int(T_win),
                    "stride": int(stride),
                    "num_classes": int(num_classes),
                }
                with open(os.path.join(sample_dir, "meta.json"), "w") as f:
                    json.dump(meta_json, f, indent=2)

                # ---- Save df_index.csv if present ----
                try:
                    df_index = sample["meta"]["df_index"][0] if isinstance(sample["meta"]["df_index"], list) else sample["meta"]["df_index"]
                    if hasattr(df_index, "to_csv"):
                        df_index.to_csv(os.path.join(sample_dir, "df_index.csv"), index=False)
                except Exception:
                    pass

                # ---- Save arrays (metadata-aligned) ----
                frame_ids = sample["meta"]["frame_ids"][0] if isinstance(sample["meta"]["frame_ids"], list) else sample["meta"]["frame_ids"]
                rel_ts    = sample["meta"]["rel_stamps"][0] if isinstance(sample["meta"]["rel_stamps"], list) else sample["meta"]["rel_stamps"]
                abs_ts    = sample["meta"]["abs_stamps"][0] if isinstance(sample["meta"]["abs_stamps"], list) else sample["meta"]["abs_stamps"]

                np.save(os.path.join(sample_dir, "frame_ids.npy"), np.asarray(frame_ids))
                np.save(os.path.join(sample_dir, "rel_stamps.npy"), np.asarray(rel_ts))
                np.save(os.path.join(sample_dir, "abs_stamps.npy"), np.asarray(abs_ts))

                np.save(os.path.join(sample_dir, "targets.npy"), labels)
                np.save(os.path.join(sample_dir, "preds.npy"),   preds)
                np.save(os.path.join(sample_dir, "probs.npy"),   probs_avg)

                # ---- Save metrics & confusion matrix ----
                with open(os.path.join(sample_dir, "metrics.json"), "w") as f:
                    json.dump({
                        "accuracy": acc,
                        "balanced_accuracy": bacc,
                        "num_frames": int(T_long),
                        "num_used_frames": int(mask.sum())
                    }, f, indent=2)

                save_confusion_png(cm, acc, bacc, os.path.join(sample_dir, "confusion.png"))
                all_cms.append(plt.imread(os.path.join(sample_dir, "confusion.png")))
                plt.close('all')
                signal.alarm(0)
            except TimeoutException:
                continue
        
        n = len(all_cms)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        confusion_fig, confusion_axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        confusion_axes = confusion_axes.ravel()

        peaks_fig, peaks_axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        peaks_axes = peaks_axes.ravel()

        for i in range(rows * cols):
            ax = confusion_axes[i]
            if i < n:
                ax.imshow(all_cms[i])
            ax.axis("off")

        for i in range(rows * cols):
            ax = peaks_axes[i]
            if i < n:
                ax.imshow(all_peaks[i])
            ax.axis("off")

        plt.tight_layout()
        title = f"acc={np.mean(all_acc)}|min={min(all_acc)}|max={max(all_acc)}|range={max(all_acc)-min(all_acc)}\nbacc={np.mean(all_bacc)}|min={min(all_bacc)}|max={max(all_bacc)}|range={max(all_bacc)-min(all_bacc)}"
        confusion_fig.suptitle(title)
        confusion_fig.savefig(os.path.join(out_dir, "combined_confusions.png"), dpi=300)
        peaks_fig.suptitle(title)
        peaks_fig.savefig(os.path.join(out_dir, "combined_p&p.png"), dpi=300)
        plt.close(confusion_fig)
        plt.close(peaks_fig)

        return all_acc, all_bacc
        


# =========================
# Main
# =========================
def main():

    BLOCKS = 6
    SPLITTER = 'PersonSplitter'
    DIM = 3

    # --- parse args ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--splitter", type=str, required=False, default=SPLITTER)
    parser.add_argument("--dim", type=int, required=False, default=DIM)
    parser.add_argument("--blocks", type=int, required=False, default=BLOCKS)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--noval", action="store_false")
    parser.add_argument("--notest", action="store_false")
    args = parser.parse_args()

    df = pd.read_csv('best_models.csv')
    MODEL_DIR  = df[(df["Splitter"] == args.splitter) & 
                    (df["Dim"] == args.dim) & 
                    (df["Blocks"] == args.blocks)]['Path'].values[0]
    OUTPUT_DIR = os.path.join('source/inference/st-gcn-plus-transitions', os.path.basename(os.path.dirname(MODEL_DIR)))
    ensure_dir(OUTPUT_DIR)
    NUM_WORKERS = 0
    DEVICE = 'cuda'
    STRIDE = 1
    VAL = args.noval
    TEST = args.notest

    # 1) Load saved training config
    cfg = load_config(MODEL_DIR)

    # 2) Reconstruct runtime objects (splitter/dataset cfg, model cfg, model)
    dataset_cfgclass, common_kwargs, val_paths, test_paths, model, model_cfg, joint_names = build_objects_from_config(cfg)

    # 3) Build inference datasets & loaders (VAL + TEST). Batch size must be 1 due to variable T_long.
    val_set,  val_loader  = build_inference_loader(dataset_cfgclass, common_kwargs, val_paths,  batch_size=1, num_workers=NUM_WORKERS)
    test_set,  test_loader = build_inference_loader(dataset_cfgclass, common_kwargs, test_paths, batch_size=1, num_workers=NUM_WORKERS)

    # 4) Load checkpoint safely (weights-only preferred, otherwise full checkpoint)
    ckpt_path = pick_checkpoint(MODEL_DIR)
    print(f"[INFO] Loading checkpoint: {os.path.basename(ckpt_path)}")
    state = torch.load(ckpt_path, map_location="cpu")
    try:
        # Try weights-only (bare state_dict)
        model.load_state_dict(state, strict=False)
    except Exception:
        # Fallbacks for full checkpoints
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        elif isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")

    device = torch.device(DEVICE)
    model.to(device)

    # 5) Window size = model's training T
    T_win = int(cfg["dataset"]["hyperparams"]["T"])
    stride = int(STRIDE)

    # 6) Run inference on both VAL and TEST
    if len(val_paths) > 0 and VAL:
        all_acc, all_bacc = run_split_inference("val",  val_loader, val_set,  model, device, T_win, stride, OUTPUT_DIR, args.video)
        mask = (
                df["Splitter"].eq(args.splitter)
                & df["Dim"].eq(args.dim)
                & df["Blocks"].eq(args.blocks)
               )
        metrics = {
                "val_acc":  np.mean(all_acc),
                "val_minacc": np.min(all_acc),
                "val_maxacc": np.max(all_acc),
                "val_rangeacc": np.max(all_acc)-np.min(all_acc),
                "val_bacc": np.mean(all_bacc),
                "val_minbacc": np.min(all_bacc),
                "val_maxbacc": np.max(all_bacc),
                "val_rangebacc": np.max(all_bacc)-np.min(all_bacc),
            }
        df.loc[mask, list(metrics)] = pd.Series(metrics).values   
    
    if len(test_paths) > 0 and TEST:
        all_acc, all_bacc = run_split_inference("test", test_loader, test_set, model, device, T_win, stride, OUTPUT_DIR, args.video)
        mask = (
                df["Splitter"].eq(args.splitter)
                & df["Dim"].eq(args.dim)
                & df["Blocks"].eq(args.blocks)
               )
        metrics = {
                "test_acc":  np.mean(all_acc),
                "test_minacc": np.min(all_acc),
                "test_maxacc": np.max(all_acc),
                "test_rangeacc": np.max(all_acc)-np.min(all_acc),
                "test_bacc": np.mean(all_bacc),
                "test_minbacc": np.min(all_bacc),
                "test_maxbacc": np.max(all_bacc),
                "test_rangebacc": np.max(all_bacc)-np.min(all_bacc),
            }
        df.loc[mask, list(metrics)] = pd.Series(metrics).values   
    
    df.to_csv('best_models.csv')
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(cfg, f)

    print(f"[DONE] Inference complete. Results saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
