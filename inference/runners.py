from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np


# ------------------------
# Config for base runner
# ------------------------
@dataclass
class InferenceRunnerConfig:
    batch_size: int = 1
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Config for evaluation runner
# ------------------------
@dataclass
class EvaluationRunnerConfig(InferenceRunnerConfig):
    target_dir: str = 'source/evaluation/unsorted_evaluation'
    compute_epe: bool = True
    compute_d1: bool = True
    compute_bad1: bool = True
    compute_bad2: bool = True
    compute_bad3: bool = True
    compute_rmse: bool = True
    compute_absrel: bool = True
    compute_fps: bool = True
    compute_memory: bool = True   # GPU peak memory usage, GPU only


# ------------------------
# Base inference runner
# ------------------------
class InferenceRunner:
    def __init__(self, cfg: InferenceRunnerConfig, model: torch.nn.Module, dataset):
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.dataset = dataset

        self.loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            pin_memory=True
        )

    @torch.no_grad()
    def run(self):
        self.model.eval()
        results = []

        for i, (sample, target) in enumerate(self.loader):
            left = sample["left"].to(self.cfg.device)
            right = sample["right"].to(self.cfg.device)
            # if left.ndim == 3:
            #     left = left.unsqueeze(0)
            # if right.ndim == 3:
            #     right = right.unsqueeze(0)
            
            pred = self.model(left=left, right=right)

            results.append({
                "pred": pred,
                "target": target,
            })

        return results


# ------------------------
# Extended evaluation runner
# ------------------------
class EvaluationRunner(InferenceRunner):
    def __init__(self, cfg: EvaluationRunnerConfig, model: torch.nn.Module, dataset):
        super().__init__(cfg, model, dataset)
        self.cfg = cfg
        os.makedirs(self.cfg.target_dir, exist_ok=True)

    
    def save_eval_results(self, metrics, results):
        # --------------------------------------------------------
        # 1) Make subdir name
        # --------------------------------------------------------
        H, W = self.dataset[0][0]["left"].shape[-2:]
        resize_to = self.dataset.cfg.resize_to
        mcfg = self.model.cfg
        # subdir = (
        #     f"H{H}xW{W}"
        #     f"_resize{resize_to}"
        #     f"_iters{mcfg['iters']}"
        #     f"_hier{mcfg['hierarchical']}"
        #     f"_sr{mcfg['small_ratio']}"
        #     f"_test{mcfg['test_mode']}"
        #     f"_lm{mcfg['low_memory']}"
        # )
        # out_dir = os.path.join(self.cfg.target_dir, subdir)
        # os.makedirs(out_dir, exist_ok=True)

        # --------------------------------------------------------
        # 2) Save numerical metrics
        # --------------------------------------------------------
        with open(os.path.join(self.cfg.target_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # --------------------------------------------------------
        # 3) Save 10 reproducible random examples
        # --------------------------------------------------------
        random.seed(42)  # ensure same samples each run
        idxs = random.sample(range(len(results)), min(10, len(results)))

        for k, idx in enumerate(idxs):
            rec = results[idx]
            pred = rec["pred"].detach().cpu().squeeze().numpy()
            gt = rec["gt"].squeeze().cpu().numpy() if rec["gt"] is not None else None
            left = rec["meta"]["left_path"]

            # Load left img for visualization
            left_img = plt.imread(left)

            # Error map
            err = None
            if gt is not None:
                err = np.abs(pred - gt)

            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(left_img)
            axs[0].set_title("Input")
            axs[1].imshow(gt, cmap="plasma") if gt is not None else axs[1].text(0.5,0.5,"No GT")
            axs[1].set_title("GT")
            axs[2].imshow(pred, cmap="plasma")
            axs[2].set_title("Prediction")
            if err is not None:
                im3 = axs[3].imshow(err, cmap="magma")
                axs[3].set_title("Error")
                axs
            for ax in axs: ax.axis("off")
            
            cbar = fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)
            cbar.set_label("Error (px)", rotation=270, labelpad=15)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg.target_dir, f"sample_{k:02d}.png"))
            plt.close(fig)


class FoundationStereoEvaluationRunner(EvaluationRunner):
    @torch.no_grad()
    def run(self):
        self.model.eval()
        results = []
        times: List[float] = []
        times_per_image: List[float] = []
        metrics: Dict[str, List[float]] = {m: [] for m in ["EPE", "D1", "Bad1", "Bad2", "Bad3", "RMSE", "AbsRel"]}

        for sample, target in self.loader: #tqdm(self.loader):
            left = sample["left"].to(self.cfg.device)   # [B,3,H,W]
            right = sample["right"].to(self.cfg.device) # [B,3,H,W]
            gt = target["disp"]
            mask = target["mask"]

            if gt is not None:
                gt = gt.to(self.cfg.device).float()     # [B,1,H,W]
                mask = mask.to(self.cfg.device)         # [B,H,W]

            # Timing
            start = time.time()
            pred = self.model(left=left, right=right)   # [B,1,H,W]
            if self.cfg.device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)

            # ---- Split batch into individual samples ----
            B = pred.shape[0]
            times_per_image.append((end-start)/B)
            for b in range(B):
                pred_b = pred[b].detach().cpu()   # [1,H,W]
                gt_b = gt[b].detach().cpu() if gt is not None else None
                mask_b = mask[b].detach().cpu() if gt is not None else None
                meta_b = {k: v[b] if isinstance(v, list) else v for k, v in target["meta"].items()} if "meta" in target else {}

                results.append({
                    "pred": pred_b,
                    "gt": gt_b,
                    "mask": mask_b,
                    "meta": meta_b,
                })

                # ---- Metrics per sample ----
                if gt_b is not None:
                    diff = pred_b - gt_b
                    abs_diff = torch.abs(diff)
                    mask_4d = mask_b.unsqueeze(0)  # [1,H,W]

                    if self.cfg.compute_epe:
                        metrics["EPE"].append(abs_diff[mask_4d].mean().item())

                    if self.cfg.compute_d1:
                        bad = ((abs_diff > 3.0) &
                            (abs_diff / gt_b > 0.05))[mask_4d].float().mean().item()
                        metrics["D1"].append(bad)

                    if self.cfg.compute_bad1:
                        metrics["Bad1"].append((abs_diff[mask_4d] > 1.0).float().mean().item())
                    if self.cfg.compute_bad2:
                        metrics["Bad2"].append((abs_diff[mask_4d] > 2.0).float().mean().item())
                    if self.cfg.compute_bad3:
                        metrics["Bad3"].append((abs_diff[mask_4d] > 3.0).float().mean().item())

                    if self.cfg.compute_rmse:
                        metrics["RMSE"].append(torch.sqrt((diff[mask_4d] ** 2).mean()).item())

                    if self.cfg.compute_absrel:
                        rel = (abs_diff / torch.clamp(gt_b, min=1e-6))[mask_4d]
                        metrics["AbsRel"].append(rel.mean().item())

        # ---- Summarize global metrics ----
        summary: Dict[str, Any] = {}
        if self.cfg.compute_fps:
            avg_time = sum(times_per_image) / len(times_per_image)
            summary["FPS"] = 1.0 / avg_time
            summary["ms/frame"] = avg_time * 1000

        if self.cfg.compute_memory and self.cfg.device.startswith("cuda"):
            summary["GPU_mem_MB"] = torch.cuda.max_memory_allocated() / (1024**2)
        elif self.cfg.compute_memory and self.cfg.device.startswith("cpu"):
            summary["GPU_mem_MB"] = -1

        for k, v in metrics.items():
            if len(v) > 0:
                summary[k] = sum(v) / len(v)

        self.save_eval_results(metrics=summary, results=results)
        return results, summary


class LightStereoEvaluationRunner(EvaluationRunner):
    @torch.no_grad()
    def run(self):
        self.model.eval()
        results = []
        times: List[float] = []
        times_per_image: List[float] = []
        metrics: Dict[str, List[float]] = {m: [] for m in ["EPE", "D1", "Bad1", "Bad2", "Bad3", "RMSE", "AbsRel"]}

        for sample, target in self.loader: #tqdm(self.loader):
            left = sample["left"].to(self.cfg.device)   # [B,3,H,W]
            right = sample["right"].to(self.cfg.device) # [B,3,H,W]
            gt = target["disp"]
            mask = target["mask"]

            if gt is not None:
                gt = gt.to(self.cfg.device).float()     # [B,1,H,W]
                mask = mask.to(self.cfg.device)         # [B,H,W]

            # Timing
            start = time.time()
            pred = self.model(left=left, right=right)   # [B,1,H,W]
            if self.cfg.device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)

            # ---- Split batch into individual samples ----
            B = pred.shape[0]
            times_per_image.append((end-start)/B)
            for b in range(B):
                pred_b = pred[b].detach().cpu()   # [1,H,W]
                gt_b = gt[b].detach().cpu() if gt is not None else None
                mask_b = mask[b].detach().cpu() if gt is not None else None
                meta_b = {k: v[b] if isinstance(v, list) else v for k, v in target["meta"].items()} if "meta" in target else {}

                results.append({
                    "pred": pred_b,
                    "gt": gt_b,
                    "mask": mask_b,
                    "meta": meta_b,
                })

                # ---- Metrics per sample ----
                if gt_b is not None:
                    diff = pred_b - gt_b
                    abs_diff = torch.abs(diff)
                    mask_4d = mask_b.unsqueeze(0)  # [1,H,W]

                    if self.cfg.compute_epe:
                        metrics["EPE"].append(abs_diff[mask_4d].mean().item())

                    if self.cfg.compute_d1:
                        bad = ((abs_diff > 3.0) &
                            (abs_diff / gt_b > 0.05))[mask_4d].float().mean().item()
                        metrics["D1"].append(bad)

                    if self.cfg.compute_bad1:
                        metrics["Bad1"].append((abs_diff[mask_4d] > 1.0).float().mean().item())
                    if self.cfg.compute_bad2:
                        metrics["Bad2"].append((abs_diff[mask_4d] > 2.0).float().mean().item())
                    if self.cfg.compute_bad3:
                        metrics["Bad3"].append((abs_diff[mask_4d] > 3.0).float().mean().item())

                    if self.cfg.compute_rmse:
                        metrics["RMSE"].append(torch.sqrt((diff[mask_4d] ** 2).mean()).item())

                    if self.cfg.compute_absrel:
                        rel = (abs_diff / torch.clamp(gt_b, min=1e-6))[mask_4d]
                        metrics["AbsRel"].append(rel.mean().item())

        # ---- Summarize global metrics ----
        summary: Dict[str, Any] = {}
        if self.cfg.compute_fps:
            avg_time = sum(times_per_image) / len(times_per_image)
            summary["FPS"] = 1.0 / avg_time
            summary["ms/frame"] = avg_time * 1000

        if self.cfg.compute_memory and self.cfg.device.startswith("cuda"):
            summary["GPU_mem_MB"] = torch.cuda.max_memory_allocated() / (1024**2)
        elif self.cfg.compute_memory and self.cfg.device.startswith("cpu"):
            summary["GPU_mem_MB"] = -1

        for k, v in metrics.items():
            if len(v) > 0:
                summary[k] = sum(v) / len(v)

        self.save_eval_results(metrics=summary, results=results)
        return results, summary






