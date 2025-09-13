# kitti12_base_dataset.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


# -----------------------------
# Small helpers
# -----------------------------
def _first_existing_dir(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.is_dir():
            return p
    return None

def _pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    """PIL (H,W,C uint8) -> float32 tensor [C,H,W] in [0,1]."""
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[..., None]
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t

def _load_disp_kitti16(path: Path) -> torch.Tensor:
    """
    Read KITTI disparity PNG (uint16, disparity*256).
    Returns float32 tensor shape [1,H,W] with disparity in pixels.
    """
    disp = Image.open(path)
    disp_np = np.array(disp, dtype=np.uint16)
    disp_f = disp_np.astype(np.float32) / 256.0
    disp_t = torch.from_numpy(disp_f)[None, ...]  # [1,H,W]
    return disp_t

def _resize_pair_and_disp(
    left: Image.Image, right: Image.Image, disp: Optional[torch.Tensor], size_hw: Tuple[int, int]
) -> Tuple[Image.Image, Image.Image, Optional[torch.Tensor]]:
    """
    Resize images to (H_new, W_new). Disparity is scaled by W_new/W_old.
    Uses bilinear for RGB and nearest for disparity.
    """
    H_new, W_new = size_hw
    W_old, H_old = left.size
    sx = W_new / float(W_old)

    left_r = left.resize((W_new, H_new), Image.BILINEAR)
    right_r = right.resize((W_new, H_new), Image.BILINEAR)

    if disp is None:
        return left_r, right_r, None

    # disp: [1,H,W] float; resize with nearest to keep discrete values, then scale by sx
    disp_np = disp.squeeze(0).numpy()
    disp_img = Image.fromarray(disp_np.astype(np.float32), mode="F")
    disp_resized = disp_img.resize((W_new, H_new), Image.NEAREST)
    disp_resized_np = np.array(disp_resized, dtype=np.float32) * sx
    disp_t = torch.from_numpy(disp_resized_np)[None, ...]  # [1,H,W]
    return left_r, right_r, disp_t

def _read_calib_dict(calib_dir: Path, stem: str) -> Dict[str, Union[np.ndarray, float, str]]:
    """
    Tries a few common calib filename patterns.
    Returns {} if not found. Parses simple "key: values" lines into arrays/floats.
    """
    candidates = [
        calib_dir / f"{stem}.txt",
        calib_dir / f"{stem[:-3]}.txt" if len(stem) > 3 else calib_dir / f"{stem}.txt",
    ]
    for p in candidates:
        if p.is_file():
            try:
                lines = p.read_text().strip().splitlines()
            except Exception:
                break
            out: Dict[str, Union[np.ndarray, float, str]] = {}
            for ln in lines:
                if ":" not in ln:
                    continue
                k, v = ln.split(":", 1)
                v = v.strip()
                # Try parse list of floats
                try:
                    arr = np.fromstring(v, sep=" ")
                    out[k.strip()] = arr
                except Exception:
                    # fallback raw string
                    out[k.strip()] = v
            return out
    return {}


# -----------------------------
# Config (optional but handy)
# -----------------------------
@dataclass
class Kitti12Config:
    root: Union[str, Path]                 # e.g., data/kitti12
    split: str = "training"                # "training" or "testing"
    use_color: bool = True                 # prefer colored_0/1 over image_0/1
    gt_type: str = "disp_noc"              # "disp_noc" or "disp_occ" (ignored for testing)
    resize_to: Optional[Tuple[int, int]] = None  # (H,W). If None: native resolution.
    # Optional hooks (run AFTER resize)
    sample_transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None
    target_transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None
    augmentation: Optional[Callable[[Dict, Dict], Tuple[Dict, Dict]]] = None  # (sample, target) -> (sample, target)


# -----------------------------
# Dataset
# -----------------------------
class Kitti12BaseDataset(Dataset):
    """
    Minimal, robust KITTI 2012 dataset for stereo benchmarking.
    __getitem__ returns (sample, target):

      sample = {
        "left":  FloatTensor [C,H,W] in [0,1],
        "right": FloatTensor [C,H,W] in [0,1],
      }

      target = {
        "disp":  FloatTensor [1,H,W] in px (or None for testing),
        "mask":  BoolTensor [H,W] where valid disparity > 0 (or None for testing),
        "calib": dict with parsed calibration (may be empty if file missing),
        "meta":  dict with paths, basename, index, split
      }

    Folder auto-detection:
      - left:  training/{colored_0 or image_0}
      - right: training/{colored_1 or image_1}
      - gt:    training/{disp_noc or disp_occ}
      - calib: {split}/calib

    Disparity scaling:
      - KITTI disparity pngs are uint16 with disparity * 256. We divide by 256.0.
    """

    def __init__(self, cfg: Kitti12Config):
        super().__init__()
        self.cfg = cfg
        self.root = Path(cfg.root)
        assert cfg.split in ("training", "testing"), "split must be 'training' or 'testing'"

        split_dir = self.root / cfg.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Choose left/right folders
        if cfg.use_color:
            left_dir = _first_existing_dir([split_dir / "colored_0", split_dir / "image_0"])
            right_dir = _first_existing_dir([split_dir / "colored_1", split_dir / "image_1"])
        else:
            left_dir = _first_existing_dir([split_dir / "image_0", split_dir / "colored_0"])
            right_dir = _first_existing_dir([split_dir / "image_1", split_dir / "colored_1"])

        if left_dir is None or right_dir is None:
            raise FileNotFoundError(
                f"Could not find stereo folders under {split_dir}. "
                f"Tried colored_0/image_0 and colored_1/image_1."
            )

        self.left_dir = left_dir
        self.right_dir = right_dir
        self.calib_dir = split_dir / "calib"
        self.has_calib = self.calib_dir.is_dir()

        # GT disparity (train only)
        self.disp_dir: Optional[Path] = None
        if cfg.split == "training":
            cand = split_dir / cfg.gt_type
            if not cand.is_dir():
                alt = "disp_occ" if cfg.gt_type == "disp_noc" else "disp_noc"
                if (split_dir / alt).is_dir():
                    cand = split_dir / alt
                else:
                    raise FileNotFoundError(
                        f"GT disparity folder not found: {split_dir / cfg.gt_type} (or '{alt}')"
                    )
            self.disp_dir = cand

        # Build file list using left images as anchor
        left_pngs = sorted(self.left_dir.glob("*.png"))

        # ⚡ Only keep *_10.png frames (GT exists only for those in KITTI12 training)
        if cfg.split == "training":
            left_pngs = [p for p in left_pngs if p.stem.endswith("_10")]

        if len(left_pngs) == 0:
            raise FileNotFoundError(f"No valid *_10.png left images found in {self.left_dir}")

        # Ensure right images exist for each left
        self.samples: List[Dict[str, Path]] = []
        for lp in left_pngs:
            name = lp.name
            rp = self.right_dir / name.replace("_10", "_11")
            if not rp.is_file():
                raise FileNotFoundError(f"Right image missing for {lp}: {rp}")

            dp = None
            if self.disp_dir is not None:
                dp = self.disp_dir / name
                if not dp.is_file():
                    raise FileNotFoundError(f"Disparity GT missing for {lp}: {dp}")

            self.samples.append(
                {
                    "left": lp,
                    "right": rp,
                    "disp": dp,  # None in testing
                    "stem": lp.stem,
                }
            )
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rec = self.samples[index]
        left = Image.open(rec["left"]).convert("RGB")
        right = Image.open(rec["right"]).convert("RGB")

        disp_t: Optional[torch.Tensor] = None
        mask_t: Optional[torch.Tensor] = None

        if rec["disp"] is not None:
            disp_t = _load_disp_kitti16(rec["disp"])  # [1,H,W] float (px)
            mask_t = (disp_t > 0.0).squeeze(0)        # [H,W] bool

        # Optional resize
        if self.cfg.resize_to is not None:
            left, right, disp_t = _resize_pair_and_disp(left, right, disp_t, self.cfg.resize_to)
            if disp_t is not None:
                mask_t = (disp_t > 0.0).squeeze(0)

        # Convert images to tensors in [0,1]
        left_t = _pil_to_tensor01(left)
        right_t = _pil_to_tensor01(right)

        sample: Dict[str, torch.Tensor] = {"left": left_t, "right": right_t}
        target: Dict[str, Union[torch.Tensor, Dict, str, int]] = {
            "disp": disp_t,
            "mask": mask_t,
            "calib": {},
            "meta": {
                "index": index,
                "split": self.cfg.split,
                "basename": rec["stem"],
                "left_path": str(rec["left"]),
                "right_path": str(rec["right"]),
                "disp_path": str(rec["disp"]) if rec["disp"] is not None else None,
            },
        }

        # Calibration (best-effort)
        if self.has_calib:
            calib = _read_calib_dict(self.calib_dir, rec["stem"])
            target["calib"] = calib

        # Optional augmentation (expects/returns dicts of tensors)
        if self.cfg.augmentation is not None:
            sample, target = self.cfg.augmentation(sample, target)

        # Optional transforms
        if self.cfg.sample_transform is not None:
            sample = self.cfg.sample_transform(sample)
        if self.cfg.target_transform is not None:
            target = self.cfg.target_transform(target)

        return sample, target


# -----------------------------
# A safe default collate_fn
# -----------------------------
def kitti12_collate(batch: List[Tuple[Dict, Dict]]):
    """
    Collate that stacks tensors of same size, leaves metas/calib in lists.
    If you use variable sizes (no resize), prefer batch_size=1 or add your own padding.
    """
    samples, targets = list(zip(*batch))
    out_sample = {k: torch.stack([s[k] for s in samples], dim=0) for k in samples[0].keys()}

    out_target: Dict[str, Union[torch.Tensor, List, Dict]] = {}
    # Stack tensors where possible
    for k in targets[0].keys():
        vals = [t[k] for t in targets]
        if isinstance(vals[0], torch.Tensor) and all((v is not None for v in vals)):
            out_target[k] = torch.stack(vals, dim=0)
        else:
            out_target[k] = vals  # e.g., mask None on test, calib/meta dicts
    return out_sample, out_target
