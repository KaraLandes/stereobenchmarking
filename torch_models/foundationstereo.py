# foundationstereo.py
from __future__ import annotations
import os, sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Locate the official FoundationStereo repo and import the model
# -----------------------------------------------------------------------------
def _append_repo_path() -> str:
    """
    Always expect FoundationStereo repo at ./third_party/foundationstereo
    relative to this file.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_path = os.path.join(here, "third_party", "foundationstereo")
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"Expected FoundationStereo repo symlink at: {repo_path}"
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    return repo_path


_REPO_ROOT = _append_repo_path()

try:
    # Official class lives here in the repo you showed
    from core.foundation_stereo import FoundationStereo as _OfficialFoundationStereo
except Exception as e:
    raise ImportError(
        f"Failed to import FoundationStereo from: {_REPO_ROOT}\nOriginal error: {e}"
    )


# -----------------------------------------------------------------------------
# Dataclass config (matches your registry pattern)
# -----------------------------------------------------------------------------
@dataclass
class FoundationStereoConfig:
    # Geometry / correlation
    max_disp: int = 192
    corr_radius: int = 4
    corr_levels: int = 2

    # GRU pyramid
    n_gru_layers: int = 3          # 1–3
    n_downsample: int = 3

    # Hidden/context dims per GRU level (len must equal n_gru_layers)
    hidden_dims: Optional[List[int]] = None

    # Vision foundation backbone
    vit_size: str = "vitl"         # {'vits','vitb','vitl'}
    mixed_precision: bool = True
    low_memory: bool = False       # default forward flag

    # Runtime
    device: str = "cuda"
    eval_mode: bool = True
    weights: Optional[str] = None  # path to .pth/.pt (state_dict or wrapped)

    # Forward_call
    iters: int = 12
    hierarchical: bool = True
    small_ratio: float = 0.5
    test_mode: bool = True
    low_memory: Optional[bool] = None

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128, 128]
        if len(self.hidden_dims) != self.n_gru_layers:
            raise ValueError(
                f"hidden_dims length ({len(self.hidden_dims)}) must equal n_gru_layers ({self.n_gru_layers})"
            )

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)
    
    # The official model reads args like a dict; provide a compatible mapping.
    def to_args(self) -> Dict[str, Any]:
        return {
            "max_disp": self.max_disp,
            "corr_radius": self.corr_radius,
            "corr_levels": self.corr_levels,
            "n_gru_layers": self.n_gru_layers,
            "n_downsample": self.n_downsample,
            "hidden_dims": self.hidden_dims,
            "vit_size": self.vit_size,
            "mixed_precision": self.mixed_precision,
            "low_memory": self.low_memory,
            # accessors used in various places
            "__getitem__": lambda k: self.__dict__[k],
            "get": lambda k, default=None: self.__dict__.get(k, default),
        }


# -----------------------------------------------------------------------------
# Wrapper nn.Module that fits your registry ('class': <nn.Module subclass>)
# -----------------------------------------------------------------------------
class FoundationStereoModel(nn.Module):
    """
    Thin nn.Module wrapper around the official FoundationStereo to fit your
    registry: the constructor accepts a dataclass config, and forward() takes
    left/right in [0,1] and returns disparity [B,1,H,W] in pixels.
    """
    def __init__(self, cfg: FoundationStereoConfig):
        super().__init__()
        self.cfg = cfg
        # args = cfg.to_args()
        self.net = _OfficialFoundationStereo(cfg)#! args 

        if cfg.weights:
            self.load_weights(cfg.weights)

        device = torch.device(cfg.device)
        self.net.to(device)
        if cfg.eval_mode:
            self.net.eval()

    def load_weights(self, path: str, strict: bool = True):
        state = torch.load(path, map_location="cpu", weights_only=False)

        # handle different checkpoint formats
        if "model" in state:
            state_dict = state["model"]
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state  # assume already a pure state_dict

        missing, unexpected = self.net.load_state_dict(state_dict, strict=strict)
        print(f"Loaded weights from {path}")
        if missing:
            print("  Missing keys:", missing)
        if unexpected:
            print("  Unexpected keys:", unexpected)

    @torch.inference_mode()
    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            left,right: [B,3,H,W] float in [0,1]
            iters: GRU iterations
            hierarchical: use run_hierachical (coarse-to-fine) if available
        Returns:
            disp: [B,1,H,W] float disparity in pixels
        """
        iters = self.cfg['iters']
        hierarchical = self.cfg['hierarchical']
        small_ratio = self.cfg['small_ratio']
        test_mode = self.cfg['test_mode']
        low_memory = self.cfg['low_memory']

        dev = next(self.net.parameters()).device
        left = left.to(dev, non_blocking=True)
        right = right.to(dev, non_blocking=True)

        # The official code expects 0–255 RGB and normalizes internally.
        left_255 = (left.clamp(0, 1) * 255.0).float()
        right_255 = (right.clamp(0, 1) * 255.0).float()

        lm = self.cfg.low_memory if low_memory is None else low_memory

        if hierarchical and hasattr(self.net, "run_hierachical"):
            disp = self.net.run_hierachical(
                left_255, right_255, iters=iters, test_mode=True, low_memory=lm, small_ratio=small_ratio
            )
        else:
            disp = self.net(left_255, right_255, iters=iters, test_mode=test_mode, low_memory=lm)

        if disp.dim() == 3:  # [B,H,W] -> [B,1,H,W]
            disp = disp.unsqueeze(1)
        return disp


