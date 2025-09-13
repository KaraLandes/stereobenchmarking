# torch_models/lightstereo.py
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import os
import sys

# Import from openstereo repo

# -----------------------------------------------------------------------------
# Locate the official FoundationStereo repo and import the model
# -----------------------------------------------------------------------------
def _append_repo_path() -> str:
    """
    Always expect OpenStereo repo at ./third_party/openstereo
    relative to this file.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_path = os.path.join(here, "third_party", "openstereo")
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"Expected OpenStereo repo symlink at: {repo_path}"
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    return repo_path


_REPO_ROOT = _append_repo_path()

try:
    # Official class lives here in the repo you showed
    from stereo.modeling.models.lightstereo.lightstereo import LightStereo as OS_LightStereo
except Exception as e:
    raise ImportError(
        f"Failed to import FoundationStereo from: {_REPO_ROOT}\nOriginal error: {e}"
    )


@dataclass
class LightStereoConfig:
    # Main hyperparameters
    max_disp: int = 192
    left_att: bool = True
    backbone: str = "MobileNetv2"  # or "EfficientNetv2"
    aggregation_blocks: tuple = (2, 2, 2)  # matches openstereo cfgs.AGGREGATION_BLOCKS
    expanse_ratio: int = 2

    # Runtime options
    device: str = "cuda"
    eval_mode: bool = True
    weights: Optional[str] = None  # path to pretrained weights

    # Forward control (not strictly needed but matches your style)
    test_mode: bool = True

    # Extra kwargs (to be flexible)
    extra: Dict[str, Any] = field(default_factory=dict)


class LightStereoModel(nn.Module):
    def __init__(self, cfg: LightStereoConfig):
        super().__init__()
        self.cfg = cfg

        # Convert dataclass → dict in openstereo style
        os_cfg = {
            "MAX_DISP": cfg.max_disp,
            "LEFT_ATT": cfg.left_att,
            "BACKCONE": cfg.backbone,
            "AGGREGATION_BLOCKS": cfg.aggregation_blocks,
            "EXPANSE_RATIO": cfg.expanse_ratio,
        }
        os_cfg.update(cfg.extra)

        self.net = OS_LightStereo(os_cfg)

        if cfg.weights is not None:
            self.load_weights(cfg.weights)

        if cfg.eval_mode:
            self.eval()

        self.to(cfg.device)

    def load_weights(self, path: str, strict: bool = False):
        print(f"Loading LightStereo weights from {path}")
        state = torch.load(path, map_location="cpu")
        # handle checkpoints saved with trainer dict wrapping
        if "model" in state:
            state = state["model"]
        self.net.load_state_dict(state, strict=strict)

    @torch.no_grad()
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        data = {"left": left, "right": right}
        out = self.net(data)  # openstereo LightStereo returns dict
        disp = out["disp_pred"]  # [B, 1, H, W]
        return disp
