from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import math
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import balanced_accuracy_score


@dataclass
class BaseTrainerConfig:
    # Core training params
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "AdamW"               # 'AdamW' or 'Adam'
    seed: int = 42
    device: str = "cuda"
    amp: bool = False                      # Automatic mixed precision (not enabled here)

    # Data loader options
    shuffle: bool = True
    num_workers: int = 12

    # Loss function
    loss_fn_name: str = "cross_entropy"    # or 'label_smoothing'

    # WandB integration
    wandb_project: str = "tennis-stgcn"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None

    # Validation/testing
    evaluate_on_test: bool = False

    # Checkpointing / early stopping
    checkpoint_dir: str = "checkpoints"
    save_checkpoint_fracs: List[float] = field(default_factory=lambda: [0.25, 0.50, 0.75, 1.0])
    early_stop_patience_frac: float = 0.15  # 15% of epochs (min 1)
    monitor_mode: str = "max"               # 'max' or 'min'
    monitor_name: str = "val/acc"           # name used in logs; best is chosen by val/acc

    # (Optional) resume training from a full checkpoint path
    resume_from: Optional[str] = None

    lr_scheduler: str = "cosineannealing"  # 'none' or 'cosineannealing'


class Metrics:
    @staticmethod
    def accuracy(target, prediction):
        target = np.asarray(target)
        prediction = np.asarray(prediction)
        return np.mean(target == prediction)

    @staticmethod
    def balanced_accuracy(target, prediction):
        return balanced_accuracy_score(
            np.asarray(target).reshape(-1),
            np.asarray(prediction).reshape(-1)
        )


class BaseTrainer:
    def __init__(self, model, trainset, valset, testset, config: BaseTrainerConfig):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.labels_format = getattr(self.trainset.config, "labels_format", "sequence_phases")

        # Optimizer
        if config.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        elif config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")

        # Loss function
        if config.loss_fn_name == "cross_entropy":
            self.loss_fn = torch.nn.functional.cross_entropy
        elif config.loss_fn_name == "label_smoothing":
            self.loss_fn = lambda logits, targets: torch.nn.functional.cross_entropy(
                logits, targets, label_smoothing=0.1
            )
        else:
            raise ValueError(f"Unknown loss_fn: {config.loss_fn_name}")

        # WandB
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            entity=config.wandb_entity,
            config=config.__dict__,
            group=config.wandb_group
        )
        wandb.define_metric("val/acc", summary="max")
        wandb.define_metric("test/acc", summary="max")

        # AMP (not enabled here)
        self.amp = config.amp
        if self.amp:
            raise NotImplementedError("AMP is not yet supported")
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

        # Checkpointing
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Epoch milestones for scheduled saves
        save_epochs = set()
        for frac in config.save_checkpoint_fracs:
            e = int(round(config.epochs * frac))
            e = min(max(e, 1), config.epochs)
            save_epochs.add(e)
        self.save_epochs = sorted(save_epochs)

        # Early stopping / best metric
        self.patience = max(1, int(round(config.early_stop_patience_frac * config.epochs)))
        self.best_metric = -float("inf") if config.monitor_mode == "max" else float("inf")
        self.best_epoch = 0
        self.best_ckpt_path: Optional[Path] = None
        self.num_bad_epochs = 0

        # Optional resume
        if config.resume_from:
            self._resume_from(Path(config.resume_from))

        # --- Scheduler ---
        if config.lr_scheduler.lower() == "none":
            self.scheduler = None
        elif config.lr_scheduler.lower() == "cosineannealing":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=1e-5  # fixed eta_min
            )
        else:
            raise NotImplementedError(f"Unknown lr_scheduler: {config.lr_scheduler}")

    def _is_improvement(self, current, best) -> bool:
        return current > best if self.config.monitor_mode == "max" else current < best

    def _save_checkpoint_pair(self, epoch: int, val_metric: Optional[float], tag: str) -> Path:
        """
        Saves two files:
        - Full checkpoint dict (for resume).
        - Weights-only file (for safe loading in PyTorch>=2.6).
        Returns the path to the full checkpoint.
        """
        # Full checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metric": val_metric,
            "config": self.config.__dict__,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        path_full = self.ckpt_dir / f"epoch{epoch:04d}_{tag}.pt"
        torch.save(ckpt, path_full)

        # Weights-only
        path_weights = self.ckpt_dir / f"epoch{epoch:04d}_{tag}_weights.pt"
        torch.save(self.model.state_dict(), path_weights)

        return path_full

    def _resume_from(self, path: Path):
        """
        Resume training from a *full* checkpoint (dict).
        Handles PyTorch 2.6+ default (weights_only=True) by forcing weights_only=False here.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and state.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        # Restore best metric if present (optional)
        if "val_metric" in state and state["val_metric"] is not None:
            if self._is_improvement(state["val_metric"], self.best_metric):
                self.best_metric = state["val_metric"]
        wandb.log({"resume/from": str(path)})

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for data, target in dataloader:
            data, target = data.to(self.device).squeeze(-1), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data)                       # (N, C, T)
            
            if self.labels_format == "sequence_phases":
                # --- old behaviour (per-frame class labels) ---
                output_reshaped = output.permute(0, 2, 1).reshape(-1, output.shape[1])  # (N*T,C)
                target_reshaped = target.reshape(-1)  # (N*T,)
                loss = self.loss_fn(output_reshaped, target_reshaped)
            elif self.labels_format == "gaussian_heatmaps":
                # --- new behaviour (transition logits vs Gaussian maps) ---
                # output: (N,4,T), target: (N,4,T)
                target = target.permute(0, 2, 1).contiguous()
                target = target.to(dtype=output.dtype)
                loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fn(output, target)
            else:
                raise NotImplementedError(f"Unknown labels_format={self.labels_format}")

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # break #! remove  for debug only

        return total_loss / max(1, len(dataloader))

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        count = 0
        all_targets, all_preds = [], []
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device).squeeze(-1), target.to(self.device)
                output = self.model(data)  # (N, C, T)
                
                if self.labels_format == "sequence_phases":
                    # Loss on per-frame logits
                    output_reshaped = output.permute(0, 2, 1).reshape(-1, output.shape[1])
                    target_reshaped = target.reshape(-1)
                    loss = torch.nn.functional.cross_entropy(output_reshaped, target_reshaped)
                    total_loss += loss.item()

                    # Predictions per frame: argmax over class dim (dim=1)
                    pred = output.argmax(dim=1)  # (N, T)
                    all_preds.append(pred.detach().cpu().numpy())
                    all_targets.append(target.detach().cpu().numpy())

                    correct += (pred == target).sum().item()
                    count += target.numel()
                
                elif self.labels_format == "gaussian_heatmaps":
                    # Loss
                    target = target.permute(0, 2, 1).contiguous()
                    target = target.to(dtype=output.dtype)
                    loss_fn = torch.nn.BCEWithLogitsLoss()
                    loss = loss_fn(output, target)
                    total_loss += loss.item()

                    # For metrics: take argmax over time per transition, then map to phase sequence
                    preds_seq = self.trainset._decode_transition_heatmaps_to_phase_sequence(output.sigmoid().cpu().numpy())
                    targs_seq = self.trainset._decode_transition_heatmaps_to_phase_sequence(target.cpu().numpy())

                    all_preds.extend(preds_seq)
                    all_targets.extend(targs_seq)
                else:
                    raise NotImplementedError(f"Unknown labels_format={self.labels_format}")

        avg_loss = total_loss / max(1, len(dataloader))
        if self.labels_format == "sequence_phases":
            accuracy = correct / count if count else 0.0
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            bacc = Metrics.balanced_accuracy(all_targets, all_preds)
        elif self.labels_format == "gaussian_heatmaps":  
            # Already converted to sequences above → can use discrete metrics
            accuracy = Metrics.accuracy(all_targets, all_preds)
            bacc = Metrics.balanced_accuracy(all_targets, all_preds)

        return avg_loss, bacc

    def fit(self):
        trainloader = DataLoader(
            self.trainset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True if str(self.device).startswith("cuda") else False,
        )
        valloader = DataLoader(
            self.valset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if str(self.device).startswith("cuda") else False,
        ) if self.valset else None

        final_epoch_ran = 0
        for epoch in tqdm(range(1, self.config.epochs + 1), desc="Training"):
            final_epoch_ran = epoch

            # ---- Train ----
            train_loss = self.train_epoch(trainloader)
            wandb.log({"train/loss": train_loss, "epoch": epoch})

            # ---- Validate ----
            val_metric_for_monitor = None
            if valloader:
                val_loss, val_bacc = self.evaluate(valloader)
                wandb.log({"val/loss": val_loss, "val/acc": val_bacc, "epoch": epoch})
                val_metric_for_monitor = val_bacc

                # Track best & early stopping
                if self._is_improvement(val_bacc, self.best_metric):
                    self.best_metric = val_bacc
                    self.best_epoch = epoch
                    self.num_bad_epochs = 0
                    self.best_ckpt_path = self._save_checkpoint_pair(epoch, val_bacc, tag="best")
                    wandb.log({"best/val_acc": val_bacc, "best/epoch": epoch})
                else:
                    self.num_bad_epochs += 1
                    if self.num_bad_epochs >= self.patience:
                        wandb.log({"early_stopping/triggered_at_epoch": epoch})
                        # Also save a final pair for where we stopped
                        self._save_checkpoint_pair(epoch, val_metric_for_monitor, tag="stopped")
                        break

            # ---- Scheduled checkpoint saves ----
            if epoch in self.save_epochs:
                self._save_checkpoint_pair(epoch, val_metric_for_monitor, tag="scheduled")

            # ---- Step scheduler (if any) ----  
            if self.scheduler is not None:
                self.scheduler.step()
                wandb.log({"lr": self.optimizer.param_groups[0]['lr'], "epoch": epoch})  

        # ---- Final save (last epoch reached or early stop) ----
        # Save a "last" pair pointing to best if we have one, else this epoch
        final_epoch_for_save = self.best_epoch if self.best_ckpt_path is not None else final_epoch_ran
        final_val_metric = self.best_metric if self.best_ckpt_path is not None else None
        final_ckpt_path = self._save_checkpoint_pair(final_epoch_for_save, final_val_metric, tag="last")

        # ---- Evaluate TEST on BEST checkpoint ----
        if self.config.evaluate_on_test and self.testset:
            testloader = DataLoader(
                self.testset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True if str(self.device).startswith("cuda") else False,
            )

            load_full = self.best_ckpt_path if self.best_ckpt_path is not None else final_ckpt_path
            load_weights = Path(str(load_full).replace(".pt", "_weights.pt"))

            if load_weights.exists():
                # Preferred path for PyTorch >= 2.6 (safe by default)
                state_dict = torch.load(load_weights, map_location=self.device)  # weights_only=True by default
                self.model.load_state_dict(state_dict)
                wandb.log({"test/loaded": "weights_only", "test/ckpt": str(load_weights)})
            else:
                # Fallback to full checkpoint (requires weights_only=False)
                state = torch.load(load_full, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state["model_state_dict"])
                wandb.log({"test/loaded": "full_ckpt", "test/ckpt": str(load_full)})

            test_loss, test_bacc = self.evaluate(testloader)
            wandb.log({"test/loss": test_loss, "test/acc": test_bacc})

        wandb.finish()


class RNNTrainer(BaseTrainer):
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss_sum = 0.0   # sum over ALL elements seen this epoch
        total_elems    = 0     # number of elements the loss averages over

        for data, mask, target in dataloader:
            data   = data.squeeze(-1).to(self.device)   # assume correct shapes from the loader
            mask   = mask.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)       # sequence_phases: (N,C,T) | gaussian_heatmaps: (N,4,T)

            if self.labels_format == "sequence_phases":
                # output: (N,C,T), target: (N,T) with class ids
                N, C, T = output.shape
                logits_flat = output.permute(0, 2, 1).reshape(-1, C)  # (N*T, C)
                target_flat = target.reshape(-1)                      # (N*T,)
                mask_flat   = mask.reshape(-1).to(dtype=torch.bool)

                logits_valid = logits_flat[mask_flat]                  # (num_valid, C)
                target_valid = target_flat[mask_flat]                  # (num_valid,)
                loss = torch.nn.functional.cross_entropy(logits_valid, target_valid, reduction="sum")
                elems = int(mask_flat.sum().item())

            elif self.labels_format == "gaussian_heatmaps":
                # Ensure (N,4,T_max) and match dtype
                if target.dim() == 3 and target.shape[1] != 4 and target.shape[2] == 4:
                    target = target.permute(0, 2, 1).contiguous()
                target = target.to(dtype=output.dtype)                    # (N,4,T_max)

                # Elementwise BCE, then zero-out padded timesteps
                loss_map = torch.nn.functional.binary_cross_entropy_with_logits(
                    output, target, reduction="none"
                )                                                         # (N,4,T_max)
                loss_map = loss_map * mask.unsqueeze(1)                   # broadcast mask over channels

                loss = loss_map.sum()
                elems = int(mask.sum().item() * output.size(1))           # valid_frames * 4 channels

            else:
                raise NotImplementedError(f"Unknown labels_format={self.labels_format}")

            loss.backward()
            self.optimizer.step()

            total_loss_sum += float(loss.item())
            total_elems    += elems

        # Average loss per element across the epoch (robust to variable T)
        return total_loss_sum / max(1, total_elems)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss_sum = 0.0
        total_elems    = 0
        all_targets, all_preds = [], []

        with torch.no_grad():
            for data, mask, target in dataloader:
                data   = data.squeeze(-1).to(self.device)    # (N,C,T_max,V[,M])
                mask   = mask.to(self.device)    # (N,T_max) bool/0-1
                target = target.to(self.device)  # phases: (N,T_max) | heatmaps: (N,4,T_max) or (N,T_max,4)

                output = self.model(data)        # phases: (N,C,T_max) | heatmaps: (N,4,T_max)

                if self.labels_format == "sequence_phases":
                    # ----- Loss (masked) -----
                    N, C, Tm = output.shape
                    logits_flat = output.permute(0, 2, 1).reshape(-1, C)   # (N*Tm, C)
                    target_flat = target.reshape(-1)                       # (N*Tm,)
                    mask_flat   = mask.reshape(-1).to(dtype=torch.bool)    # (N*Tm,)

                    logits_valid = logits_flat[mask_flat]                  # (num_valid, C)
                    target_valid = target_flat[mask_flat]                  # (num_valid,)
                    loss = torch.nn.functional.cross_entropy(logits_valid, target_valid, reduction="sum")
                    elems = int(mask_flat.sum().item())

                    total_loss_sum += float(loss.item())
                    total_elems    += elems

                    # ----- Metrics (masked) -----
                    pred = output.argmax(dim=1)                            # (N, T_max)
                    preds_valid = pred[mask].detach().cpu().numpy()
                    targs_valid = target[mask].detach().cpu().numpy()
                    all_preds.append(preds_valid)
                    all_targets.append(targs_valid)

                elif self.labels_format == "gaussian_heatmaps":
                    # Ensure target is (N,4,T_max) and correct dtype
                    if target.dim() == 3 and target.shape[1] != 4 and target.shape[2] == 4:
                        target = target.permute(0, 2, 1).contiguous()
                    target = target.to(dtype=output.dtype)                 # (N,4,T_max)

                    # ----- Loss (masked) -----
                    loss_map = torch.nn.functional.binary_cross_entropy_with_logits(
                        output, target, reduction="none"
                    )                                                      # (N,4,T_max)
                    loss_map = loss_map * mask.unsqueeze(1)                # zero-out pads
                    loss = loss_map.sum()
                    elems = int(mask.sum().item() * output.size(1))        # valid_frames * 4 channels

                    total_loss_sum += float(loss.item())
                    total_elems    += elems

                    # ----- Metrics: decode sequences (mask out pads first) -----
                    probs = output.sigmoid() * mask.unsqueeze(1)           # (N,4,T_max)
                    preds_seq = self.trainset._decode_transition_heatmaps_to_phase_sequence(
                        probs.cpu().numpy()
                    )
                    # Targets may already be zero on pads; keep shape (N,4,T_max)
                    targs_seq = self.trainset._decode_transition_heatmaps_to_phase_sequence(
                        target.cpu().numpy()
                    )
                    all_preds.extend(preds_seq)
                    all_targets.extend(targs_seq)

                else:
                    raise NotImplementedError(f"Unknown labels_format={self.labels_format}")

        avg_loss = total_loss_sum / max(1, total_elems)

        if self.labels_format == "sequence_phases":
            # Concatenate masked per-frame arrays
            all_preds   = np.concatenate(all_preds) if len(all_preds) else np.array([])
            all_targets = np.concatenate(all_targets) if len(all_targets) else np.array([])
            bacc = Metrics.balanced_accuracy(all_targets, all_preds) if all_targets.size else 0.0
        else:  # gaussian_heatmaps
            bacc = Metrics.balanced_accuracy(all_targets, all_preds) if len(all_targets) else 0.0

        return avg_loss, bacc

