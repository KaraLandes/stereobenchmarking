from torch_models import *
from classes_collections import *
from utils import *

from pathlib import Path
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore")


from configurators.inference_conf_lightstereo import ALL_CONFIGS

for INFERENCE_CONFIG in tqdm(ALL_CONFIGS):
    try:
        # --------------------------------------------------------
        # Resolve model registry entry
        # --------------------------------------------------------
        model_name = INFERENCE_CONFIG['model']['name']
        model_entry = COLLECTION['models'][model_name]
        ModelCfgClass = model_entry['cfgclass']
        ModelClass = model_entry['class']
        # --------------------------------------------------------
        # Case: instantiate from checkpoint directory
        # --------------------------------------------------------
        if INFERENCE_CONFIG['model']['from_chkpt']['on']:
            chkpt_dir = Path(INFERENCE_CONFIG['model']['from_chkpt']['chkpt_path'])
            cfg_file = next(
                (f for ext in ("*.yml", "*.yaml", "*.json") for f in chkpt_dir.glob(ext)),
                None
            )
            if cfg_file is None:
                raise FileNotFoundError(f"No config file (.yml/.yaml/.json) found in {chkpt_dir}")

            ckpt_file = next(
                (f for ext in ("*.pth", "*.ckpt") for f in chkpt_dir.glob(ext)),
                None
            )
            if ckpt_file is None:
                raise FileNotFoundError(f"No checkpoint file (.pth/.ckpt) found in {chkpt_dir}")

            # Load YAML config
            cfg_raw = OmegaConf.load(cfg_file)

            # Map to dataclass
            if model_name == 'FoundationStereoModel':
                cfg = ModelCfgClass(
                        max_disp=int(cfg_raw.get("max_disp", 192)),
                        corr_radius=int(cfg_raw.get("corr_radius", 4)),
                        corr_levels=int(cfg_raw.get("corr_levels", 2)),
                        n_gru_layers=int(cfg_raw.get("n_gru_layers", 3)),
                        n_downsample=int(cfg_raw.get("n_downsample", 3)),
                        hidden_dims=list(cfg_raw.get("hidden_dims", [128, 128, 128])),
                        vit_size=str(cfg_raw.get("vit_size", "vitl")),
                        mixed_precision=bool(cfg_raw.get("mixed_precision", True)),
                        low_memory=bool(cfg_raw.get("low_memory", False)),
                        weights=str(ckpt_file),
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
            elif model_name == 'LightStereoModel':
                cfg = ModelCfgClass(
                        max_disp=int(cfg_raw.MODEL.get("MAX_DISP", 192)),
                        left_att=bool(cfg_raw.MODEL.get("LEFT_ATT", True)),
                        backbone=str(cfg_raw.MODEL.get("BACKBONE", "MobileNetv2")),
                        aggregation_blocks=tuple(cfg_raw.MODEL.get("AGGREGATION_BLOCKS", [1, 2, 4])),
                        expanse_ratio=int(cfg_raw.MODEL.get("EXPANSE_RATIO", 2)),
                        weights=str(ckpt_file),
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        eval_mode=bool(INFERENCE_CONFIG['model']['from_chkpt'].get("eval_mode", True)),
                        test_mode=bool(INFERENCE_CONFIG['model']['from_chkpt'].get("test_mode", True)),
                    )

            model = ModelClass(cfg)

        # --------------------------------------------------------
        # Case: (future) instantiate directly from hyperparams
        # --------------------------------------------------------
        elif INFERENCE_CONFIG['model']['from_hyperparams']['on']:
            raise NotImplementedError("Hyperparams-based instantiation not implemented yet")

        else:
            raise RuntimeError("No valid instantiation mode selected in INFERENCE_CONFIG")

        print(f"Loaded model: {model_name}")
        print(f"    from: {INFERENCE_CONFIG['model']['from_chkpt']['chkpt_path']}")
        print(cfg)
        # print(model)

        # --------------------------------------------------------
        # Resolve dataset registry entry
        # --------------------------------------------------------
        dataset_cfg = INFERENCE_CONFIG['dataset']
        ds_name = dataset_cfg['name']
        ds_entry = COLLECTION['datasets'][ds_name]

        # Build dataset config
        DataCfgClass = ds_entry['class']
        DatasetClass = ds_entry['cfgclass']

        cfg = DataCfgClass(**dataset_cfg['hyperparameters'])
        dataset = DatasetClass(cfg)

        print(f"Loaded dataset: {ds_name}")
        print(f"  Number of samples: {len(dataset)}")
        print(cfg)
        if len(dataset) > 0:
            sample, target = dataset[0]
            print(f"  Example sample keys: sample={list(sample.keys())}, target={list(target.keys())}")
        else:
            print("  EMPTY")


        # --------------------------------------------------------
        # Resolve runner registry entry
        # --------------------------------------------------------
        runner_cfg_entry = COLLECTION["runners"][INFERENCE_CONFIG["runner"]["name"]]

        # Build runner config
        RunnerCfgClass = runner_cfg_entry["cfgclass"]

        runner_cfg = RunnerCfgClass(**INFERENCE_CONFIG["runner"]["hyperparameters"])
        runner_cfg.target_dir = os.path.join(INFERENCE_CONFIG["runner"]["hyperparameters"]['target_dir'],
                                             INFERENCE_CONFIG['config_name'])

        # Build runner
        RunnerClass = runner_cfg_entry["class"]
        runner = RunnerClass(runner_cfg, model, dataset)

        print(f"Loaded runner: {INFERENCE_CONFIG['runner']['name']}")
        print(f"Config: {runner_cfg}")

        # --------------------------------------------------------
        # R.U.N.
        # --------------------------------------------------------
        runner.run()
        collect_metrics(INFERENCE_CONFIG["runner"]["hyperparameters"]['target_dir'])
    except Exception as e:
        print(f"Failed {INFERENCE_CONFIG['config_name']}")
        raise e