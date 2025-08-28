from torch_datasets import *
from torch_models import *
from training import *

from training.configurator_rnngcnplustransitions_grid import ALL_CONFIGS #! adjust file to import from
from classes_collections import COLLECTION

import os
import json
import torch
import wandb
import argparse

print(f"Found {len(ALL_CONFIGS)} runs")
parser = argparse.ArgumentParser()
parser.add_argument("--runids", type=str, default=None,
                    help="Config range to run, format: start-end (inclusive-exclusive, e.g., 0-10).")
args = parser.parse_args()

if args.runids:
    try:
        start, end = [int(x) for x in args.runids.split('-')]
        selected_configs = ALL_CONFIGS[start:end]
        config_offset = start  # to keep the correct config_name/indexing
    except Exception as e:
        raise ValueError(f"Invalid --runids format: {args.runids}") from e
else:
    selected_configs = ALL_CONFIGS
    config_offset = 0

SAVE_DIR = "source/models/rnn-transitions/6blocks_2d_ClipSplitter_gridbest"
os.makedirs(SAVE_DIR, exist_ok=True)

for i, cfg in enumerate(selected_configs):
    # --- Save config as JSON ---
    model_dir = os.path.join(SAVE_DIR, cfg['config_name'])
    os.makedirs(model_dir, exist_ok=True)
    json_path = os.path.join(model_dir, f"config.json")
    with open(json_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved config {i} to {json_path}")

    # --- Instantiate splitter config and splitter class dynamically ---
    splitter_section = cfg['splitter']
    splitter_cfgclass = COLLECTION['datasets'][splitter_section['name']]['cfgclass']
    splitter_class = COLLECTION['datasets'][splitter_section['name']]['class']
    splitter_cfg = splitter_cfgclass(**splitter_section['hyperparams'])

    # Instantiate the splitter (adjust args if your Splitter supports val_portion)
    splitter = splitter_class(splitter_cfg)
    trainpath, valpath, testpath = splitter.train, splitter.val, splitter.test

    # --- Instantiate pose normalizer config ---
    pose_norm_section = cfg['pose_norm']
    pose_norm_cfgclass = COLLECTION['datasets'][pose_norm_section['name']]['cfgclass']
    pose_norm_cfg = pose_norm_cfgclass(**pose_norm_section['hyperparams'])

    # --- Instantiate dataset config and dataset class ---
    dataset_section = cfg['dataset']
    dataset_cfgclass = COLLECTION['datasets'][dataset_section['name']]['cfgclass']
    dataset_class = COLLECTION['datasets'][dataset_section['name']]['class']

    # If dataset config expects a pose_norm_config argument, pass it;
    # else, just pass dataset config fields (depends on your dataclass)
    trainset_cfg = dataset_cfgclass(
        **dataset_section['hyperparams'],
        pose_norm_config=pose_norm_cfg,  # Pass as argument if needed
        path_list=trainpath,
        caching_path='source/cache/train',
        split_type=splitter_section['name'],
        random_seed=splitter.config.random_seed
    )
    trainset = dataset_class(trainset_cfg)

    testset_cfg = dataset_cfgclass(
        **dataset_section['hyperparams'],
        pose_norm_config=pose_norm_cfg,  # Pass as argument if needed
        path_list=testpath,
        caching_path='source/cache/test'
    )
    testset = dataset_class(testset_cfg)

    valset_cfg = dataset_cfgclass(
        **dataset_section['hyperparams'],
        pose_norm_config=pose_norm_cfg,  # Pass as argument if needed
        path_list=valpath,
        caching_path='source/cache/val'
    )
    valset = dataset_class(valset_cfg)

    # --- Model instantiation from COLLECTION ---

    model_section = cfg['model']
    model_cfgclass = COLLECTION['models'][model_section['name']]['cfgclass']
    model_class = COLLECTION['models'][model_section['name']]['class']

    # Build model config dataclass, injecting joint_names and bones from dataset
    model_cfg = model_cfgclass(
        **model_section['hyperparams'],
        num_point=len(trainset_cfg.all_joint_names_ordered),  #FIXME   #! not dynamic yet, stgcn/plus specific
        joint_names=trainset_cfg.all_joint_names_ordered,     #FIXME   #! not dynamic yet, stgcn/plus specific
        bones=trainset_cfg.all_bones_names                    #FIXME   #! not dynamic yet, stgcn/plus specific
    )

    model = model_class(model_cfg)

    # --- Trainer instantiation from COLLECTION ---
    trainer_section = cfg['trainer']
    trainer_cfgclass = COLLECTION['trainers'][trainer_section['name']]['cfgclass']
    trainer_class = COLLECTION['trainers'][trainer_section['name']]['class']

    # Instantiate trainer config dataclass
    trainer_cfg = trainer_cfgclass(**trainer_section['hyperparams'])
    trainer_cfg.wandb_run_name = cfg['config_name']
    trainer_cfg.checkpoint_dir = model_dir

    trainer = trainer_class(
                            model=model,
                            trainset=trainset,
                            valset=valset,
                            testset=testset,
                            config=trainer_cfg
                        )

    print(f"\n=== Starting training for config {cfg['config_name']} ===")
    trainer.fit()

    # Optionally save checkpoint
    ckpt_path = os.path.join(model_dir, f"model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model checkpoint to {ckpt_path}\n")