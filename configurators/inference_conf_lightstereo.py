import copy
import itertools

BASE_CONFIG = {
    'model': {
        'name': 'FoundationStereoModel',
        'from_chkpt': {
            'on': True,  # if true then instantiate from chkpt
            'chkpt_path': 'source/checkpoints/fouindationstereo/large',
            
            # # backbone / architecture
            # 'vit_size': 'vitl',   # change to vitb, vitl, etc. to match checkpoint
            # 'max_disp': 416,
            # 'corr_implementation': 'reg',
            # 'corr_levels': 2,
            # 'corr_radius': 4,
            # 'hidden_dims': [128, 128, 128],
            # 'n_downsample': 2,
            # 'n_gru_layers': 3,
            # 'slow_fast_gru': False,

            # # inference & memory
            # 'iters': 12,               # forward call iterations
            # 'hierarchical': True,
            # 'small_ratio': 0.5,
            # 'low_memory': False,
        },
        'from_hyperparams': {
            'on': False,  # set True if training from scratch
        }
    },
    'dataset': {
        'name': 'Kitti12BaseDataset',
        'hyperparameters': {
            'root': 'source/kitti12/data_stereo_flow',   # required
            'split': 'training',          # "training" or "testing"
            'use_color': True,            # colored_0/1 instead of image_0/1
            'gt_type': 'disp_occ',        # disparity type (only training split)
            'resize_to': None,            # keep KITTI native resolution
            'sample_transform': None,
            'target_transform': None,
            'augmentation': None,
        }
    },
    'runner': {
        'name': 'FoundationStereoEvaluationRunner',
        'hyperparameters': {
            'target_dir': 'source/evaluation/foundationstereo_kitti12-training',
            'batch_size': 1,
            'num_workers': 12,
            'device': "cuda"
        }
    }
}


# -------------------------
# Grid search definition (structured by config hierarchy)
# -------------------------
GRID = {
    'dataset': {
        "hyperparameters": {
            'resize_to': [(384,1248),(192,624)]
        }
    },
    "model": {
        "from_chkpt": {
            "iters": [2, 12],
            "hierarchical": [True, False],
            "small_ratio": [0.5, 1.0],
            "low_memory": [False, True],
        }
    },
    "runner": {
        "hyperparameters": {
            "batch_size": [1, 3],
        }
    }
}

# -------------------------
# Grid generator
# -------------------------
def generate_grid(base_config, grid):
    def flatten_dict(d, parent_key=""):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key))
            else:
                items.append((new_key, v))
        return items

    flat_grid = flatten_dict(grid)
    keys, values = zip(*flat_grid)

    all_cfgs = []
    for combo in itertools.product(*values):
        cfg = copy.deepcopy(base_config)
        name_parts = []
        for k, v in zip(keys, combo):
            # drill down into config dict
            parts = k.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = v
            # build name
            if isinstance(v,(tuple,list)):
                flatval = str(v).replace(" ","")
                name_parts.append(f"{parts[-1]}{flatval}")
            else:
                name_parts.append(f"{parts[-1]}{v}")
        # attach config name
        cfg["config_name"] = "_".join(name_parts)
        all_cfgs.append(cfg)

    return all_cfgs

# -------------------------
# Generate all configs
# -------------------------
ALL_CONFIGS = generate_grid(BASE_CONFIG, GRID)


