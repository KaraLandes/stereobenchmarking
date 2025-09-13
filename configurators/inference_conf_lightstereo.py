import copy
import itertools

BASE_CONFIG = {
    'model': {
        'name': 'LightStereoModel',
        'from_chkpt': {
            'on': True,  # if true then instantiate from chkpt
            'chkpt_path': 'source/checkpoints/lightstereo/l',  # <-- adjust

            # runtime args that go into LightStereoConfig
            'eval_mode': True,
            'test_mode': True,
            'device': "cuda",
        },
        'from_hyperparams': {
            'on': False,  # set True if training from scratch
            'hyperparameters': {
                'max_disp': 192,
                'left_att': True,
                'backbone': "MobileNetv2",   # or "EfficientNetv2"
                'aggregation_blocks': (1, 2, 4),
                'expanse_ratio': 4,
                'eval_mode': True,
                'test_mode': True,
                'device': "cuda",
            }
        }
    },
    'dataset': {
        'name': 'Kitti12BaseDataset',
        'hyperparameters': {
            'root': 'source/kitti12/data_stereo_flow',   # required
            'split': 'training',          # "training" or "testing"
            'use_color': True,            # colored_0/1 instead of image_0/1
            'gt_type': 'disp_occ',        # disparity type (only training split)
            'resize_to': None,            # LightStereo was trained on Images 256*512
            'sample_transform': None,
            'target_transform': None,
            'augmentation': None,
        }
    },
    'runner': {
        'name': 'LightStereoEvaluationRunner',
        'hyperparameters': {
            'target_dir': 'source/evaluation/lightstereo_kitti12-training_l',
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
            'resize_to': [(256, 512)] #! LightStereo checkpoints we have were trained on 256x512
        }
    },
    "model": {
        "from_chkpt": {
        }
    },
    "runner": {
        "hyperparameters": {
            "batch_size": [1, 5],
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


