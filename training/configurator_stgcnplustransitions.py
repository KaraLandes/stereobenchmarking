import copy

# --- 1. Define your base (default) config as nested dicts with `name` and `hyperparams` fields ---

BASE_CONFIG = {
    'config_name': "base",
    'dataset': {
        'name': 'TennisSTGCNDataset',
        'hyperparams': {
            'use_visibility': True,
            'start_in_middle_of_shot': False,
            'T': 205,
            'cropping_strategy': 'downsample',
            'onfly': False,
            'max_zeros': 10,
            'labels_format': 'gaussian_heatmaps',
            'pose_dimesionality':2
        },
    },
    'pose_norm': {
        'name': 'PoseNormalizer',
        'hyperparams': {
            'enable': True,
            'center': False,
            'align': False,
            'scale': 'avg_bone',
        }
    },
    'splitter': {
        'name': 'PersonSplitter',
        'hyperparams': {
            'test_portion': 0.1,
            'val_portion': 0.1,
            'data_dir': 'source/data'
        }
    },
    'model': {
        'name': 'STGCNPlusTransitionsModel',
        'hyperparams': {
            # window_size and in_channels will be set below
            'graph_strategy': "spatial",        # "uniform", "distance", "spatial"
            'edge_importance_weighting': True, 
            'channels': [32]*6,
            
            # Temporal strides per block; len must match channels. Use 2 to downsample T.
            'strides': [1]*6, #keep 1,1,1... for full sequence outputs
            'dropout': 0.0,
            'temporal_kernel_size': 7,
            'dilations': [2]*6,
            
            # Adjacency options
            'A_self_connections': False    
            
        }
    },
    'trainer': {
        'name': 'BaseTrainer',
        'hyperparams': {
            'epochs':250,
            'learning_rate': 5e-4,
            'lr_scheduler': 'cosineannealing',
            'optimizer': 'AdamW',
            'loss_fn_name': 'cross_entropy',
            'num_workers':12,
            'batch_size':32,
            'evaluate_on_test':True,
            'wandb_project':"stgcnplustransitions",
            'wandb_group': '6blocks-PersonSplitter-2d-sweep',
        }
    }
}

# --- 2. Define the hyperparameters to sweep and their values (using hyperparams subdicts) ---
channels_3blocks_options = [
                            [16, 32, 64],     # 1. ultra-light
                            [24, 48, 96],     # 2. extra-light
                            [32, 64, 128],    # 3. light
                            [48, 96, 192],    # 4. narrow-0.75×
                            [64, 128, 256],   # 5. baseline-1.0×
                            [80, 160, 320],   # 6. wide-1.25×
                            [96, 192, 384],   # 7. ultra-1.5×
                            [128, 128, 128],  # 8. flat-mid
                            [128, 160, 192],  # 9. early-heavy
                            [64, 192, 320],   # 10. late-heavy
                            [64, 128, 192],   # 11. pyramid-up
                            [192, 128, 64],   # 12. pyramid-down
                            [64, 96, 64]      # 13. bottleneck-wave
                            ]
channels_6block_options = [
                            [16, 16, 32, 32, 64, 64],      # 1. ultra-light
                            [24, 24, 48, 48, 96, 96],      # 2. extra-light
                            [32, 32, 64, 64, 128, 128],    # 3. light
                            [48, 48, 96, 96, 192, 192],    # 4. light-mid
                            [64, 64, 128, 128, 256, 256],  # 5. baseline
                            [80, 80, 160, 160, 320, 320],  # 6. wide
                            [96, 96, 192, 192, 384, 384],  # 7. ultra-wide
                            [128, 128, 128, 128, 128, 128],# 8. flat-mid
                            [96, 96, 128, 160, 192, 256],  # 9. early-heavy
                            [48, 64, 128, 256, 320, 384],  # 10. late-heavy
                        ]

dilations_3blocks_options = [
                            [1, 1, 1],     # 1. uniform small
                            [2, 2, 2],     # 2. uniform medium
                            [4, 4, 4],     # 3. uniform large
                            [1, 2, 4],     # 4. progressive small→large
                            [4, 2, 1],     # 5. progressive large→small
                            [1, 4, 1],     # 6. mixed center-heavy
                            [2, 4, 2],     # 7. alternating low-high
                            [1, 2, 8],     # 8. stepwise doubling
                            [2, 8, 2],     # 9. middle emphasis
                            [1, 8, 16]     # 10. high spread                   
                            ]
dilation_6block_options = [
                            [1, 1, 1, 1, 1, 1],       # 1. no dilation (baseline)
                            [1, 2, 1, 2, 1, 2],       # 2. alternating small dilation
                            [1, 2, 4, 1, 2, 4],       # 3. repeating wave
                            [1, 2, 4, 8, 4, 2],       # 4. pyramid up-down
                            [1, 2, 4, 8, 16, 32],     # 5. exponential growth
                            [1, 1, 2, 2, 4, 4],       # 6. step growth
                            [1, 3, 9, 1, 3, 9],       # 7. triple pattern
                            [1, 2, 2, 4, 4, 8],       # 8. gradual growth
                            [1, 4, 8, 4, 8, 16],      # 9. wide alternating
                            [2, 4, 8, 16, 32, 64],    # 10. large receptive field
                        ]

selected_channels_options = channels_6block_options
selected_dilations_options = dilation_6block_options

SWEEP_PARAMETERS = {
    # ('dataset', 'use_visibility'): [True, False],
    # ('dataset', 'start_in_middle_of_shot'): [True, False],
    # ('dataset', 'T'): [50, 75, 100, 150, 205],
    # ('dataset', 'cropping_strategy'): ['center', 'end', 'downsample', 'discard_long'],
    # ('dataset', 'max_zeros'): [3, 5, 10, 20],
    # ('pose_norm', 'enable'): [True, False],
    # ('pose_norm', 'center'): [True, False],
    # ('pose_norm', 'align'): [True, False],
    # ('pose_norm', 'scale'): ['height', 'width', 'avg_bone', None],
    ('model', 'graph_strategy'): ['spatial', 'uniform', 'distance'],
    # ('model', 'edge_importance_weighting'): [True, False],
    ('model', 'channels'): selected_channels_options,
    ('model', 'dilations'): selected_dilations_options,   
    ('model', 'temporal_kernel_size'): [7,9,15],                    
    ('model', 'dropout'): [0, 0.1, 0.2, 0.5],
    ('model', 'A_self_connections'): [True, False],
    ('trainer', 'learning_rate'): [1e-3, 5e-4],
    ('trainer', 'lr_scheduler'): ['cosineannealing', 'none'],
    # ('trainer', 'optimizer'): ['AdamW', 'Adam'],
    # ('trainer', 'loss_fn_name'): ['cross_entropy'],
}

# --- 3. Linear sweep config generator ---

def generate_linear_sweep_configs(base_config, sweep_params):
    """
    Generates a list of configs, where for each parameter, only that parameter is varied, all else at base value.
    Returns: List[dict]
    """
    sweep_configs = []

    for (section, field), values in sweep_params.items():
        base_val = base_config[section]['hyperparams'][field]
        for v in values:
            # if v == base_val:
            #     continue  # skip base, it's included as default run
            cfg = copy.deepcopy(base_config)
            cfg[section]['hyperparams'][field] = v
            config_name = f"{section}-{field}={v}"
            cfg['config_name'] = config_name

            # Ensure model config is always valid:
            # -- in_channels depends on use_visibility
            use_visibility = cfg['dataset']['hyperparams']['use_visibility']
            pose_dim =  cfg['dataset']['hyperparams']['pose_dimesionality']
            cfg['model']['hyperparams']['in_channels'] = pose_dim+1 if use_visibility else pose_dim
            # -- window_size depends on T
            T = cfg['dataset']['hyperparams']['T']
            cfg['model']['hyperparams']['window_size'] = T

            sweep_configs.append(cfg)

    # Add the base config as the control run
    base_cfg = copy.deepcopy(base_config)
    use_visibility = base_cfg['dataset']['hyperparams']['use_visibility']
    pose_dim =  cfg['dataset']['hyperparams']['pose_dimesionality']
    base_cfg['model']['hyperparams']['in_channels'] = pose_dim+1 if use_visibility else pose_dim
    T = base_cfg['dataset']['hyperparams']['T']
    base_cfg['model']['hyperparams']['window_size'] = T
    base_cfg['config_name'] = "base"
    sweep_configs.append(base_cfg)
    return sweep_configs

# --- 4. Generate and save configs as JSON ---

ALL_CONFIGS = generate_linear_sweep_configs(BASE_CONFIG, SWEEP_PARAMETERS)