from torch_datasets import *
from torch_models import *
from training import *


COLLECTION = {
    "datasets":{
        'Kitti12BaseDataset':{
            'cfgclass':Kitti12BaseDataset,
            'class':Kitti12Config
        },

    },
    'models':{
        'FoundationStereoModel':{
            'cfgclass':FoundationStereoConfig,
            'class':FoundationStereoModel
        },

    },
    'trainers':{
        'BaseTrainer':{
            'cfgclass':BaseTrainerConfig,
            'class':BaseTrainer
        },

    }
}