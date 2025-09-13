from torch_datasets import *
from torch_models import *
from inference import *


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
    'runners':{
        'InferenceRunner':{
            'cfgclass':InferenceRunnerConfig,
            'class':InferenceRunner
        },
        'EvaluationRunner':{
            'cfgclass':EvaluationRunnerConfig,
            'class':EvaluationRunner
        },
        'FoundationStereoEvaluationRunner':{
            'cfgclass':EvaluationRunnerConfig,
            'class':FoundationStereoEvaluationRunner
        },
        'LightStereoEvaluationRunner':{
            'cfgclass':EvaluationRunnerConfig,
            'class':LightStereoEvaluationRunner
        },
    }
}