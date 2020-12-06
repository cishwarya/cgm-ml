class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


EVAL_CONFIG = dot_dict(dict(
    # Name of evaluation
    # NAME='q3-depthmap-plaincnn-height-100-95k-run_17',
    NAME='rgbtrain-poseest-95k-run_1',

    # Experiment in Azure ML which will be used for evaluation
    # EXPERIMENT_NAME="QA-pipeline",
    EXPERIMENT_NAME="anonrgbtrain_poseestimation_ps",
    CLUSTER_NAME="gpu-cluster",

    # Used for Debug the QA pipeline
    DEBUG_RUN=False,
    # DEBUG_RUN = True,

    # Will run eval on specified # of scan instead of full dataset
    DEBUG_NUMBER_OF_SCAN=50,

    SPLIT_SEED=0,
))

# Details of Evaluation Dataset
DATA_CONFIG = dot_dict(dict(
    # Name of training dataset
    NAME='anon_rgb_training',

    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,

    # Batch size for evaluation
    BATCH_SIZE=256,
    NORMALIZATION_VALUE=7.5,

    # Parameters for dataset generation.
    TARGET_INDEXES=[0],  # 0 is height, 1 is weight.

    NUM_SCANFILES=4,  # process one file for testing
    # NUM_SCANFILES = 0, # meaning process all files

    CODE_TO_SCANTYPE={
        '100': '_front',
        '101': '_360',
        '102': '_back',
        '200': '_lyingfront',
        '201': '_lyingrot',
        '202': '_lyingback',
    },
    POSEROOT_PATH='pose',
    POSETYPE_PATH='coco',
    PROTOTXT_PATH='deploy_coco.prototxt',
    MODELTYPE_PATH='pose_iter_440000.caffemodel',
    DATASETTYPE_PATH='COCO'

))

# Result configuration for result generation after evaluation is done
RESULT_CONFIG = dot_dict(dict(
    # Error margin on various ranges
    # ACCURACIES=[.2, .4, .6, 1, 1.2, 2., 2.5, 3., 4., 5., 6.],
    # COLUMNS=['qrcode', 'artifact', 'scantype', 'GT'],
    COLUMNS=['artifact'],
    # path of csv file in the experiment which final result is stored
    # SAVE_PATH='./outputs/result.csv',
    SAVE_PATH='outputs/',
))
