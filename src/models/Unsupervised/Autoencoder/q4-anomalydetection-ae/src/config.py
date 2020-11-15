class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DATASET_MODE_DOWNLOAD = "dataset_mode_download"
DATASET_MODE_MOUNT = "dataset_mode_mount"

CONFIG = dotdict(dict(
    #DATASET_MODE=DATASET_MODE_DOWNLOAD,
    #DATASET_NAME="anon-depthmap-95k",
    #DATASET_NAME_LOCAL="anon-depthmap-mini",
    SPLIT_SEED=0,
   
    #IMAGE_TARGET_HEIGHT=256,
    #IMAGE_TARGET_WIDTH=256,
    #IMAGE_TARGET_DEPTH=3,
    #MODEL_SIZE="big",
    #FILTERS=[32, 32, 64, 64, 128, 128, 256],
    #LATENT_DIM=128,
    
    #IMAGE_TARGET_HEIGHT=64,
    #IMAGE_TARGET_WIDTH=64,
    #IMAGE_TARGET_DEPTH=1,
    #MODEL_SIZE="small",
    #FILTERS=[32, 32, 64, 64],
    #LATENT_DIM=64,

    IMAGE_TARGET_HEIGHT=28,
    IMAGE_TARGET_WIDTH=28,
    IMAGE_TARGET_DEPTH=1,
    MODEL_SIZE="tiny",
    FILTERS=[32, 64],
    LATENT_DIM=2,

    KL_LOSS_FACTOR=0.5,
    EPOCHS=100,
    BATCH_SIZE=256,
    SHUFFLE_BUFFER_SIZE=2560,
    #NORMALIZATION_VALUE=7.5,
    #LEARNING_RATE=0.01,

    # Parameters for dataset generation.
    #TARGET_INDEXES=[0],  # 0 is height, 1 is weight.
))
