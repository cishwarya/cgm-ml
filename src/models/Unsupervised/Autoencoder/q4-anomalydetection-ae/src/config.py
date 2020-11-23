class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DATASET_MODE_DOWNLOAD = "dataset_mode_download"
DATASET_MODE_MOUNT = "dataset_mode_mount"

CONFIG = dotdict(dict(
    DATASET_MODE=DATASET_MODE_MOUNT,
    #DATASET_MODE=DATASET_MODE_DOWNLOAD,
    DATASET_NAME="anon_rgb_training",
    DATASET_NAME_LOCAL="anon_rgb_training",
    DATASET_MAX_SCANS=100,
    SPLIT_SEED=0,
   
    #MODEL_SIZE="big",
    #IMAGE_TARGET_HEIGHT=256,
    #IMAGE_TARGET_WIDTH=256,
    #IMAGE_TARGET_DEPTH=3,
    #FILTERS=[32, 32, 64, 64, 128, 128, 256],
    #LATENT_DIM=128,
    
    #MODEL_SIZE="small",
    #IMAGE_TARGET_HEIGHT=64,
    #IMAGE_TARGET_WIDTH=64,
    #IMAGE_TARGET_DEPTH=3,
    #FILTERS=[32, 32, 64, 64],
    #LATENT_DIM=64,

    MODEL_SIZE="tiny",
    IMAGE_TARGET_HEIGHT=32,
    IMAGE_TARGET_WIDTH=32,
    IMAGE_TARGET_DEPTH=3,
    FILTERS=[32, 64],
    LATENT_DIM=2,

    KL_LOSS_FACTOR=0.5,
    EPOCHS=100,
    BATCH_SIZE=1024,
    SHUFFLE_BUFFER_SIZE=2560,
    #NORMALIZATION_VALUE=7.5,
    #LEARNING_RATE=0.01,

))
