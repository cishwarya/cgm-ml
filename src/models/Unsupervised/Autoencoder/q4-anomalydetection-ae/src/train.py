

from pathlib import Path
import os
import pickle
import random
import shutil

import glob2 as glob
import tensorflow as tf
import tensorflow_datasets as tfds
from azureml.core import Experiment, Workspace
from azureml.core.run import Run

from config import CONFIG, DATASET_MODE_DOWNLOAD, DATASET_MODE_MOUNT
from constants import DATA_DIR_ONLINE_RUN, MODEL_CKPT_FILENAME, REPO_DIR

# Get the current run.
run = Run.get_context()

if run.id.startswith("OfflineRun"):
    utils_dir_path = REPO_DIR / "src/common/model_utils"
    utils_paths = glob.glob(os.path.join(utils_dir_path, "*.py"))
    temp_model_util_dir = Path(__file__).parent / "tmp_model_util"
    # Remove old temp_path
    if os.path.exists(temp_model_util_dir):
        shutil.rmtree(temp_model_util_dir)
    # Copy
    os.mkdir(temp_model_util_dir)
    os.system(f'touch {temp_model_util_dir}/__init__.py')
    for p in utils_paths:
        shutil.copy(p, temp_model_util_dir)

from model import VariationalAutoencoder 
from tmp_model_util.preprocessing import preprocess_depthmap, preprocess_targets  # noqa: E402
from tmp_model_util.utils import download_dataset, get_dataset_path, AzureLogCallback, create_tensorboard_callback  # noqa: E402

# Make experiment reproducible
tf.random.set_seed(CONFIG.SPLIT_SEED)
random.seed(CONFIG.SPLIT_SEED)

DATA_DIR = REPO_DIR / 'data' if run.id.startswith("OfflineRun") else Path(".")
print(f"DATA_DIR: {DATA_DIR}")

# Offline run. Download the sample dataset and run locally. Still push results to Azure.
if run.id.startswith("OfflineRun"):
    print("Running in offline mode...")

    # TODO Make this work again.

    # Access workspace.
    #print("Accessing workspace...")
    #workspace = Workspace.from_config()
    #experiment = Experiment(workspace, "training-junkyard")
    #run = experiment.start_logging(outputs=None, snapshot_directory=None)

    #dataset_name = CONFIG.DATASET_NAME_LOCAL
    #dataset_path = get_dataset_path(DATA_DIR, dataset_name)
    #download_dataset(workspace, dataset_name, dataset_path)

# Online run. Use dataset provided by training notebook.
else:
    print("Running in online mode...")
    experiment = run.experiment
    workspace = experiment.workspace

    # TODO Make this work again.
    #dataset_name = CONFIG.DATASET_NAME

    # Mount or download
    #if CONFIG.DATASET_MODE == DATASET_MODE_MOUNT:
    #    dataset_path = run.input_datasets["dataset"]
    #elif CONFIG.DATASET_MODE == DATASET_MODE_DOWNLOAD:
    #    dataset_path = get_dataset_path(DATA_DIR_ONLINE_RUN, dataset_name)
    #    download_dataset(workspace, dataset_name, dataset_path)
    #else:
    #    raise NameError(f"Unknown DATASET_MODE: {CONFIG.DATASET_MODE}")

# Get the QR-code paths.
#dataset_path = os.path.join(dataset_path, "scans")
#print("Dataset path:", dataset_path)
#print(glob.glob(os.path.join(dataset_path, "*"))) # Debug
#print("Getting QR-code paths...")
#qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
#print("qrcode_paths: ", len(qrcode_paths))
#assert len(qrcode_paths) != 0

# Shuffle and split into train and validate.
#random.shuffle(qrcode_paths)
#split_index = int(len(qrcode_paths) * 0.8)
#qrcode_paths_training = qrcode_paths[:split_index]
#qrcode_paths_validate = qrcode_paths[split_index:]
#qrcode_paths_activation = random.choice(qrcode_paths_validate)
#qrcode_paths_activation = [qrcode_paths_activation]

#del qrcode_paths

# Show split.
#print("Paths for training:")
#print("\t" + "\n\t".join(qrcode_paths_training))
#print("Paths for validation:")
#print("\t" + "\n\t".join(qrcode_paths_validate))
#print("Paths for activation:")
#print("\t" + "\n\t".join(qrcode_paths_activation))

#print(len(qrcode_paths_training))
#print(len(qrcode_paths_validate))

#assert len(qrcode_paths_training) > 0 and len(qrcode_paths_validate) > 0


#def get_depthmap_files(paths):
#    pickle_paths = []
#    for path in paths:
#        pickle_paths.extend(glob.glob(os.path.join(path, "**", "*.p")))
#    return pickle_paths


# Get the pointclouds.
#print("Getting depthmap paths...")
#paths_training = get_depthmap_files(qrcode_paths_training)
#paths_validate = get_depthmap_files(qrcode_paths_validate)
#paths_activate = get_depthmap_files(qrcode_paths_activation)

#del qrcode_paths_training
#del qrcode_paths_validate
#del qrcode_paths_activation

#print("Using {} files for training.".format(len(paths_training)))
#print("Using {} files for validation.".format(len(paths_validate)))
#print("Using {} files for validation.".format(len(paths_activate)))


# Function for loading and processing depthmaps.
#def tf_load_pickle(path, max_value):
#    def py_load_pickle(path, max_value):
#        depthmap, targets = pickle.load(open(path.numpy(), "rb"))
#        depthmap = preprocess_depthmap(depthmap)
#        depthmap = depthmap / max_value
#        depthmap = tf.image.resize(depthmap, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH))
#        targets = preprocess_targets(targets, targets_indices)
#        return depthmap, targets

#    depthmap, targets = tf.py_function(py_load_pickle, [path, max_value], [tf.float32, tf.float32])
#    depthmap.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1))
#    targets.set_shape((len(targets_indices,)))
#    return depthmap, targets


#def tf_flip(image):
#    image = tf.image.random_flip_left_right(image)
#    return image


# Parameters for dataset generation.
#targets_indices = [0]  # 0 is height, 1 is weight.

# Create dataset for training.
#paths = paths_training
#dataset = tf.data.Dataset.from_tensor_slices(paths)
#dataset_norm = dataset.map(lambda path: tf_load_pickle(path, CONFIG.NORMALIZATION_VALUE))
#dataset_norm = dataset_norm.cache()
#dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
#dataset_norm = dataset_norm.shuffle(CONFIG.SHUFFLE_BUFFER_SIZE)
#dataset_training = dataset_norm
#del dataset_norm

# Create dataset for validation.
# Note: No shuffle necessary.
#paths = paths_validate
#dataset = tf.data.Dataset.from_tensor_slices(paths)
#dataset_norm = dataset.map(lambda path: tf_load_pickle(path, CONFIG.NORMALIZATION_VALUE))
#dataset_norm = dataset_norm.cache()
#dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
#dataset_validation = dataset_norm
#del dataset_norm

# Create dataset for activation
#paths = paths_activate
#dataset = tf.data.Dataset.from_tensor_slices(paths)
#dataset_norm = dataset.map(lambda path: tf_load_pickle(path, CONFIG.NORMALIZATION_VALUE))
#dataset_norm = dataset_norm.cache()
#dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
#dataset_activation = dataset_norm
#del dataset_norm

# TODO Use CGM data.
# https://www.tensorflow.org/datasets/catalog/cats_vs_dogs
dataset_train, dataset_validate = tfds.load("mnist", split=["train[:80%]", "train[80%:]"])
#dataset_train, dataset_validate = tfds.load("cats_vs_dogs", split=["train[:20%]", "train[25%:30%]"])

def tf_preprocess(image):
    image = tf.image.resize(image, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH)) / 255.0
    return image

dataset_train = dataset_train.map(lambda sample: tf_preprocess(sample["image"]))
dataset_train = dataset_train.cache()
dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

dataset_validate = dataset_validate.map(lambda sample: tf_preprocess(sample["image"]))
dataset_validate = dataset_validate.cache()
dataset_validate = dataset_validate.prefetch(tf.data.experimental.AUTOTUNE)

# Note: Now the datasets are prepared.

# Create the model.
model = VariationalAutoencoder(
    input_shape=(CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, CONFIG.IMAGE_TARGET_DEPTH), 
    filters=CONFIG.FILTERS, 
    latent_dim=CONFIG.LATENT_DIM,
    size=CONFIG.MODEL_SIZE
)
#model.summary()

#model = VariationalAutoencoder(
#    input_shape=(CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, CONFIG.IMAGE_TARGET_DEPTH),
#    latent_dim=CONFIG.LATENT_DIM
#)

best_model_path = str(DATA_DIR / f'outputs/{MODEL_CKPT_FILENAME}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
training_callbacks = [
    AzureLogCallback(run),
    create_tensorboard_callback(),
    checkpoint_callback,
]

# Train the model.
#model.train(
#    dataset_train.batch(CONFIG.BATCH_SIZE), 
#    epochs=CONFIG.EPOCHS, 
#    kl_loss_factor=CONFIG.KL_LOSS_FACTOR
#)

model.train(
    dataset_train, 
    dataset_validate, 
    epochs=CONFIG.EPOCHS,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle_buffer_size=CONFIG.SHUFFLE_BUFFER_SIZE,
    render=True
    #kl_loss_factor=CONFIG.KL_LOSS_FACTOR
)

# Done.
run.complete()
