import os

# path stuff
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
RAW_TRAINING_DATA_DIR = os.path.join(BASE_DIR, "antrenare")
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")
CHARACTERS = ["daphne", "fred", "shaggy", "velma"]

POS_EXAMPLES_DIR = os.path.join(TRAINING_DATA_DIR, 'positives')
NEG_EXAMPLES_DIR = os.path.join(TRAINING_DATA_DIR, 'negatives')

FEATURES_DIR = os.path.join(TRAINING_DATA_DIR, 'features')
POS_FEATURES_PATH = os.path.join(FEATURES_DIR,'positive_features.npy')
NEG_FEATURES_PATH = os.path.join(FEATURES_DIR, 'negative_features.npy')

MODELS_DIR = os.path.join(BASE_DIR, "models")

# testing the models

# THIS IS THE FOLDER PATH WHERE GROUND TRUTH IS PLACED (image folder and annotation files)
# (modify "validation" to the appropriate path from project root)
VALIDATION_DATA_DIR = os.path.join(BASE_DIR, 'validation')
# THIS IS THE IMAGE FOLDER PATH
# (modify "test_images" to the image folder name)
IMAGES_DIR = os.path.join(VALIDATION_DATA_DIR, "test_images")

DETECTIONS_DIR = os.path.join(BASE_DIR, "evaluare", "fisiere_solutie", "333_Ionescu_Andrei",)
TASK1_DIR = os.path.join(DETECTIONS_DIR, "task1")
TASK2_DIR = os.path.join(DETECTIONS_DIR, "task2")

# training data stuff
NEG_EXAMPLES_PER_IMAGE = 10
USE_CACHED_FEATURES = True

# training stuff
IMAGE_RESCALES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
DETECTION_THRESHOLD = -0.5

# hog stuff
HOG_WINDOW_WIDTH = 36
HOG_WINDOW_HEIGHT = 48
HOG_CELL_SIZE = [6, 6] # 6x6 pixels
HOG_CELLS_PER_BLOCK = [2, 2] # 2x2 cells
USE_FLIPPED_IMAGES = True

