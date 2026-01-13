import os

BASE_DIR = "../"
TRAINING_DIR = os.path.join(BASE_DIR, "antrenare")
CHARACTERS = ["daphne", "fred", "shaggy", "velma"]

DIR_POS_EXAMPLES = os.path.join(TRAINING_DIR, 'faces')
DIR_NEG_EXAMPLES = os.path.join(TRAINING_DIR, 'negatives')

if not os.path.exists(DIR_POS_EXAMPLES): os.makedirs(DIR_POS_EXAMPLES)
if not os.path.exists(DIR_NEG_EXAMPLES): os.makedirs(DIR_NEG_EXAMPLES)

hog_width = 36
hog_height = 48