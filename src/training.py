import os
import cv2 as cv
from skimage.feature import hog
from sklearn.svm import LinearSVC
import pickle
import numpy as np
import parameters
import time
import datetime
from detection import run_sliding_window

def get_hog_features(image):
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    features = hog(image,
                   pixels_per_cell=parameters.HOG_CELL_SIZE,
                   cells_per_block=parameters.HOG_CELLS_PER_BLOCK,
                   feature_vector=True)
    return features


def get_hog_features_from_folder(folder_path, also_use_flipped:bool = True):
    folder_features = []
    for image_name in os.listdir(folder_path):
        image = cv.imread(os.path.join(folder_path, image_name), cv.IMREAD_GRAYSCALE)
        if image is None: continue
        features = get_hog_features(image)
        folder_features.append(features)

        if parameters.USE_FLIPPED_IMAGES and also_use_flipped:
            folder_features.append(get_hog_features(np.fliplr(image)))
    return folder_features

def load_positive_features():
    print("* Loading positive features...\n")
    positive_features = []
    for character in parameters.CHARACTERS + ["unknown"]:
        character_dir = os.path.join(parameters.POS_EXAMPLES_DIR, character)
        if not os.path.exists(character_dir): continue

        number_of_images_processed = len(positive_features)

        for image_name in os.listdir(character_dir):
            image_path = os.path.join(character_dir, image_name)
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            if image is None: continue

            image_features = get_hog_features(image)
            positive_features.append(image_features)

            if parameters.USE_FLIPPED_IMAGES:
                image_flipped = np.fliplr(image)
                image_flipped_features = get_hog_features(image_flipped)
                positive_features.append(image_flipped_features)

        print(f"{len(positive_features) - number_of_images_processed} {character} images")

    print(f"\n> Done. Processed {len(positive_features)} total images.")
    print()
    return positive_features

def load_negative_features():
    print("* Loading negative features...\n\nestimated time remaining:")
    negative_features = []
    negatives_dir = parameters.NEG_EXAMPLES_DIR
    image_count = 0
    start_time = 0
    total_images_count = len(os.listdir(negatives_dir))
    for image_name in os.listdir(negatives_dir):
        if image_count % 1000 == 0:
            start_time = time.time()

        image_path = os.path.join(negatives_dir, image_name)
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if image is None: continue

        image_features = get_hog_features(image)
        negative_features.append(image_features)

        if image_count % 2000 == 0:
            end_time = time.time()
            seconds_left = (end_time - start_time) * (total_images_count - image_count)
            image_count_str = str(image_count)
            while len(image_count_str) < len(str(total_images_count)):
                image_count_str = " " + image_count_str
            print(f"    L {str(datetime.timedelta(seconds=int(seconds_left)))} - {image_count_str} / {total_images_count}")
        image_count += 1
    print(f"\n> Done. Processed {len(negative_features)} total images.")
    print()
    return negative_features


def get_positive_features_data(use_cached_data:bool = False):
    if os.path.exists(parameters.POS_FEATURES_PATH) and use_cached_data:
        print(f"[Cache Hit] Loading positive features from {parameters.POS_FEATURES_PATH}...")
        positive_features = np.load(parameters.POS_FEATURES_PATH)
    else:
        positive_features = load_positive_features()
        np.save(parameters.POS_FEATURES_PATH, positive_features)
        print(f"> Saved positive features to {parameters.POS_FEATURES_PATH}")
        print()
    return positive_features

def get_negative_features_data(use_cached_data:bool = False):
    if os.path.exists(parameters.NEG_FEATURES_PATH) and use_cached_data:
        print(f"[Cache Hit] Loading negative features from {parameters.NEG_FEATURES_PATH}...")
        negative_features = np.load(parameters.NEG_FEATURES_PATH)
    else:
        negative_features = load_negative_features()
        np.save(parameters.NEG_FEATURES_PATH, negative_features)
        print(f"> Saved negative features to {parameters.NEG_FEATURES_PATH}")
        print()
    return negative_features

# makes the model for faces (task 1)
def train_general_classifier():
    positive_features = get_positive_features_data(parameters.USE_CACHED_FEATURES)
    negative_features = get_negative_features_data(parameters.USE_CACHED_FEATURES)

    pos_labels = np.ones(len(positive_features))
    neg_labels = np.zeros(len(negative_features))
    labels = np.concatenate((pos_labels, neg_labels), axis=0)
    data = np.concatenate((positive_features, negative_features), axis=0)


    print(f"* Training Linear SVM...\n")
    classifier = LinearSVC(C=1.0, dual='auto')
    classifier.fit(data,labels)

    model_name = "model_all_faces.pickle"
    model_path = os.path.join(parameters.MODELS_DIR, model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)

    print(f"> Done!")
    print(f"> Model saved to {model_path}")
    print("    L Training Accuracy:", classifier.score(data, labels), '\n')

    return classifier


# makes models for characters (task 2)
def train_classifier(target_character):
    negative_features = get_negative_features_data(parameters.USE_CACHED_FEATURES)[::10]
    print(f"> Loaded {len(negative_features)} background negatives.")

    positive_features = []

    for character_name in parameters.CHARACTERS + ["unknown"]:
        character_pos_examples_dir = os.path.join(parameters.POS_EXAMPLES_DIR, character_name)
        character_features = get_hog_features_from_folder(character_pos_examples_dir)
        if character_name != target_character:
            print(f" - Loaded {len(character_features)} NEGATIVES ({character_name}).")
            np.concatenate((negative_features, character_features), axis=0)
        else:
            print(f" + Loaded {len(character_features)} POSITIVES ({character_name}).")
            positive_features = character_features

    positive_labels = np.ones(len(positive_features))
    negative_labels = np.zeros(len(negative_features))
    labels = np.concatenate((positive_labels, negative_labels), axis=0)
    data = np.concatenate((positive_features, negative_features), axis=0)

    print(f"* Training Linear SVM...\n")
    classifier = LinearSVC(C=1.0, dual='auto')
    classifier.fit(data, labels)

    model_name = f"model_{target_character}.pickle"
    model_path = os.path.join(parameters.MODELS_DIR, model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)

    print(f"> Done!")
    print(f"> Model saved to {model_path}")
    print("    L Training Accuracy:", classifier.score(data, labels), '\n')

