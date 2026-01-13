import os
import cv2 as cv
import random
from helper_functions import *

def load_positives():
    print("\n* Loading positives...")

    number_of_found_characters = {character: 1 for character in parameters.CHARACTERS + ["unknown"]}

    for character in parameters.CHARACTERS:
        annotation_path = annotations_path(character)
        if not os.path.exists(annotation_path):
            continue
        with open(annotation_path) as character_annotations:
            character_image_dir = image_dir(character)
            for line in character_annotations:
                face_annotation = line.split()

                if len(face_annotation) < 6: continue

                image_name, character_name = face_annotation[0], face_annotation[-1]
                coords = [int(x) for x in face_annotation[1:5]]
                xmin, ymin, xmax, ymax = coords

                image_path = os.path.join(character_image_dir, image_name)
                img = cv.imread(image_path)
                if img is None:
                    print(f"Warning: Could not read {image_path}")
                    continue

                h, w, _ = img.shape
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)
                if xmin >= xmax or ymin >= ymax:
                    continue

                face = img[ymin:ymax, xmin:xmax]
                face_resized = cv.resize(face, (parameters.hog_width, parameters.hog_height))

                save_dir = os.path.join(parameters.DIR_POS_EXAMPLES, character_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_name = f"{number_of_found_characters[character_name]:04d}.jpg"
                save_path = os.path.join(save_dir, save_name)

                cv.imwrite(save_path, face_resized)
                number_of_found_characters[character_name] += 1

    print(f"\n> Done!\n> Total character finds:\n{number_of_found_characters}")

def build_negatives():
    print("\n* Building negatives...")

    negatives_per_img = 5
    negative_count = 0

    save_dir = parameters.DIR_NEG_EXAMPLES

    random.seed(42)

    for character in parameters.CHARACTERS:
        img_dir = image_dir(character)
        annotation_path = annotations_path(character)

        ground_truth_boxes = {}

        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as annotation_file:
                for line in annotation_file:

                    face_annotation = line.split()
                    if len(face_annotation) < 6: continue

                    img_name = face_annotation[0]

                    coords = [int(x) for x in face_annotation[1:5]]
                    if img_name not in ground_truth_boxes:
                        ground_truth_boxes[img_name] = []
                    ground_truth_boxes[img_name].append(coords)

        for img_name in os.listdir(img_dir):
            if not img_name.endswith(".jpg"): continue

            img_path = os.path.join(img_dir, img_name)
            img = cv.imread(img_path)
            if img is None: continue

            h_img, w_img, _ = img.shape
            faces = ground_truth_boxes.get(img_name, [])
            for _ in range(negatives_per_img):
                for attempt in range(67):
                    rand_x = random.randint(0, w_img - parameters.hog_width)
                    rand_y = random.randint(0, h_img - parameters.hog_height)

                    cand_xmin, cand_ymin = rand_x, rand_y
                    cand_xmax = rand_x + parameters.hog_width
                    cand_ymax = rand_y + parameters.hog_height

                    has_overlap = False
                    for face in faces:
                        fx_min, fy_min, fx_max, fy_max = face

                        ix_min = max(cand_xmin, fx_min)
                        iy_min = max(cand_ymin, fy_min)
                        ix_max = min(cand_xmax, fx_max)
                        iy_max = min(cand_ymax, fy_max)

                        iw = max(0, ix_max - ix_min)
                        ih = max(0, iy_max - iy_min)

                        intersection_area = iw * ih

                        if intersection_area > 0:
                            has_overlap = True
                            break
                    if not has_overlap:
                        crop = img[cand_ymin:cand_ymax, cand_xmin:cand_xmax]
                        save_name = f"{negative_count:05d}.jpg"
                        cv.imwrite(os.path.join(save_dir, save_name), crop)
                        negative_count += 1
                        break
    print(f"\n> Done!\n> Generated {negative_count} negative examples")