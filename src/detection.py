import cv2 as cv
import numpy as np
import os
import pickle
import parameters
from skimage.feature import hog
import time
import datetime


def run_sliding_window(image, model, threshold):

    detections = []

    if len(image.shape) > 2:
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        image_gray = image

    for scale in parameters.IMAGE_RESCALES:
        image_resized = cv.resize(image_gray, (0, 0), fx=scale, fy=scale)

        if image_resized.shape[0] < parameters.HOG_WINDOW_HEIGHT or image_resized.shape[1] < parameters.HOG_WINDOW_WIDTH:
            break

        image_height, image_width = image_resized.shape

        window_width = parameters.HOG_WINDOW_WIDTH
        window_height = parameters.HOG_WINDOW_HEIGHT
        step_x = parameters.HOG_CELL_SIZE[0]
        step_y = parameters.HOG_CELL_SIZE[1]

        for y in range (0, image_height - window_height, step_y):
            for x in range (0, image_width - window_width, step_x):
                patch = image_resized[y: y + window_height, x: x + window_width]
                features = hog(patch,
                               pixels_per_cell=parameters.HOG_CELL_SIZE,
                               cells_per_block=parameters.HOG_CELLS_PER_BLOCK,
                               feature_vector=True)
                features = features.reshape(1, -1)
                score = model.decision_function(features)[0]
                if score > threshold:
                    real_x = int(x / scale)
                    real_y = int(y / scale)
                    real_w = int(window_width / scale)
                    real_h = int(window_height / scale)
                    detections.append([real_x, real_y, real_x + real_w, real_y + real_h, score])
    return detections

def non_maximal_suppression(detections, overlap_threshold = 0.3):
    if len(detections) == 0:
        return []

    detections = np.array(detections)
    indices = np.argsort(detections[:, -1])
    kept_boxes = []
    while len(indices) > 0:
        best_idx = indices[-1]
        best_box = detections[best_idx]
        kept_boxes.append(best_box)
        indices = indices[:-1]
        surviving_indices = []
        if len(indices) == 0:
            break
        bestx_min, besty_min, bestx_max, besty_max, _ = best_box
        best_area = (bestx_max - bestx_min) * (besty_max - besty_min)
        for index in indices:
            current_box = detections[index]
            currentx_min, currenty_min, currentx_max, currenty_max, _ = current_box
            current_area = (currentx_max - currentx_min) * (currenty_max - currenty_min)

            intersectionX_min = max(currentx_min, bestx_min)
            intersectionY_min = max(currenty_min, besty_min)
            intersectionX_max = min(currentx_max, bestx_max)
            intersectionY_max = min(currenty_max, besty_max)

            intersection_width = max(0, intersectionX_max - intersectionX_min)
            intersection_height = max(0, intersectionY_max - intersectionY_min)
            intersection_area = intersection_width * intersection_height
            union_area = best_area + current_area - intersection_area
            overlap = intersection_area / union_area

            if overlap < overlap_threshold:
                surviving_indices.append(index)
        indices = surviving_indices
    return kept_boxes


def oneimg_test(test_image_path, character = "faces"):
    print(f"* Loading {character} model...")
    with open(os.path.join(parameters.MODELS_DIR, f"model_{character}.pickle"), "rb") as file:
        model = pickle.load(file)


    image = cv.imread(test_image_path)
    print(f"* Scanning image for {character}...")
    detections = run_sliding_window(image, model, threshold=parameters.DETECTION_THRESHOLD)
    print(f"found {len(detections)} potential boxes.")
    detections = non_maximal_suppression(detections)
    # draw box
    for det in detections:
        x1, y1, x2, y2, score = det
        cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv.putText(image, f"{score:.3f}", (int(x1), int(y2) + 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv.imshow("Detections", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def save_npy_files(detections, scores, file_names, output_dir, character):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, f"detections_{character}.npy"), np.array(detections))
    np.save(os.path.join(output_dir, f"scores_{character}.npy"), np.array(scores))
    np.save(os.path.join(output_dir, f"file_names_{character}.npy"), np.array(file_names))

    print(f"Saved {len(scores)} detections for '{character}' in {output_dir}")


def generate_task_files(character = "all_faces"):
    model_path = os.path.join(parameters.MODELS_DIR, f"model_{character}.pickle")
    print(f"* Processing: {parameters.IMAGES_DIR}\n     L using {model_path}...")


    if not os.path.exists(model_path):
        print(f"ERROR: Model {model_path} not found.")
        return
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    all_detections = []
    all_scores = []
    all_filenames = []

    images_dir = parameters.IMAGES_DIR
    image_count = 0
    total_images = len(os.listdir(images_dir))
    start_time = time.time()
    for image_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_name)
        image = cv.imread(image_path)
        if image is None: continue
        image_count += 1
        raw_detections = run_sliding_window(image, model, threshold=parameters.DETECTION_THRESHOLD)
        final_detections = non_maximal_suppression(raw_detections, overlap_threshold=0.3)

        for det in final_detections:
            x1, y1, x2, y2, score = det
            all_detections.append([x1, y1, x2, y2])
            all_scores.append(score)
            all_filenames.append(image_name)

        if image_count % 2 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time

            seconds_per_img = elapsed_time / image_count
            remaining_images = total_images - image_count
            seconds_left = remaining_images * seconds_per_img

            time_left_str = str(datetime.timedelta(seconds=int(seconds_left)))
            count_str = str(image_count).rjust(len(str(total_images)))

            print(f"[{time_left_str} left] - Processed {count_str}/{total_images} images")
    save_dir = parameters.TASK2_DIR
    if character == "all_faces":
        save_dir = parameters.TASK1_DIR
    save_npy_files(all_detections, all_scores, all_filenames, save_dir, character)


def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou


# def cleanup_task2_conflicts():
#
#     base_dir = parameters.TASK2_DIR
#     all_detections = []
#     for char_name in parameters.CHARACTERS:
#         det_path = os.path.join(base_dir, f"detections_{char_name}.npy")
#         score_path = os.path.join(base_dir, f"scores_{char_name}.npy")
#         name_path = os.path.join(base_dir, f"file_names_{char_name}.npy")
#         if not os.path.exists(det_path): continue
#
#         d = np.load(det_path, allow_pickle=True, encoding='latin1')
#         s = np.load(score_path, allow_pickle=True, encoding='latin1')
#         n = np.load(name_path, allow_pickle=True, encoding='latin1')
#
#         for i in range(len(d)):
#             all_detections.append({
#                 "box": d[i],
#                 "score": s[i],
#                 "file": n[i],
#                 "char": char_name
#             })
#     print(f"> Loaded {len(all_detections)} total detections.\n* resolving conflicts...")
#     all_detections.sort(key=lambda x: x["score"], reverse=True)
#
#     final_detections = []
#
#     for detection in all_detections:
#         is_conflict = False
#
#         for kept in final_detections:
#             if kept["file"] != detection["file"]:
#                 continue
#             iou = intersection_over_union(detection["box"], kept["box"])
#             if iou > 0.3:
#                 is_conflict = True
#                 break
#         if not is_conflict:
#             final_detections.append(detection)
#
#     print(f"Reduced to {len(final_detections)} clean detections.")
#     for character_name in parameters.CHARACTERS:
#         character_subset = [d for d in final_detections if d["char"] == character_name]
#         new_detections = [d["box"] for d in character_subset]
#         new_scores = [d["score"] for d in character_subset]
#         new_filenames = [d["file"] for d in character_subset]
#         save_npy_files(new_detections, new_scores, new_filenames, parameters.TASK2_DIR, character_name)