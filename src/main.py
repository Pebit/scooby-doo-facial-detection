import building_dataset
import training
import parameters
import detection

build_dataset = False
train_models = False
calculate_all_detections = False
test_main_detections = True

if build_dataset:
    building_dataset.load_positives()
    building_dataset.build_negatives()

if train_models:
    training.train_general_classifier()
    for character_name in parameters.CHARACTERS:
        training.train_classifier(character_name)

if calculate_all_detections and not test_main_detections:
    detection.generate_task_files(character="all_faces")
    for character_name in parameters.CHARACTERS:
        detection.generate_task_files(character_name)

if test_main_detections:
    detection.generate_task_files(character="all_faces")
    detection.generate_task_files(character=parameters.CHARACTERS[1])

