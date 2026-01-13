import parameters

def image_dir(name):
    if name in parameters.CHARACTERS:
        return f"{parameters.TRAINING_DIR}/{name}/"
    return None

def annotations_path(name):
    if name in parameters.CHARACTERS:
        return f"{parameters.TRAINING_DIR}/{name}_annotations.txt"
    return None