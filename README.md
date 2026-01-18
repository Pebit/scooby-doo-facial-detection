1. the libraries required to run the project including the full version of each library.
```
numpy==2.2.6  
opencv_python==4.12.0  
scikit_image==0.26.0  
scikit_learn==1.8.0
```

2. how to run your code and where to look for the output files.

To run the detection script on a new set of images, follow these steps:

**Step 1:** Configure Paths Open the file src/parameters.py and update the following variables to match your machine:
- VALIDATION_DATA_DIR: Set this to the parent folder containing your test data.
- IMAGES_DIR: Set this to the specific folder containing the .jpg images you want to test.

**Step 2:** Configure Execution Mode Open src/main.py and check the flags at the top of the file:
- Set calculate_all_detections = True (This runs the detection and generates solution files).
- Set build_dataset = False (Unless you want to regenerate training data but you need to add the images back to antrenare yourself).
- Set train_models = False (Unless you want to retrain the SVMs).

**Step 3:** Execute the script from the root directory:
```
python src/main.py
```

The script will generate the solution .npy files in the following directories:  
**Task 1:** evaluare/fisiere_solutie/333_Ionescu_Andrei/task1  
**Task 2:** evaluare/fisiere_solutie/333_Ionescu_Andrei/task2
