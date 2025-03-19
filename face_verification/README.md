# Face Verification

This repository contains various scripts and models for face verification using different methods. The primary focus is on leveraging TFLite models for efficient face verification, suitable for deployment on edge devices.

## Additional Models for Download
VGG16 Models: https://drive.google.com/drive/folders/1eMMwx83z2gOaQSYb-AWKj05oLTAsBKvx?usp=drive_link

## Folder Structure

```
face_verification/
├── archive/
│   ├── compare_vgg_tflite.py
│   ├── convert_to_tflite.py
│   ├── face_verification_old.py
│   ├── face_verification_only_tflite.py
│   └── face_verification.py
├── saved_faces/
├── debug_mobilefacenet.py
├── face_embeddings.pkl
├── face_profiling.py
├── face_verification_mobilefacenet.py
├── mobilefacenet_tf.pb
├── mobilefacenet.tflite
├── motion_detection.py
├── vgg16_feature_extractor.h5 (to download if profiling)
├── vgg16_feature_extractor.tflite (to download if profiling)
└── requirements.txt
```

## Files Description

### Archive

-   **compare_vgg_tflite.py**: Script to compare the performance and output similarity between VGG16 and TFLite models.
-   **convert_to_tflite.py**: Script to convert the VGG16 model to TFLite format and analyze tensor types before and after optimization.
-   **face_verification_old.py**: Old version of the face verification system using DeepFace.
-   **face_verification_only_tflite.py**: Face verification system using only TFLite model.
-   **face_verification.py**: Face verification system using VGG16 and TFLite models.

### Main Files

-   **debug_mobilefacenet.py**: Script to debug and get details on the TensorFlow and TFLite MobileFaceNet model.
-   **face_profiling.py**: Performance profiling tool for face detection and embedding generation pipelines.
-   **face_verification_mobilefacenet.py**: Main script for face verification using MobileFaceNet TFLite model and MediaPipe for face detection.
-   **motion_detection.py**: Script for motion detection using OpenCV.

### Models and Data

-   **mobilefacenet_tf.pb**: MobileFaceNet model in TensorFlow PB format.
-   **mobilefacenet.tflite**: MobileFaceNet model in TFLite format.
-   **face_embeddings.pkl**: Pickle file containing face embeddings.
-   **vgg16_feature_extractor.h5**: VGG16 model.
-   **vgg16_feature_extractor.tflite**: VGG16 TFLite quantized format.

### Directories

-   **saved_faces/**: Directory to store saved face images.

### Requirements

-   **requirements.txt**: List of required Python packages.

## Setup

To get the environment ready, install the required packages:

```
pip install -r requirements.txt
```

## Functionality

### 1. Face Verification with MobileFaceNet TFLite and MediaPipe

This section details how the face verification system works using the `face_verification_mobilefacenet.py` script. It uses the MobileFaceNet TFLite model for generating face embeddings and MediaPipe for face detection.

#### Workflow:

1.  **Face Detection**: Utilizes MediaPipe's face detection model to locate faces within an image or video frame.
2.  **Face Extraction**: Extracts the detected face from the image, applying padding to include contextual information.
3.  **Preprocessing**: Preprocesses the extracted face to match the input requirements of the MobileFaceNet model, including resizing and normalization.
4.  **Embedding Generation**: Generates a face embedding vector using the MobileFaceNet TFLite model.
5.  **Verification**: Compares the generated embedding with a database of known face embeddings to identify the person.

#### Usage:

To run the face verification script:

```
python face_verification_mobilefacenet.py
```

The script performs the following actions:

1.  Captures a face image using the webcam and saves it to the `saved_faces` folder with the filename "junxian.jpg".
2.  Builds face embeddings from the images in the `saved_faces` folder.
3.  Waits for a face to appear in the webcam feed and attempts to verify it against the generated embeddings.

#### Key Functions:

-   `detect_face()`: Detects faces in a frame using MediaPipe.
-   `extract_face()`: Extracts a face from a frame with padding.
-   `preprocess_face()`: Preprocesses the face image for the MobileFaceNet model.
-   `get_face_embedding()`: Generates the face embedding vector using the MobileFaceNet TFLite model.
-   `build_embeddings_from_folder()`: Builds a database of face embeddings from images in a folder.
-   `verify_face()`: Verifies a face against the embeddings database.
-   `wait_for_face_and_verify()`: Waits for a face to appear and then verifies it against the database.

### 2. Motion Detection

The `motion_detection.py` script implements motion detection using OpenCV.

#### Workflow:

1.  **Frame Capture**: Captures video frames from a specified camera.
2.  **Preprocessing**: Converts each frame to grayscale and applies a Gaussian blur to reduce noise.
3.  **Motion Detection**: Compares the current frame with the previous frame to identify differences. Thresholding is applied to highlight significant changes.
4.  **Contour Detection**: Identifies contours in the thresholded image, representing areas of motion.
5.  **Motion Indication**: If the area of any contour exceeds a predefined minimum area, motion is considered detected, and a message is printed.

#### Usage:

To run the motion detection script:

```
python motion_detection.py
```

This will start capturing video from the default camera and print a message to the console whenever motion is detected.

#### Key Parameters:

-   `camera_id`: ID of the camera to use.
-   `resolution`: Resolution of the captured video.
-   `threshold`: Threshold value for motion detection.
-   `min_area`: Minimum area of a contour to be considered motion.
-   `blur_size`: Size of the Gaussian blur kernel.
-   `check_interval`: Time interval between motion checks.

### 3. Profiling

The `face_profiling.py` script is designed to profile the performance of different face detection and embedding methods. It measures the time taken for each step in the face processing pipeline and collects system metrics such as CPU usage, memory usage, and I/O operations.

#### Workflow:

1.  **Initialization**: Initializes the FaceProfiler with specified models and configurations.
2.  **Image Loading**: Loads an image for processing.
3.  **Face Detection**: Detects faces in the image using either MediaPipe or OpenCV.
4.  **Face Extraction**: Extracts the detected face from the image.
5.  **Preprocessing**: Preprocesses the extracted face to match the input requirements of the chosen embedding model.
6.  **Embedding Generation**: Generates a face embedding vector using the specified model (MobileFaceNet, VGG16, etc.).
7.  **Metrics Collection**: Collects timing information for each step and system metrics such as CPU usage, memory usage, and I/O operations.
8.  **Reporting**: Prints a detailed report of the profiling results, including timing information and system metrics.

#### Usage:

To run the face profiling script:

```
python face_profiling.py
```

The script allows you to select different embedding methods and face detectors to compare their performance.

#### Key Components:

-   `FaceProfiler` class: Manages the face detection, extraction, preprocessing, and embedding generation processes.
-   `profile_pipeline()` function: Executes the face processing pipeline and collects timing and system metrics.
-   `print_profiling_results()` function: Prints a detailed report of the profiling results.