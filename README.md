
# **INF2009-Team6**: Dual-Gate Access Authentication with 2-Factor Authentication (Voice + Camera)

## Project Description
This project is a dual-gate access authentication system that uses 2-factor authentication. The system is composed of two gates. The first camera on first gate is the face recognition gate, which uses the camera to capture the face of the person trying to enter. The second camera on first gate will prevent tailgating. The second gate is the face and voice recognition gate, which uses the camera to capture the face of the person trying to enter and the microphone to capture the voice of the person trying to enter. The system will only allow access if the person is recognized by both gates. The system will also have a web interface that will allow the user to add new users, delete users, and view the logs of the system.

![Frame 1 (3)](https://github.com/user-attachments/assets/4ae08189-72d1-4743-97dc-dbc8322823b2)


## 🔗 Links
![Static Badge](https://img.shields.io/badge/Poster_Link-Canva-blue?style=for-the-badge&logo=canva&link=https%3A%2F%2Fwww.canva.com%2Fdesign%2FDAGcoSWfQ7E%2F206BhRh_AZRSofwBzGcc8g%2Fview%3Futm_content%3DDAGcoSWfQ7E%26utm_campaign%3Ddesignshare%26utm_medium%3Dlink2%26utm_source%3Duniquelinks%26utlId%3Dhceeff75b2a)

![Static Badge](https://img.shields.io/badge/Demo_Video_Link-Youtube-red?style=for-the-badge&logo=youtube&link=https%3A%2F%2Fwww.canva.com%2Fdesign%2FDAGcoSWfQ7E%2F206BhRh_AZRSofwBzGcc8g%2Fview%3Futm_content%3DDAGcoSWfQ7E%26utm_campaign%3Ddesignshare%26utm_medium%3Dlink2%26utm_source%3Duniquelinks%26utlId%3Dhceeff75b2a)

# Setup

### Fog Device (PC)
1. Install Docker on the fog device.
2. Run the following command to start the MQTT broker container.
    ```bash
    docker run -d --name mqtt-broker \
    -p 1883:1883 \
    -e MOSQUITTO_USERNAME=mosquitto \
    -e MOSQUITTO_PASSWORD=mosquitto \
    ekiost/mqtt-broker:latest\


### Edge Device (Raspberry PI)

1. Install all required libraries from the `requirements.txt` by running this command:
```
pip install -r requirements.txt
```

2. Facial verification has multiple models you may choose to run. Additional models for download can be found [here](https://drive.google.com/drive/folders/1eMMwx83z2gOaQSYb-AWKj05oLTAsBKvx?usp=drive_link)

- The System can be further configured through a dictionary containing various parameters should you choose to:
```python
face_verification_config = {
    "model_path": "mobilefacenet.tflite",
    "database_path": "face_embeddings.pkl",
    # Face detection & preprocessing settings:
    "model_selection": 0,
    "min_detection_confidence": 0.7,
    "padding": 0.2,
    "face_required_size": (512, 512),
    "target_size": (112, 112),
    # Verification settings:
    "verification_threshold": 0.7,
    "verification_timeout": 30,
    "verification_max_attempts": 3,
    # Camera settings:
    "camera_id": 0,
}
```
3. Human detection has multiple models to be chosen too. Navigate to yolov5 folder

```
-- File Structure --
INF2009-Team6/
├── humanDetection/
│   ├── tflite-human-tracking/
│   │   ├── models/
│   │   │   └── yolov5 <--
│   │   └── ...
│   └── ...
└── ...
```

To get yolov5 tflite models, you can use yolov5/export.py by running these commands:
```
git clone git@github.com:ultralytics/yolov5.git

cd yolov5

pip install -r requirements.txt

python export.py --weights yolov5n.pt --include  tflite
```

# Documentation

## 1. Features
These are the libraries and hardware modules used within this project. (Not exhaustive)
### 1.1 💻Software
- paho-mqtt
- Flask, React (Next js)
- noisereduce (noise reduction for sound)
- resemblyzer (Deep-learning sound embedding)
- Google Speech Recognition
- Librosa
- MediaPipe
- Yolov5
- Docker

### 1.2 🪫Hardware
- Raspberry Pi 4/5
- Webcam w/ inbuilt microphone
- I2C LCD
- Buzzer
- Ultrasonic sensor
- Servo

## 2. 🗣️ Communication
The system will use MQTT and REST to communicate between the two gates and the fog device. 

### 2.1 MQTT Topics
The system will use the following topics to communicate between the two gates and the fog device.
- `gate_1\status`: This topic is used by the first gate to publish its status. 
    ```json
    {
        "opened": "2025-02-28 21:52:39"
    }
    ```
    ```json
    {
        "closed": "2025-02-28 21:52:39"
    }
    ```
- `gate_2\status`: This topic is used by the second gate to publish its status. 
    ```json
    {
        "opened": "2025-02-28 21:52:39"
    }
    ```
    ```json
    {
        "closed": "2025-02-28 21:52:39"
    }
    ```
- `verified`: This topic is used to tell the second gate that the person has been verified by the first gate. The message will contain the personnel ID of the person.
    ```json
    {
        "personnel_id": "123456"
    }
    ```
- `alert`: This topic is used to send an alert to the fog device and let first gate proceed to the next state.
    ```json
    {
        "message": "multi",
        "picture": "base64 encoded picture"
    }
    ```
    ```json
    {
        "message": "diff",
        "picture": "base64 encoded picture"
    }
    ```
- `command`: This topic is used to send a command to the first gate to open or close the gate.
    ```json
    {
        "command": "open"
    }
    ```
    ```json
    {
        "command": "close"
    }
    ```
- `update/embeddings`: This topic is used to tell the pi to update the embeddings of the personnel.
    ```json
    {
        "face": "filename.pkl",
        "voice": "filename.pkl"
    }

### 2.2 API (TODO: Add more details) 
The system will use the following API to get the embeddings of the personnel.
- `GET /embeddings/face/filename.pkl`: This API is used to get the face embeddings of the personnel.
- `GET /embeddings/voice/filename.pkl`: This API is used to get the voice embeddings of the personnel.


### 2.3 Classes
- Gate: this class will represent the gate and will have the following class
    - EventManager: this class will manage the events that occur at the gate from the MQTT broker
    - StateManager: this class will manage the state of the gate
    - UpdateManager: this class will manage the updates that occur at the gate from the MQTT broker
- Network: 
    - MQTT:
        - Publisher: this class will publish messages to the MQTT broker
        - Subscriber: this class will subscribe to messages from the MQTT broker
    - API:
        - UpdateDownloader: this class will download the embeddings of the personnel from the API
- Utils:
    - LoggerMixin: this class will log the messages to the console 
- Enums:
    - GateState: this enum will represent the state of the gate
    - GateType: this enum will represent the type of the gate

## 3. 🪪 Face Verification
### 3.1 Main Files

-   **debug_mobilefacenet.py**: Script to debug and get details on the TensorFlow and TFLite MobileFaceNet model.
-   **face_profiling.py**: Performance profiling tool for face detection and embedding generation pipelines.
-   **face_verification_mobilefacenet.py**: Main script for face verification using MobileFaceNet TFLite model and MediaPipe for face detection.
-   **motion_detection.py**: Script for motion detection using OpenCV.

### 3.2 Models and Data

-   **mobilefacenet_tf.pb**: MobileFaceNet model in TensorFlow PB format.
-   **mobilefacenet.tflite**: MobileFaceNet model in TFLite format.
-   **face_embeddings.pkl**: Pickle file containing face embeddings.
-   **vgg16_feature_extractor.h5**: VGG16 model.
-   **vgg16_feature_extractor.tflite**: VGG16 TFLite quantized format.

### 3.3 Directories

-   **saved_faces/**: Directory to store saved face images.

### 3.4 Functionality

#### 3.4.1 Face Verification with MobileFaceNet TFLite and MediaPipe

This section details how the face verification system works using the `face_verification_mobilefacenet.py` script. It uses the MobileFaceNet TFLite model for generating face embeddings and MediaPipe for face detection.

#### 3.4.2 Configuration

The system can be configured through a dictionary containing various parameters:

```python
face_verification_config = {
    "model_path": "mobilefacenet.tflite",
    "database_path": "face_embeddings.pkl",
    # Face detection & preprocessing settings:
    "model_selection": 0,
    "min_detection_confidence": 0.7,
    "padding": 0.2,
    "face_required_size": (512, 512),
    "target_size": (112, 112),
    # Verification settings:
    "verification_threshold": 0.7,
    "verification_timeout": 30,
    "verification_max_attempts": 3,
    # Camera settings:
    "camera_id": 0,
}
```

#### 3.4.3 Enhanced Image Preprocessing

The system includes advanced image preprocessing techniques to improve face recognition accuracy:

1. **White Balance Correction**: Adjusts color balance using LAB color space
2. **Adaptive Gamma Correction**: Dynamically adjusts image brightness based on average luminance
3. **Contrast Normalization**: Enhances image contrast using LAB color space
4. **Model-Specific Normalization**: Applies final normalization based on model requirements

#### 3.4.4 Workflow:

1. **Face Detection**: Utilizes MediaPipe's face detection model to locate faces with configurable confidence threshold
2. **Face Extraction**: Extracts the detected face with configurable padding and resizing
3. **Preprocessing**: Applies enhanced image preprocessing pipeline
4. **Embedding Generation**: Generates a 512-dimensional face embedding vector using MobileFaceNet
5. **Verification**: Compares embeddings using cosine similarity with configurable threshold

#### 3.4.5 Key Functions:

- `detect_face()`: Face detection with MediaPipe
- `extract_face()`: Face extraction with padding
- `preprocess_face()`: Enhanced image preprocessing pipeline
- `get_face_embedding()`: Face embedding generation
- `verify_face()`: Face verification against database
- `wait_for_face_and_verify()`: Interactive verification with timeout and retry
- `build_embedding_from_folder()`: Build embeddings database from folder structure
- `build_embedding_from_images()`: Build embeddings from image list
- `capture_mismatch()`: Capture and encode mismatched faces

#### 3.4.6 Embeddings Storage and Database Management

The system uses a flexible database structure for storing face embeddings:

```python
# Example embeddings database structure
{
    "person1_name": [
        embedding1,  # numpy array of shape (512,)
        embedding2,  # Multiple embeddings per person
        ...
    ],
    "person2_name": [
        embedding1,
        ...
    ]
}
```

#### 3.4.7 Key features:
- **Multiple Embeddings**: Stores multiple embeddings per person for better recognition
- **Persistent Storage**: Saves embeddings to disk using pickle serialization
- **Database Management**:
  - `save_face_embeddings()`: Merges new embeddings with existing database
  - `load_face_embeddings()`: Loads embeddings from disk with error handling
  - `build_embedding_from_folder()`: Builds database from folder structure
  - `build_embedding_from_images()`: Builds database from image list

Folder structure for training data:
```
faces/
├── person1_name/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2_name/
│   ├── image1.jpg
│   └── ...
```

#### 3.4.8 Key Functions:

-   `detect_face()`: Detects faces in a frame using MediaPipe.
-   `extract_face()`: Extracts a face from a frame with padding.
-   `preprocess_face()`: Preprocesses the face image for the MobileFaceNet model.
-   `get_face_embedding()`: Generates the face embedding vector using the MobileFaceNet TFLite model.
-   `build_embeddings_from_folder()`: Builds a database of face embeddings from images in a folder.
-   `verify_face()`: Verifies a face against the embeddings database.
-   `wait_for_face_and_verify()`: Waits for a face to appear and then verifies it against the database.

#### 3.4.9 Profiling
The `face_profiling.py` script is designed to profile the performance of different face detection and embedding methods. It measures the time taken for each step in the face processing pipeline and collects system metrics such as CPU usage, memory usage, and I/O operations.

##### 3.4.9.1 Workflow

1.  **Initialization**: Initializes the FaceProfiler with specified models and configurations.
2.  **Image Loading**: Loads an image for processing.
3.  **Face Detection**: Detects faces in the image using either MediaPipe or OpenCV.
4.  **Face Extraction**: Extracts the detected face from the image.
5.  **Preprocessing**: Preprocesses the extracted face to match the input requirements of the chosen embedding model.
6.  **Embedding Generation**: Generates a face embedding vector using the specified model (MobileFaceNet, VGG16, etc.).
7.  **Metrics Collection**: Collects timing information for each step and system metrics such as CPU usage, memory usage, and I/O operations.
8.  **Reporting**: Prints a detailed report of the profiling results, including timing information and system metrics.

##### 3.4.9.2 Key Components

-   `FaceProfiler` class: Manages the face detection, extraction, preprocessing, and embedding generation processes.
-   `profile_pipeline()` function: Executes the face processing pipeline and collects timing and system metrics.
-   `print_profiling_results()` function: Prints a detailed report of the profiling results.

##### 3.4.9.3 Results

The following results were obtained when performing profiling on a Raspberry Pi 5.

**Model Loading**
| Model              | Model Size (MB) | Memory Usage (MB) | CPU Usage (%) | Load Time (seconds) |
| ------------------ | --------------- | ----------------- | ------------- | ------------------- |
| MobileFaceNet (TFLite) | 5.0             | 17.5              | 98.9          | 0.01                |
| VGG (TFLite)       | 128.2           | 283.5             | 100.9         | 0.38                |
| MobileFaceNet (PB)   | 5.2             | 52.73             | 99.9          | 1.48                |
| VGG (Keras)        | 512.2           | 585.83            | 91.7          | 5.4                 |

**Pipeline**
| Framework | Model | Detection (seconds) | Extraction (seconds) | Preprocessing (seconds) | Embedding (seconds) | Total Pipeline Time (seconds) | Function Calls | Primitive Calls |
|---|---|---|---|---|---|---|---|---|
| MediaPipe | MobileFaceNet (TFLite) | 0.0207 | 0 | 0.0038 | 0.0246 | 0.049 | 8003 | 7768 |
| MediaPipe | MobileFaceNet (PB) | 0.0213 | 0 | 0.0039 | 0.0357 | 0.0609 | 8604 | 8203 |
| OpenCV | MobileFaceNet (TFLite) | 0.0465 | 0.0001 | 0.0037 | 0.0246 | 0.0749 | 377 | 372 |
| OpenCV | MobileFaceNet (PB) | 0.0452 | 0.0001 | 0.0039 | 0.0353 | 0.0845 | 682 | 669 |
| MediaPipe | VGG (TFLite) | 0.0211 | 0 | 0.0094 | 0.2534 | 0.2839 | 8004 | 7769 |
| OpenCV | VGG (TFLite) | 0.0545 | 0.0001 | 0.0095 | 0.2516 | 0.3156 | 378 | 373 |
| MediaPipe | VGG (Keras) | 0.0227 | 0 | 0.0101 | 0.7241 | 0.7569 | 473804 | 454125 |
| OpenCV | VGG (Keras) | 0.0552 | 0.0001 | 0.0099 | 0.7708 | 0.836 | 71530 | 68144 |

Based on the profiling results:

*   **Model Loading:** VGG (Keras) has the largest model size and load time, while MobileFaceNet (TFLite) is the smallest and fastest to load.
*   **Pipeline Performance:** MobileFaceNet (TFLite) with MediaPipe offers the fastest total pipeline time. VGG (Keras) is significantly slower, especially in the embedding generation step.
*   **Framework Impact:** MediaPipe generally results in faster face detection compared to OpenCV.
*   **Model Format Matters:** TFLite models are generally faster than their TensorFlow Frozen Graph (PB) or Keras counterparts.
*   **Function Calls:** VGG (Keras) has a significantly higher number of function calls, indicating greater complexity or overhead.

## 4. 🚶‍♂️‍➡️ Motion Detection

The `motion_detection.py` script implements motion detection using OpenCV.

### 4.1 Workflow:

1.  **Frame Capture**: Captures video frames from a specified camera.
2.  **Preprocessing**: Converts each frame to grayscale and applies a Gaussian blur to reduce noise.
3.  **Motion Detection**: Compares the current frame with the previous frame to identify differences. Thresholding is applied to highlight significant changes.
4.  **Contour Detection**: Identifies contours in the thresholded image, representing areas of motion.
5.  **Motion Indication**: If the area of any contour exceeds a predefined minimum area, motion is considered detected, and a message is printed.

### 4.2 Key Parameters:

-   `camera_id`: ID of the camera to use.
-   `resolution`: Resolution of the captured video.
-   `threshold`: Threshold value for motion detection.
-   `min_area`: Minimum area of a contour to be considered motion.
-   `blur_size`: Size of the Gaussian blur kernel.
-   `check_interval`: Time interval between motion checks.

## 5. 🧑‍🤝‍🧑 Human detection
Within the airlock itself, we should only allow 1 person in at a single moment. Hence, we have to detect if multiple people are within it. 

There are multiple ways of doing so: we can use opencv natively and code out a human detection. This is simple and extremely lightweight, but highly inefficient due to how insufficiently trained it is.

![photo_2025-03-25_02-24-36](https://github.com/user-attachments/assets/60ce28ef-7a29-418d-9d54-2f971941e4ff)

*OpenCV incorrectly detecting humans*

Another method of approaching this is to use Tensorflow Lite human detection system built on top of YOLOv5 tflite model. Utilizing both Tensorflow lite and YOLOv5 models, this human detection system is much more effective in accruately detecting people, especially when there is multiple people (i.e. detecting a city square). Although, a stronger hardware would be required.

![Screenshot 2025-03-25 022614](https://github.com/user-attachments/assets/581d26d3-ecfb-495b-86bd-ebff2fd88918)

*YOLOv5 tflite detecting humans more efficiently in a crowded square*

## 6. 📌 State Diagram
The state table and state diagram are shown below. The state diagram is a visual representation of the state table. The state table shows the states of the system, the conditions that trigger the transitions between states, and the actions to take when transitioning between states. The state diagram shows the states of the system as nodes and the transitions between states as edges. The state diagram is a useful tool for understanding the behavior of the system and for designing the system.

```mermaid
stateDiagram
    [*] --> 1
    1 --> 2: Motion detected (Cam 1)
    2 --> 3: Face verified
    3 --> 4: Ultrasonic detected personnel entered
    4 --> 7: Multiple personnel detected in mantrap
    4 --> 8: Mantrap only one personnel and received the personnel ID
    8 --> 9: Face or voice not verified X number of times
    8 --> 10: Face verified
    10 --> 11: Voice verified
    11 --> 1: Voice verified
    1 --> 5: Security guard came and checked
    5 --> 1: Wait for all personnel to leave
    4 --> 6: Unauthorized access in mantrap
    6 --> 5: Alert (incl buzzer and led) and wait for security guard
    7 --> 6: Send alert and picture to fog
    9 --> 6: Send alert and picture to fog
```


| State ID | Gate 1 Status | Mantrap Status | Gate 2 Status | Condition | Preceding States | Next States | Action to take |
|----------|--------------|----------------|---------------|-----------|------------------|-------------|----------------|
| 1 | Close | Idle | Close | Default, Face not verified | *, 2, 5, 11 | 2 | Wait for personnel to approach |
| 2 | Face | Idle | Close | Motion detected (Cam 1) | 1 | 1, 3 | Enable facial verification (Cam 1) |
| 3 | Open | Idle | Close | Face verified | 2 | 4 | Wait for personnel to enter mantrap |
| 4 | Close | Scan | Close | Ultrasonic detected personnel entered | 3 | 7, 8 | Enable human detection scan (Cam 2) |
| 5 | Open | Checked | Close | Security guard came and checked | 6 | 1 | Wait for all personnel to leave |
| 6 | Close | Alert | Close | Unauthorized access in mantrap | 7 | 5 | Alert (incl buzzer and led) and wait for security guard |
| 7 | Close | Multi | Close | Multiple personnel detected in mantrap | 4 | 6 | Send alert and picture to fog |
| 8 | Close | Idle | Face | Mantrap only one personnel and received the personnel ID | 4 | 9, 10 | Enable facial verification (Cam 3) |
| 9 | Close | Idle | Diff | Face or voice not verified X number of times | 8 | 6 | Send alert and picture to fog |
| 10 | Close | Idle | Voice | Face verified | 8 | 9, 11 | Enable voice verification (Cam 3) |
| 11 | Close | Idle | Open | Voice verified | 10 | 1 | Wait for personnel to exit the mantrap |


# 🚀 Developer's Diary
## 🔊 Voice Authentication
As voice authentication was a vital part of the authentication process, including both the voice password and the voice signature analysis, the system was broken down into two parts accordingly.

### 1️⃣ Starting Premise
The edge devices would be connected through LAN cables to the internet, providing secured access through the premise's internet and firewall. 

The voice authentication process occurs **after** confirming the identity of the personnel, meaning there is no need to determine their identity beforehand.

Voice enrollment is performed on the fog device, while authentication is carried out on the edge. Thus, updates to personnel’s unique voice passwords are transferred from the fog device to the edge device.

Ensuring the **security** of voice authentication during these updates is **critical**.

### ⚙️ Approach to Premise
A **simple solution** was chosen to match voice signatures and passwords **without** relying on deep learning models (which can be heavyweight and have costlier inference). 

The approach involves matching voice signatures by calculating the **linear normalized distance** of the MFCC features between the enrolled and authentication voice samples. This allows developers to **customize a threshold value** based on testing to separate voice signatures.

A **speech recognition library** was used to capture the voice passwords. Both speed ⚡ and accuracy 🎯 were crucial, leading to testing (below) to identify the most suitable library.

After recognition, **voice passwords are hashed and serialized** for secure transmission to the edge device.

### 🧪 Testing
🛠️ Five speech recognition models were tested:
- Sphinx
- Google Speech Recognition
- Wit.ai
- Houdify
- Whisper

**Google Speech Recognition** provided the **best accuracy** and **speed** for word inference.

Since authentication involves waiting, minimizing **wait times** was a priority.
![image](https://github.com/user-attachments/assets/e269a903-95e3-4ff1-92c3-15adabe6ca51)

### 🚨 Problems Encountered
🛠️ Due to differences in **hardware** on edge and fog devices, the team suspected a **significant amount of noise** was picked up by the edge device. 

**Fog devices (often laptops) apply noise reduction automatically**, leading to cleaner recordings. Additionally, the differences in hardware (I.E. the microphone) might affect the output. 

Edge devices, however, recorded **noisier audio**, causing **failed authentication attempts**.

### 🛠️ Solutions Employed
The team **split development** into two alternate paths:

1️⃣ **Lightweight approach** (MFCC features comparison)
   - **Noise reduction** using the `noisereduce` library when recording on the edge device.
   - A **first authentication pass** using the noise-reduced WAV file to extract MFCC features.
   - If this **fails**, a **second authentication pass** applies **noise-reduction on MFCC features** instead.

2️⃣ **Noise-reduction profile development**
   - **Recorded samples** from fog and edge devices were analyzed.
   - **Signal-to-noise ratio** was calculated for each feature.
   - Features were adjusted by a **scale factor** to improve authentication accuracy.

📊 **Example testing results**:
![image](https://github.com/user-attachments/assets/9214537e-35ec-4c56-b9db-9228a104f752)

📌 **Observations**:
- ✅ **Same individual authentication**: After denoising, embedding distance **reduced by 57 points** (125.39 → 67.61).
- ❌ **Different individual authentication**: Embedding distance **only reduced by 10 points** (104.06 → 93.69).

This **simpler algorithm** selectively **reduces embeddings using noise estimates**, enabling **more reliable authentication** across **Fog and Edge devices** without deep learning.

### 🧠 Alternative Deep Learning Approach
**Deep-learning-based method**:
- Utilized `resemble-ai/Resemblyzer` ([GitHub](https://github.com/resemble-ai/Resemblyzer))
- Generates a **256-value summary vector** per audio file.

**Comparison**:
| Approach | Features Used | Speed | Storage  | Noise Handling  |
|----------|--------------|---------|----------|----------------|
| **Lightweight (MFCC)** | 20 values | ✅ Fast | ✅ Low | ⚠️ Moderate |
| **Deep Learning (Resemblyzer)** | 256 values | ❌ Slower | ❌ High | ✅ Strong |

🔍 The **deep-learning approach** accounts **better** for:
- **Different recording hardware**
- **Environmental noise**
  
However, it has **higher inference time** and **storage costs**.

---
✅ **Final Decision:**  
For **lightweight** and **efficient** authentication across devices, **the MFCC-based approach was prioritized**. Deep learning remains an **alternative** for future enhancements.

🚀 **Next Steps:**
- Optimize **threshold calibration** for different noise profiles.
- Improve **real-time denoising** on edge devices.
- Evaluate **hybrid approaches** for further robustness.

---
🎉 **Conclusion:**  
Through **adaptive noise reduction** and **threshold tuning**, a **lightweight, scalable** voice authentication system was implemented—ensuring **fast, reliable** authentication without **heavy AI dependency**.
