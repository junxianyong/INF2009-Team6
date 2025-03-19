import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import psutil
import os
import cProfile
import pstats
import tracemalloc

class FaceProfiler:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.target_size = (112, 112)
        self.padding = 0.2
        self._interpreter = None
        self.model_path = "mobilefacenet.tflite"
        self.pb_model_path = "mobilefacenet_tf.pb"
        self._pb_session = None  # Add session for .pb model
        self.vgg_model_path = "vgg16_feature_extractor.h5"
        self.vgg_tflite_path = "vgg16_feature_extractor.tflite"
        self.vgg_target_size = (224, 224)  # VGG required input size
        self._vgg_model = None
        self._vgg_interpreter = None
        # Add OpenCV face detector
        self.opencv_face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def get_tflite_interpreter(self):
        if self._interpreter is None:
            print("Loading MobileFaceNet TFLite model...")
            self._interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self._interpreter.allocate_tensors()
            print("Model loaded successfully.")
        return self._interpreter

    def get_pb_session(self):
        """Load and return the TensorFlow session for the .pb model."""
        if self._pb_session is None:
            print("Loading MobileFaceNet .pb model...")
            with tf.io.gfile.GFile(self.pb_model_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            
            with tf.compat.v1.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name='')
                self._pb_session = tf.compat.v1.Session(graph=graph)
                # Initialize variables if any
                self._pb_session.run(tf.compat.v1.global_variables_initializer())
            print("PB Model loaded successfully.")
        return self._pb_session

    def get_vgg_model(self):
        if self._vgg_model is None:
            print("Loading VGG16 model...")
            self._vgg_model = tf.keras.models.load_model(self.vgg_model_path)
            print("VGG model loaded successfully.")
        return self._vgg_model

    def get_vgg_tflite_interpreter(self):
        if self._vgg_interpreter is None:
            print("Loading VGG16 TFLite model...")
            self._vgg_interpreter = tf.lite.Interpreter(model_path=self.vgg_tflite_path)
            self._vgg_interpreter.allocate_tensors()
            print("VGG TFLite model loaded successfully.")
        return self._vgg_interpreter

    def detect_face_mediapipe(self, frame):
        """MediaPipe face detection implementation"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.7,
        ) as face_detection:
            results = face_detection.process(frame_rgb)
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = min(int(bbox.width * w), w - x)
                height = min(int(bbox.height * h), h - y)
                return (x, y, width, height)
        return None

    def detect_face_opencv(self, frame):
        """OpenCV face detection implementation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.opencv_face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        if len(faces) > 0:
            x, y, width, height = faces[0]
            return (x, y, width, height)
        return None

    def extract_face(self, frame, face_location):
        x, y, width, height = face_location
        h, w, _ = frame.shape
        
        # Add padding
        padding_x = int(width * self.padding)
        padding_y = int(height * self.padding)
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(w, x + width + padding_x)
        y2 = min(h, y + height + padding_y)
        
        # Extract face with padding
        face = frame[y1:y2, x1:x2]
                
        return face

    def preprocess_face(self, face_img, model_type='mobilefacenet'):
        """Preprocess face image according to model requirements."""
        if model_type in ['mobilefacenet', 'mobilefacenet_pb']:
            target_size = self.target_size
            face_img = cv2.resize(face_img, target_size)
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb.astype("float32")
            face_normalized = (face_normalized - 127.5) / 128.0
            return np.expand_dims(face_normalized, axis=0)
        else:  # VGG preprocessing
            target_size = self.vgg_target_size
            face_img = cv2.resize(face_img, target_size)
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb.astype("float32")
            # Use ImageNet preprocessing for VGG
            face_normalized[..., 0] -= 103.939
            face_normalized[..., 1] -= 116.779
            face_normalized[..., 2] -= 123.68
        
        return np.expand_dims(face_normalized, axis=0)

    def get_face_embedding(self, face_img, method='mobilefacenet'):
        if method == 'mobilefacenet':
            interpreter = self.get_tflite_interpreter()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], face_img)
            interpreter.invoke()
            embedding = interpreter.get_tensor(output_details[0]["index"])
        elif method == 'mobilefacenet_pb':
            session = self.get_pb_session()
            # Use correct tensor names from freeze.pb
            input_tensor = session.graph.get_tensor_by_name("input:0")
            output_tensor = session.graph.get_tensor_by_name("embeddings:0")
            
            # Apply same preprocessing as TFLite
            embedding = session.run(output_tensor, feed_dict={input_tensor: face_img})
            
            # Apply L2 normalization to make it comparable with TFLite output
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        elif method in ['vgg', 'vgg_tflite']:
            if method == 'vgg':
                model = self.get_vgg_model()
                embedding = model.predict(face_img, verbose=0)
            else:  # vgg_tflite
                interpreter = self.get_vgg_tflite_interpreter()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]["index"], face_img)
                interpreter.invoke()
                embedding = interpreter.get_tensor(output_details[0]["index"])
            
            # Apply L2 normalization followed by scaling
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            # Add scaling factor to increase distances between embeddings
            embedding = embedding * 10.0
        
        return embedding.flatten()

def profile_pipeline(profiler, image_path, detector='mediapipe', embedding_method='mobilefacenet'):
    """Profile each step of the face processing pipeline with system metrics."""
    timings = {}
    metrics = {
        'cpu': [],
        'memory': [],
        'io': {'read_bytes': 0, 'write_bytes': 0}
    }
    
    try:
        # Start resource monitoring
        process = psutil.Process()
        initial_io = process.io_counters()
        tracemalloc.start()
            
        # CPU profiler setup - wrap in try/except to avoid conflicts
        cpu_profiler = None
        try:
            cpu_profiler = cProfile.Profile()
            cpu_profiler.enable()
        except RuntimeError as e:
            print(f"Warning: Could not enable CPU profiler: {e}")
        
        # Read image and record initial metrics
        metrics['cpu'].append(process.cpu_percent())
        metrics['memory'].append(process.memory_info().rss / 1024 / 1024)  # MB
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return None, None
        
        # Face detection timing and metrics
        start_time = time.time()
        if detector == 'mediapipe':
            face_location = profiler.detect_face_mediapipe(frame)
        else:
            face_location = profiler.detect_face_opencv(frame)
        timings['detection'] = time.time() - start_time
        
        metrics['cpu'].append(process.cpu_percent())
        metrics['memory'].append(process.memory_info().rss / 1024 / 1024)
        
        if not face_location:
            print(f"No face detected in image using {detector}")
            return None, None
        
        # Face extraction timing and metrics
        start_time = time.time()
        face = profiler.extract_face(frame, face_location)
        timings['extraction'] = time.time() - start_time
        
        metrics['cpu'].append(process.cpu_percent())
        metrics['memory'].append(process.memory_info().rss / 1024 / 1024)
        
        # Preprocessing timing and metrics
        start_time = time.time()
        preprocessed_face = profiler.preprocess_face(face, 'mobilefacenet' if 'mobilefacenet' in embedding_method else embedding_method)
        timings['preprocessing'] = time.time() - start_time
        
        metrics['cpu'].append(process.cpu_percent())
        metrics['memory'].append(process.memory_info().rss / 1024 / 1024)
        
        # Embedding generation timing and metrics
        start_time = time.time()
        embedding = profiler.get_face_embedding(preprocessed_face, embedding_method)
        timings['embedding'] = time.time() - start_time
        
        # Final metrics collection
        metrics['cpu'].append(process.cpu_percent())
        metrics['memory'].append(process.memory_info().rss / 1024 / 1024)
        
        # Stop profilers and collect results
        if cpu_profiler:
            try:
                cpu_profiler.disable()
                stats = pstats.Stats(cpu_profiler)
                metrics['function_calls'] = stats.total_calls
                metrics['primitive_calls'] = stats.prim_calls
            except Exception as e:
                print(f"Warning: Could not disable CPU profiler: {e}")
                metrics['function_calls'] = 0
                metrics['primitive_calls'] = 0
        else:
            metrics['function_calls'] = 0
            metrics['primitive_calls'] = 0
            
        final_io = process.io_counters()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate I/O metrics
        metrics['io']['read_bytes'] = final_io.read_bytes - initial_io.read_bytes
        metrics['io']['write_bytes'] = final_io.write_bytes - initial_io.write_bytes
        metrics['peak_memory'] = peak / 1024 / 1024  # MB
        
        return timings, metrics
    
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        try:
            tracemalloc.stop()
        except:
            pass
        return None, None

def print_profiling_results(timings, metrics):
    """Print detailed profiling results including system metrics."""
    print("\nTiming Results:")
    print("-" * 60)
    for step, time_taken in timings.items():
        print(f"{step.replace('_', ' ').title():20}: {time_taken:.4f} seconds")
    
    print("\nSystem Metrics:")
    print("-" * 60)
    print(f"Average CPU Usage:      {np.mean(metrics['cpu']):6.2f}%")
    print(f"Peak Memory Usage:      {metrics['peak_memory']:6.2f} MB")
    print(f"Total I/O Read:         {metrics['io']['read_bytes']/1024:6.2f} KB")
    print(f"Total I/O Write:        {metrics['io']['write_bytes']/1024:6.2f} KB")
    print(f"Function Calls:         {metrics['function_calls']:6d}")
    print(f"Primitive Calls:        {metrics['primitive_calls']:6d}")

def compute_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    from scipy.spatial.distance import cosine
    return 1 - cosine(embedding1, embedding2)

def get_embedding_from_image(profiler, image_path, detector='mediapipe', embedding_method='mobilefacenet'):
    """Get embedding from a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    if detector == 'mediapipe':
        face_location = profiler.detect_face_mediapipe(frame)
    else:
        face_location = profiler.detect_face_opencv(frame)
    
    if not face_location:
        print(f"No face detected in image using {detector}")
        return None
    
    face = profiler.extract_face(frame, face_location)
    preprocessed_face = profiler.preprocess_face(face, embedding_method)
    embedding = profiler.get_face_embedding(preprocessed_face, embedding_method)
    
    return embedding

def measure_model_load_metrics():
    """Measure memory and CPU usage during model loading."""
    process = psutil.Process()
    tracemalloc.start()
    
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    initial_virtual_memory = process.memory_info().vms / 1024 / 1024  # MB
    initial_cpu = process.cpu_percent()
    initial_io = process.io_counters()
    
    metrics = {
        'initial_memory': initial_memory,
        'initial_virtual_memory': initial_virtual_memory,
        'initial_cpu': initial_cpu,
        'peak_memory': 0,
        'current_memory': 0,
        'virtual_memory': 0,
        'cpu_usage': 0,
        'io_read': 0,
        'io_write': 0,
        'load_time': 0
    }
    
    return process, tracemalloc, initial_io, metrics

def print_model_load_metrics(metrics):
    """Print metrics related to model loading."""
    print("\nModel Loading Metrics:")
    print("-" * 60)
    print(f"Initial Memory:         {metrics['initial_memory']:6.2f} MB")
    print(f"Current Memory:         {metrics['current_memory']:6.2f} MB")
    print(f"Memory Usage Change:    {metrics['memory_change']:6.2f} MB")
    print(f"Peak Memory Usage:      {metrics['peak_memory']:6.2f} MB")
    print(f"Virtual Memory:         {metrics['virtual_memory']:6.2f} MB")
    print(f"CPU Usage:             {metrics['cpu_usage']:6.2f}%")
    print(f"I/O Read:              {metrics['io_read']/1024:6.2f} KB")
    print(f"I/O Write:             {metrics['io_write']/1024:6.2f} KB")
    print(f"Load Time:             {metrics['load_time']:6.2f} seconds")

def main():
    print("\nInitializing face profiler...")
    profiler = FaceProfiler()

    # Available models
    print("\nAvailable embedding methods:")
    print("1. MobileFaceNet (TFLite)")
    print("2. VGG (Keras)")
    print("3. VGG (TFLite)")
    print("4. MobileFaceNet (Original PB)")  # Add new option
    
    while True:
        choice = input("\nSelect embedding method (1-4) or 'q' to quit: ")
        if choice.lower() == 'q':
            break
            
        try:
            choice = int(choice)
            if choice not in [1, 2, 3, 4]:  # Update valid choices
                print("Invalid choice. Please select 1-4.")
                continue
                
            embedding_method = {
                1: 'mobilefacenet',
                2: 'vgg',
                3: 'vgg_tflite',
                4: 'mobilefacenet_pb'  # Add new method
            }[choice]
            
            # Start measuring metrics before model loading
            process, tracemalloc, initial_io, metrics = measure_model_load_metrics()
            
            # Load the selected model
            print(f"\nLoading {embedding_method.upper()} model...")
            start_time = time.time()
            if embedding_method == 'mobilefacenet':
                profiler.get_tflite_interpreter()
            elif embedding_method == 'vgg':
                profiler.get_vgg_model()
            elif embedding_method == 'vgg_tflite':
                profiler.get_vgg_tflite_interpreter()
            else:  # mobilefacenet_pb
                profiler.get_pb_session()
            
            # Collect metrics after model loading
            final_memory = process.memory_info().rss / 1024 / 1024
            final_virtual_memory = process.memory_info().vms / 1024 / 1024
            current, peak = tracemalloc.get_traced_memory()
            final_io = process.io_counters()
            
            metrics.update({
                'current_memory': final_memory,
                'memory_change': final_memory - metrics['initial_memory'],
                'peak_memory': final_memory + (peak / 1024 / 1024),  # Add traced memory to RSS
                'virtual_memory': final_virtual_memory,
                'cpu_usage': process.cpu_percent(),
                'io_read': final_io.read_bytes - initial_io.read_bytes,
                'io_write': final_io.write_bytes - initial_io.write_bytes,
                'load_time': time.time() - start_time
            })
            
            tracemalloc.stop()
            print_model_load_metrics(metrics)

            test_images = [
                "saved_faces/billgates.jpg",
                "saved_faces/justinbieber.jpg",
                "saved_faces/selenagomez.jpg",
                "saved_faces/shawnmendes.jpg",
                "saved_faces/taylor.jpg",
            ]
            
            detectors = ['mediapipe', 'opencv']
            
            # Wrap the entire profiling section in a try-except to catch any failures
            try:
                for detector in detectors:
                    print(f"\n=== Testing {detector.upper()} + {embedding_method.upper()} ===")
                    
                    # Initial run
                    print(f"\nPerforming initial {detector} + {embedding_method} run...")
                    initial_timings, initial_metrics = profile_pipeline(profiler, test_images[0], detector, embedding_method)
                    if initial_timings:
                        print_profiling_results(initial_timings, initial_metrics)
                    else:
                        print(f"Warning: Initial run with {detector} + {embedding_method} failed")
                        continue
                    
                    # Multiple runs for average
                    print(f"\nPerforming multiple {detector} + {embedding_method} runs...")
                    all_timings = []
                    all_metrics = []
                    for i, image_path in enumerate(test_images[1:], 1):
                        print(f"\nProcessing image {i}/4...")
                        timings, metrics = profile_pipeline(profiler, image_path, detector, embedding_method)
                        if timings:
                            all_timings.append(timings)
                            all_metrics.append(metrics)
                        else:
                            print(f"Warning: Failed to process image {i}")
                    
                    if all_timings:
                        avg_timings = {
                            key: np.mean([t[key] for t in all_timings]) 
                            for key in all_timings[0].keys()
                        }
                        std_timings = {
                            key: np.std([t[key] for t in all_timings]) 
                            for key in all_timings[0].keys()
                        }
                        
                        print(f"\n{detector.upper()} + {embedding_method.upper()} Average Results (over {len(all_timings)} runs):")
                        print("-" * 60)
                        total_avg = sum(avg_timings.values())
                        for step in avg_timings.keys():
                            avg = avg_timings[step]
                            std = std_timings[step]
                            print(f"{step.replace('_', ' ').title():20}: {avg:.4f} ± {std:.4f} seconds")
                        print("-" * 60)
                        print(f"Total Pipeline Time:    {total_avg:.4f} seconds")
                    else:
                        print(f"No valid timing data collected for {detector} + {embedding_method}")
                
                print("\n" + "="*80 + "\n")
                
                # After the performance testing, add similarity testing
                print("\n=== Similarity Testing ===")
                test_image = "saved_faces/selenagomez_test.jpg"
                same_person = "saved_faces/selenagomez.jpg"
                different_person = "saved_faces/billgates.jpg"
                
                for detector in ['mediapipe', 'opencv']:
                    print(f"\nTesting similarities using {detector} detector:")
                    
                    # Get embeddings with error handling
                    test_embedding = get_embedding_from_image(profiler, test_image, detector, embedding_method)
                    same_embedding = get_embedding_from_image(profiler, same_person, detector, embedding_method)
                    diff_embedding = get_embedding_from_image(profiler, different_person, detector, embedding_method)
                    
                    if test_embedding is not None and same_embedding is not None and diff_embedding is not None:
                        # Compare with same person
                        same_similarity = compute_similarity(test_embedding, same_embedding)
                        print(f"Similarity with same person (Selena Gomez): {same_similarity:.4f}")
                        
                        # Compare with different person
                        diff_similarity = compute_similarity(test_embedding, diff_embedding)
                        print(f"Similarity with different person (Bill Gates): {diff_similarity:.4f}")
                    else:
                        print("Error: Could not compute similarities due to face detection failure")
                
                print("\n" + "="*80 + "\n")
            except Exception as e:
                print(f"Error during profiling process: {e}")
                import traceback
                traceback.print_exc()
            
        except ValueError:
            print("Invalid input. Please enter a number between 1-4.")

if __name__ == "__main__":
    main()