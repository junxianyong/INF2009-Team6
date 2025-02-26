# Run on Raspberry Pi, not on your main desktop.
# This script will load both the VGG16 and TFLite models and compare their performance using a random test image.
# The script will output the average inference time for both models, the speedup factor, and the output similarity between the two models.

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import time
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def analyze_tensor_types(interpreter=None, model=None):
    tensor_details = {}
    
    if interpreter:
        details = interpreter.get_tensor_details()
        for detail in details:
            dtype = str(detail['dtype']).split('.')[-1].strip("'>")  # Clean up dtype string
            tensor_details[dtype] = tensor_details.get(dtype, 0) + 1
    else:
        for layer in model.layers:
            for weight in layer.weights:
                dtype = str(weight.dtype).split('.')[-1]  # Just get the dtype name directly
                tensor_details[dtype] = tensor_details.get(dtype, 0) + 1
    
    return tensor_details

def generate_test_image():
    """Generate a random test image of size 224x224x3"""
    # Generate integers first, then convert to float32
    return np.random.randint(0, 256, (1, 224, 224, 3)).astype(np.float32)

def prepare_image(image_array):
    """Prepare image array for model input"""
    # Ensure values are between 0 and 255
    image_array = np.clip(image_array, 0, 255)
    return image_array

def analyze_model(model_path=None, model_object=None, sample_image_path=None):
    if model_path:
        size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        interpreter = tf.lite.Interpreter(model_path=model_path)
        tensor_types = analyze_tensor_types(interpreter=interpreter)
    else:
        size = model_object.count_params() * 4 / (1024 * 1024)  # Rough estimation in MB
        tensor_types = analyze_tensor_types(model=model_object)
    
    return size, tensor_types

def convert_to_tflite():
    print("Loading VGG16 model...")
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    
    # Analyze original model
    original_size, original_types = analyze_model(model_object=model)
    print(f"\nBefore Optimization:")
    print(f"Model size: {original_size:.2f} MB")
    print("Tensor types:")
    for dtype, count in original_types.items():
        print(f"- {dtype}: {count} tensors")
    
    # Convert the model to TFLite format
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save the TFLite model
    tflite_path = 'vgg16_feature_extractor.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Analyze optimized model
    optimized_size, optimized_types = analyze_model(model_path=tflite_path)
    
    print("\nAfter Optimization:")
    print("Tensor types:")
    for dtype, count in optimized_types.items():
        print(f"- {dtype}: {count} tensors")
    
    print("\nOptimization Summary:")
    print(f"- Original model size: {original_size:.2f} MB")
    print(f"- Optimized TFLite model size: {optimized_size:.2f} MB")
    print(f"- Size reduction: {((original_size - optimized_size) / original_size * 100):.1f}%")

def load_models():
    """Load both VGG16 and TFLite models"""
    print("Loading VGG16 model...")
    base_model = VGG16(weights='imagenet')
    vgg_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    
    print("\nLoading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path='vgg16_feature_extractor.tflite')
    interpreter.allocate_tensors()
    
    return vgg_model, interpreter

def compare_performance(vgg_model, tflite_interpreter, image_array, num_runs=5):
    """Compare performance between VGG16 and TFLite"""
    # Prepare image
    input_data = prepare_image(image_array)
    
    # Get TFLite input/output details
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    
    # Warm-up run
    _ = vgg_model.predict(input_data)
    tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
    tflite_interpreter.invoke()
    
    # Test VGG16
    vgg_times = []
    vgg_results = []
    print("\nTesting VGG16...")
    for i in range(num_runs):
        start_time = time.time()
        result = vgg_model.predict(input_data)
        end_time = time.time()
        vgg_times.append(end_time - start_time)
        vgg_results.append(result)
        print(f"Run {i+1}: {vgg_times[-1]:.4f} seconds")
    
    # Test TFLite
    tflite_times = []
    tflite_results = []
    print("\nTesting TFLite...")
    for i in range(num_runs):
        start_time = time.time()
        tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
        tflite_interpreter.invoke()
        result = tflite_interpreter.get_tensor(output_details[0]['index'])
        end_time = time.time()
        tflite_times.append(end_time - start_time)
        tflite_results.append(result)
        print(f"Run {i+1}: {tflite_times[-1]:.4f} seconds")
    
    # Calculate statistics
    print("\nPerformance Summary:")
    print(f"VGG16 Average: {np.mean(vgg_times):.4f} seconds (±{np.std(vgg_times):.4f})")
    print(f"TFLite Average: {np.mean(tflite_times):.4f} seconds (±{np.std(tflite_times):.4f})")
    print(f"Speedup: {np.mean(vgg_times)/np.mean(tflite_times):.2f}x")
    
    # Compare results
    vgg_mean = np.mean(vgg_results, axis=0)
    tflite_mean = np.mean(tflite_results, axis=0)
    similarity = np.corrcoef(vgg_mean.flatten(), tflite_mean.flatten())[0,1]
    print(f"\nOutput Similarity: {similarity:.4f}")

if __name__ == "__main__":
    # Check if TFLite model exists
    if not os.path.exists('vgg16_feature_extractor.tflite'):
        print("Error: TFLite model not found. Please run conversion script first.")
        exit(1)
    
    # Load models
    vgg_model, tflite_interpreter = load_models()
    
    # Generate random test image
    print("Generating random test image...")
    test_image = generate_test_image()
    
    # Run comparison
    compare_performance(vgg_model, tflite_interpreter, test_image)