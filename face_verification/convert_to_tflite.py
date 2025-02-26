# Run on your main desktop, not on the Raspberry Pi.
# This script will convert the VGG16 model to TFLite format and analyze the tensor types before and after optimization. The script will output the tensor types and the size of the original and optimized models.
# The size reduction percentage will also be displayed.

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

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

def prepare_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

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

if __name__ == "__main__":
    convert_to_tflite()