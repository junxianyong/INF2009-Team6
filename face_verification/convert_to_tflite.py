import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

def convert_to_tflite():
    # Load the VGG16 model
    print("Loading VGG16 model...")
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    
    # Convert the model to TFLite format
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimize for latency
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open('vgg16_feature_extractor.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite model saved as 'vgg16_feature_extractor.tflite'")

if __name__ == "__main__":
    convert_to_tflite()
