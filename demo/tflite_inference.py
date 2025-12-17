import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import argparse

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128,128))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = preprocess_image(args.input)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    print("Prediction:", "Deepfake" if output[0][0] > 0.5 else "Real")
