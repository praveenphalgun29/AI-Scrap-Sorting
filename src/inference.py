
import onnxruntime as ort
import numpy as np
from PIL import Image

class ScrapperONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Feed the raw image array directly into the model
        ort_inputs = {self.ort_session.get_inputs()[0].name: img_array}
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        # --- THE FINAL FIX ---
        # The ONNX model's output is ALREADY the final probabilities.
        # We do not need to apply softmax again.
        probs = ort_outs[0][0]

        prediction_index = np.argmax(probs)
        confidence = float(probs[prediction_index]) # Just grab the highest probability
        predicted_class = self.class_names[prediction_index]

        return predicted_class, confidence
