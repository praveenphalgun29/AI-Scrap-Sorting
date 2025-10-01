
import os
import time
import pandas as pd
from datetime import datetime
from inference import ScrapperONNXPredictor

# --- CONFIGURATION ---
MODEL_PATH = '/content/drive/MyDrive/AI_Scrap_Sorting/models/scrap_classifier.onnx'
IMAGE_FOLDER = '/content/drive/MyDrive/AI_Scrap_Sorting/test_images'
RESULTS_CSV_PATH = '/content/drive/MyDrive/AI_Scrap_Sorting/results/simulation_log.csv'
RETRANING_QUEUE_PATH = '/content/drive/MyDrive/AI_Scrap_Sorting/results/retraining_queue.csv'
CONFIDENCE_THRESHOLD = 0.80 
DELAY_BETWEEN_ITEMS = 1 # Reduced delay for a quick run

def main():
    predictor = ScrapperONNXPredictor(MODEL_PATH)
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('jpg', 'jpeg', 'png'))]
    
    log_data = []
    retraining_queue = []

    print("--- Starting SHORT simulation to generate log files ---")

    # MODIFICATION: We will only loop through the few images
    for image_file in image_files[:10]:
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        
        predicted_class, confidence = predictor.predict(image_path)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        low_confidence_flag = confidence < CONFIDENCE_THRESHOLD
        
        print(f"[{timestamp}] Image: {image_file} -> Prediction: {predicted_class} (Confidence: {confidence:.2f})")
        
        log_entry = {
            'timestamp': timestamp,
            'image_file': image_file,
            'prediction': predicted_class,
            'confidence': confidence,
            'low_confidence_flag': low_confidence_flag
        }
        log_data.append(log_entry)
        
        if low_confidence_flag:
            print(f"  -> WARNING: Low confidence detected!")
            # user_input = input("  -> Is the prediction correct? (y/n) or 'skip' (s): ").lower()
            # if user_input == 'n':
            #     print(f"  -> LOGGING FOR RETRAINING: {image_file} was misclassified.")
            #     retraining_queue.append({'image_file': image_file, 'predicted_class': predicted_class})

        time.sleep(DELAY_BETWEEN_ITEMS)

    # Save the logs to CSV files
    pd.DataFrame(log_data).to_csv(RESULTS_CSV_PATH, index=False)
    if retraining_queue:
        pd.DataFrame(retraining_queue).to_csv(RETRANING_QUEUE_PATH, index=False)

    print("\n--- Simulation Complete ---")
    print(f"Log files generated successfully!")

if __name__ == '__main__':
    main()
