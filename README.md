# End-to-End ML Pipeline for Real-Time Material Classification

## Overview

This project is a complete, end-to-end machine learning pipeline that classifies scrap materials from images. It simulates a real-time conveyor belt system. The pipeline includes data loading, model training, evaluation, and a final deployment simulation using an ONNX model. The project was built using TensorFlow/Keras.

## Project Structure

The repository is organized as follows:

- **/data/**: Contains the TrashNet dataset used for training.
- **/models/**: Contains the saved trained model in both Keras (`.keras`) and ONNX (`.onnx`) formats.
- **/results/**: Contains the output logs from the simulation, including a log of all predictions and a queue of items flagged for retraining.
- **/src/**: Contains all Python source code.
  - `inference.py`: A module with a class to run predictions using the ONNX model.
  - `simulation.py`: The main application script that simulates the conveyor belt.
- **/test_images/**: A sample of images used for running the simulation.
- `performance_report.md`: A detailed report on the model's performance metrics.

## How to Run

The project was developed in Google Colab.

1.  **Setup**: The initial cells of the notebook set up the environment, download the data, and copy it to the local Colab disk for performance.
2.  **Training**: The training cells load the data, build the MobileNetV2 transfer learning model, and train it for 10 epochs.
3.  **Evaluation & Export**: After training, the model's performance is visualized, and the final model is saved as `scrap_classifier.keras` and exported to `scrap_classifier.onnx`.
4.  **Simulation**: To run the final application, execute the simulation script from the command line:
    ```bash
    python src/simulation.py
    ```

## Results

The final model achieved **88% accuracy** on the validation set. The simulation script successfully classifies items in real-time, logs results, and uses a confidence threshold to flag uncertain predictions for manual review, demonstrating a complete production-like cycle.
