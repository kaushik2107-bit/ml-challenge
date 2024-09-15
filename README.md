# Amazon ML Challenge Submission: Model Mavericks

### Team Members:
- **Darshan Patel**
- **Divy Awasthi**
- **Kaushik Bhowmick**

## Introduction

This repository contains the solution submitted by **Model Mavericks** for the Amazon ML Challenge. Our team primarily worked on building the `main.py` script and training a custom YOLO model for object detection using **Roboflow.ai**. Other files such as helper functions and utility scripts were part of the sample code provided by the competition organizers.

## File Structure

### Project Root
- **`main.py`**: Main script to perform object detection, OCR, and entity prediction. This was written by our team.
- **`dataset/`**: Contains training and testing datasets in `.csv` format.
- **`images/`**: Directory containing the images corresponding to dataset entries.
- **`src/`**: Contains utility scripts provided by the competition organizers, such as sanity checks and image download scripts.

## Core Code Explanation

### Main Script: `main.py`
The script begins by importing key libraries like `torch`, `cv2`, `easyocr`, and `ultralytics` for object detection and OCR. It uses the YOLO model (`best.pt`) trained on Roboflow.ai to detect bounding boxes in the images and extract text using EasyOCR.

The script sets the device to use CUDA if available, otherwise it defaults to CPU. The YOLO model is loaded with the `best.pt` weights, and the OCR reader is initialized to extract English text.

## Workflow Explanation

### Data Preprocessing
The datasets (`train.csv` and `test.csv`) contain image links and entities to predict. The images are stored in the `images/` directory.

### Prediction Workflow
- **YOLO Detection**: The YOLO model detects bounding boxes and object classes.
- **OCR Extraction**: Text is extracted from the bounding boxes using EasyOCR.
- **Text Parsing**: Extracted text is parsed to get numerical values and their corresponding units.

### Post-processing
- Confidence filtering is applied, and the text with the highest confidence is kept for each entity type.

## YOLO Model Training

We used **Roboflow.ai** to train our YOLO model (`best.pt`). The process involved manually annotating around 300 images and exporting the dataset for training. The model was trained using transfer learning, and the weights were exported as `best.pt`. This model was then used in the `main.py` script for object detection.

## Additional Files Provided by the Competition

- **`sanity.py`**: Sanity checker to ensure the final output file passes all formatting checks.
- **`utils.py`**: Contains utility functions, such as downloading images from URLs.
- **`constants.py`**: Defines the allowed units for each entity type and contains mappings used in text parsing.

## Dataset Information

### Training Data
- **`train.csv`**: Contains labeled data for supervised learning (`entity_value`).

### Test Data
- **`test.csv`**: Contains unlabeled test data, used to generate predictions.

### Sample Data
- **`sample_test.csv`**: Sample test input file provided to showcase the format of test data.
- **`sample_test_out.csv`**: Sample output file demonstrating the correct output format for predictions.

## Output Format

The output file must match the format of `sample_test_out.csv`. Each row should include:
- **Index**: Row index from the test data.
- **Prediction**: The predicted entity value, formatted as "number unit".

## Execution Instructions

### Dependencies
To install the required dependencies, run:
```bash
pip install torch ultralytics easyocr pandas tqdm opencv-python-headless pillow numpy
```

## Running the Script
To run the script and generate predictions, use the following command:

```bash
python main.py
```

## Best Practices

1. **Error Handling**: The script includes error handling to ensure that processing continues even if individual rows or images encounter issues.
2. **Memory Efficiency**: The dataset is processed in chunks to avoid excessive memory usage.
3. **Parallel Processing**: Multiprocessing is used to speed up computation, especially for large datasets.
4. **Modular Code**: Functions are modular and reusable for better maintainability.


