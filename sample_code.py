import os
import random
import pandas as pd

def predictor(image_link, category_id, entity_name):
    '''
    Call your model/approach here
    '''
    #TODO
    return "" if random.random() > 0.5 else "10 inch"

if __name__ == "__main__":
    DATASET_FOLDER = '../dataset/'
    
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)

# 
# 
# 


# def process_dataset(input_csv, output_csv, image_folder):
#     data = pd.read_csv(input_csv)
#     # data = data.head(20)
#     print("HELLO WORLD")

#     process_row_partial = partial(process_row, image_folder=image_folder)

#     with multiprocessing.Pool(64) as pool:
#         predictions = list(tqdm(pool.imap(process_row_partial, [row for idx, row in data.iterrows()]), total=len(data)))
#         pool.close()
#         pool.join()

#     pred_df = pd.DataFrame(predictions)
#     pred_df.to_csv(output_csv, index=False)


# 
# 
# 

import cv2
import pandas as pd
import pytesseract
import re
import os
from ultralytics import YOLO 
from PIL import Image 
import numpy as np 
from utils import parse_string
model = YOLO('../best (1).pt')
classes = model.names


def perform_ocr_on_bbox(image, bbox):
    """
    Perform OCR on the given bounding box of the image.
    :param image: The input image (PIL.Image or numpy array)
    :param bbox: Bounding box coordinates (x_center, y_center, width, height)
    :return: Extracted text
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert YOLO format (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox # Changed line
    # x_min = int((x_center - width / 2) * w)
    # y_min = int((y_center - height / 2) * h)
    # x_max = int((x_center + width / 2) * w)
    # y_max = int((y_center + height / 2) * h)

    # Crop the bounding box region
    cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]

    # Convert cropped image back to PIL Image
    cropped_image_pil = Image.fromarray(cropped_image)

    # Perform OCR
    text = pytesseract.image_to_string(cropped_image_pil)
    return text

# Create a function to process the dataset and generate predictions
def process_dataset(input_csv, output_csv, image_folder):
    data = pd.read_csv(input_csv)
    predictions = []
    counter = 1 
    for idx, row in data.iterrows():
        # index = row['index']
        if idx == 11 : break;
        print(counter)
        counter+=1
        image_url = row['image_link'].split('/')[-1]
        entity_name = row['entity_name']
        entity_value = ""
        # print(image_url)
        # break;
        image_path = os.path.join(image_folder, f'{image_url}')
        # print(image_path)
        # break;
        if os.path.exists(image_path):
            results = model.predict(source=image_path, imgsz=640)
            results = results[0]
            textdict = dict()
            # print('result'results.bbox)
            # break;
            original_image = Image.open(image_path).convert("RGB")
            for i, (box, conf, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
                x_min, y_min, x_max, y_max = map(float, box)
                bbox = (x_min, y_min, x_max, y_max) # Changed line
                text = perform_ocr_on_bbox(np.array(original_image), bbox)
                print(text)
                # try : 
                #     num,unit = parse_string(text)
                #     print(str(num)+' ' + unit_symbol_map[unit] )
                #     textdict[classes[int(cls)]]=str(num)+' ' + unit_symbol_map[unit] 
                # except : 
                #     print('inside except ',idx)
                    # pass 
            if entity_name in textdict : entity_value = textdict[entity_name]
        predictions.append({
            'index': idx,
            'prediction': entity_value
        })
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_csv, index=False)

# Create a function to generate the final output file for submission
def generate_submission_file(test_csv, output_csv, image_folder):
    """
    Generates the final submission file by processing test images.
    """
    # Load the test data
    test_data = pd.read_csv(test_csv)

    # Initialize list to store output
    output_data = []

    for idx, row in test_data.iterrows():
        index = row['index']
        entity_name = row['entity_name']

        # Assuming images have been downloaded to a local folder
        image_path = os.path.join(image_folder, f'{index}.jpg')
        
        if os.path.exists(image_path):
            # Extract text from image
            text = extract_text_from_image(image_path)

            # Extract the entity value from the text
            entity_value = extract_entity_from_text(text, entity_name)
        else:
            entity_value = ""

        # Append the result
        output_data.append({
            'index': index,
            'prediction': entity_value if entity_value else ""
        })

    # Convert to DataFrame and save to CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False)

train_csv = '../dataset/train.csv'
test_csv = '../dataset/test.csv'
image_folder = '../images/train/'  # Ensure you have downloaded the images to this folder
output_train_csv = 'train_predictions.csv'
output_test_csv = 'final_submission.csv'

    # Process the training dataset
process_dataset(train_csv, output_train_csv, image_folder)

    # Generate the final submission file for the test dataset
    # generate_submission_file(test_csv, output_test_csv, image_folder)

print("Processing complete. Files generated successfully.")
