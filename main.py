import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from ultralytics import YOLO 
from functools import partial
import cv2
import pandas as pd
from tqdm import tqdm
import multiprocessing
from PIL import Image
import numpy as np
import easyocr
import re
from utils import parse_string
from units import unit_symbol_map, most_common_unit_map
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

reader = easyocr.Reader(['en'], gpu=True)
model = YOLO('best.pt')
model.to(device)
classes = model.names

def text_parse_string(text, text_class):
    text = text.lower()

    number_pattern = r'[-+]?\d*\.?\d+'
    match = re.search(number_pattern, text)
    if not match:
        raise ValueError("Invalid text. No number found")

    number = float(match.group()) if '.' in match.group() else int(match.group())
    unit = most_common_unit_map[text_class]
    for key in unit_symbol_map:
        if key in text:
            unit = unit_symbol_map[key]

    return number, unit

def perform_ocr_on_bbox(image, bbox):
    if isinstance(image, Image.Image):
        image = np.array(image)

    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox 

    cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]

    result = reader.readtext(cropped_image, detail=0)
    text = ' '.join(result)
    return text

def process_row(row, image_folder):
    idx = row.get('index', row.name)
    image_url = row['image_link'].split('/')[-1]
    entity_name = row['entity_name']
    entity_value = ""

    image_path = os.path.join(image_folder, f'{image_url}')
    if os.path.exists(image_path):
        results = model.predict(source=image_path, imgsz=640, device=device, verbose=False)
        results = results[0]
        textdict = dict()

        original_image = Image.open(image_path).convert("RGB")
        for i, (box, conf, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
            x_min, y_min, x_max, y_max = map(float, box)
            bbox = (x_min, y_min, x_max, y_max)
            text = perform_ocr_on_bbox(np.array(original_image), bbox)

            try:
                class_name = classes[int(cls)]
                
                num, unit = text_parse_string(text, class_name)
                final_text = str(num) + ' ' + unit_symbol_map[unit]

                if class_name not in textdict or conf > textdict[class_name]['confidence']:
                    textdict[class_name] = {
                        'text': final_text,
                        'confidence': conf
                    }

            except Exception as e:
                # print(f'Error in row {idx}: {e}')
                pass

        if entity_name in textdict:
            entity_value = textdict[entity_name]['text']
    else: 
        print(f"{image_path} path not exist")

    return {
        'index': idx,
        'prediction': entity_value
    }

def process_dataset(input_csv, output_csv, image_folder):
    chunksize = 1024
    with open(output_csv, 'w', newline='') as f_out:
        counter = 0
        csv_writer = None

        for data in pd.read_csv(input_csv, chunksize=chunksize):

            process_row_partial = partial(process_row, image_folder=image_folder)

            with multiprocessing.Pool(8) as pool:
                predictions = list(tqdm(pool.imap(process_row_partial, [row for idx, row in data.iterrows()]), total=len(data)))
                pool.close()
                pool.join()

            pred_df = pd.DataFrame(predictions)
    
            if csv_writer is None:
                pred_df.to_csv(f_out, index=False)
                csv_writer = True  # After writing the header once
            else:
                pred_df.to_csv(f_out, index=False, header=False)

            counter += 1
        
            print(f"Batch {counter} completed. {counter * chunksize} rows completed")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    train_csv = 'dataset/train.csv'
    test_csv = 'dataset/test.csv'
    train_image_folder = 'images/train/'
    test_image_folder = 'images/test/'
    output_train_csv = 'train_predictions.csv'
    output_test_csv = 'final_submission.csv'

    # process_dataset(train_csv, output_train_csv, train_image_folder)
    process_dataset(test_csv, output_test_csv, test_image_folder)