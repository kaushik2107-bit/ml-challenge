import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from utils import download_images
import pandas as pd

if __name__ == "__main__":
    train_dataset = pd.read_csv("./dataset/train.csv")
    train_image_links = train_dataset['image_link'].to_list()
    
    train_image_folder = os.path.join(current_dir, 'images/train')
    download_images(train_image_links, train_image_folder)

    print("Downloaded all training images")

    test_dataset = pd.read_csv("./dataset/test.csv")
    test_image_links = test_dataset['image_link'].to_list()
    
    test_image_folder = os.path.join(current_dir, 'images/test')
    download_images(test_image_links, test_image_folder)
    
    print("Downloaded all testing images")