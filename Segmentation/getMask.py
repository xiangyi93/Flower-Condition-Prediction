import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image  #, ImageDraw


def labelme_json_to_dataset(json_dir):
    output_mask_dir = json_dir + "/masks"
    os.makedirs(output_mask_dir, exist_ok=True)
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    # 類別映射表 (根據你的標註名稱修改)
    label_map = {"background": 0, "cherry": 1, "daylily": 2, "hydrangeas": 3} 

    for json_file in json_files:
        with open(os.path.join(json_dir, json_file), 'r') as f:
            data = json.load(f)
            
        img_h = data['imageHeight']
        img_w = data['imageWidth']
        
        # 建立空白遮罩 (背景為 0)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            if label in label_map:
                # 將多邊形座標轉為整數
                poly = np.array(points, dtype=np.int32)
                # 在遮罩上填充該類別的數值 (例如 1)
                cv2.fillPoly(mask, [poly], color=label_map[label])
        
        # 儲存為 PNG
        mask_name = json_file.replace('.json', '.png')
        Image.fromarray(mask).save(os.path.join(output_mask_dir, mask_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Labelme JSON files to masks.")
    parser.add_argument("json_dir", help="Directory containing paired images and Labelme JSON files")
    arguments = parser.parse_args()
    labelme_json_to_dataset(arguments.json_dir)