import os
import csv
from tkinter import filedialog, Tk
import argparse
from PIL import Image
import torch
from omnigui_parser.utils import (check_ocr_box, 
                                  get_yolo_model, 
                                  get_caption_model_processor, 
                                  get_som_labeled_img,
                                  generate_element_id,
                                  normalize_bbox,
                                  compute_iou,
                                  get_dominant_color,
                                  categorize_interactivity
)
from pathlib import Path
import numpy as np

# Set device for model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to search for the model in parent directories
def find_model_path(model_filename='best.pt', search_folder='weights/icon_detect'):
    current_dir = Path(__file__).resolve().parent # Start from the script directory
    
    while current_dir !=current_dir.root: # Traverse upwards till root
        model_path = current_dir / search_folder / model_filename
        if model_path.exists():
            return str(model_path) # Return the absolute path if found
        current_dir = current_dir.parent # Move one level up

    raise FileNotFoundError(f"Model file '{model_filename}' not found in any parent directory.")

# Initialize models
ICON_DETECT_MODEL_PATH = find_model_path()
ICON_CAPTION_MODEL_NAME = 'florence2'
ICON_CAPTION_MODEL_PATH = 'microsoft/Florence-2-base'

# Load YOLO model for icon detection
yolo_model = get_yolo_model(ICON_DETECT_MODEL_PATH)
caption_model_processor = get_caption_model_processor(ICON_CAPTION_MODEL_NAME, ICON_CAPTION_MODEL_PATH)

def process_image(image_path, previous_elements):
    image = Image.open(image_path)
    box_threshold = 0.05
    iou_threshold = 0.1
    use_paddleocr = False
    imgsz = 1920
    icon_process_batch_size = 64
    
    image_width, image_height = image.size
    image_np = np.array(image)
    
    ocr_bbox_rslt, _ = check_ocr_box(
        image_path,
        display_img=False,
        output_bb_format='xyxy',
        goal_filtering=None,
        easyocr_args={'paragraph': False, 'text_threshold': 0.9},
        use_paddleocr=use_paddleocr
    )
    text, ocr_bbox = ocr_bbox_rslt
    
    _, _, parsed_content_list = get_som_labeled_img(
        image_path,  
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config={},
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        batch_size=icon_process_batch_size
    )
    
    if not parsed_content_list:
        print(f"WARNING: No elements detected in {image_path}")
        return []
    
    structured_data = []
    for element in parsed_content_list:
        bbox = element.get('bbox', None)
        if bbox in None or not isinstance(bbox, list) or any(np.isnan(bbox)):
            print(f"WARNING: Skipping element with missing bbox in {image_path}")
            continue # Skip if bbox is None or contains NaN values
        bbox = [max(0, int(v)) for v in bbox] # Ensure bbox values are integers and >= 0
        
        element_id = generate_element_id(bbox, element.get('content', ''))
        normalized_bbox = normalize_bbox(bbox, image_width, image_height)
        dominant_color = get_dominant_color(image_np, bbox)
        interactivity_type = categorize_interactivity(element['type'])
        ocr_confidence = element.get('ocr_confidence', None)
        
        # Ensure OCR Confidence is a float, otherwise set it to 0.0
        if ocr_confidence is None or not isinstance(ocr_confidence, (float, int)) or np.isnan(ocr_confidence):
            ocr_confidence = 0.0
        
        # Compute IOU with previous element
        max_iou = 0
        for prev_element in previous_elements:
            iou = compute_iou(bbox, prev_element['bbox'])
            max_iou = max(max_iou, iou)
            
        structured_data.append({
            "Image Name": os.path.basename(image_path),
            "Element ID": element_id,
            "Type": element['type'],
            "Bounding Box": bbox,
            "Normalized Bounding Box": normalized_bbox,
            "Interactivity": element['interactivity'],
            "Interaction Type": interactivity_type,
            "Content": element.get('content', ''),
            "OCR Confidence": ocr_confidence,
            "IOU with Previous": max_iou,
            "Dominant Color": dominant_color,
        })
    
    return structured_data

def select_folder():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select folder with images")
    return folder_path

def process_folder(folder_path, output_csv_path):
    headers = ["Image Name", 
                   "Element ID", 
                   "Type", 
                   "Bounding Box", 
                   "Normalized Bounding Box", 
                   "Interactivity", 
                   "Interaction Type", 
                   "Content", 
                   "OCR Confidence", 
                   "IOU with Previous", 
                   "Dominant Color"
    ]
    
    previous_elements = []
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                try:
                    parsed_data = process_image(image_path, previous_elements)
                    previous_elements = parsed_data # Update for next image
                    writer.writerows(parsed_data)
                    print(f'Processed {filename}')
                except Exception as e:
                    print(f'Failed to process {filename}: {str(e)}')
                    
def process_folder_cli():
    parser = argparse.ArgumentParser(description='Process images from folder.')
    parser.add_argument('folder_path', type=str, help='Path to folder with images')
    parser.add_argument('output_csv_path', type=str, help='Path to save output CSV file')
    args = parser.parse_args()
    
    process_folder(args.folder_path, args.output_csv_path)
                    
def main():
    folder_path = select_folder()
    if folder_path:
        output_csv_path = os.path.join(folder_path, 'parsed_output.csv')
        process_folder(folder_path, output_csv_path)
        print('All images processed. Results saved to', output_csv_path)
    else:
        print('No folder selected. Exiting.')

if __name__ == "__main__":
    main()
