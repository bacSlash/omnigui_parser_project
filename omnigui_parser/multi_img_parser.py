import os
import csv
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
                                  categorize_interactivity,
                                  detect_interface_regions,  # NEW
                                  get_region_importance      # NEW
)
from pathlib import Path
import numpy as np
import cv2  # NEW

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
ICON_CAPTION_MODEL_PATH = 'Microsoft/Florence-2-base'

# Load YOLO model for icon detection
yolo_model = get_yolo_model(ICON_DETECT_MODEL_PATH)
caption_model_processor = get_caption_model_processor(ICON_CAPTION_MODEL_NAME, ICON_CAPTION_MODEL_PATH)

def crop_image_region(image_path, region_bbox):
    """Crop image to specific region"""
    image = Image.open(image_path)
    x1, y1, x2, y2 = region_bbox
    return image.crop((x1, y1, x2, y2))

def process_region_specific(image_path, region_name, region_bbox, previous_elements):
    """Process specific region with appropriate strategy"""
    if region_bbox is None:
        return []
    
    # Crop to region
    cropped_path = f"temp_region_{region_name}.png"
    cropped_img = crop_image_region(image_path, region_bbox)
    cropped_img.save(cropped_path)
    
    try:
        # Adjust processing parameters based on region
        if region_name == 'viewport':
            # Lower sensitivity for viewport (reduce noise from camera movements)
            box_threshold = 0.15  # Higher threshold
            iou_threshold = 0.3   # More aggressive overlap removal
        elif region_name == 'ribbon':
            # High sensitivity for tool detection
            box_threshold = 0.03  # Lower threshold for small tool icons
            iou_threshold = 0.1   # Keep more overlapping elements
        else:
            # Default parameters for other regions
            box_threshold = 0.05
            iou_threshold = 0.1
        
        # Process the cropped region
        region_data = process_image_region(cropped_path, previous_elements, 
                                         box_threshold, iou_threshold, region_bbox)
        
        # Add region context to each element
        for element in region_data:
            element['Region'] = region_name
            element['Region Importance'] = get_region_importance(region_name)
            
            # Adjust bounding box coordinates back to full image space
            bbox = element['Bounding Box']
            element['Bounding Box'] = [
                bbox[0] + region_bbox[0], bbox[1] + region_bbox[1],
                bbox[2] + region_bbox[0], bbox[3] + region_bbox[1]
            ]
        
        return region_data
    
    finally:
        # Clean up temp file
        if os.path.exists(cropped_path):
            os.remove(cropped_path)

def process_image_region(image_path, previous_elements, box_threshold=0.05, iou_threshold=0.1, region_bbox=None):
    """Modified version of original process_image for region-specific processing"""
    image = Image.open(image_path)
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
        BOX_TRESHOLD=box_threshold,  # Use region-specific threshold
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config={},
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,  # Use region-specific threshold
        imgsz=imgsz,
        batch_size=icon_process_batch_size
    )
    
    if not parsed_content_list:
        return []
    
    structured_data = []
    for element in parsed_content_list:
        bbox = element.get('bbox', None)
        if bbox is None or not isinstance(bbox, list):
            continue
        
        if isinstance(bbox, list) and any(isinstance(v, float) and np.isnan(v) for v in bbox):
            continue
            
        bbox = [max(0, int(v)) for v in bbox]
        
        element_id = generate_element_id(bbox, element.get('content', ''))
        normalized_bbox = normalize_bbox(bbox, image_width, image_height)
        dominant_color = get_dominant_color(image_np, bbox)
        interactivity_type = categorize_interactivity(element['type'])
        ocr_confidence = element.get('ocr_confidence', None)
        
        if ocr_confidence is None or not isinstance(ocr_confidence, (float, int)) or np.isnan(ocr_confidence):
            ocr_confidence = 0.0
        
        # Compute IOU with previous element
        max_iou = 0
        for prev_element in previous_elements:
            if 'bbox' in prev_element:
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

def process_image(image_path, previous_elements):
    """Enhanced main processing function with region segmentation"""
    # Detect interface regions
    regions = detect_interface_regions(image_path)
    
    all_structured_data = []
    region_elements = {}
    
    # Process each region separately
    for region_name, region_bbox in regions.items():
        if region_bbox is not None:
            # Get previous elements for this region
            prev_region_elements = [elem for elem in previous_elements 
                                  if elem.get('Region') == region_name]
            
            region_data = process_region_specific(image_path, region_name, 
                                                region_bbox, prev_region_elements)
            all_structured_data.extend(region_data)
            region_elements[region_name] = region_data
    
    # If no regions detected, fall back to original processing
    if not all_structured_data:
        print(f"No regions detected in {image_path}, using original processing")
        original_data = process_image_original(image_path, previous_elements)
        for element in original_data:
            element['Region'] = 'unknown'
            element['Region Importance'] = 1.0
        all_structured_data = original_data
    
    return all_structured_data

def process_image_original(image_path, previous_elements):
    """Original process_image function as fallback"""
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
        if bbox is None or not isinstance(bbox, list):
            print(f"WARNING: Skipping element with missing bbox in {image_path}")
            continue
        
        if isinstance(bbox, list) and any(isinstance(v, float) and np.isnan(v) for v in bbox):
            print(f" WARNING: Skipping element with NaN bbox in {image_path}")
            continue
            
        bbox = [max(0, int(v)) for v in bbox]
        
        element_id = generate_element_id(bbox, element.get('content', ''))
        normalized_bbox = normalize_bbox(bbox, image_width, image_height)
        dominant_color = get_dominant_color(image_np, bbox)
        interactivity_type = categorize_interactivity(element['type'])
        ocr_confidence = element.get('ocr_confidence', None)
        
        if ocr_confidence is None or not isinstance(ocr_confidence, (float, int)) or np.isnan(ocr_confidence):
            ocr_confidence = 0.0
        
        # Compute IOU with previous element
        max_iou = 0
        for prev_element in previous_elements:
            if 'bbox' in prev_element:
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
               "Dominant Color",
               "Region",           # NEW
               "Region Importance"  # NEW
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
                    
def main():
    parser = argparse.ArgumentParser(description='Process images from folder.')
    parser.add_argument('folder_path', type=str, help='Path to folder with images')
    parser.add_argument('output_csv_path', type=str, help='Path to save output CSV file')
    args = parser.parse_args()
    
    if not os.path.exists(args.folder_path):
        print(f"Error: Folder path '{args.folder_path}' does not exist.")
        return
    
    process_folder(args.folder_path, args.output_csv_path)
    print('All images processed. Results saved to', args.output_csv_path)

if __name__ == "__main__":
    main()
