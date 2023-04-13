import os
import cv2
import torch
import argparse
from PIL import Image
from tqdm import tqdm

def main(args):
    # Load YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5", "yolov5l6")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Iterate over images in the input folder
    for img_name in tqdm(os.listdir(args.input_folder)[19000:]):
        img_path = os.path.join(args.input_folder, img_name)

        # Load image
        image = Image.open(img_path)

        # Detect person instances
        try:
            results = model(image)
        except:
            print(img_path)
            continue
        results = results.pandas().xyxy[0]
        person_detections = results[results['name'] == 'person'].copy()
    
        # If there are more than one person instances, process the image
        if len(person_detections) > 1:
            # Find the largest bounding box
            person_detections['area'] = (person_detections['xmax'] - person_detections['xmin']) * (person_detections['ymax'] - person_detections['ymin'])
            largest_bbox = person_detections.loc[person_detections['area'].idxmax()]

            # Read the image using OpenCV
            img = cv2.imread(img_path)

            # Mask the areas of the image with smaller bounding boxes
            for index, row in person_detections.iterrows():
                if row['area'] < largest_bbox['area']:
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    img[y1:y2, x1:x2] = 0

            # Save the processed image
            output_path = os.path.join(args.output_folder, img_name)
            cv2.imwrite(output_path, img)

        # If there's only one person instance, copy the image to the output folder
        # Images with no person instances are not copied over
        elif len(person_detections) == 1:
            output_path = os.path.join(args.output_folder, img_name)
            
            # Convert the image to 'RGB' mode if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            image.save(output_path)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mask smaller person instances in a folder of images.")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the input images folder.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the output images folder.")
    args = parser.parse_args()
    main(args)
