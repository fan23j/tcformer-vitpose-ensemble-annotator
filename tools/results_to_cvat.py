import json
import os
from lxml import etree
import argparse
from itertools import zip_longest

# COCO Whole Body Keypoints mapping
coco_keypoints = {
    '0': 'nose',
    '1': 'left_eye',
    '2': 'right_eye',
    '3': 'left_ear',
    '4': 'right_ear',
    '5': 'left_shoulder',
    '6': 'right_shoulder',
    '7': 'left_elbow',
    '8': 'right_elbow',
    '9': 'left_wrist',
    '10': 'right_wrist',
    '11': 'left_hip',
    '12': 'right_hip',
    '13': 'left_knee',
    '14': 'right_knee',
    '15': 'left_ankle',
    '16': 'right_ankle',
    '17': 'left_big_toe',
    '18': 'left_small_toe',
    '19': 'left_heel',
    '20': 'right_big_toe',
    '21': 'right_small_toe',
    '22': 'right_heel'
}

def keypoint(iterable):
    """
    Grouper function to add [x,y,score] as single iterable in keypoints list.
    """
    args = [iter(iterable)] * 3
    return zip_longest(fillvalue=None, *args)

def json_to_cvat(args):
    """
    Function to convert input json to cvat xml.
    """
    with open(args.input_json, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    images = data['images']
    root = etree.Element('annotations')
    
    # assumes 1:1 mapping between images and annotations
    for image, annotation in zip(images,annotations):
        image_node = etree.SubElement(root, 'image')
        image_node.set('id', str(image['id']))
        image_node.set('name', image['file_name'])
        
        # add bbox
        bbox = annotation['cvat_bbox']
        box_node = etree.SubElement(image_node, 'box')
        box_node.set('label', 'person')
        box_node.set('occluded', "2")
        box_node.set('xtl', str(bbox[0]))
        box_node.set('ytl', str(bbox[1]))
        box_node.set('xbr', str(bbox[2]))
        box_node.set('ybr', str(bbox[3]))
        
        count = 0
        
        # add keypoints
        for x,y,occlusion in keypoint(annotation['keypoints']):
            if(count < 17):
                count += 1
                continue
            kpts_node = etree.SubElement(image_node, 'points')
            # if annotation exists, set occluded to 2
            kpts_node.set('occluded', str(occlusion))
            kpts_node.set('points', str(round(x,2))+","+str(round(y,2)))
            kpts_node.set('label', coco_keypoints[str(count)])
            count += 1
            
    with open(args.output_xml, 'wb') as f:
        f.write(etree.tostring(root, pretty_print=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert results JSON from ensemble to CVAT XML.")
    parser.add_argument('--input_json', type=str, required=True, help="Path to the input JSON.")
    parser.add_argument('--output_xml', type=str, required=True, help="Path to the output XML.")
    args = parser.parse_args()
    
    json_to_cvat(args)
