'''
딥러닝 메인 로직 외에 유틸리티성 기능을 모아놓은 로직
'''
import cv2
import numpy as np
import tensorflow as tf
import colorsys
from operator import itemgetter

def draw_bounding_box_and_label_info(frame, x_min, y_min, x_max, y_max, label, confidence, color):
    draw_bounding_box(frame, x_min, y_min, x_max, y_max,color)
    draw_label_info(frame, x_min, y_min, x_max, y_max,color)

def draw_bounding_box(frame, x_min, y_min, x_max, y_max, color):
    cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),color,3)
    
    
def draw_label_info(frame, x_min, y_min, label, confidence, color):
    text = label + ' ' + str('%.3f' % confidence)
    bottomLeftCornerOfText = (x_min, y_min)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    fontColor = color
    lineType = 2
    
    cv2.putText(frame, text,bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
    
    
def find_max_confidence_bounding_box(bounding_box_info_list):
    bounding_box_info_list_sorted = sorted(bounding_box_info_list,
                                           key=itemgetter('confidence'),
                                           reverse=True)
    max_confidence_bounding_box = bounding_box_info_list_sorted[0]
    
    return max_confidence_bounding_box

def yolo_format_to_bounding_box_dict(xcenter, ycenter,box_w, box_h, class_name, confidence):
    bounding_box_info = {}
    bounding_box_info['left'] = int(xcenter - (box_w/2))
    bounding_box_info['top'] = int(ycenter - (box_h/2))
    bounding_box_info['right'] = int(xcenter + (box_w/2))
    bounding_box_info['bottom'] = int(ycenter + (box_h/2))
    bounding_box_info['class_name'] = class_name
    bounding_box_info['confidence'] = confidence
    return bounding_box_info

def iou(yolo_pred_boxes, grounding_truth_boxes):
    """
    calculate ious
    Args:
        yolo_pred_boxes : 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL,4] ===>(x_center, y_center,w,h)
        ground_truth_boxes : 1-D tensor [4] ===> (x_center, y_center, w, h)
        
    Return:
        iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    
    boxes1 = yolo_pred_boxes
    boxes2 = grounding_truth_boxes
    
    boxes1 = tf.stack([boxes1[:,:,:,0] - boxes1[:,:,:,2]/2, boxes1[:,:,:,1] - boxes1[:,:,:,3]/2,
                       boxes1[:,:,:,0] + boxes1[:,:,:,2]/2, boxes1[:,:,:,1] + boxes1[:,:,:,3]/2])
    boxes1 = tf.transpose(boxes1,[1,2,3,0])    
    boxes2 = tf.stack([boxes2[0] - boxes2[2]/2 , boxes2[1] -boxes2[3]/2,
                      boxes2[0] + boxes2[2]/2, boxes2[1] + boxes2[3]/2])
    boxes2 = tf.cast(boxes2, tf.float32)
    
    # calculate the left up point
    lu = tf.maximum(boxes1[:,:,:,0:2], boxes2[0:2])
    rd = tf.minimum(boxes1[:,:,:,0:2], boxes2[0:2])
    
    intersection = rd - lu
    
    
    inter_square = intersection[:,:,:,0] * intersection[:,:,:,1]
    
    mask = tf.cast(intersection[:,:,:,0]>0, tf.float32) * tf.cast(intersection[:,:,:,1]>0, tf.float32)
                      
    inter_square = mask * inter_square
    
    # calculate the box1 square and boxs2 square
    square1 = (boxes1[:,:,:,2] - boxes1[:,:,:,0]) * (boxes1[:,:,:,3] - boxes1[:,:,:,1])
    square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
    
    return inter_square / (square1 + square2 - inter_square + 1e-6)


def generate_color(num_classes):
    # Generate colors for drawing bounding boxes.
    
    hsv_tuples = [(x / num_classes, 1., 1.)
                  for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    
    np.random.seed(10101)
    np.random.shuffle(colors)
    np.random.seed(None)
    
    return colors