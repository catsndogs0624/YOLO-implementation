'''
데이터 전처리 및 batch 단위로 묶는 로직
'''
import tensorflow as tf
import numpy as np

def bounds_per_dimension(ndarray):
    return map(
        lambda e: range(e.min(), e.max() + 1),
        np.where(ndarray != 0)
    )
    
def zero_trim_ndarray(ndarray):
    return ndarray[np.ix_(*bounds_per_dimension(ndarray))]


# process ground-truth data for YOLO format
def process_each_ground_truth(original_image,
                              bbox,
                              class_labels,
                              input_width,
                              input_height
                              ):
    
    '''
    Reference: 
    bbox return : (ymin / height, xmin / wdith, ymax / height, xmax / width)
    
    Args:
        original_image : (original_height, original_width, channel) image tensor
        bbox : (max_object_num_in_batch, 4) = (ymin/height, xmin/width, ymax/hegiht, xmax/width)
        class_labels : (max_object_num_in_batch) = class labels without one-hot-encoding
        input_width : original input width
        input_height : original input height
        
    Returns:
        image : (resized_height, resized_width, channel) image ndarray
        labels : 2-D list [object_num, 5] (xcenter (Absolute Coordinate), ycenter (Absolute Coordinate),width (Absolute Coordinate), height(Absolute Coordinate))
        object_num : total object number in image
    '''
    
    image = original_image.numpy()
    image = zero_trim_ndarray(image)
    
    # set original width height
    original_h = image.shape[0]
    original_w = image.shape[0]
    
    width_rate = input_width * 1.0 / original_w
    height_rate = input_height * 1.0 / original_h
    
    image = tf.image.resize(image, [input_height, input_width])
    
    object_num = np.count_nonzero(bbox, axis=0)[0]
    labels = [0,0,0,0,0] * object_num
    
    for i in range(object_num):
        xmin = bbox[i][1] * original_w
        ymin = bbox[i][0] * original_h
        xmax = bbox[i][3] * original_w
        ymax = bbox[i][4] * original_w
        
        class_num = class_labels[i]
        
        xcenter = (xmin + xmax) * 1.0 / 2 * width_rate
        ycenter = (ymin + ymax) * 1.0 / 2 * width_rate
        
        box_w = (xmax - xmin) * width_rate
        box_h = (ymax - ymin) * width_rate
        
        labels[i] = [xcenter, ycenter, box_w, box_h, class_num]
    
    return [image.numpy(), labels, object_num]