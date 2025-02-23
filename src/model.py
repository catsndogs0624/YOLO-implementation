'''
Keras Subclassing형태의 모델 구조 class 정의
'''

import tensorflow as tf

# Implementation using tf.keras.application
# & Keras Functikonal API

class YOLOv1(tf.keras.Model):
    def __init__(self, input_height, input_width,cell_size, boxes_per_cell, num_classes):
        super(YOLOv1, self).__init__()
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',
                                                       input_shape=(input_height,input_width,3))
        base_model.trainable = True
        x = base_model.output
        
        # Global Average Pooling
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        output = tf.keras.layers.Dense(cell_size* cell_size * (num_classes + (boxes_per_cell*5)), activation=None)(x)
        model = tf.keras.Model(inputs=base_model.input, output=output)
        self.model = model
        self.model.summary()
        
        
    def call(self, x):
        return self.model(x)