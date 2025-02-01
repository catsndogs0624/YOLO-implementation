'''
모델 클래스를 인스턴스로 선언하고
For loop을 돌면서 gradient descent를 수행하면서
파라미터를 업데이터하는 로직
'''

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import random

from absl import flags
from absl import app

from loss import yolo_loss
from model import YOLOv1
from dataset import process_each_ground_truth
from utils import draw_bounding_box_and_label_info, generate_color, find_max_confidence_bounding_box,yolo_format_to_bounding_box_dict

# set cat label dictionary
cat_label_dict = {
    0:"cat"
}
cat_class_to_label_dict = {v: k for k, v in cat_label_dict.items()}

flags.DEFINE_string('checkpoint_path',default='saved_model',help='path to a directory to save model checkpoints during training.')
flags.DEFINE_integer('save_checkpoint_steps',default=50,help='period at which checkpoints are saved (dafault to every 50 steps).')
flags.DEFINE_string('tensorboard_log_path',default='tensorboard_log',help='path to a directory to save tensorboard log.')
flags.DEFINE_integer('validation_steps',default=50,help='period at which test prediction result and save image.')
flags.DEFINE_integer('num_epochs',default=135,help='training epoch.')
flags.DEFINE_float('init_learning_rate',default=0.0001,help='initial learning rate.')
flags.DEFINE_float('lr_decay_rate',default=0.5,help='decay rate for the learning rate.')
flags.DEFINE_integer('lr_decay_steps',default=2000,help='number of steps after which the learning rate is decayed by decay rate.')
flags.DEFINE_integer('checkpoint_path',default=8,help='number of visualize image for validation.')

FLAGS = flags.FLAGS

# set configuration value
batch_size = 24 # original paper : 64
input_width = 224 # original paper : 448
input_height = 224 # original paper : 448
cell_size = 7
num_classes = 1 # original paper : 20
boxes_per_cell = 2

# set color_list for drawing
color_list = generate_color(num_classes)

# set loss functino coefficients
coord_scale = 10 # original paper : 5
class_scale = 0.1 # original paper : 1
object_scale = 1
noobject_scale = 0.5

# load pascal voc2007/voc2012 dataset using tfds
# notice : voc2007 train data(=2,501 images) for test & voc2007 test data(=4,952 images) for training

voc2007_test_split_data = tfds.load("voc/2007",split=tfds.Split.TEST, batch_size=1)
voc2007_train_split_data = tfds.load("voc/2012",split=tfds.Split.TRAIN, batch_size=1)
voc2012_validation_split_data = tfds.load("voc/2012",split=tfds.VALIDATION, batch_size=1)
train_data = voc2007_test_split_data.concatenate(voc2012_validation_split_data).concatenate(voc2012_validation_split_data)

# set validation data
voc2007_validation_split_data = tfds.load("voc/2007", split=tfds.VALIDATION, batch_size=1)
validation_data = voc2007_validation_split_data

# label 7 : cat
def predicate(x, allowed_labels=tf.constant([7.0])):
    label = x['objects']['label']
    isallowed = tf.equal(allowed_labels, tf.cast(label, tf.float32))
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    
    return tf.greater(reduced, tf.constant(0.))

train_data = train_data.filter(predicate)
train_data = train_data.padded_batch(batch_size)

validation_data = validation_data.filter(predicate)
validation_data = validation_data.filter(batch_size)


def reshape_yolo_preds(preds):
    # flatten vector -> cell_size x cell_size x (num_classes + 5 * boxes_per_cell)
    return tf.reshape(preds, [tf.shape(preds)[0], cell_size,cell_size,num_classes + 5 * boxes_per_cell])

def calculate_loss(model, batch_image, batch_bbox, batch_labels):
    total_loss = 0.0
    coord_loss = 0.0
    object_loss = 0.0
    noobject_loss = 0.0
    class_loss = 0.0
    
    for batch_index in range(batch_image.shape[0]):
        image, labels, object_num = precess_each_ground_truth(batch_image[batch_index], batch_labels[batch_index],input_width, input_height)
        image = tf.expand_dims(image, axis=0)
        
        predict = model(image)
        predict = reshape_yolo_preds(predict)
        
        for object_num_index in range(object_num):
            each_object_total_loss, each_object_coord_loss, each_object_object_loss, each_object_noobject_loss, each_object_class_loss = yolo_loss(predict[0],
                                                                                                                        labels,
                                                                                                                        object_num_index,
                                                                                                                        num_classes,
                                                                                                                        boxes_per_cell,
                                                                                                                        cell_size,
                                                                                                                        input_width,
                                                                                                                        input_height,
                                                                                                                        coord_scale,
                                                                                                                        object_scale,
                                                                                                                        noobject_scale,
                                                                                                                        class_scale
                                                                                                                        )
            total_loss = total_loss + each_object_total_loss
            coord_loss = coord_loss + each_object_coord_loss
            object_loss = object_loss + each_object_object_loss
            noobject_loss = noobject_loss + each_object_noobject_loss
            class_loss - class_loss + each_object_class_loss
            
    return total_loss, coord_loss, object_loss, noobject_loss, class_loss

def train_step():
    pass

def save_validation_result():
    pass

def main(_):
    # set learning rate decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        FLAGS.init_learning_rate,
        decay_steps=FLAGS.lr_decay_steps,
        decay_rate=FLAGS.lr_decay_rate,
        staircase=True
    )
    
    # set optimizer
    optimizer = tf.optimizers.Adam(lr_schedule) # original paper: SGD with momentum 0.9, decay 0.0005

    # check if checkpoint path exists
    if not os.path.exists(FLAGS.checkpoint_path):
        os.mkdir(FLAGS.checkpoint_path)
        
    # create YOLO model
    YOLOv1_model = YOLOv1(input_height, input_width, cell_size, boxes_per_cell, num_classes)
    
    # set checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=YOLOv1_model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=FLAGS.checkpoint_path,
                                              max_to_keep=None)
    latest_ckpt =  tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    
    # restore latest checkpoint
    if latest_ckpt:
        ckpt.restore(latest_ckpt)
        print(f'global_step : {int(ckpt.step)}, chekpoint is restored!')
        
    # set tensorboard log
    train_summary_writer = tf.summary.create_file_writer(FLAGS.tensorboard_log_path + '/train')
    validation_summary_writer = tf.summary.create_file_writer(FLAGS.tensorboard_log_path + '/train')
    
    for epoch in range(FLAGS.num_epochs):
        num_batch = len(list(train_data))
        for iter, features in enumerate(train_data):
            batch_image = features['image']
            batch_bbox = features['objects']['bbox']
            batch_labels = features['objects']['label']
            
            batch_image = tf.squeeze(batch_image, axis=1)
            batch_bbox = tf.squeeze(batch_bbox, axis=1)
            batch_labels = tf.squeeze(batch_labels, axis=1)
            
            # runs optimization and calculate loss
            total_loss, coord_loss, object_loss, noobject_loss, class_loss = train_step(optimizer, YOLOv1_model, batch_image, batch_image, batch_bbox, batch_labels)
            
            # print log
            print(f"Epoch: {epoch+1}, Iter: {iter+1}/{num_batch}, Loss : {total_loss.numpy()}")
            
            # save tensorboard log
            with train_summary_writer.as_default():
                tf.summary.scalar('learning_rate', optimizer.lr(ckpt.step).numpy(),step=int(ckpt.step))
                tf.summary.scalar('total_loss', total_loss, step=int(ckpt.step))
                tf.summary.scalar('coord_loss', coord_loss, step=int(ckpt.step))
                tf.summary.scalar('object_loss', object_loss, step=int(ckpt.step))
                tf.summary.scalar('noobject_loss', noobject_loss, step=int(ckpt.step))
                tf.summary.scalar('class_loss', total_loss, step=int(ckpt.step))
                
            # save checkpoint
            if ckpt.step % FLAGS.save_checkpoint_steps == 0:
                # save checkpoint
                ckpt_manager.save(f'global step : {ckpt.step}, steppoint is saved!')
                
            ckpt.step.assign_add(1)
            
            # occassionally check validation data and save tenosorboard log
            if iter % FLAGS.validation_steps == 0:
                save_validation_result(YOLOv1_model, ckpt, validation_summary_writer, FLAGS.num_visualize_image)
    
if __name__ == "__main__":
    app.run(main)
