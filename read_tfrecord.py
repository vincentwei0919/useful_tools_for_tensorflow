#coding:utf-8
import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow as tf
slim = tf.contrib.slim
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ROWS, COLS = 224, 224

def read_and_decode(record, rows, cols):
    """
    # 解析tfrecord
    # 如果将数据集制作成了多个tfrecord，那么需要使用
    # file_list = tf.train.match_filenames_once(record)来生成一个列表，那么在下面的
    # tf.train.string_input_producer()函数中需要传入file_list，设置shuffle=True，则将
    # 这些tfrecord打乱顺序读入缓存列表，如果设置为False，则顺序读入
    # 如果使用了tf.train.match_filenames_once()函数，则需要在session中进行局部初始化，否则会报错
    """
    filename_queue = tf.train.string_input_producer([record], shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        "label": tf.FixedLenFeature([], tf.int64),
        "image": tf.FixedLenFeature([], tf.string)
    })
    imgs = tf.decode_raw(features["image"], tf.uint8)
    imgs = tf.reshape(imgs, [rows, cols, 3])
    imgs = tf.cast(imgs, tf.float32)
    # 这里加入了预处理过程，也可以在制作的时候就处理
    imgs =  imgs / 255.
    imgs = imgs - tf.reduce_mean(imgs, axis=0)
    labels = tf.cast(features['label'], tf.int64)
    return imgs, labels


def create_batch(record_path, batchsize, rows, cols,  is_train=True):
    images, labels = read_and_decode(record_path, rows, cols)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batchsize
    # 不同的训练阶段，传入的数据不一样，
    if is_train:
        # 在线程从缓存列表中取一个batch的时候，通过设置tf.train.shuffle_batch（）来达到
        # 随机选择的目的
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batchsize, num_threads=4,
                                                      capacity=capacity, min_after_dequeue=min_after_dequeue)
    # 测试
    else:
        image_batch, label_batch = tf.train.batch([images, labels], batch_size=batchsize, capacity=capacity,
                                                  allow_smaller_final_batch=True)
    # 如果需要进行one-hot编码
    label_batch = slim.one_hot_encoding(label_batch, 2)
    return image_batch, label_batch


if __name__ == '__main__':
    imgs, labels = create_batch(files, 32, ROWS, COLS)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        images = sess.run(imgs)
        print(images.shape)
    coord.request_stop()
    coord.join(threads)
