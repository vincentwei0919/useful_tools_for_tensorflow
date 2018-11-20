# coding: utf-8
import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image

"""
本程序将图片转化成tfrecord格式用于神经网络训练
参数:
img_list:将图片制作成图片列表
label_list:将label做成label列表
save_name:制作的tfrecord的名称
save_path:tfrecord存放的路径
"""

def convert1(img_list, label_list, save_name, save_path):
    save_filename = save_path+ '\\' + save_name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(save_filename)
    # 用来对图片进行索引
    # 图像的个数
    for number in range(30000):
        print(number)
        image = Image.open(img_list[number])
        image = image.resize((224, 224))
        # 将图像转化为byte
        img_raw = image.tobytes()
        # 对图像进行编码
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label_list[number])]))
        }))
        # 将序列化的图像和标签进行写入
        writer.write(example.SerializeToString())
        number += 1
    writer.close()
    print("Done!")

if __name__ == '__main__':
    # txt_path中包含图像路径与图像标签
    txt_path = r'W:\train_label.txt'
    # 想要保存tfrecord的路径
    tfrecord_save_path = r'E:'
    # 要保存的tfrecord 的名称
    tfrecord_name = 'train'
    image_list = []
    label_list = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        # 如果数据没有经过打乱，那么在这步对列表进行打乱，原地打乱，没有返回值
        random.shuffle(lines)
        for line in lines:
            splited = line.strip().split(' ')
            img_name = splited[0]
            label = splited[1]
            image_list.append(img_name)
            label_list.append(label)
    # 传入两个list， 便于下步对其进行索引
    convert1(image_list,label_list, tfrecord_name, tfrecord_save_path)
