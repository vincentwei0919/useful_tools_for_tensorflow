#coding:utf-8
import tensorflow as tf
import os
import numpy as np
from PIL import Image
from tensorflow.contrib.slim import nets
import shutil
slim = tf.contrib.slim
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

img_path = '要测试的图像存放路径'
result_save_path = '结果保存路径'
imgs = os.listdir(img_path)
img_num = len(imgs)

'''
    使用微调好的网络进行测试
'''
'''
1.设置参数，并加载数据
'''
# 微调后的检查点文件和日志文件路径
model_path = '已经训练好的ckpt模型'
batch_size = 1
rows, cols, channels = 224, 224, 3
num_classes = 2

# 获取模型参数的命名空间
arg_scope = nets.vgg.vgg_arg_scope()

# 创建网络
with  slim.arg_scope(arg_scope):
    '''
    2.定义占位符和网络结构
    '''
    # 输入图片
    input_images = tf.placeholder(dtype=tf.float32, shape=[batch_size, rows, cols, channels])
    # 训练还是测试？测试的时候不进行梯度回传
    is_training = tf.placeholder(dtype=tf.bool)
    # 创建vgg16网络
    logits, end_points = nets.vgg.vgg_16(input_images, is_training=is_training, num_classes=num_classes)
    # 预测标签
    pred = tf.argmax(logits, axis=1)
    # 定义saver来进行模型恢复
    saver = tf.train.Saver()
    # 设置gpu占用比例，
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        # 恢复模型
        saver.restore(sess, model_path)
        graph = tf.get_default_graph()
        print("模型加载完毕.")
        print("开始测试")
        # 创建一个协调器，管理线程
        coord = tf.train.Coordinator()
        # 启动QueueRunner, 此时文件名才开始进队。
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(img_num):
            image = Image.open(img_path + '/' + imgs[i])
            image = image.resize((224, 224))
            image = np.array(image)/255. - 0.5
            pic = np.expand_dims(image, axis=0)
            pred_value = sess.run(pred, feed_dict={input_images: pic, is_training: False})
            # print('result:', pred_value[0] == 0)
            if pred_value[0] == 1:
                shutil.copy(img_path + '/' + imgs[i], save_path + '/' + imgs[i])
            print(i)
        print("Done!")
        # 终止线程
        coord.request_stop()
        coord.join(threads)
