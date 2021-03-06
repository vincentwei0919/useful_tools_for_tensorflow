#coding:utf-8

import tensorflow as tf
import numpy as np
import os
import read_tfrecord
from tensorflow.contrib.slim import nets
import model

slim = tf.contrib.slim
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
    1.设置参数，并加载数据
'''
# 用于保存微调后的检查点文件和日志文件路径
train_log_dir = '/home/.../train_log/'
test_log_dir = '/home/.../test_log/'

save_model_path = '/home/.../model.ckpt'

# 官方下载的检查点文件路径
checkpoint_file = '/home/.../vgg_16.ckpt'
train_record_path = '/home/.../train.tfrecords'
test_record_path = '/home/.../val.tfrecords'
# 设置训练参数
num_classes = 2
batch_size = 64
learning_rate = 1e-3
weight_decay = 0.0005
rows, cols, channels = 224, 224, 3
# 训练集数据长度
n_train = 192000
# 测试集数据长度
n_test = 24000
# 迭代轮数
training_epochs = 8
# 每100个batch显示一下训练状态
disp_batch = 100

# 加载数据
train_images, train_labels = read_tfrecord.create_batch(train_record_path, batch_size, rows, cols, True)

test_images, test_labels = read_tfrecord.create_batch(test_record_path, batch_size, rows, cols, False)

# 获取模型参数的命名空间
arg_scope = nets.vgg.vgg_arg_scope()

# 创建网络
with  slim.arg_scope(arg_scope):
    '''
    2.定义占位符和网络结构
    '''
    # 输入图片
    input_images = tf.placeholder(dtype=tf.float32, shape=[batch_size, rows, cols, 3], name='input_x')
    # 图片标签
    input_labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, num_classes], name='input_y')
    # 训练还是测试？测试的时候其权参数会设置为1.0
    is_training = tf.placeholder(dtype=tf.bool)

    # 创建vgg16网络  如果想冻结所有层，可以指定slim.conv2d中的 trainable=False

    logits, end_points =model.vgg_16(input_images, is_training=is_training, num_classes=0, spatial_squeeze=False, global_pool=False)
    # Restore only the convolutional layers: 从检查点载入当前图除了fc8层之外所有变量的参数
    # params = slim.get_variables_to_restore(include=['vgg16/conv1', 'vgg16/conv2','vgg_16/conv3', 'vgg_16/conv4', 'vgg_16/conv5'])
    params = slim.get_variables_to_restore(exclude=['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8'])
    restorer = tf.train.Saver(params)

    #每个元素都是以vgg_16/xx命名

    '''
    #从当前图中搜索指定scope的变量，然后从检查点文件中恢复这些变量(即vgg_16网络中定义的部分变量)
    #如果指定了恢复检查点文件中不存在的变量，则会报错 如果不知道检查点文件有哪些变量，我们可以打印检查点文件查看变量名
    params = []
    conv1 = slim.get_variables(scope="vgg_16/conv1")
    params.extend(conv1)
    conv2 = slim.get_variables(scope="vgg_16/conv2")
    params.extend(conv2)
    conv3 = slim.get_variables(scope="vgg_16/conv3")
    params.extend(conv3)
    conv4 = slim.get_variables(scope="vgg_16/conv4")
    params.extend(conv4)
    conv5 = slim.get_variables(scope="vgg_16/conv5")
    params.extend(conv5)
    fc6 = slim.get_variables(scope="vgg_16/fc6")
    params.extend(fc6)
    fc7 = slim.get_variables(scope="vgg_16/fc7")
    params.extend(fc7)
    '''
    #新建命名空间，来将修改之后的网络进行统一存放
    with tf.variable_scope('after_vgg16'):
        logits = slim.flatten(logits, scope='flatten')
        logits = slim.fully_connected(logits, 128, activation_fn=tf.nn.relu, scope='fc1')
        logits = slim.fully_connected(logits, num_classes, activation_fn=None, scope='fc2')

    # 用于恢复模型  如果使用这个保存或者恢复的话，只会保存或者恢复指定的变量

    global_step = tf.Variable(0, trainable=False)
    '''
    3 定义代价函数和优化器
    '''
    # 代价函数
    learn_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_steps=6000, decay_rate=0.8,
                                            staircase=True)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_labels, logits=logits))
    # 将l2_loss整合
    with tf.variable_scope('l2_loss'):
        var_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        l2_loss = var_loss * weight_decay
    total_loss = cost + l2_loss
    #cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_labels, logits=logits))

    # 设置优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(total_loss)

    # 预测结果评估
    pred = tf.nn.softmax(logits, name='softmax')
    classes = tf.argmax(pred, axis=1, name='prediction')
    correct = tf.equal(tf.cast(classes, dtype=tf.int64), tf.argmax(input_labels, axis=1))  # 返回一个数组 表示统计预测正确或者错误
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # 求准确率

    train_batch = int(np.ceil(n_train / batch_size))
    test_batch = int(np.ceil(n_test / batch_size))

    # 用于保存检查点文件
    save = tf.train.Saver(max_to_keep=3)


    # 恢复模型
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # 检查最近的检查点文件
        ckpt = tf.train.latest_checkpoint(train_log_dir)
        if ckpt != None:
            save.restore(sess, ckpt)
            print('从上次训练保存后的模型继续训练！')
        else:
            restorer.restore(sess, checkpoint_file)
            print('从官方模型加载训练！')

        # 创建一个协调器，管理线程
        coord = tf.train.Coordinator()
        # 启动QueueRunner, 此时文件名才开始进队。
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        test_writer = tf.summary.FileWriter(test_log_dir)
        # 保存模型参数，用来进行可视化
        tf.summary.scalar('loss', cost)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learn rate', learn_rate)
        merged_summary = tf.summary.merge_all()
        '''
        4 查看预处理之后的图片
        '''
        # imgs, labs = sess.run([train_images, train_labels])
        # print('原始训练图片信息：', imgs.shape, labs.shape)
        # show_img = np.array(imgs[0], dtype=np.uint8)
        # plt.imshow(show_img)
        # plt.title('Original train image')
        # plt.show()
        #
        # imgs, labs = sess.run([test_images, test_labels])
        # print('原始测试图片信息：', imgs.shape, labs.shape)
        # show_img = np.array(imgs[0], dtype=np.uint8)
        # plt.imshow(show_img)
        # plt.title('Original test image')
        # plt.show()

        print('开始训练！')
        cnt  = 0
        for epoch in range(training_epochs):
            train_cost = 0.0
            train_acc = 0.0
            test_cost = 0.0
            test_acc = 0.0

            for i in range(train_batch):
                imgs, labs = sess.run([train_images, train_labels])
                _, summary, loss, acc = sess.run([optimizer, merged_summary, cost, accuracy],
                                                           feed_dict={input_images: imgs, input_labels: labs, is_training: True})
                train_cost += loss
                train_acc += acc
                # 打印信息
                if (cnt+1) % 10 == 0:
                    print('step {}/{}  average cost: {:.4f}  accuracy:  {:.4f}'.format(i, train_batch, loss, acc))
                    train_writer.add_summary(summary, global_step=i)
                    train_writer.flush()
                cnt += 1
            # 进行预测处理

            print("开始测试")
            for j in range(test_batch):
                imgs, labs = sess.run([test_images, test_labels])

                summary, cost_values, accuracy_value = sess.run([merged_summary, cost, accuracy], feed_dict={input_images: imgs, input_labels: labs, is_training: False})
                test_acc += accuracy_value
                test_cost += cost_values
                if (j+1) % 10 == 0:
                    test_writer.add_summary(summary, j)
                    test_writer.flush()

            print('Epoch {}/{}  Test cost {:.4f}'.format(epoch + 1, training_epochs, test_cost / test_batch))
            print('准确率:', test_acc/test_batch)

            # 保存模型
            save.save(sess, save_model_path, global_step=epoch)
            print('Epoch {}/{}  模型保存成功'.format(epoch + 1, training_epochs))
        train_writer.close()
        test_writer.close()
        print('训练完成')

        # 终止线程
        coord.request_stop()
        coord.join(threads)
