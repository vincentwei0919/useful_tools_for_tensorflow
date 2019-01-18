# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.contrib.slim import nets
slim = tf.contrib.slim

import read_tfrecord


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

rows, cols, channels = 224, 224, 3

epochs = 50000
batch_size = 64

num_classes = 1
learn_rate = 0.001
weight_decay = 0.0005


if __name__ == '__main__':
    resnet_model_path = './resnet_v1_50.ckpt'
    train_record = './female_train.tfrecords'
    val_record = './female_val.tfrecords'
    model_save_path = './ResNet50.ckpt'
    train_dir = './train_logs/'
    val_dir = './val_logs/'

    x = tf.placeholder(tf.float32, (None, rows, cols, channels), name='inputs')
    y = tf.placeholder(tf.float32, (None, 1), name='labels')
    is_train = tf.placeholder(tf.bool, name='is_training')

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        net, endpoints = nets.resnet_v1.resnet_v1_50(x, num_classes=None, global_pool=True, is_training=is_train)
    with variable_scope.variable_scope('resnet_v1_50', reuse=True):
            net = slim.flatten(net, scope='flatten')
            logits = slim.fully_connected(net, num_outputs=num_classes, activation_fn=None, scope='fc')

    checkpoint_exclude_scopes = "logits"

    train_data, train_labels = read_tfrecord.create_batch([train_record], batch_size, rows, cols, True)
    val_data, val_labels = read_tfrecord.create_batch([val_record], batch_size, rows, cols, False)


    with tf.name_scope("Loss"):
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = tf.sqrt(tf.reduce_sum(tf.square(logits - y)))
        total_loss = loss + l2_loss * weight_decay

    global_step = tf.Variable(0, trainable=False)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=0.9, use_nesterov=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
        train_step = optimizer.minimize(total_loss, global_step=global_step)


    with tf.name_scope("Accuracy"):
        accuracy = tf.reduce_mean(tf.square(logits-y))
        # accuracy = tf.reduce_mean(tf.abs(logits-y))

    saver_restore = tf.train.Saver(var_list=variables_to_restore)
    saver = tf.train.Saver(tf.global_variables())
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        saver_restore.restore(sess, resnet_model_path)
        train_writer = tf.summary.FileWriter(train_dir, sess.graph)
        test_writer = tf.summary.FileWriter(val_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        try:
            for epoch in range(1, epochs+1):
                if coord.should_stop():
                     break
                if epoch == (epochs* 0.25) or epoch == (epochs * 0.5) or epoch == (epochs * 0.75):
                    learn_rate = learn_rate / 10
                train_data_np, train_labels_np = sess.run([train_data, train_labels])
                _, train_los, train_acc = sess.run([train_step, loss, accuracy], feed_dict={x: train_data_np, y:train_labels_np, is_train:True})
                train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_los),
                                                  tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
                print("epoch: %d/%d: train loss is :%.4f, train accuracy is %.4f"%(epoch, epochs, train_los, train_acc))

                if epoch % 200 == 0:
                    total_acc = 0
                    total_los = 0
                    for j in range(156):

                        val_data_np, val_labels_np = sess.run([val_data, val_labels])
                        val_los, val_acc = sess.run([loss, accuracy], feed_dict={x: val_data_np, y: val_labels_np, is_train: True})
                        print("val iteration:%d, val loss is :%.4f, val accuracy is %.4f" % (j, val_los, val_acc))
                        total_acc += val_acc
                        total_los += val_los

                    val_summary = tf.Summary(value=[tf.Summary.Value(tag='val_loss', simple_value=total_los/100),
                                                    tf.Summary.Value(tag='val_accuracy', simple_value=total_acc/100)])
                    train_writer.add_summary(summary=train_summary, global_step=epoch)
                    test_writer.add_summary(summary=val_summary,  global_step=epoch)
                    train_writer.flush()
                    test_writer.flush()

                    print("**************epoch: %d/%d, val_loss is: %.4f, val_acc is: %.4f\n" % (epoch, epochs,total_los/156, total_acc/156))

                if (epoch + 1) % 200 == 0:
                    saver.save(sess, model_save_path, global_step=global_step)
                    print('save mode to {}'.format(model_save_path))

        except tf.errors.OutOfRangeError:
            print("Done trainning!")
        finally:
            coord.request_stop()
            coord.join(threads)






