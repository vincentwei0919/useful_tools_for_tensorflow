#coding:utf-8
import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow
ckpt='/home/model.ckpt'
reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key))
