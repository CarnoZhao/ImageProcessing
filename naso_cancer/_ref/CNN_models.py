# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:05:37 2019

@author: Zhong.Lianzhen
"""

import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np

class_num = 2
cardinality = 8

def conv_layer(input_data, filter_num, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs = input_data, use_bias = False, filters = filter_num, kernel_size = kernel, strides=stride, padding=padding)
        return network

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Average_pooling(x, pool_size=[2,2], stride = 2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs = x, pool_size = pool_size, strides = stride, padding = padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope = scope,
                   updates_collections = None,
                   decay = 0.9,
                   center = True,
                   scale = True,
                   zero_debias_moving_mean = True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs = x, is_training = training, reuse = None),
                       lambda : batch_norm(inputs = x, is_training = training, reuse = True))

def Relu(x):
    return tf.nn.relu(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs = x, use_bias = False, units = class_num, name = 'linear')



class ResNeXt():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_ResNext(x)

    def first_layer(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter_num = 32, kernel=[3, 3], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            
            x = conv_layer(x, filter_num = 64, kernel=[3, 3], stride = 2, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            
            x = tf.layers.max_pooling2d(inputs = x, pool_size = [3,3], strides = 2, padding = 'SAME')

            return x

    def transform_layer(self, x, stride, scope):
        input_dim = x.get_shape().as_list()[-1]
        nn = input_dim/(2*cardinality)
        with tf.name_scope(scope) :
            x = conv_layer(x, filter_num = nn, kernel=[1,1], stride=stride, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter_num = nn, kernel=[3,3], stride=1, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter_num = out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            # x = Relu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name) :
            layers_split = list()
            for i in range(cardinality) :
                splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def residual_layer(self, input_x, out_dim, layer_num, res_block = 3):
        # split + transform(bottleneck) + transition + merge

        for i in range(res_block):
            # input_dim = input_x.get_shape().as_list()[-1]
            input_dim = input_x.get_shape().as_list()[-1]

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1
            x = self.split_layer(input_x, stride = stride, layer_name = 'split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(x, out_dim = out_dim, scope = 'trans_layer_'+layer_num+'_'+str(i))

            if flag is True :
                pad_input_x = Average_pooling(input_x)
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
            else :
                pad_input_x = input_x

            input_x = Relu(x + pad_input_x)

        return input_x 


    def Build_ResNext(self, input_x):
        # only cifar10 architecture

        input_x = self.first_layer(input_x, scope='first_layer')

        x = self.residual_layer(input_x, out_dim = 64, layer_num = '1', res_block = 3)
        x = self.residual_layer(x, out_dim = 128, layer_num = '2', res_block = 4)
        x = self.residual_layer(x, out_dim = 256, layer_num = '3', res_block = 6)
        x = self.residual_layer(x, out_dim = 512, layer_num = '4', res_block = 3)

        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)

        # x = tf.reshape(x, [-1,10])
        return x


class DenseNet():
#    DenseNet-121
    def __init__(self, x, training, sita = 0.5):
        self.training = training
        self.growthRate = 12
        self.sita = sita
        self.model = self.Build_DenseNet(x)
    
    def first_layer(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter_num = 32, kernel=[3, 3], stride=1, layer_name = scope + '_conv1')
            x = Batch_Normalization(x, training = self.training, scope=scope + '_batch1')
            x = Relu(x)
            
            x = conv_layer(x, filter_num = 64, kernel=[3, 3], stride = 2, layer_name = scope + '_conv2')
            x = Batch_Normalization(x, training = self.training, scope = scope + '_batch2')
            x = Relu(x)
            
            x = tf.layers.max_pooling2d(inputs = x, pool_size = [3,3], strides = 2, padding = 'SAME')

            return x

    def add_layer(self, scope, l):
        with tf.name_scope(scope):
            c = Batch_Normalization(l, training = self.training, scope = scope + '_batch1')
            c = Relu(c)
            c = conv_layer(c,filter_num = 4*self.growthRate, kernel = [1,1], stride = 1, layer_name = scope+'_conv1')
            c = Batch_Normalization(l, training = self.training, scope=scope + '_batch2')
            c = Relu(c)
            c = conv_layer(c,filter_num = self.growthRate, kernel = [3,3], stride = 1, layer_name = scope+'_conv2')
            l = tf.concat([c, l], 3)
        return l
    
    def add_transition(self,scope, l):
        in_channel = l.get_shape().as_list()[-1]
        with tf.name_scope(scope):
            l = Batch_Normalization(l, training = self.training, scope = scope + '_batch')
            l = Relu(l)
            l = conv_layer(l,filter_num = int(self.sita * in_channel), kernel = [1,1], stride = 1, layer_name = scope+'_conv')
            l = Average_pooling(l)
        return l
    
    def dense_layer(self,x,layer_num,res_block,is_last = False):
        for i in range(res_block):
            x = self.add_layer('dense_layer_' + layer_num + str(i), x)
        if  not is_last:
            x = self.add_transition('transition_' + layer_num, x)
        
        return x
    
    def Build_DenseNet(self, input_x):
#       DenseNet-121
        input_x = self.first_layer(input_x, scope='first_layer')

        x = self.dense_layer(input_x, layer_num='1', res_block = 6)
        x = self.dense_layer(x, layer_num='2', res_block = 12)
        x = self.dense_layer(x, layer_num='3', res_block = 24)
        x = self.dense_layer(x, layer_num='4', res_block = 16,is_last = True)

        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)

        return x

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def Fully_connected(x, units=class_num, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

class SE_ResNeXt():
    def __init__(self, x, training, reduction_ratio = 4):
        self.reduction_ratio = reduction_ratio
        self.training = training
        self.model = self.Build_SE_ResNext(x)

    def first_layer(self, x, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter_num = 32, kernel=[3, 3], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            
            x = conv_layer(x, filter_num = 64, kernel=[3, 3], stride = 2, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            
            x = tf.layers.max_pooling2d(inputs = x, pool_size = [3,3], strides = 2, padding = 'SAME')

            return x

    def transform_layer(self, x, stride, scope):
        input_dim = int(np.shape(x)[-1])
        nn = input_dim/(2*cardinality)
        with tf.name_scope(scope):
            x = conv_layer(x, filter_num = nn, kernel=[1,1], stride=stride, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter_num = nn, kernel=[3,3], stride=1, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter_num = out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            # x = Relu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name):
            layers_split = list()
            for i in range(cardinality):
                splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)
        
    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):

            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units = out_dim / ratio, layer_name = layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units = out_dim, layer_name = layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation

            return scale

    def SE_residual_layer(self, input_x, out_dim, layer_num, res_block = 3):
        # split + transform(bottleneck) + transition + merge

        for i in range(res_block):
            # input_dim = input_x.get_shape().as_list()[-1]
            input_dim = input_x.get_shape().as_list()[-1]

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1
            x = self.split_layer(input_x, stride = stride, layer_name = 'split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(x, out_dim = out_dim, scope = 'trans_layer_'+layer_num+'_'+str(i))
            x = self.squeeze_excitation_layer(x, out_dim = out_dim, ratio = self.reduction_ratio, layer_name = 'squeeze_layer_'+layer_num+'_'+str(i))
   
            if flag is True:
                    pad_input_x = Average_pooling(input_x)
                    pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
            else:
                pad_input_x = input_x
            
            input_x = Relu(x + pad_input_x)
    
        return input_x 


    def Build_SE_ResNext(self, input_x):
#       SE_resNext-50
        input_x = self.first_layer(input_x, scope = 'first_layer')

        x = self.SE_residual_layer(input_x, out_dim = 64, layer_num='1', res_block = 3)
        x = self.SE_residual_layer(x, out_dim = 128, layer_num='2', res_block = 4)
        x = self.SE_residual_layer(x, out_dim = 256, layer_num='3', res_block = 6)
        x = self.SE_residual_layer(x, out_dim = 512, layer_num='4', res_block = 3)

        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)

        return x