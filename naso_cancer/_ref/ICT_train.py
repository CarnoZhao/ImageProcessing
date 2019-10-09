# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:41:35 2019

@author: Zhong.Lianzhen
"""
import tensorflow as tf
import numpy as np
import os
import scipy.io
from CNN_models import ResNeXt,DenseNet,SE_ResNeXt
from datetime import datetime
from tensorflow.python.framework import graph_util
from Prepare_data import preprocess

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
model_path = 'SE_ResNeXt'
weight_decay = 0.0005
momentum = 0.9
base_learning_rate = 0.01
label_smoothing = 0.01
gamma = 2
belta = 2.0


batch_size = 16
batch_size_v = 8
num_classes = 2
num_epochs = 100
img_width = 128
img_heigh = 128


output_path = os.path.join(r"/home/competition/ZlZ-ShiPin/ICT-PNC", model_path)
checkpoint_path = os.path.join(output_path, "checkpoint_path")
if not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

main_path = r'/home/competition/ZlZ-ShiPin/ICT-PNC/group_data1.mat'
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    data1 = scipy.io.loadmat(main_path)
    tra_n_path = data1['train_slice_0']
    tra_p_path = data1['train_slice_1']
    train_pat = data1['train_pat'].tolist()
    validation_pat = data1['validation_pat'].tolist()
    test_pat = data1['test_pat'].tolist()
    tra_n_path = tra_n_path.tolist()
    tra_p_path = tra_p_path.tolist()
    num_p = int(batch_size * 0.67)
    num_n = batch_size - num_p
    p_batchs = int(len(tra_p_path) / num_p)
    n_batchs = int(len(tra_n_path) / num_n)
    num_batchs = min(p_batchs, n_batchs)

    tra_path = tra_n_path + tra_p_path
    ind1 = sorted(range(len(tra_path)), key = lambda k: tra_path[k].split('_')[1])
    train_order_file = []
    for i in range(len(ind1)):
        train_order_file.append(tra_path[ind1[i]])

    val_path = data1['val_slice_0'].tolist() + data1['val_slice_1'].tolist()
    ind1 = sorted(range(len(val_path)), key = lambda k: val_path[k].split('_')[1])
    val_order_file = []
    for i in range(len(ind1)):
        val_order_file.append(val_path[ind1[i]])

    test_path = data1['test_slice_0'].tolist() + data1['test_slice_1'].tolist()
    ind1 = sorted(range(len(test_path)), key = lambda k: test_path[k].split('_')[1])
    test_order_file = []
    for i in range(len(ind1)):
        test_order_file.append(test_path[ind1[i]])



def main():
        
    with tf.device('/gpu:3'):
        x = tf.placeholder(tf.float32, [None, img_width, img_heigh, 2], name = 'input')
        y = tf.placeholder(tf.float32, [None, num_classes], name = 'label')
        training_flag = tf.placeholder(tf.bool)
        if model_path == 'SE_ResNeXt':
            logits = SE_ResNeXt(x, training = training_flag).model
        if model_path == 'DenseNet':
            logits = DenseNet(x, training = training_flag, sita = 0.8).model
        if model_path == 'ResNeXt':
            logits = ResNeXt(x, training = training_flag,).model
        final_tensor = tf.nn.softmax(logits)
        #label smooth
        new_labels = (1.0 - label_smoothing) * y + label_smoothing / num_classes
        bias_base = tf.tile([belta,1.0], [batch_size])
        bias_class = tf.reshape(bias_base, shape = [-1,2], name = 'bias_class')
        weight2 = tf.multiply(bias_class, new_labels)
        cross_entropy = -tf.reduce_mean(tf.multiply(weight2, tf.log(tf.clip_by_value(final_tensor,1e-6,1.0))))
        # 定义交叉熵损失函数。
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        global_step = tf.Variable(0)
        learning_rate = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(base_learning_rate, global_step,500,0.96,staircase = True)
        optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = momentum, use_nesterov = True)
        train_step = optimizer.minimize(cross_entropy_mean + l2_loss * weight_decay, global_step = global_step)

        # 计算正确率。
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(tf.argmax(tf.reduce_mean(final_tensor, axis = 0), 0), tf.argmax(y[0], 0))
            evaluation_step = tf.cast(correct_prediction, tf.float32)
            
            
        # Initialize an saver for store model checkpoints
        saver = tf.train.Saver(max_to_keep = 5)
            
        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)#设置每个GPU使用率0.7代表70%
        #
        config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
        with tf.Session(config = config) as sess:
            
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    #        saver.restore(sess, "/home/competition/ZlZ-ShiPin/PythonApplication1/experiment_test/checkpoints/model_epoch10.ckpt") # 导入训练好的参数
            
            # Loop over number of epochs
            for epoch in range(num_epochs):
            
                print("{} Epoch number: {}".format(datetime.now(), epoch+1))

                # Initialize iterator with the training dataset
                train_loss = 0.0
                kk = 0
                np.random.shuffle(tra_p_path)
                np.random.shuffle(tra_n_path)
                for i in range(num_batchs-1):
                    img_path = tra_p_path[i*num_p:(i+1)*num_p] + tra_n_path[i*num_n:(i+1)*num_n]
                    lab = np.zeros([batch_size,], dtype = np.float32)
                    lab[:num_p] = 1
                    ind = np.random.permutation(batch_size)
                    label_batch1 = lab[ind]
                #    加主路径
                    img_paths = np.array(img_path)[ind]
                    img_paths = img_paths.tolist()
                    label_batch = tf.one_hot(label_batch1, depth = num_classes)
                    img_batch = preprocess(img_paths)
                    
                    x_in, y_in = sess.run([img_batch,label_batch])
                    # And run the training op
                    # import pdb;pdb.set_trace()
                    _,batch_loss,gs = sess.run([train_step, cross_entropy_mean, global_step], feed_dict = {x: x_in, y: y_in, training_flag: True})
                    # import pdb;pdb.set_trace()
                    if not np.isnan(batch_loss):
                        kk += 1
                        train_loss += batch_loss
                    else:
                        print('******')

                train_loss /= kk

                print("epoch: %d/%d, train_loss: %.4f"  % (epoch, num_epochs, train_loss))

                    
                if (epoch+1) % 10 == 0:
                    print("Start validation0: {}".format(datetime.now()))
                    tra_acc = 0.
                    tra_count = 0

                    xd = train_order_file[0].split('_')[1]
                    img_path1 = []
                    for item in train_order_file:
                        temp = item.split('_')[1]
                        if temp == xd:
                            img_path1.append(item)
                        else:
                            img_batch = preprocess(img_path1, False)
                            yd = int(img_path1[0].split('_')[0])
                            label_batch = tf.one_hot(yd, depth = num_classes)
                            # import pdb;pdb.set_trace()
                            label_batch=  tf.expand_dims(label_batch,0)
                            x_in, y_in = sess.run([img_batch,label_batch])
                            train_accuracy = sess.run(evaluation_step, feed_dict = {x: x_in, y: y_in, training_flag: False})
                            tra_acc += train_accuracy
                            tra_count += 1

                            xd = temp
                            img_path1 = []
                            img_path1.append(item)
                
                    tra_acc /= tra_count

                    print("Start validation1: {}".format(datetime.now()))
                    
                    val_acc = 0.
                    val_count = 0

                    xd = val_order_file[0].split('_')[1]
                    img_path1 = []
                    for item in val_order_file:
                        temp = item.split('_')[1]
                        if temp == xd:
                            img_path1.append(item)
                        else:
                            img_batch = preprocess(img_path1, False)
                            yd = int(img_path1[0].split('_')[0])
                            label_batch = tf.one_hot(yd, depth = num_classes)
                            label_batch=  tf.expand_dims(label_batch,0)
                            x_in, y_in = sess.run([img_batch,label_batch])
                            val_accuracy = sess.run(evaluation_step, feed_dict = {x: x_in, y: y_in, training_flag: False})
                            val_acc += val_accuracy
                            val_count += 1

                            xd = temp
                            img_path1 = []
                            img_path1.append(item)
                
                    val_acc /= val_count
                    
                    print("Start validation3: {}".format(datetime.now()))

                    test_acc = 0.
                    test_count = 0

                    xd = test_order_file[0].split('_')[1]
                    img_path1 = []
                    for item in test_order_file:
                        temp = item.split('_')[1]
                        if temp == xd:
                            img_path1.append(item)
                        else:
                            img_batch = preprocess(img_path1, False)
                            yd = int(img_path1[0].split('_')[0])
                            label_batch = tf.one_hot(yd, depth = num_classes)
                            label_batch=  tf.expand_dims(label_batch,0)
                            x_in, y_in = sess.run([img_batch,label_batch])
                            test_accuracy = sess.run(evaluation_step, feed_dict = {x: x_in, y: y_in, training_flag: False})
                            test_acc += test_accuracy
                            test_count += 1

                            xd = temp
                            img_path1 = []
                            img_path1.append(item)
                
                    test_acc /= test_count

                    lr = sess.run(learning_rate)
                    line = "epoch: %d/%d, global_step: %d, learning_rate: %.4f, tra_acc: %.4f, val_acc: %.4f, test_acc: %.4f"  % (epoch, num_epochs, gs, lr, tra_acc, val_acc, test_acc)
                    print(line)
                    with open(os.path.join(output_path,'logs_1-2.txt'), 'a') as f:
                        f.write(line)
                    
                    # # save checkpoint of the model
                    checkpoint_name = os.path.join(checkpoint_path,
                                            'model_epoch'+str(num_epochs)+'.ckpt')
                    saver.save(sess, checkpoint_name)
                
                    print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                                    checkpoint_name))
            
                


if __name__ == '__main__':
    main()