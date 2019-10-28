# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:41:35 2019

@author: Zhong.Lianzhen
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf
import numpy as np
import scipy.io
#ResNeXt,DenseNet,
from CNN_models import ResNeXt,DenseNet,SE_ResNeXt
from datetime import datetime
from Prepare_data1 import preprocess
import pandas as pd
from lifelines.utils import concordance_index


model_path = 'SE_ResNeXt'
# weight_decay = 0.005
momentum = 0.9
base_learning_rate = 1e-3
num_classes = 2


batch_size = 64
num_epochs = 100
img_width = 128
img_heigh = 128


output_path = os.path.join("/wangshuo/ZhongLianzhen/ICT-NPC/T1C_result/", model_path + '_New_64_Surv-3_basedOnSurv')
checkpoint_path = os.path.join(output_path, "checkpoint_path")
if not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

main_path = r"/wangshuo/ZhongLianzhen/ICT-NPC/summary_T1C.mat"
clinic_path = r"/home/competition/ZlZ-ShiPin/ICT-PNC/classify_FFS3.csv"
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    data1 = scipy.io.loadmat(main_path)
    clinic_msag = pd.read_csv(clinic_path, header=0, index_col = 0)
    tra_msag = clinic_msag[clinic_msag.data_cohort == 0]
    val_msag = clinic_msag[clinic_msag.data_cohort == 1]
    test_msag = clinic_msag[clinic_msag.data_cohort == 2]
    tra_Pat_ID = np.array(tra_msag.index)
    tra_FFS_time = np.array(tra_msag.loc[:, 'FFS.time'], np.float32)
    tra_FFS_event = np.array(tra_msag.loc[:, 'FFS.event'], np.int32)
    
    val_Pat_ID = np.array(val_msag.index)
    val_FFS_time = np.array(val_msag.loc[:, 'FFS.time'], np.float32)
    val_FFS_event = np.array(val_msag.loc[:, 'FFS.event'], np.int32)
    
    test_Pat_ID = np.array(test_msag.index)
    test_FFS_time = np.array(test_msag.loc[:, 'FFS.time'], np.float32)
    test_FFS_event = np.array(test_msag.loc[:, 'FFS.event'], np.int32)
    num_batchs = int(len(tra_Pat_ID) / batch_size)

def _prepare_surv_data(surv_time, surv_event):
    surv_data_y = surv_time * ([item == 1 and 1.0 or -1.0 for item in surv_event])
    surv_data_y = np.array(surv_data_y, np.float32)
    T = - np.abs(np.squeeze(surv_data_y))
    sorted_idx = np.argsort(T)
    _Y = surv_data_y[sorted_idx]
    
    return _Y

def DeepSurv_loss(Y, Y_hat):
    # Obtain T and E from self.Y
    # NOTE: negtive value means E = 0
    Y_c = tf.squeeze(Y)
    Y_hat_c = tf.squeeze(Y_hat)
    Y_label_T = tf.abs(Y_c)
    Y_label_E = tf.cast(tf.greater(Y_c, 0), dtype=tf.float32)
    Obs = tf.reduce_sum(Y_label_E)
    
    Y_hat_hr = tf.exp(Y_hat_c)
    Y_hat_cumsum = tf.log(tf.cumsum(Y_hat_hr))
    
    # Start Computation of Loss function
    
    # Get Segment from T
    _, segment_ids = tf.unique(Y_label_T)
    # Get Segment_max
    loss_s2_v = tf.segment_max(Y_hat_cumsum, segment_ids)
    # Get Segment_count
    loss_s2_count = tf.segment_sum(Y_label_E, segment_ids)
    # Compute S2
    loss_s2 = tf.reduce_sum(tf.multiply(loss_s2_v, loss_s2_count))
    # Compute S1
    loss_s1 = tf.reduce_sum(tf.multiply(Y_hat_c, Y_label_E))
    # Compute Breslow Loss
    loss_breslow = tf.divide(tf.subtract(loss_s2, loss_s1), Obs)
    
    return loss_breslow

def _create_fc_layer(x, output_dim, activation, scope, keep_prob):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        w = tf.get_variable('weights', [x.shape[1], output_dim], 
            initializer=tf.truncated_normal_initializer(stddev=0.1))
    
        b = tf.get_variable('biases', [output_dim], 
            initializer=tf.constant_initializer(0.0))
    
        # add weights and bias to collections
        tf.add_to_collection("var_weight", w)
        tf.add_to_collection("var_bias", b)
    
        layer_out = tf.nn.dropout(tf.matmul(x, w) + b, keep_prob)
    
        if activation == 'relu':
            layer_out = tf.nn.relu(layer_out)
        elif activation == 'sigmoid':
            layer_out = tf.nn.sigmoid(layer_out)
        elif activation == 'tanh':
            layer_out = tf.nn.tanh(layer_out)
        else:
            raise NotImplementedError('activation not recognized')
    
        return layer_out

def main():
        
    with tf.device('/gpu:3'):
        
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        x = tf.placeholder(tf.float32, [None, img_width, img_heigh, 2], name = 'input')
        y = tf.placeholder(tf.float32, [None,], name = 'label')
        training_flag = tf.placeholder(tf.bool)
        dropout_rate = tf.placeholder(tf.float32, [], name = 'dropout_rate')
        if model_path == 'SE_ResNeXt':
            _, Global_Average = SE_ResNeXt(x, training = training_flag).model
        if Global_Average == 'DenseNet':
            _, Global_Average = DenseNet(x, training = training_flag, sita = 0.8).model
        if model_path == 'ResNeXt':
            _, Global_Average = SE_ResNeXt(x, training = training_flag).model
#         variables = tf.contrib.framework.get_variables_to_restore()
# #       if v.name.split('/')[0]!='linear'
#         variables_to_restore = [v for v in variables]
#         variables_to_restore = tf.stop_gradient(variables_to_restore)

        fc1 =  _create_fc_layer(Global_Average, 128, 'tanh', 'FC_layer1', dropout_rate)
        output = _create_fc_layer(fc1, 1, 'tanh', 'FC_layer2', 1.0)
        with tf.name_scope('MIL'):
            instance_index = tf.argmax(output, 0)
            pred_value = tf.reduce_max(output, 0)
            pred_value = tf.squeeze(pred_value)
        with tf.name_scope('DeepSurv_loss'):
            loss1 = DeepSurv_loss(y, output)
        reg_item = tf.contrib.layers.l1_l2_regularizer(0.0, 0.0)
        loss_reg = tf.contrib.layers.apply_regularization(reg_item, tf.get_collection("var_weight"))
        global_step = tf.Variable(0)
        # learning_rate = tf.train.exponential_decay(base_learning_rate, global_step,num_batchs * 5, 0.9, staircase = True)
        optimizer = tf.train.MomentumOptimizer(learning_rate = base_learning_rate, momentum = momentum, use_nesterov = True)
        # , var_list = tf.get_collection("var_weight")
        train_step = optimizer.minimize(loss1 + loss_reg, global_step = global_step)

        saver = tf.train.Saver(max_to_keep = 30)
        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)#设置每个GPU使用率0.7代表70%
        #
        config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
        with tf.Session(config = config) as sess:
            
            # Initialize all variables
            # sess.run(tf.global_variables_initializer())
            # "/wangshuo/ZhongLianzhen/ICT-NPC/T1C_result/SE_ResNeXt_Surv-3_64_1-1p5/checkpoint_path/model_epoch-40"
            # "/wangshuo/ZhongLianzhen/ICT-NPC/T2_result/SE_ResNeXt_Surv-3_16_4_1-1p5/checkpoint_path/model_epoch-11"
            saver.restore(sess, "/wangshuo/ZhongLianzhen/ICT-NPC/T1C_result/SE_ResNeXt_Surv-3_64_1-1p5/checkpoint_path/model_epoch-40") # 导入训练好的参数
            
            # print("{} start compute CI".format(datetime.now()))

            # tra_pred = []
            # for Pat_path in tra_Pat_ID:
            #     Pat_path = str(Pat_path)
            #     instance_paths = list(data1[Pat_path])
            #     instance_batch = preprocess(instance_paths, False)
            #     Pat_pred = sess.run(pred_value, feed_dict = {x: instance_batch, training_flag: False, dropout_rate: 1.0})
            #     tra_pred.append(Pat_pred)
            # tra_pred = np.array(tra_pred, np.float32)
            # ci_value = concordance_index(tra_FFS_time, -1.0 * tra_pred, tra_FFS_event)
            # line = 'pre-train cohort, CI: %.4f' % ci_value
            # print(line)
            # with open(os.path.join(output_path,'logs.txt'), 'a') as f:
            #     f.write(line + '\n')


            # val_pred = []
            # for Pat_path in val_Pat_ID:
            #     Pat_path = str(Pat_path)
            #     instance_paths = list(data1[Pat_path])
            #     instance_batch = preprocess(instance_paths, False)
            #     Pat_pred = sess.run(pred_value, feed_dict = {x: instance_batch, training_flag: False, dropout_rate: 1.0})
            #     val_pred.append(Pat_pred)
            # val_pred = np.array(val_pred, np.float32)
            # ci_value = concordance_index(val_FFS_time,  -1.0 * val_pred, val_FFS_event)
            # line = 'pre-vallidation cohort, CI: %.4f' % ci_value
            # print(line)
            # with open(os.path.join(output_path,'logs.txt'), 'a') as f:
            #     f.write(line + '\n')

            # test_pred = []
            # for Pat_path in test_Pat_ID:
            #     Pat_path = str(Pat_path)
            #     instance_paths = list(data1[Pat_path])
            #     instance_batch = preprocess(instance_paths, False)
            #     Pat_pred = sess.run(pred_value, feed_dict = {x: instance_batch, training_flag: False, dropout_rate: 1.0})
            #     test_pred.append(Pat_pred)
            # test_pred = np.array(test_pred, np.float32)
            # ci_value = concordance_index(test_FFS_time,  -1.0 * test_pred, test_FFS_event)
            # line = 'pre-test cohort, CI: %.4f' % ci_value
            # print(line)
            # with open(os.path.join(output_path,'logs.txt'), 'a') as f:
            #     f.write(line + '\n')
            # import pdb;pdb.set_trace()
            # Loop over number of epochs
            for epoch in range(num_epochs):
            
                print("{} Start epoch number: {}".format(datetime.now(), epoch+1))
                Pat_ind = np.arange(len(tra_Pat_ID))
                np.random.shuffle(Pat_ind)
                # Initialize iterator with the training dataset
                for i in range(num_batchs):
                    ind_start = i*batch_size
                    ind_end = (i+1)*batch_size
                    img_path = tra_Pat_ID[Pat_ind[ind_start:ind_end]]
                    vd_img = []
                    time_batch = tra_FFS_time[Pat_ind[ind_start:ind_end]]
                    event_batch = tra_FFS_event[Pat_ind[ind_start:ind_end]]
                    for Pat_path in img_path:
                        Pat_path = str(Pat_path)
                        instance_paths = list(data1[Pat_path])
                        instance_batch = preprocess(instance_paths, True)
                        IID = sess.run(instance_index, feed_dict = {x: instance_batch, training_flag: False, dropout_rate: 1.0})
                        vd_img.append(instance_batch[IID[0],:,:,:])
                    img_batch = np.stack(vd_img)
                    # import pdb;pdb.set_trace()
                    label_batch = _prepare_surv_data(time_batch, event_batch)
                    _, batch_loss, gs = sess.run([train_step, loss1, global_step], feed_dict = {x: img_batch, y: label_batch, training_flag: True, dropout_rate: 1.0})
                    line = 'batch_loss: %.4f, gs: %d' % (batch_loss, gs)
                    print(line)
                    with open(os.path.join(output_path,'logs.txt'), 'a') as f:
                        f.write(line + '\n')

                print("{} start compute CI".format(datetime.now()))

                tra_pred = []
                for Pat_path in tra_Pat_ID:
                    Pat_path = str(Pat_path)
                    instance_paths = list(data1[Pat_path])
                    instance_batch = preprocess(instance_paths, False)
                    Pat_pred = sess.run(pred_value, feed_dict = {x: instance_batch, training_flag: False, dropout_rate: 1.0})
                    tra_pred.append(Pat_pred)
                tra_pred = np.array(tra_pred, np.float32)
                tra_ci_value = concordance_index(tra_FFS_time, -1.0 * tra_pred, tra_FFS_event)
                line = 'train cohort, CI: %.4f, epoch: %d' % (tra_ci_value, epoch)
                print(line)
                with open(os.path.join(output_path,'logs.txt'), 'a') as f:
                    f.write(line + '\n')


                val_pred = []
                for Pat_path in val_Pat_ID:
                    Pat_path = str(Pat_path)
                    instance_paths = list(data1[Pat_path])
                    instance_batch = preprocess(instance_paths, False)
                    Pat_pred = sess.run(pred_value, feed_dict = {x: instance_batch, training_flag: False, dropout_rate: 1.0})
                    val_pred.append(Pat_pred)
                val_pred = np.array(val_pred, np.float32)
                val_ci_value = concordance_index(val_FFS_time,  -1.0 * val_pred, val_FFS_event)
                line = 'vallidation cohort, CI: %.4f' % val_ci_value
                print(line)
                with open(os.path.join(output_path,'logs.txt'), 'a') as f:
                    f.write(line + '\n')

                test_pred = []
                for Pat_path in test_Pat_ID:
                    Pat_path = str(Pat_path)
                    instance_paths = list(data1[Pat_path])
                    instance_batch = preprocess(instance_paths, False)
                    Pat_pred = sess.run(pred_value, feed_dict = {x: instance_batch, training_flag: False, dropout_rate: 1.0})
                    test_pred.append(Pat_pred)
                test_pred = np.array(test_pred, np.float32)
                test_ci_value = concordance_index(test_FFS_time,  -1.0 * test_pred, test_FFS_event)
                line = 'test cohort, CI: %.4f' % test_ci_value
                print(line)
                with open(os.path.join(output_path,'logs.txt'), 'a') as f:
                    f.write(line + '\n')

                if (tra_ci_value + val_ci_value) > 1.18 and test_ci_value > 0.57:
                    checkpoint_name = os.path.join(checkpoint_path,
                                                'model_epoch')
                    saver.save(sess, checkpoint_name, global_step = epoch)

                    print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_path))
                
                    
                    

if __name__ == '__main__':
    main()
