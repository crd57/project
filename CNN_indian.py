# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   CNN_indian
   Author:      crd
   date:        2018/3/25
-------------------------------------------------
"""

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import scipy.io as scio
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_mat(filename, matname, n1, n2):
    load_data = scio.loadmat(filename)
    data_mat = load_data[matname]
    data_mat = np.float64(data_mat)
    data_reshape = data_mat.reshape(n1, n2)
    return data_reshape


def data_pre(X_datas, Y_datas):
    X1 = MinMaxScaler().fit_transform(X_datas)  # 每一列最大值最小值标准化
    Y1 = OneHotEncoder().fit_transform(Y_datas).todense()  # one-hot编码
    X1 = X1.reshape(-1, 1, 200, 1)
    return X1, Y1


def generatebatch_MBGD(X2, Y2, n_examples, batch_size=8):
    for batch_i in range(n_examples // batch_size):  # //整数除法
        start = batch_i * batch_size
        end = start + batch_size
        batch_x = X2[start:end]
        batch_y = Y2[start:end]
        yield batch_x, batch_y


def generatebatch_RBGD(X2, Y2, n_examples, n):
    for i in range(n):
        batch_i = np.random.randint(0, n_examples)
        batch_x = X2[batch_i]
        batch_y = Y2[batch_i]
        yield batch_x, batch_y


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 20, 1], strides=[1, 1, 2, 1], padding='SAME')
    #  [batch, height, width, channel] 池化窗口大小
    #  [batch, height, width, channel] 窗口滑动步长


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)


if __name__ == "__main__":
    train_path = 'Indian_pines_corrected.mat'
    label_path = 'Indian_pines_gt.mat'
    X_data = read_mat(train_path, 'indian_pines_corrected', 145 * 145, 200)
    Y_data = read_mat(label_path, 'indian_pines_gt', 145 * 145, 1)
    X, Y = data_pre(X_data, Y_data)
    tf.reset_default_graph()
    with tf.name_scope('input'):
        tf_X = tf.placeholder(tf.float32, [None, 1, 200, 1])
        tf_Y = tf.placeholder(tf.float32, [None, 17])
    with tf.name_scope('conv1'):
        with tf.name_scope('weights_conv1'):
            W_conv1 = weight_variable([1, 20, 1, 5])  # 前两维patch的大小，输入通道数，输出的通道数。
            # tf.summary.scalar('W_conv1', W_conv1)
        with tf.name_scope('biases_conv1'):
            b_conv1 = bias_variable([5])
            # variable_summaries(b_conv1)
        with tf.name_scope('relu_conv1'):
            h_conv1 = tf.nn.relu(conv2d(tf_X, W_conv1) + b_conv1)
            tf.summary.histogram('h_conv1', h_conv1)
        with tf.name_scope('pool1'):
            h_pool1 = max_pool_3x3(h_conv1)
            # tf.summary.histogram('h_pools', h_pool1)
    with tf.name_scope('conv2'):
        with tf.name_scope('weights_conv2'):
            W_conv2 = weight_variable([1, 20, 5, 5])
            # variable_summaries(W_conv2)
        with tf.name_scope('biases_conv2'):
            b_conv2 = bias_variable([5])
            # variable_summaries(b_conv2)
        with tf.name_scope('relu_conv2'):
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            tf.summary.histogram('h_conv2', h_conv2)
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_3x3(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 50 * 5])
        tf.summary.histogram('h_pool2', h_pool2)
    with tf.name_scope('fc'):
        with tf.name_scope('weights_fc'):
            W_fc = tf.Variable(tf.random_normal([1 * 50 * 5, 25]))
            # variable_summaries(W_fc)
        with tf.name_scope('biases_fc'):
            b_fc = tf.Variable(tf.random_normal([25]))
            # variable_summaries(b_fc)
        with tf.name_scope('relu_fc'):
            h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)
            tf.summary.histogram('h_fc', h_fc)
    with tf.name_scope('out_layer'):
        with tf.name_scope('weights_out'):
            W_out = tf.Variable(tf.random_normal([25, 17]))
            # variable_summaries(W_out)
        with tf.name_scope('biases_out'):
            b_out = tf.Variable(tf.random_normal([17]))
            variable_summaries(b_out)
        with tf.name_scope('relu_fc'):
            out_layer = tf.nn.softmax(tf.matmul(h_fc, W_out) + b_out)
            # tf.summary.histogram('h_out', out_layer)
    with tf.name_scope('loss'):
        loss = -tf.reduce_mean(tf_Y * tf.log(tf.clip_by_value(out_layer, 1e-11, 1.0)))  # 损失函数
        tf.summary.scalar('loss', loss)
    # clip_by_value = 输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
    # reduce_mean 求平均值  reduce_max 求最大值
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)  # Adam 算法优化器
    with tf.name_scope('accuracy'):
        y_pred = tf.arg_max(out_layer, 1)  # 返回最大的那个数值所在的下标
        bool_pred = tf.equal(tf.arg_max(tf_Y, 1), y_pred)  # 是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True
        accuracy = tf.reduce_mean(tf.cast(bool_pred, tf.float32))  # cast 数据格式转换
        tf.summary.scalar('accuracy', accuracy)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # merged = tf.summary.merge_all()
        for epoch in range(1000):
            for batch_xs, batch_ys in generatebatch_MBGD(X, Y, Y.shape[0]):
                sess.run(train_step, feed_dict={tf_X: batch_xs, tf_Y: batch_ys})
                if epoch % 100 == 0:
                    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    # run_metadata = tf.RunMetadata()
                    # summary_str = sess.run(merged)
                    # writer.add_summary(summary_str, epoch)
                    res = sess.run(accuracy, feed_dict={tf_X: X, tf_Y: Y})
                    print(epoch, res)
        res_ypred = y_pred.eval(feed_dict={tf_X: X, tf_Y: Y}).flatten()
        print(res_ypred)
