# -*- coding: utf-8 -*-
"""
# @Time    : 18-2-2 下午4:46
# @Author  : Crd
# @Email   : crd57@126.com
# @File    : LSTM.py
# @Software: PyCharm
"""

import pandas as pd
import numpy as np
import random
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Dropout


class NeuralNetwork():
    def __init__(self, **kwargs):
        """
        :param
        **kwargs:
        output_dim=4: output dimension of LSTM layer;
        activation_LSTM='tanh': activation function for LSTM layers;
        activation_dense='relu' : activation function for Dense layer;
        activation_last='sigmoid': activation function for last layer;
        drop_out=0.2: fraction of input units to drop;
        np_epoch=10, the number of epoches to train the model.epoch is one forward pass and one backward pass of all the training examples;
        batch_size=32: number of samples per gradient update. The higher the batch size, the more memory space you'll need;
        loss='mean_square_error': loss function;
        optimizer='Rmsprop'
        """
        self.output_dim = kwargs.get('output_dim', 8)
        self.activation_lstm = kwargs.get('activation_lstm', 'relu')
        self.activation_dense = kwargs.get('activation_dense', 'relu')
        self.activation_last = kwargs.get('activation_last', 'softmax')  # softmax for multiple output
        self.dense_layer = kwargs.get('dense_layer', 2)  # at least 2 layers
        self.lstm_layer = kwargs.get('lstm_layer', 2)
        self.drop_out = kwargs.get('drop_out', 0.2)
        self.nb_epoch = kwargs.get('nb_epoch', 10)
        self.batch_size = kwargs.get('batch_size', 100)
        self.loss = kwargs.get('loss', 'categorical_crossentropy')
        self.optimizer = kwargs.get('optimizer', 'rmsprop')

    def NN_model(self, trainX, trainY, testX, testY):
        """
        :param trainX: training data set
        :param trainY: expect value of training data
        :param testX: test data set
        :param testY: epect value of test data
        :return: model after training
        """
        print("Training model is LSTM network!")
        input_dim = trainX[1].shape[1]
        output_dim = trainY.shape[1]  # one-hot label
        # print predefined parameters of current model:
        model = Sequential()
        # applying a LSTM layer with x dim output and y dim input. Use dropout parameter to avoid overfitting
        model.add(LSTM(output_dim=self.output_dim,
                       input_dim=input_dim,
                       activation=self.activation_lstm,
                       dropout_U=self.drop_out,
                       return_sequences=True))
        for i in range(self.lstm_layer - 2):
            model.add(LSTM(output_dim=self.output_dim,
                           input_dim=self.output_dim,
                           activation=self.activation_lstm,
                           dropout_U=self.drop_out,
                           return_sequences=True))
        # argument return_sequences should be false in last lstm layer to avoid input dimension incompatibility with dense layer
        model.add(LSTM(output_dim=self.output_dim,
                       input_dim=self.output_dim,
                       activation=self.activation_lstm,
                       dropout_U=self.drop_out))
        for i in range(self.dense_layer - 1):
            model.add(Dense(output_dim=self.output_dim,
                            activation=self.activation_last))
        model.add(Dense(output_dim=output_dim,
                        input_dim=self.output_dim,
                        activation=self.activation_last))
        # configure the learning process
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        # train the model with fixed number of epoches
        model.fit(x=trainX, y=trainY, nb_epoch=self.nb_epoch, batch_size=self.batch_size,
                  validation_data=(testX, testY))
        # store model to json file
        model_json = model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        # store model weights to hdf5 file
        if model_weight_path:
            if os.path.exists(model_weight_path):
                os.remove(model_weight_path)
            model.save_weights(model_weight_path)  # eg: model_weight.h5
        return model