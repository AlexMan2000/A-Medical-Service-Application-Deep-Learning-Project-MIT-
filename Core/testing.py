import tensorflow as tf2
tf = tf2.compat.v1
import csv
import sys
import numpy as np

x = tf.placeholder('float')
y = tf.placeholder('float')

class CNN:
    """
    This class forms the base of the 3D CCN network
    """

    def __init__(self):
        self.n_classes = 2
        self.batch_size = 10
        self.img_size_px = 50
        self.slice_count = 20
        self.keep_rate = 0.8

    def __conv3d(self, x, W):
        """
        This function computes a 3-D convolution.
        :param x: A float placeholder (tensor)
        :param W: The weights that should be used
        :return: A 3D CNN based comprising the provided weights
        """
        return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

    def __maxpool3d(self, x):
        """
        This function performs 3D max pooling
        :param x: A float placeholder (tensor)
        :return: The 3D pool
        """
        return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    def __get_weights(self):
        """
        This function will return the weight used in the model
        :return: The model
        """
        return {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
                   #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
                   'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
                   #                                  64 features
                   'W_fc': tf.Variable(tf.random_normal([54080, 1024])),
                   'out': tf.Variable(tf.random_normal([1024, self.n_classes]))}

    def __get_biases(self):
        """
        This function returns the biases used in the model
        :return: The biases
        """
        return {'b_conv1': tf.Variable(tf.random_normal([32])),
                  'b_conv2': tf.Variable(tf.random_normal([64])),
                  'b_fc': tf.Variable(tf.random_normal([1024])),
                  'out': tf.Variable(tf.random_normal([self.n_classes]))}


    def __convolutional_neural_network(self, x):
        """
        This function uses the conv3d, maxpool3d, get_weights, and get_biases functions to
        actually build the model, which can be trained hereafter (using the train_neural_network
        function)
        :param x: A float placeholder
        :return:A 3D CNN model which can be trained.
        """
        #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
        weights = self.__get_weights()
        biases =  self.__get_biases()

        # reshaping the image to the provided pixels (see init)
        x = tf.reshape(x, shape=[-1, self.img_size_px, self.img_size_px, self.slice_count, 1]) #image x, y, z
        conv1 = self.__maxpool3d(tf.nn.relu(self.__conv3d(x, weights['W_conv1']) + biases['b_conv1']))
        conv2 = self.__maxpool3d(tf.nn.relu(self.__conv3d(conv1, weights['W_conv2']) + biases['b_conv2']))

        fc = tf.reshape(conv2, [-1, 54080])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc, self.keep_rate)

        output = tf.matmul(fc, weights['out']) + biases['out']
        return output

    def getModelShape(self):
        result = self.__convolutional_neural_network(self,)

if __name__ =="__main__":
    cnn = CNN()
