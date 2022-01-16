import tensorflow as tf2

tf = tf2.compat.v1
tf.disable_eager_execution()
import timeit
import traceback
import csv
import sys
import numpy as np


class CNN:
    """
    This class forms the base of the 3D CCN network
    """

    def __init__(self, img_size_px, slice_count, n_classes, batch_size, keep_rate,
                 hm_epochs, gpu=True, model_name='model'):

        self.gpu = gpu
        self.hm_epochs = hm_epochs
        self.img_size_px = img_size_px
        self.slice_count = slice_count
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.keep_rate = keep_rate
        self.model_name = model_name

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
                'W_fc': tf.Variable(tf.random_normal([22528, 1024])),
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
        biases = self.__get_biases()

        # reshaping the image to the provided pixels (see init)
        x = tf.reshape(x, shape=[-1, self.img_size_px, self.img_size_px, self.slice_count, 1])  # image x, y, z
        conv1 = self.__maxpool3d(tf.nn.relu(self.__conv3d(x, weights['W_conv1']) + biases['b_conv1']))
        conv2 = self.__maxpool3d(tf.nn.relu(self.__conv3d(conv1, weights['W_conv2']) + biases['b_conv2']))

        fc = tf.reshape(conv2, [-1, 22528])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc, self.keep_rate)

        output = tf.matmul(fc, weights['out']) + biases['out']
        return output

    def __build_output(self, run_time, accuracy, fitment_perc):
        """
        This function will return all parameters and the newly
        obatined parameters from a training run.
        :param run_time: The run time in seconds
        :param accuracy:  The accuracy of the training
        :param fitment_perc: The fitment percentage (decimal)
        :return: The model parameters and statistics for training
        """
        return {
            'run_time': run_time,
            'accuracy': accuracy,
            'fitment_percent': fitment_perc,
            'number_epochs': self.hm_epochs,
            'img_size_px': self.img_size_px,
            'slice_count': self.slice_count,
            'batch_size': self.batch_size,
            'keep_rate': self.keep_rate
        }

    def __get_processor(self):
        """
        This function can be used to switch between CPU and GPU
        :return: configuration indicating which processor should be used.
        """
        if self.gpu:
            return tf.ConfigProto()  # just return default config
        else:
            return tf.ConfigProto(
                device_count={'GPU': 0}
            )

    def __reverse_label(self, labels):
        """
        Reverse the input format for tensorflow to human readable format
        :param labels:
        :return:
        """
        return [1 if label[1] == 1 else 0 for label in labels]

    def __write_stats(self, ids, labels, predicted, output_path):
        """
        This function writes the testing stats to the given output path
        :param ids: The patient ids
        :param labels: The orginal labels of the patients (1 = sick, 0 = norma)
        :param predicted:  The predicted class for the patient
        :param output_path: The output path to which the results should be written
        """
        with open(output_path, 'w') as out_file:
            header = ['id', 'label', 'predicted']
            writer = csv.writer(out_file, delimiter='\t')
            writer.writerows([header])
            writer.writerows(zip(*[ids, self.__reverse_label(labels), predicted]))

    def test_neural_network(self, test_data, model_name, model_folder, output_path=None):
        """
        This function can be used to test the neural network on a training set
        :param test_data: The test data (obtained from splitter.py)
        :param model_name: The name of the model that should be used
        :param model_folder: The folder in which the model is saved
        :param output_path: The path to which the output should be written (see write stats)
        """
        x = tf.placeholder('float')
        y = tf.placeholder('float')
        prediction = self.__convolutional_neural_network(x)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()

            for epoch in range(self.hm_epochs):
                try:
                    # this will load the trained model
                    saver.restore(sess, model_folder + model_name + '.ckpt')
                except Exception as e:
                    print(str(e))

            # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y[:290], 1))
            # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            prediction_result = tf.argmax(prediction,1)

            test_data = np.load(test_data, allow_pickle=True)
            ids, features, labels = [], [], []

            for item in test_data:
                data = item[0]
                label = item[1]
                id = item[2]
                ids.append(id)
                features.append(data)
                labels.append(label)

            test_x = np.array(features)
            test_y = np.array(labels)

            # this will predicted the classes and will write the given
            # and predicted classes to the output file (if specified)
            # if output_path:
            #     predicted = sess.run(tf.argmax(test_y, 1), feed_dict={x: test_x})
            #     self.__write_stats(ids, labels, predicted, output_path)
            #     print('[INFO] test stats written to: {}'.format(output_path))

            print('[INFO] Accuracy:', prediction_result.eval({x: test_x, y: test_y}),1)

    def __load_data(self, train_data, validation_data):
        """
        This function can be used to load the training and validation data
        :param train_data: The training data that should be loaded
        :param validation_data: The validation data that should be loaded
        :return:
        """
        try:
            train_data = np.load(train_data, allow_pickle=True)
            validation_data = np.load(validation_data, allow_pickle=True)
            return train_data, validation_data
        except FileNotFoundError as e:
            print('[ERROR] The file {} was not found!'.format(e.filename))
            sys.exit(1)


