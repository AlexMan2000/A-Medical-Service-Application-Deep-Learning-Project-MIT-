import tensorflow as tf2
from cnnTest import CNN
import pydicom as dicom
import os
import sys
tf = tf2.compat.v1
tf.disable_eager_execution()

path = r"G:\tempout\model"
testing_data_path = r"G:\tempout\smaller_dataset\test.npy"

x = tf.placeholder("float")
y = tf.placeholder("float")
param_list = [30,20,2,1,1,10]
ins = CNN(*param_list)

res = ins.test_neural_network(testing_data_path,"model",path)


print(res)