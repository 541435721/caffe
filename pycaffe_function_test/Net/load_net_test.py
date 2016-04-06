# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016/04/06_20:15 '

from numpy import *
import caffe
from caffe import layers as L
from caffe import params as P
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model

caffe.set_device(0)
caffe.set_mode_gpu()

x = random.randint(0, 100, (10, 2))
y = array(map(lambda x: int(x[-1] > x[0]), x))

# 让caffe以测试模式读取网络参数
net = caffe.Net('Net.proto', 'snap_iter_10000.caffemodel', caffe.TEST)

for k, v in net.blobs.items():
    print (k, v.data.shape)
