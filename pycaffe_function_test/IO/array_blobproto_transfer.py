# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016/04/06_10:50 '

from numpy import *
import caffe
from caffe import layers as L
from caffe import params as P


data = random.randint(0, 100, (100, 4))
proto = caffe.io.array_to_blobproto(data)
with open('data.prototxt', 'w') as f:
    f.write(str(proto))


data2 = caffe.io.blobproto_to_array(proto)

print data2
