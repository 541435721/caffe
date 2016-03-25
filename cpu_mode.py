# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016_2016/2/28_15:30 '

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import caffe
import time

caffe_root = 'E:/caffe-windows-master/'

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

start = time.clock()

# 设置CPU模式
caffe.set_mode_cpu()
# 用模型创建初始化网络
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# 输入预处理 'data'是输入blob对象的名字，blobs['data']==net.input[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data',
                     np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))  # mean pixel
transformer.set_raw_scale('data', 255)  # 参考模型作用于像素值为[0,255]的范围，而不是[0,1]
transformer.set_channel_swap('data', (2, 1, 0))  # 参考模型作用的像素通道顺序为BGR而不是RGB

# 设置bacth大小为50
net.blobs['data'].reshape(50, 3, 227, 227)

# 读入将要进行分类的图片
net.blobs['data'].data[...] = transformer.preprocess('data',
                                                     caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'))
# 图像往上层前进
out = net.forward()
print("Predicted class is #{}.".format(out['prob'][0].argmax()))

# 显示图片
plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
# 加载分类标签
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print labels[top_k]

print time.clock() - start
