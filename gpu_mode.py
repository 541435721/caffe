# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016_2016/2/28_15:30 '

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import caffe
import time
import cv2

caffe_root = 'E:/caffe-windows-master/'

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

start = time.clock()

# 设置GPU模式
caffe.set_device(0)
caffe.set_mode_gpu()
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
cv2.imshow('test', transformer.deprocess('data', net.blobs['data'].data[0]))
cv2.waitKey()

# 加载分类标签
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print labels[top_k]

# print time.clock() - start

net.forward()

# 打印出每层输出名称和形状
print [(k, v.data.shape) for k, v in net.blobs.items()]


# 可视化函数；获取每层输出的array.
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))  # 获取图像的宽度尺寸
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    # plt.imshow(data)
    cv2.imshow('test', data)
    cv2.waitKey(3000)


# 卷积核，[weights,biases]构成的list
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))

# 卷积后的输出图像，先输出前36张
feat = net.blobs['conv1'].data[0, :36]
vis_square(feat, padval=1)

# 第二个卷积核,一共256个
filters = net.params['conv2'][0].data
vis_square(filters[:48].reshape(48 ** 2, 5, 5))

# 卷积核2的输出结果 共256个
feat = net.blobs['conv2'].data[0, :36]
vis_square(feat, padval=1)

# 卷积核3输出结果
feat = net.blobs['conv3'].data[0]
vis_square(feat, padval=0.5)

# 卷积核4输出结果
feat = net.blobs['conv4'].data[0]
vis_square(feat, padval=0.5)

# 卷积核5输出结果
feat = net.blobs['conv5'].data[0]
vis_square(feat, padval=0.5)

# 池化层5输出结果
feat = net.blobs['pool5'].data[0]
vis_square(feat, padval=1)

# 第一层全连接层输出结果
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.show()

# 第二层全连接层输出结果
feat = net.blobs['fc7'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.show()

# 最终结果输出
feat = net.blobs['prob'].data[0]
plt.plot(feat.flat)
plt.show()
