# coding:utf-8
# @Author:bianxuesheng

from numpy import *
import caffe
from caffe import layers as L
from caffe import params as P


# 定义minst网络
# lmdb:路径名
# batch_size:数据量
# source: 数据库文件的路径
# backend [default LEVELDB]: 选择使用 LEVELDB 还是 LMDB
def lenet(lmdb, batch_size):
    # 创建网络
    n = caffe.NetSpec()
    # 定义数据集
    n.data, n.label = L.Data(batch_size=batch_size,
                             backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), ntop=2)
    # 定义第一层卷积层
    # 参数1：输入数据
    # kernel_size:卷积核尺寸
    # num_output:卷积核数量
    # weight_fillter:指定参数的初始化方案，default type: 'constant' value: 0
    # bias_term [default true]: 指定是否给卷积输出添加偏置项
    # pad (或者 pad_h 和 pad_w) [default 0]: 指定在输入图像周围补 0 的像素个数
    # stride (或者 stride_h 和 stride_w) [default 1]: 指定卷积核在输入 图像上滑动的步长
    # group (g) [default 1]: 指定分组卷积操作的组数，默认为 1 即不分组。具体地说, 输入图像和输出图像在通道维度上分别被
    # 分成 g 个组, 输出图像的第 i 组只与输入图像第 i 组连接（ 即输入图像的第 i 组与相应的卷积核卷积得到第 i组输出）。
    n.conv1 = L.Convolution(n.data, kernel_size=5,
                            num_output=20, weight_filler=dict(type='xavier'))
    # 定义第一层池化层
    # 参数1：输入数据
    # kernel_size:池化核尺寸
    # stride[default 1]: 指定池化窗口在输入数据上滑动的步长
    # pad[default 0]: 指定在输入图像周围补 0 的像素个数
    #
    n.pool1 = L.Pooling(n.conv1, kernel_size=2,
                        stride=2, pool=P.Pooling.MAX)
    # 定义第二层卷积层
    n.conv2 = L.Convolution(n.pool1, kernel_size=5,
                            num_output=50, weight_filler=dict(type='xavier'))
    # 定义第二层池化层
    n.pool2 = L.Pooling(n.conv2, kernel_size=2,
                        stride=2, pool=P.Pooling.MAX)
    # 定义第一层内积层
    n.ip1 = L.InnerProduct(n.pool2, num_output=500,
                           weight_filler=dict(type='xavier'))
    # 定义激活函数层
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    # 定义第二层内积层
    n.ip2 = L.InnerProduct(n.relu1, num_output=10,
                           weight_filler=dict(type='xavier'))
    # 定义输出层
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)

    return n.to_proto()


# 保存网络定义
with open('F:/data/mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('F:/data/mnist/mnist_train_lmdb', 64)))
with open('F:/data/mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('F:/data/mnist/mnist_test_lmdb', 100)))
