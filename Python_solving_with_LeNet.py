# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016_2016/2/28_15:30 '

from numpy import *
from caffe import layers as L
from caffe import params as P
import caffe
from pylab import *


def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()


with open('F:/data/mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('F:/data/mnist/mnist_train_lmdb', 64)))

with open('F:/data/mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('F:/data/mnist/mnist_test_lmdb', 100)))

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('F:/data/mnist/lenet_auto_solver.prototxt')  # 用solver的prototxt文件初始化sovler

for k, v in solver.net.blobs.items():
    print (k, v.data.shape)

solver.net.forward()  # 训练网络
solver.test_nets[0].forward()  # 测试网络

# 测试训练样本
imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8 * 28), cmap='gray')
print solver.net.blobs['label'].data[:8]
# show()

# 验证测试样本
imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8 * 28), cmap='gray')
print solver.test_nets[0].blobs['label'].data[:8]
# show()

solver.step(1)
imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5)
       .transpose(0, 2, 1, 3).reshape(4 * 5, 5 * 5), cmap='gray')
# show()

niter = 200
test_interval = 25
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# 网络的迭代过程
for it in range(niter):
    solver.step(1)  # 单步随机梯度下降

    # 保存训练中的损失函数值
    train_loss[it] = solver.net.blobs['loss'].data

    # 保存每次迭代结果
    # 开始第一层卷积层
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['ip2'].data[:8]

    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
# show()
'''
for i in range(8):
    figure(figsize=(2, 2))
    imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    figure(figsize=(10, 2))
    imshow(output[:50, i].T, interpolation='nearest', cmap='gray')
    xlabel('iteration')
    ylabel('label')
    # show()
'''
for i in range(8):
    figure(figsize=(2, 2))
    imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    figure(figsize=(10, 2))
    imshow(exp(output[:50, i].T) / exp(output[:50, i].T).sum(0), interpolation='nearest', cmap='gray')
    xlabel('iteration')
    ylabel('label')
    show()
