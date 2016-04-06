# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016/04/05_10:06 '

from numpy import *
from caffe import layers as L
from caffe import params as P
import caffe
from pylab import *

import h5py
import shutil
import tempfile

import sklearn
import sklearn.datasets
import sklearn.linear_model

import pandas as pd

# 创建一个数据集，样本数量为10000，特征维度为4，其中有效特征为2，另外两个为噪声特征
# prarm1:样本数量，param2:特征维度，param3：冗余数，param4:有效特征数，param5:类别数
X, y = sklearn.datasets.make_classification(n_samples=10000, n_features=4, n_redundant=0, n_informative=2,
                                            n_clusters_per_class=2, hypercube=False, random_state=0)
# 把样本分成训练姐和数据集
X, Xt, y, yt = sklearn.cross_validation.train_test_split(X, y)

# 可视化样本
ind = np.random.permutation(X.shape[0])[:1000]
df = pd.DataFrame(X[ind])
_ = pd.scatter_matrix(df, figsize=(9, 9), diagonal='kde', marker='o', s=40, alpha=.4, c=y[ind])
# show()

# 使用随机梯度下降算法学习和评估sklearn的逻辑回归，检查准确度
# 训练和测试 sklearn SGD逻辑回归
clf = sklearn.linear_model.SGDClassifier(loss='log', n_iter=1000, penalty='l2', alpha=1e-3, class_weight='auto')
clf.fit(X, y)  # 拟合数据
yt_pred = clf.predict(Xt)  # 预测数据
print('Accuracy: {:.3f}'.format(sklearn.metrics.accuracy_score(yt, yt_pred)))

# 把数据集保存为HDF5文件 以便Caffe读入

with h5py.File('train.h5', 'w') as f:
    f['data'] = X
    f['label'] = y.astype(np.float32)
with open('train.txt', 'w') as f:
    f.write('train.h5' + '\n')
    f.write('train.h5' + '\n')

comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File('test.h5', 'w') as f:
    f.create_dataset('data', data=Xt, **comp_kwargs)
    f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
with open('test.txt', 'w') as f:
    f.write('test.h5' + '\n')


# 在Caffe中创建逻辑回归模型
def logreg(hdf5, batch_size):
    n = caffe.NetSpec()  # 创建神经网络
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)  # 读入数据集
    n.ip1 = L.InnerProduct(n.data, num_output=2, weight_filler=dict(type='xavier'))  # 输入层定义
    n.accuracy = L.Accuracy(n.ip1, n.label)  # 准确度评估值
    n.loss = L.SoftmaxWithLoss(n.ip1, n.label)  # 输出层
    return n.to_proto()


# 写入数据集到prototxt文件
with open('logreg_auto_train.prototxt', 'w') as f:
    f.write(str(logreg('train.txt', 10)))

with open('logreg_auto_test.prototxt', 'w') as f:
    f.write(str(logreg('test.txt', 10)))

# 训练网络
caffe.set_mode_cpu()  # 设置GPU模式
solver = caffe.get_solver('solver.prototxt')  # 设置优化器配置文件
solver.solve()

accuracy = 0
batch_size = solver.test_nets[0].blobs['data'].num
test_iters = int(len(Xt) / batch_size)
for i in range(test_iters):
    solver.test_nets[0].forward()
    accuracy += solver.test_nets[0].blobs['accuracy'].data
accuracy /= test_iters

print("Accuracy: {:.3f}".format(accuracy))

