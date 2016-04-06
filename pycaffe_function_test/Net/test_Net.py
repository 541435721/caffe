# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016/04/06_15:00 '

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

x, y = sklearn.datasets.make_classification(n_samples=100, n_features=4, n_redundant=0, n_informative=2,
                                            n_clusters_per_class=2, hypercube=False, random_state=0)

x = random.randint(0, 100, (100, 2))
y = array(map(lambda x: int(x[-1] > x[0]), x))

x, xt, y, yt = sklearn.cross_validation.train_test_split(x, y)

with h5py.File('train.h5', 'w') as f:
    f['data'] = x
    f['label'] = y.astype(float32)
with open('train.txt', 'w') as f:
    f.write('train.h5' + '\n')
    f.write('train.h5' + '\n')

comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File('test.h5', 'w') as f:
    f.create_dataset('data', data=xt, **comp_kwargs)
    f.create_dataset('label', data=yt.astype(float32), **comp_kwargs)
with open('test.txt', 'w') as f:
    f.write('test.h5' + '\n')

solver = caffe.get_solver('solver.proto')  # 设置优化器配置文件
solver.solve()

accuracy = 0
batch_size = solver.test_nets[0].blobs['data'].num
test_iters = int(len(xt) / batch_size)
for i in range(test_iters):
    solver.test_nets[0].forward()
    accuracy += solver.test_nets[0].blobs['accuracy'].data
accuracy /= test_iters

print '精确度：' + str(accuracy)
print solver.test_nets[0].blobs['data'].data
print solver.test_nets[0].blobs['label'].data
