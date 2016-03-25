import caffe

caffe.set_device(0)
caffe.set_mode_gpu()  # 设置GPU模式

solver = caffe.SGDSolver('F:/data/mnist/lenet_auto_solver.prototxt')  # 设置solver参数文件
solver.solve()
