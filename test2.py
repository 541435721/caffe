import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('F:/data/mnist/lenet_auto_solver.prototxt')
solver.solve()
