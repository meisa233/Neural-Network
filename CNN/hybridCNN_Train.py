# 训练Siamese网络
# -*- coding: utf-8 -*-
import numpy as np
import caffe
import sys
import matplotlib.pyplot as plt
import os

caffe_root = 'C:\\caffe'
sys.path.insert(0, caffe_root + '\\caffe\\python')

caffe.set_mode_cpu()

os.chdir('E:\\Meisa_SiameseNetwork\\hybridCNN\\hybridCNN_Reference')
weights = 'hybridCNN_iter_700000.caffemodel'
SolverPath = 'solver.prototxt'

solver = caffe.get_solver(SolverPath)
solver.net.copy_from(weights)

niter = 1000
disp_interval = 1

loss = np.zeros(niter)
testloss = np.zeros(10)

for it in range(niter):
    solver.step(1)
    #loss[it]  = solver.net.blobs['loss'].data.copy()
    #print '%s: loss=%.3f' % (it, loss[it])
    #if it % disp_interval == 0 or it + 1 == niter:
    #    loss_disp = '%s: loss=%.3f' % (it, loss[it])
    #    for test_it in range(10):
    #        solver.test_nets[0].forward()
    #        test_loss[test_it] = solver.test_nets[0].blobs['loss'].data
    #    print '%s: test_loss=%.3f' % (it, np.mean(test_loss))
