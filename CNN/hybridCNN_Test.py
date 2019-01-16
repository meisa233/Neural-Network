# -*- coding: utf-8 -*-
import numpy as np
import caffe
import sys
import matplotlib.pyplot as plt
import os

caffe_root = 'C:\\caffe'
sys.path.insert(0, caffe_root + '\\caffe\\python')

os.chdir('E:\\Meisa_SiameseNetwork\\hybridCNN\\hybridCNN_Reference')
hybridCNN_Weight = 'hybridCNN_iter_700000.caffemodel'
hybridCNN_Def = 'hybridCNN_deploy_new.prototxt'
net = caffe.Net(hybridCNN_Def,      
                hybridCNN_Weight,  
                caffe.TEST)


#将mean.binaryproto转换为mean.npy
blob=caffe.proto.caffe_pb2.BlobProto()
data=open('hybridCNN_mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
array = np.array(caffe.io.blobproto_to_array(blob))#j转换为numpy格式
mean_npy = array[0]
np.save('place205.npy',mean_npy)
mu=np.load('place205.npy')

mu = mu.mean(1).mean(1)  

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))


net.blobs['data'].reshape(10,
                          3,
                          227,227)

image = caffe.io.load_image('hybridCNN_testpicture.jpg')
transformed_image = transformer.preprocess('data', image)

plt.imshow(image)
plt.show()

net.blobs['data'].data[...] = transformed_image
output = net.forward()
output_prob = output['prob'][0]
print 'predicted class is:', output_prob.argmax()

for layer_name, blob in net.blobs.iteritems():
    print layer_name+'\t'+str(blob.data.shape)

for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
#labels_file = 'E:\\Meisa_SiameseNetwork\\VGG19-2\\synset_words.txt'

#labels = np.loadtxt(labels_file, str, delimiter = '\t')

#print 'output label: ', labels[output_prob.argmax()]
