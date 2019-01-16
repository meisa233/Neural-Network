import sys
import os
import cv2
import numpy as np
if __name__ == '__main__':
    sys.path.insert(0,'/data/caffe/python')
    import caffe
    import math
    import csv

    caffe.set_mode_gpu()
    caffe.set_device(0)
    # load model(.prototxt) and weight (.caffemodel)
    os.chdir('/data/Meisa/ResNet/ResNet-50')
    ResNet_Weight = './resnet50_cvgj_iter_320000.caffemodel'#pretrained on il 2012 and place 205

    ResNet_Def = 'deploynew_nosoftmax.prototxt'
    net = caffe.Net(ResNet_Def,
                    ResNet_Weight,
                    caffe.TEST)

    # load video
    i_Video = cv2.VideoCapture('/data/RAIDataset/Video/2.mp4')

    # get width of this video
    wid = int(i_Video.get(3))

    # get height of this video
    hei = int(i_Video.get(4))

    # get the number of frames of this video
    framenum = int(i_Video.get(7))

    if i_Video.isOpened():
        success = True
    else:
        success = False
        print('Can\' open this video!')

    ret, frame = i_Video.read()

    # Frame stores all frames of this video
    Frame = []
    while success:
        success, frame = i_Video.read()
        Frame.append(frame)





    # # Convert .binaryproto to .npy file
    # blob = caffe.proto.caffe_pb2.BlobProto()
    # data = open('hybridCNN_mean.binaryproto', 'rb').read()
    # blob.ParseFromString(data)
    # array = np.array(caffe.io.blobproto_to_array(blob))
    # mean_npy = array[0]
    # np.save('place205.npy', mean_npy)
    # mu = np.load('place205.npy')
    #
    # mu = mu.mean(1).mean(1)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2, 0, 1))
    #transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))



    net.blobs['data'].reshape(1,
                              3,
                              224, 224)

    FrameV = []
    # for i in Frame:
    #     #cv2.imwrite('tmp.jpg', Frame[i])
    #     #image = caffe.io.load_image('tmp.jpg')
    #     transformed_image = transformer.preprocess('data', i)
    #     net.blobs['data'].data[...] = transformed_image
    #     output = net.forward()
    #     FrameV.append(output['fc8'][0])
    #     #print FrameV[-1]
    i = 0

    #InputFrame = np.array(Frame[0])
    for i in range(len(Frame)):
        if Frame[i] is None:
            print i
            continue
        transformed_image = transformer.preprocess('data', Frame[i])
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        FrameV.append(output['score'][0].tolist())




    # write the features to .csv file
    with open('/data/Meisa/ResNet/ResNet-50/ResNetFeatures/Video2Features.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for item in FrameV:
            csvwriter.writerow(item)
