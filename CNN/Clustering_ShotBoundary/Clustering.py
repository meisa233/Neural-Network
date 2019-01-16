
import cv2
import sys
import os
import numpy as np

class Clus():
    w = 0.5
    C = []
    Frame = []
    def __init__(self, Frame):
         self.Frame = Frame

    #Calculate the difference between frame a and frame b (frame's dimension is 1183)
    def difference(self, a, b):
        # Chi-square distance

        # a = np.array(a)
        # b = np.array(b)
        # a_minus_b = a - b
        # a_add_b = a + b
        # #print (a_minus_b ** 2) / a_add_b
        # return np.sum(np.true_divide(a_minus_b ** 2,  a_add_b))

        #OUSHI JULI
        a = np.array(a)
        b = np.array(b)

        a_minus_b = a - b
        return np.sqrt(np.sum(a_minus_b ** 2))

    #Use linear interpolation to calculate the n (it is index) frame
    def Mnw(self, n, w):
        # if n+w == int(n+w):
        #     if int(n+w) >= len(self.Frame) or int(n-w) >= len(self.Frame):
        #         return -1
        #     return self.difference(self.Frame[int(n-w)], self.Frame[int(n+w)])
        # else:
        #     return (1. / 2.) * (self.Mnw(n - 0.5, w) + self.Mnw(n + 0.5, w))

        if n+w == int(n+w):
            if int(n + w) >= len(self.Frame) or int(n - w) >= len(self.Frame):
                return -1
            return self.difference(self.Frame[int(n-w)], self.Frame[int(n+w)])
        else:
            return (1. / 2.) * (self.Mnw(n - 0.5, w) + self.Mnw(n + 0.5, w))

    #Calculate all frames of the video the difference(from 1 to the end)
    # between (0.5,1.5), (1.5, 2.5), (2.5, 3.5), ....
    def CalculateDiff(self):
        diff = []
        i = 1
        while i < len(self.Frame):
            #diff.append(self.Mnw(i, self.w))
            diff.append(self.Mnw(i, self.w))
            i = i + 1
        return diff

    def CalculateDiff2(self):
        i = 2
        D = []
        while i < len(self.Frame):
            D.append(self.difference(self.Frame[i], self.Frame[i-2]))
            i = i + 2
        return D
    def Kmeans(self):







if __name__ == '__main__':
    sys.path.insert(0,'/data/caffe/python')
    import caffe
    import math
    import csv

    caffe.set_mode_gpu()
    caffe.set_device(0)
    # # load model(.prototxt) and weight (.caffemodel)
    # os.chdir('/data/Meisa/hybridCNN')
    # hybridCNN_Weight = './snapshots/snapshot_iter_10000.caffemodel'
    #
    # hybridCNN_Def = 'Shot_hybridCNN_deploy_new.prototxt'
    # net = caffe.Net(hybridCNN_Def,
    #                 hybridCNN_Weight,
    #                 caffe.TEST)
    #
    # # load video
    # i_Video = cv2.VideoCapture('/data/RAIDataset/1.mp4')
    #
    # # get width of this video
    # wid = int(i_Video.get(3))
    #
    # # get height of this video
    # hei = int(i_Video.get(4))
    #
    # # get the number of frames of this video
    # framenum = int(i_Video.get(7))
    #
    # if i_Video.isOpened():
    #     success = True
    # else:
    #     success = False
    #     print('Can\' open this video!')
    #
    # ret, frame = i_Video.read()
    #
    # # Frame stores all frames of this video
    # Frame = []
    # while success:
    #     success, frame = i_Video.read()
    #     Frame.append(frame)
    #
    #
    #
    #
    # #Frame.reshape(framenum, wid, hei, 3)
    #
    # #print Frame.shape
    # #print np.shape(Frame)
    #
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
    # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    #
    # transformer.set_transpose('data', (2, 0, 1))
    # transformer.set_mean('data', mu)
    # transformer.set_raw_scale('data', 255)
    # transformer.set_channel_swap('data', (2, 1, 0))
    #
    # net.blobs['data'].reshape(100,
    #                           3,
    #                           227, 227)
    #
    # FrameV = []
    # # for i in Frame:
    # #     #cv2.imwrite('tmp.jpg', Frame[i])
    # #     #image = caffe.io.load_image('tmp.jpg')
    # #     transformed_image = transformer.preprocess('data', i)
    # #     net.blobs['data'].data[...] = transformed_image
    # #     output = net.forward()
    # #     FrameV.append(output['fc8'][0])
    # #     #print FrameV[-1]
    # i = 0
    # while i < math.ceil(len(Frame) / 100.0):
    # # Each loop processes 100 images
    #     #ArrayFrame100 = np.array([])
    #     #for i in range(100):
    #        #ArrayFrame100 = np.vstack(ArrayFrame100, Frame[i])
    #     #print np.array(Frame[i: i+100]).shape
    #
    #
    #     Trans = []
    #     # if the no. of images left >= 100
    #     for j in range(100):
    #         if (j + i * 100)>=len(Frame):
    #             break
    #         if Frame[j + i * 100] is None:
    #             continue
    #         transformed_image = transformer.preprocess('data', Frame[j + i * 100])
    #         Trans.append(transformed_image)
    #     # if the no. of images left < 100
    #     if len(Trans) < 100:
    #         break
    #     net.blobs['data'].data[...] = np.array(Trans[:])
    #     output = net.forward()
    #     FrameV.extend(output['fc8'].tolist())
    #     i = i + 1
    # net.blobs['data'].reshape(len(Trans),
    #                           3,
    #                           227, 227)
    # net.blobs['data'].data[...] = np.array(Trans[:])
    # output = net.forward()
    # FrameV.extend(output['fc8'].tolist())
    #
    # # write the features to .csv file
    # with open('/data/Meisa/hybridCNN/out.csv', 'wb') as csvfile:
    #     csvwriter = csv.writer(csvfile, delimiter=',')
    #     for item in FrameV:
    #         csvwriter.writerow(item)

    #Read the features from .csv file
    with open('/data/Meisa/hybridCNN/out.csv', 'r') as csvfile:
        Framereader = []
        rows = csv.reader((csvfile))
        for row in rows:
            Framereader.append(row)

    FrameR = []
    for i in Framereader:
        FrameR.append([float(j) for j in i])


    # # Spectral Clustering
    # from sklearn.cluster import SpectralClustering
    #
    # Y = []
    # Maybe = []
    # for i in range(len(FrameR) - 100):
    #     y_pred = SpectralClustering(n_clusters = 2, gamma = 0.1).fit_predict(FrameR[i:i+100])
    #     Y.append(y_pred)
    # for i in range(len(Y)):
    #     initial = Y[i][0]
    #     diff = -1
    #     diffI = -1
    #     for j in range(100):
    #         if Y[i][j] != initial and diff == -1:
    #             diff = Y[i][j]
    #             diffI = j - 1
    #         if (Y[i][j] != diff) and diff != -1:
    #             break
    #         if j == 99:
    #             Maybe.append([i+1+diffI, i+2+diffI])

    # Kmeans Clustering
    # from cluster import KMeansClustering
    from sklearn.cluster import KMeans
    Y = []
    Maybe = []
    for i in range(len(FrameR) - 100):
        y_pred_100 = KMeans(n_clusters=2).fit_predict(FrameR[i:i+100])
        Y.append(y_pred_100)

    for i in range(len(Y)):
        initial = Y[i][0]
        diff = -1
        diffI = -1
        for j in range(100):
            if Y[i][j] != initial and diff == -1:
                diff = Y[i][j]
                diffI = j - 1
            if (Y[i][j] != diff) and diff != -1:
                break
            if j == 99:
                Maybe.append([i+1+diffI, i+2+diffI])






    Clustering = Clus(FrameR)
    a = Clustering.CalculateDiff() # The Mnw of all frames ( from frame1 to frameN) N = the number of all frames -1
    D = Clustering.CalculateDiff2()

    #print test1.difference(a,b)

    # with open('/data/Meisa/hybridCNN/gt_1.txt', 'r') as f:
    #     groundtruth = f.readlines()
    #
    # for i in groundtruth:
    #     if (int(i.strip().split('\t')[-1])+3) < len(a):
    #         print 'i = ', i, '\n',a[int(i.strip().split('\t')[-1])-4 : int(i.strip().split('\t')[-1])+2 ]
    #     else:
    #         print 'i = ', i, '\n', a[int(i.strip().split('\t')[-1]) - 4: int(i.strip().split('\t')[-1])]
    import matplotlib.pyplot as plt

    x = range(len(a))

    plt.figure()
    plt.plot(x, a)

    plt.show()

    FrameI = [] # The number of frames whose Mnw > 0.5
    FrameD = [] # The Mnw of every frame whose Mnw > 0.5
    FrameM = [1] # It is used to merge consecutive frame
    for i in range(len(a)):
        if a[i] > 0.5:
            #print 'i is', i, 'a[i] is', a[i]
            FrameI.append(i)
            FrameD.append(a[i])
    for i in range(1,len(FrameI)):
        if FrameI[i] - FrameI[i-1] == 1:
            FrameM[-1] = FrameM[-1] + 1
        else:
            FrameM.append(1)

    ii = 0
    P = []
    for i in FrameM:
        iter = [ii + j for j in range(i)]
        P.append(max([FrameD[k] for k in iter]) - min(a[FrameI[iter[0]] - 1], a[FrameI[iter[-1]] + 1]))
        ii = ii + i
    for i in range(len(P)):
        if P[i] > 0.5:
            print 'Frame number is', [FrameI[sum(FrameM[:i]) + j] for j in range(FrameM[i])],', P is', P[i]
            for k in range(FrameM[i]):
                if a[FrameI[sum(FrameM[:i]) + k]] - a[FrameI[sum(FrameM[:i]) + k] - 1] > 0.5:
                    print 'i is', FrameI[sum(FrameM[:i]) + k]

    for i in range(1,len(a)):
        if a[i] - a[i-1] > 0.5:
            print 'i is', i, 'a[i] is', a[i], '; a[i-1] is', a[i-1]



    print 'a'

