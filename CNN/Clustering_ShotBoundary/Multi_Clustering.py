import numpy as np
import csv
import random
class Cluster():

    w = 0.5

    # Calculate the Euclidean distance
    # Input:vecA and vecB are Frames (type is np.array)
    def CalculateEuclid(self, vecA, vecB):
        return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

    # KMeans Clusterring
    def KMeans(self, subFrame):
        lens = len(subFrame[0])
        subFrameA = np.array(subFrame[0]).reshape((1, lens))

        for i in range(1,len(subFrame)):
            numpyarray = np.array(subFrame[i]).reshape((1, lens))
            subFrameA = np.vstack((subFrameA, numpyarray))

        # # Cend is start and end
        # ClusterA = subFrameA[0]
        # ClusterB = subFrameA[-1]
        # Cend is random
        LabelS = random.randint(0, len(subFrame) - 1)
        LabelE = random.randint(0, len(subFrame) - 1)
        while(LabelS==LabelE):
            LabelE = random.randint(0, len(subFrame) - 1)

        Label = np.ones(len(subFrame)) * (-1)

        # # Cend is start and end
        # Label[0] = 0
        # Label[-1] = 1

        # # Random Cend
        ClusterA = subFrameA[LabelS]
        ClusterB = subFrameA[LabelE]

        Label[LabelS] = 0
        Label[LabelE] = 1

        for i in range(0,len(subFrame)):
            if i == LabelS or i == LabelE:
                continue
            first = self.CalculateEuclid(subFrameA[i], ClusterA)
            second = self.CalculateEuclid(subFrameA[i], ClusterB)
            if first < second:
                Label[i] = 0
                ClusterA = np.mean(np.array(subFrameA[Label==0]), axis=0)
            else:
                Label[i] = 1
                ClusterB = np.mean(np.array(subFrameA[Label==1]), axis=0)
        return Label

    # DBScan Clustering
    def DBScan(self, subFrame):
        from sklearn.cluster import DBSCAN
        clst = DBSCAN(eps=0.2)
        return clst.fit_predict(subFrame)

    # Agglomerative Clustering
    def Agglomerative(self, subFrame):
        from sklearn.cluster import AgglomerativeClustering
        clst = AgglomerativeClustering()
        return clst.fit_predict(subFrame)


    # Spectral Clustering
    def Spectral(self, subFrame):
        from sklearn.cluster import SpectralClustering
        clst = SpectralClustering(n_clusters=2)
        return clst.fit_predict(subFrame)


    ###############################################################
    def difference(self, a, b):
        a = np.array(a)
        b = np.array(b)

        a_minus_b = a - b
        return np.sqrt(np.sum(a_minus_b ** 2))

    def Mnw(self, n, w, Frame):
        if n+w == int(n+w):
            if int(n + w) >= len(Frame) or int(n - w) >= len(Frame):
                return -1
            return self.difference(Frame[int(n-w)], Frame[int(n+w)])
        else:
            return (1. / 2.) * (self.Mnw(n - 0.5, w, Frame)) + self.Mnw(n + 0.5, w, Frame)

    #Calculate all frames of the video the difference(from 1 to the end)
    # between (0.5,1.5), (1.5, 2.5), (2.5, 3.5), ....
    def CalculateDiff(self, Frame):
        diff = []
        i = 1
        while i < len(Frame):
            #diff.append(self.Mnw(i, self.w))
            diff.append(self.Mnw(i, self.w, Frame))
            i = i + 1
        return diff
    ##############################################################
    def cosin_distance(self, vector1, vector2):
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return None
        else:
            return dot_product / ((normA * normB) ** 0.5)

    def AllFrames_cosin_distance(self, Frame):
        Alldistance = []
        for i in range(1, len(Frame)):
            Alldistance.append(self.cosin_distance(Frame[i], Frame[i-1]))
        return Alldistance

    def DualThreshold(self, Frame):
        Tc = 0.8
        Tgh = 0.95
        Tgl = 0.9

        for i in range(len(Frame) - 1):

            st1 = self.cosin_distance(Frame[i], Frame[i+1])
            if(i+2 >= len(Frame)):
                break
            st2 = self.cosin_distance(Frame[i], Frame[i+2])
            st_11 = self.cosin_distance(Frame[i-1], Frame[i])

            if st1<Tc and st2<Tc and st_11 > Tgh:

                print [i+1, i+2], '\n'

            elif st1<=Tgh and st1>=Tgl:
                n = i+1
                while self.cosin_distance(Frame[n], Frame[n+1])<=Tgh and self.cosin_distance(Frame[n], Frame[n+1])>= Tgl:
                    n = n+1
                if self.cosin_distance(Frame[i], Frame[n])< Tc and self.cosin_distance(Frame[n], Frame[n+1])>Tgh:
                    print 'gradual transition:', [i, n]



    # The Method from fast video shot transition localization with deep structured models
    def getT(self, Frame, n, a, t, sigma):
        All1_Si = 0
        i = n - a + 1
        while i <= n + a - 1:
            All1_Si = All1_Si + 1 - self.cosin_distance(Frame[n], Frame[i+1])
            i = i + 1
        T = t + (sigma / 1) * (All1_Si)
        return T

    def InitialFiltering(self, Frame):
        t = 0.5
        sigma = 0.05
        a = 1

        n = 0
        i = n + a - 1

        for i in range(len(Frame)):
            if (1 - self.cosin_distance(Frame[i], Frame[i+1])) > self.getT(Frame, i, a, t, sigma):
                print i



if __name__ == '__main__':

    # # Read the features from .csv file
    # with open('/data/Meisa/hybridCNN/out.csv', 'r') as csvfile:
    #     Framereader = []
    #     rows = csv.reader((csvfile))
    #     for row in rows:
    #         Framereader.append(row)

    # trained by siamese network
    # with open('/data/Meisa/hybridCNN/out.csv', 'r') as csvfile:

    # Original weight(only pretrained on ilsvrc12 and place 205)
    # with open('/data/Meisa/hybridCNN/out_2.csv', 'r') as csvfile:

    # pre-trained ResNet 50
    with open ('/data/Meisa/ResNet/ResNet-50/ResNetFeatures/Video2Features.csv', 'r') as csvfile:
        Framereader = []
        rows = csv.reader((csvfile))
        TempFrame = []
        i = 0
        for row in rows:
            Framereader.append(row)

    FrameR = []
    for i in Framereader:
        FrameR.append([float(j.strip('[').strip(']')) for j in i])

    #############################################################

    test1 =Cluster()



    # # #The method from fast video shot transition localilzation with deep structured models
    # test1.InitialFiltering(FrameR)







    # Threshold Method
    D1 = test1.CalculateDiff(FrameR)

    D2 = test1.AllFrames_cosin_distance(FrameR)
    import matplotlib.pyplot as plt

    x1 = range(len(D1))
    x2 = range(len(D2))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plt.plot(x1, D1)
    ax2 = fig.add_subplot(212)
    plt.plot(x2, D2)

    plt.show()

    # test1.DualThreshold(FrameR)
    # load the 2th Video ground truth label
    # with open('/data/RAIDataset/Video/gt_2.txt', 'r') as f:
    #     groundtruth = f.readlines()
    #
    # for i in range(len(groundtruth)):
    #     print 'index is', groundtruth[i],'\n'
    #     if int(groundtruth[i].split('\t')[0]) != 0:
    #         print D2[int(groundtruth[i].split('\t')[0]) - 1], '\n'
    #     print D2[int(groundtruth[i].split('\t')[1]) - 1], '\n'


        # Test the Claster
    # KMeans

    # Y = []
    # Maybe = []
    #
    #
    #
    # for i in range(len(FrameR) - 100):
    #     y_pred_100 = test1.KMeans(FrameR[i:i+100])
    #     Y.append(y_pred_100)
    #
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
    # KmeansResult = []
    # for i in Maybe:
    #     if i not in KmeansResult:
    #         KmeansResult.append(i)
    # a = test1.CalculateDiff(FrameR) # The Mnw of all frames ( from frame1 to frameN) N = the number of all frames -1

    # DBScan
    #a = test1.DBScan(FrameR[1000:1100])




    # # Agglomerative
    # Y = []
    # Maybe = []
    #
    # i = 0
    # while i < (len(FrameR) - 100):
    #     TempResult = test1.Spectral(FrameR[i:i+100])
    #     Order = np.where(TempResult == 0)[0].tolist()
    #     if len(set(range(Order[0], Order[-1])) - set(Order)) == 0:
    #         if Order[0] == 0:
    #             Maybe.append([i+Order[-1] + 1, i+Order[-1]+2])
    #             i = i + Order[0] + 1
    #         else:
    #             Maybe.append([i+Order[0], i+Order[0] + 1])
    #             i = i + Order[-1] + 1
    #     else:
    #         New = list(set(range(Order[0], Order[-1])) - set(Order))
    #         if Order[0]==0:
    #             i = i + New[0] + 1
    #         else:
    #             Order2 = np.where(TempResult == 1)[0].tolist()
    #             New2 = list(set(range(Order2[0], Order2[-1])) - set(Order2))
    #             i = i + New2[0] + 1
    #     i = i + 100


    # List = []
    # for i in Maybe:
    #     if i not in List:
    #         List.append(i)
    #
    # print List



    print 'a'
