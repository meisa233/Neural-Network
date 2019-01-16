#-*- coding: UTF-8 -*-
import numpy as np
import os
from scipy.special import comb, perm
import cv2


class RAI():

    def __init__(self):
        self.Datapath = 'E:\\Meisa_SiameseNetwork\\RAIDataset'

    #The number of shot or scene in videos (shots per video)
    #Output (if shot):[113.  61. 105. 147.  81.  55. 110. 197.  62.  64.]
    #Output (if scenes):[ 7. 12. 16. 22. 13.  5.  9. 12. 14. 16.]
    #当option为shot时，计算的是每个视频的shot的个数
    #当option为scene时，计算的是每个视频的scene的个数
    def EveryVideoShotAndScenesNumbers(self, option):
        # 10 is the no. of videos
        a = np.zeros(10)
        b = ''
        os.chdir(self.Datapath)
        if option == 'shot':
            b = 'gt'
        else:
            b = 'scenes'
        for i in range(1, 11):
            with open('%s_%s.txt' % (b, i), 'r') as f:
                a[i-1] = len(f.readlines())
        return a

    #The number of shots in every scene
    #计算的是每个场景的shot的数目
    def TheNumOfShotsInEveryScene(self, Scene):

        os.chdir(self.Datapath)
        # index: The index of Scene
        # the content of it: the shots
        h = [0]
        with open('scenes_%s.txt' % Scene, 'r') as f:
            AllScenes = f.readlines()
        with open('gt_%s.txt' % Scene, 'r') as f:
            AllShots = f.readlines()

        for i in range(len(AllScenes)):
            AllScenes[i] = AllScenes[i].strip('\n')
        for i in range(len(AllShots)):
            AllShots[i] = AllShots[i].strip('\n')
        for i in range(len(AllScenes)):
            #*Notice*:The split in shots(.txt) should be '\t'
            h.append(0)
            for j in AllShots:
                if int(j.split('\t')[1]) <= int(AllScenes[i].split(' ')[1]):
                    h[i] = h[i]+1
                else:
                    break
        #The last element is 0 which doesn't make any sense
        del h[-1]
        #The range is from the last index in list h to the second index in list h
        for i in range(len(h)-1, 0, -1):
            h[i] = h[i] - h[i-1]

        return h

    #实现排列组合
    def ThePermAndComb(self, h):
        return comb(h, 2)

    def GenerateNegativepairs(self):
        import cv2
        import random
        #图像数据集的存储地址
        save_path = 'E:\\Meisa_SiameseNetwork\\RAIDataset\\images'
        os.chdir(self.Datapath)
        #视频数目
        VideoNum = 10
        #存储全部的shot数目
        AllShot = self.EveryVideoShotAndScenesNumbers('shot')
        #存储全部的scenes数目
        AllScenes = self.EveryVideoShotAndScenesNumbers('scenes')
        V = []
        V2 = []
        for i in range(VideoNum):
            #V内装的是每个视频里面的每个场景里含多少个shot
            V.append(self.TheNumOfShotsInEveryScene(i+1))
            #V2内装的是每个视频里的每个场景的里的shot编号（第x个shot）
            V2.append(self.TheNumOfShotsInEveryScene(i+1))
            #对V2进行修改方便后面的计算
            for j in range(1,len(V2[i])):
                V2[i][j] = V2[i][j] + V2[i][j-1]
            V2[i].insert(0, 0)


        Shots = []
        #读取每个视频的shot
        for i in range(VideoNum):
            with open('gt_%s.txt' % str(i+1), 'r') as f:
                Shots.append(f.readlines())
        
        #ii代表的是第ii个视频
        ii = 0

        for i in V:
            #选取第i个视频,V内装的是每个视频的场景的shot数目
            #例
            #V[0] = i = [5, 13, 13, 3, 5, 34, 40]
            #V2内装的是每个视频的场景的shot编号
            #第一个场景从0到4，第二个场景从5到17，第三个场景从18到30，...
            #V2[0] = [0, 5, 18, 31, 34, 39, 73, 113]
            i_Video = cv2.VideoCapture('E:\\Meisa_SiameseNetwork\\RAIDataset\\videos\\'+'%s.mp4' % str(ii+1))
            #关于VideoCapture的用法：https://blog.csdn.net/u013539952/article/details/79349098
            #关于如何获取特定帧的方法：https://cloud.tencent.com/developer/ask/133766/answer/237878
            for j in range(len(i)):#选取第j个场景
                temp = np.arange(V2[ii][j], V2[ii][j+1])
                for k in range(V2[ii][j], V2[ii][j+1]):#从第k个镜头开始，与其他镜头组成负样本
                    temp = np.delete(temp, 0)#删除第一个索引上的数据
                    for p in temp:#k是该场景的第k个镜头（从该场景的第一个镜头开始到最后一个镜头）
                        start_1 = int(Shots[ii][k].strip().split('\t')[0])
                        end_1 = int(Shots[ii][k].strip().split('\t')[1])
                        if start_1 > end_1:#去除标注中可能有错误的部分，即起始帧＞末尾帧，不再进行生成样本的操作
                            print 'video_',str(ii), 'shot_', str(k)
                            continue
                        #取中间帧的方法
                        #frame1_number = (int(start_1) + int(end_1)) / 2 + random.randint(-3, 3)、
                        #取镜头中的任意一个帧的方法
                        frame1_number = random.randint(start_1, end_1)
                        i_Video.set(1, frame1_number)#定位到该帧的位置，取出图片
                        ret1, frame_1 = i_Video.read()

                        start_2 = int(Shots[ii][p].strip().split('\t')[0])
                        end_2 = int(Shots[ii][p].strip().split('\t')[1])
                        #frame2_number = (int(start_2) + int(end_2)) / 2  + random.randint(-3, 3)
                        if start_2 > end_2:
                            print 'video_',str(ii), 'shot_', str(p)
                            continue
                        frame2_number = random.randint(start_2, end_2)
                        i_Video.set(1, frame2_number)
                        ret2, frame_2 =i_Video.read()

                        assert (temp[-1] - k) >= 0 and (temp[-1] - k < i[j])#注意这个括号是必须要有的

                        cv2.imwrite(save_path + '\\video_%s_frame_%s.jpg' % (str(ii), str(frame1_number)), frame_1)
                        cv2.imwrite(save_path + '\\video_%s_frame_%s.jpg' % (str(ii), str(frame2_number)), frame_2)
                        with open('./txtFiles/trainleft.txt','a') as f:
                            f.write(save_path + '\\video_%s_frame_%s.jpg 0\n'% (str(ii), str(frame1_number)))
                        with open('./txtFiles/trainright.txt','a') as f:
                            f.write(save_path + '\\video_%s_frame_%s.jpg 0\n'% (str(ii), str(frame2_number)))
            ii = ii + 1
    
    #生成正样本
    def GeneratePositivePairs(self):
        import cv2
        import random
        os.chdir(self.Datapath)
        save_path = 'E:\\Meisa_SiameseNetwork\\RAIDataset\\images2'
        Shots = []
        for i in range(10):
            i_Video = cv2.VideoCapture('E:\\Meisa_SiameseNetwork\\RAIDataset\\videos\\'+'%s.mp4' % str(i+1))
            with open('gt_%s.txt' % str(i+1), 'r') as f:
                #读取所有shot
                Shots.append(f.readlines())
                for j in Shots[i]:
                    start = int(j.strip().split('\t')[0])
                    end = int(j.strip().split('\t')[1])
                    #如果读取到错误的标注，不再进行生成正样本的操作
                    if start >= end:
                        continue
                    for k in range((end - start) / 20):
                        frame1_number = random.randint(start, end)
                        frame2_number = random.randint(start, end)
                        while frame1_number == frame2_number:
                            frame2_number = random.randint(start, end)
                        
                        assert (frame1_number <= end) and (frame1_number >= start)
                        assert (frame2_number <= end) and (frame2_number >= start)
                        i_Video.set(1, frame1_number)
                        ret1, frame_1 = i_Video.read()
                        i_Video.set(1, frame2_number)
                        ret1, frame_2 = i_Video.read()
                        cv2.imwrite(save_path + '\\video_%s_frame_%s.jpg' % (str(i), str(frame1_number)), frame_1)
                        cv2.imwrite(save_path + '\\video_%s_frame_%s.jpg' % (str(i), str(frame2_number)), frame_2)
                        with open('./txtFiles/trainleft2.txt','a') as f:
                            f.write(save_path + '\\video_%s_frame_%s.jpg 1\n'% (str(i), str(frame1_number)))
                        with open('./txtFiles/trainright2.txt','a') as f:
                            f.write(save_path + '\\video_%s_frame_%s.jpg 1\n'% (str(i), str(frame2_number)))
    #检测错误的镜头
    def DetectError(self):
        os.chdir(self.Datapath)
        Shots = []
        for i in range(10):
            with open('gt_%s.txt' % str(i+1), 'r') as f:
                Shots.append(f.readlines())
                for j in Shots[i]:
                    start = j.strip().split('\t')[0]
                    end = j.strip().split('\t')[1]
                    if int(start) > int(end):
                        print 'video_', str(i), 'shot_', str(j)

    #打乱样本
    def ShuffleExamples(self):
        import random
        os.chdir(self.Datapath)
        NegativeExamplesLeft = []
        NegativeExamplesRight = []
        PositiveExamplesLeft = []
        PositiveExamplesRight = []
        with open('./txtFiles/trainleft.txt', 'r') as f:
            NegativeExamplesLeft = f.readlines()
        with open('./txtFiles/trainright.txt', 'r') as f:
            NegativeExampleRight = f.readlines()
        with open('./txtFiles/trainleft2.txt', 'r') as f:
            PositiveExamplesLeft = f.readlines()
        with open('./txtFiles/trainright2.txt', 'r') as f:
            PositiveExamplesRight = f.readlines()
        ExamplesAllLeft = []
        ExamplesAllRight = []
        ExamplesAllLeft.extend(NegativeExamplesLeft)
        ExamplesAllLeft.extend(PositiveExamplesLeft)
        ExamplesAllRight.extend(NegativeExampleRight)
        ExamplesAllRight.extend(PositiveExamplesRight)
        #以相同的次序打乱两个列表，来源：https://blog.csdn.net/yideqianfenzhiyi/article/details/79197570
        ExamplesAll = list(zip(ExamplesAllLeft, ExamplesAllRight))
        np.random.shuffle(ExamplesAll)
        ExamplesAllLeft[:], ExamplesAllRight[:] = zip(*ExamplesAll)#[*参数]会将这些传入参数转化成一个元组
        
        TrainShuffleLeft = []
        TrainShuffleRight =[]
        ValShuffleLeft = []
        ValShuffleRight = []
        for i in range(len(ExamplesAllLeft)):
            if(random.random() > 0.3):
                TrainShuffleLeft.append(ExamplesAllLeft[i])
                TrainShuffleRight.append(ExamplesAllRight[i])
            else:
                ValShuffleLeft.append(ExamplesAllLeft[i])
                ValShuffleRight.append(ExamplesAllRight[i])
        with open('./txtFiles/TrainShuffleLeft.txt', 'a') as f:
            for i in TrainShuffleLeft:
                f.write(i)
        with open('./txtFiles/TrainShuffleRight.txt', 'a') as f:
            for i in TrainShuffleRight:
                f.write(i)
        with open('./txtFiles/ValShuffleLeft.txt', 'a') as f:
            for i in ValShuffleLeft:
                f.write(i)
        with open('./txtFiles/ValShuffleRight.txt', 'a') as f:
            for i in ValShuffleRight:
                f.write(i)


    #增强数据
    def DataAugmentation(self):
        os.chdir(self.Datapath)
        from glob import glob
        from PIL import Image
        import cv2
        #from imgaug import augmenters as iaa
        #from PIL import Image

        #seq = iaa.Sequential([
        #    iaa.Fliplr(0.5)
        #    ])
        #ExamplesLeft = []
        #ExamplesRight = []
        #with open('./txtFiles/ShffuleLeft.txt', 'r') as f:
        #    ExamplesLeft = f.readlines()
        #with open('./txtFiles/ShffuleRight.txt','r') as f:
        #    ExamplesRight = f.readlines()
        #for i in ExamplesLeft:
        #    img = Image.open(i.strip().split(' ')[0])
        #    img = np.array(img)
        #    images_aug = seq.augment_images(img)

        #读取文件夹下所有.jpg文件
        NegativeImage = glob('E:\\Meisa_SiameseNetwork\\RAIDataset\\images2\\*.jpg')
        for i in NegativeImage:
            img = cv2.imread(i)
            FlipVerImg = cv2.flip(img, -1)#0表示纵向翻转（Vertical），1表示横向翻转，-1表示横向纵向同时反转
            cv2.imwrite('E:\\Meisa_SiameseNetwork\\RAIDataset\\VerAndHor2\\%s' % i.split('\\')[-1], FlipVerImg)
        #print NegativeImage

    def ResizeImage(self):
        import cv2
        from glob import glob
        os.chdir(self.Datapath)
        NegativeImage = glob('E:\\Meisa_SiameseNetwork\\RAIDataset\\TestImage\\*.jpg')
        ResizeImage = []
        for i in NegativeImage:
            img = cv2.imread(i)
            ResizeImg = cv2.resize(img, (227, 227), cv2.INTER_AREA)
            cv2.imwrite('E:\\Meisa_SiameseNetwork\\RAIDataset\\TestImage_resize\\%s' % i.split('\\')[-1], ResizeImg)

    def ImgaugDataAugmentation(self):
        from glob import glob
        import cv2
        from imgaug import augmenters as iaa
        import numpy as np
        import os

        os.chdir(self.Datapath)

        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Crop(px=(0,16))
            #iaa.ContrastNormalization((0.5,1.5))
            ])
             
        NegativeImage = glob('E:\\Meisa_SiameseNetwork\\RAIDataset\\TestImage_resize\\*.jpg')
        AllImgs = []
        for i in NegativeImage:
            img = cv2.imread(i)
            AllImgs.append(img)
        NdarrayAllImgs = np.array(AllImgs)
        AllImgs_aug = seq.augment_images(NdarrayAllImgs)
        #AllImgs_aug_list = AllImgs_aug.tolist()
        for i in range(10):
            os.mkdir('./TestImage_aug/%s' % str(i))
            for index in range(AllImgs_aug.shape[0]):
                cv2.imwrite('E:\\Meisa_SiameseNetwork\\RAIDataset\\TestImage_aug\\%s\\%s' % (str(i),NegativeImage[index].split('\\')[-1]), AllImgs_aug[index])
        print NdarrayAllImgs.shape

if __name__ == '__main__':
    test1 = RAI()

    #Count the number of shots and scenes
    #计算shots的数目和scenes
    #test1.EveryVideoShotAndScenesNumbers('scenes')

    #Count the number of shots in scene (argument)
    #计算每个场景中有几个shots
    #print test1.TheNumOfShotsInEveryScene(1)

    #打印第一个电影的可得到的负(negative)样本数（按场景分）
    #print test1.ThePermAndComb(test1.TheNumOfShotsInEveryScene(1))
    #打印第一个电影可得到的负样本总数
    #print sum(test1.ThePermAndComb(test1.TheNumOfShotsInEveryScene(1)))

    #得到每个视频里面的每个场景的shot
    '''
    b = []
    for i in range(1,11):
        b.append(test1.TheNumOfShotsInEveryScene(i))
        assert sum(b[i-1]) == test1.EveryVideoShotAndScenesNumbers('shot')[i-1]
    '''
    #print b

    '''
    c = []
    d = []
    for i in range(10):
        c.append(test1.ThePermAndComb(b[i]))
        d.append(sum(c[i]))
        print 'The %s'% i, 'video is', c[i], '\n and the sum is', sum(c[i])
    print 'All videos\' examples is ', np.sum(d)
    '''
#    videoCapture = cv2.
    test1.GenerateNegativepairs()
    #test1.DetectError()
    #test1.GeneratePositivePairs()
    #test1.ShuffleExamples()
    #test1.DataAugmentation()
    #test1.ImgaugDataAugmentation()
    #test1.ResizeImage()
