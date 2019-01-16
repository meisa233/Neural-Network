class RAI():

    Imagesuffix = '.jpg'
    DatasetPath = '/data/RAIDataset'

    def ResizeDataset(self):
        import cv2
        from glob import glob

        NegativeImage = glob(self.DatasetPath + '/images/*' + self.Imagesuffix)
        PositiveImage = glob(self.DatasetPath + '/images2/*' + self.Imagesuffix)

        ResizeN1 = []
        ResizeN2 = []

        for i in NegativeImage:
            img = cv2.imread(i)
            #Resize the dataset to 227 227 or 224 224
            ResizeImg = cv2.resize(img, (224, 224), cv2.INTER_AREA)
            cv2.imwrite(self.DatasetPath + '/ResNetRimages/' + i.split('/')[-1], ResizeImg)

        for i in PositiveImage:
            img = cv2.imread(i)
            ResizeImg = cv2.resize(img, (224, 224), cv2.INTER_AREA)
            cv2.imwrite(self.DatasetPath + '/ResNetRimages2/' + i.split('/')[-1], ResizeImg)

    # Detect the error file ( size < 1kb )
    def DetectError(self):
        import os
        from glob import glob
        NegativeImage = glob(self.DatasetPath + '/Rimages/*' + self.Imagesuffix)
        PositiveImage = glob(self.DatasetPath + '/Rimages2/*' + self.Imagesuffix)
        for i in NegativeImage:
            if os.path.getsize(i) < 1024:
                print i
        print 'NegativeImage No. is', len(NegativeImage)

        for i in PositiveImage:
            if os.path.getsize(i) < 1024:
                print i
        print 'PositiveImage No. is', len(PositiveImage)

    # DataAugmentation
    def ImgaugDataAugmentation(self):
        from glob import glob
        import cv2
        import numpy as np
        from imgaug import augmenters as iaa
        import os

        seq = iaa.Sequential([
            #iaa.Fliplr(1),
            #iaa.Flipud(1),
            iaa.Crop(px=(0,16))
            #iaa.ContrastNormalization((0.5,1.5))
            ])

        NegativeImage = glob(self.DatasetPath + '/ResNetRimages/*' + self.Imagesuffix)
        PositiveImage = glob(self.DatasetPath + '/ResNetRimages2/*' + self.Imagesuffix)

        NegativeImageFiles = []
        for i in NegativeImage:
            img = cv2.imread(i)
            NegativeImageFiles.append(img)

        PositiveImageFiles = []
        for i in PositiveImage:
            img = cv2.imread(i)
            PositiveImageFiles.append(img)

        NdarrayN = np.array(NegativeImageFiles)
        NdarrayP = np.array(PositiveImageFiles)


        #for i in range(20):
        NdarrayN_aug = seq.augment_images(NdarrayN)
        NdarrayP_aug = seq.augment_images(NdarrayP)

        os.mkdir('/data/RAIDataset/ResNetDataset/CropImage')
        for i in range(NdarrayN_aug.shape[0]):
            cv2.imwrite('/data/RAIDataset/ResNetDataset/CropImage/' + NegativeImage[i].split('/')[-1], NdarrayN_aug[i])

        os.mkdir('/data/RAIDataset/ResNetDataset/CropImage_2')
        for i in range(NdarrayP_aug.shape[0]):
            cv2.imwrite('/data/RAIDataset/ResNetDataset/CropImage_2/' + PositiveImage[i].split('/')[-1], NdarrayP_aug[i])

    # Shuffle All Images(having been croped)
    def ShuffExamples2(self):
        import os
        import numpy as np
#        import random
#        import copy

        os.chdir('/data/RAIDataset/OriginalLabels')
#        trainNega = []
#        trainNega_r =[]
#        trainPosi = []
#        trainPosi_r = []

        with open('./CroptrainNega.txt', 'r') as f:
            CroptrainNega = f.readlines()
        with open('./CroptrainNega_r.txt', 'r') as f:
            CroptrainNega_r = f.readlines()
        with open('./CroptrainPosi.txt', 'r') as f:
            CroptrainPosi = f.readlines()
        with open('./CroptrainPosi_r.txt', 'r') as f:
            CroptrainPosi_r = f.readlines()

#        H_trainNega = copy.deepcopy(trainNega)
#        H_trainNega_r = copy.deepcopy(trainNega_r)
#        H_trainPosi = copy.deepcopy(trainPosi)
#        H_trainPosi_r = copy.deepcopy(trainPosi_r)
        with open('./H_CroptrainNega.txt', 'r') as f:
            H_CroptrainNega = f.readlines()
        with open('./H_CroptrainNega_r.txt', 'r') as f:
            H_CroptrainNega_r = f.readlines()
        with open('./H_CroptrainPosi.txt', 'r') as f:
            H_CroptrainPosi = f.readlines()
        with open('./H_CroptrainPosi_r.txt', 'r') as f:
            H_CroptrainPosi_r = f.readlines()

        with open('./V_CroptrainNega.txt', 'r') as f:
            V_CroptrainNega = f.readlines()
        with open('./V_CroptrainNega_r.txt', 'r') as f:
            V_CroptrainNega_r = f.readlines()
        with open('./V_CroptrainPosi.txt', 'r') as f:
            V_CroptrainPosi = f.readlines()
        with open('./V_CroptrainPosi_r.txt', 'r') as f:
            V_CroptrainPosi_r = f.readlines()

        with open('./H_V_CroptrainNega.txt', 'r') as f:
            H_V_CroptrainNega = f.readlines()
        with open('./H_V_CroptrainNega_r.txt', 'r') as f:
            H_V_CroptrainNega_r = f.readlines()
        with open('./H_V_CroptrainPosi.txt', 'r') as f:
            H_V_CroptrainPosi = f.readlines()
        with open('./H_V_CroptrainPosi_r.txt', 'r') as f:
            H_V_CroptrainPosi_r = f.readlines()

        Left = []
        Right = []
        ExamplesAll = []

        Left.extend(CroptrainNega)
        Left.extend(CroptrainPosi)
        Left.extend(H_CroptrainNega)
        Left.extend(H_CroptrainPosi)
        Left.extend(V_CroptrainNega)
        Left.extend(V_CroptrainPosi)
        Left.extend(H_V_CroptrainNega)
        Left.extend(H_V_CroptrainPosi)

        Right.extend(CroptrainNega_r)
        Right.extend(CroptrainPosi_r)
        Right.extend(H_CroptrainNega_r)
        Right.extend(H_CroptrainPosi_r)
        Right.extend(V_CroptrainNega_r)
        Right.extend(V_CroptrainPosi_r)
        Right.extend(H_V_CroptrainNega_r)
        Right.extend(H_V_CroptrainPosi_r)

        ExamplesAll = list(zip(Left, Right))
        np.random.shuffle(ExamplesAll)
        Left[:], Right[:] = zip(*ExamplesAll)

        with open('../left2.txt', 'a') as f:
            for i in Left:
                f.write(i)

        with open('../right2.txt', 'a') as f:
            for i in Right:
                f.write(i)

    # Shuffle All Images (having not been croped)
    def ShuffExamples(self):
        import os
        import numpy as np
        #        import random
        #        import copy

        os.chdir('/data/RAIDataset/OriginalLabels')
        #        trainNega = []
        #        trainNega_r =[]
        #        trainPosi = []
        #        trainPosi_r = []

        with open('./trainNega.txt', 'r') as f:
            trainNega = f.readlines()
        with open('./trainNega_r.txt', 'r') as f:
            trainNega_r = f.readlines()
        with open('./trainPosi.txt', 'r') as f:
            trainPosi = f.readlines()
        with open('./trainPosi_r.txt', 'r') as f:
            trainPosi_r = f.readlines()

        #        H_trainNega = copy.deepcopy(trainNega)
        #        H_trainNega_r = copy.deepcopy(trainNega_r)
        #        H_trainPosi = copy.deepcopy(trainPosi)
        #        H_trainPosi_r = copy.deepcopy(trainPosi_r)
        with open('./H_trainNega.txt', 'r') as f:
            H_trainNega = f.readlines()
        with open('./H_trainNega_r.txt', 'r') as f:
            H_trainNega_r = f.readlines()
        with open('./H_trainPosi.txt', 'r') as f:
            H_trainPosi = f.readlines()
        with open('./H_trainPosi_r.txt', 'r') as f:
            H_trainPosi_r = f.readlines()

        with open('./V_trainNega.txt', 'r') as f:
            V_trainNega = f.readlines()
        with open('./V_trainNega_r.txt', 'r') as f:
            V_trainNega_r = f.readlines()
        with open('./V_trainPosi.txt', 'r') as f:
            V_trainPosi = f.readlines()
        with open('./V_trainPosi_r.txt', 'r') as f:
            V_trainPosi_r = f.readlines()

        with open('./H_V_trainNega.txt', 'r') as f:
            H_V_trainNega = f.readlines()
        with open('./H_V_trainNega_r.txt', 'r') as f:
            H_V_trainNega_r = f.readlines()
        with open('./H_V_trainPosi.txt', 'r') as f:
            H_V_trainPosi = f.readlines()
        with open('./H_V_trainPosi_r.txt', 'r') as f:
            H_V_trainPosi_r = f.readlines()

        Left = []
        Right = []
        ExamplesAll = []

        Left.extend(trainNega)
        Left.extend(trainPosi)
        Left.extend(H_trainNega)
        Left.extend(H_trainPosi)
        Left.extend(V_trainNega)
        Left.extend(V_trainPosi)
        Left.extend(H_V_trainNega)
        Left.extend(H_V_trainPosi)

        Right.extend(trainNega_r)
        Right.extend(trainPosi_r)
        Right.extend(H_trainNega_r)
        Right.extend(H_trainPosi_r)
        Right.extend(V_trainNega_r)
        Right.extend(V_trainPosi_r)
        Right.extend(H_V_trainNega_r)
        Right.extend(H_V_trainPosi_r)

        ExamplesAll = list(zip(Left, Right))
        np.random.shuffle(ExamplesAll)
        Left[:], Right[:] = zip(*ExamplesAll)

        with open('../left.txt', 'a') as f:
            for i in Left:
                f.write(i)

        with open('../right.txt', 'a') as f:
            for i in Right:
                f.write(i)

    # Check the folder path between left.txt and right.txt (whether is same)
    def CheckLabels(self):
        left = []
        right = []
        with open('/data/RAIDataset/left2.txt', 'r') as f:
            left = f.readlines()

        with open('/data/RAIDataset/right2.txt', 'r') as f:
            right = f.readlines()

        for i in range(len(left)):
            #print left[i].split('/')[-2::-1]
            if cmp(left[i].split('/')[-2::-1], right[i].split('/')[-2::-1]) != 0:
                print left[i],'\n'
                print right[i], '\n'

    def MergeAllExamples(self):
        import  random

        AllLeft = []
        AllRight = []
        with open('/data/RAIDataset/left.txt', 'r') as f:
            left = f.readlines()

        with open('/data/RAIDataset/left2.txt', 'r') as f:
            left2 = f.readlines()

        with open('/data/RAIDataset/right.txt', 'r') as f:
            right = f.readlines()

        with open('/data/RAIDataset/right2.txt', 'r') as f:
            right2 = f.readlines()

        AllLeft.extend(left)
        AllLeft.extend(left2)

        AllRight.extend(right)
        AllRight.extend(right2)


        TrainLeft = []
        TrainRight = []

        ValLeft = []
        ValRight =[]
        # Group the data into val and train data
        for i in range(len(AllLeft)):
            if random.random() > 0.15:
                TrainLeft.append(AllLeft[i])
                TrainRight.append(AllRight[i])
            else:
                ValLeft.append(AllLeft[i])
                ValRight.append(AllRight[i])

        with open('/data/RAIDataset/NewLabels/TrainLeft3.txt', 'w') as f:
            f.writelines(TrainLeft)
        with open('/data/RAIDataset/NewLabels/TrainRight3.txt', 'w') as f:
            f.writelines(TrainRight)
        with open('/data/RAIDataset/NewLabels/ValLeft3.txt', 'w') as f:
            f.writelines(ValLeft)
        with open('/data/RAIDataset/NewLabels/ValRight3.txt', 'w') as f:
            f.writelines(ValRight)

    def CheckNewLabels(self):
        TrainLeft  = []
        TrainRight = []
        ValLeft = []
        ValRight = []
        with open('/data/RAIDataset/NewLabels/TrainLeft.txt', 'r') as f:
            TrainLeft = f.readlines()

        with open('/data/RAIDataset/NewLabels/TrainRight.txt', 'r') as f:
            TrainRight = f.readlines()

        with open('/data/RAIDataset/NewLabels/ValLeft.txt', 'r') as f:
            ValLeft = f.readlines()

        with open('/data/RAIDataset/NewLabels/ValRight.txt', 'r') as f:
            ValRight = f.readlines()

        assert len(TrainLeft) == len(TrainRight)
        assert len(ValLeft) == len(ValRight)

        for i in range(len(TrainLeft)):
            if cmp(TrainLeft[i].split('/')[-2::-1], TrainRight[i].split('/')[-2::-1]) != 0:
                print i

            if cmp(TrainLeft[i].split('/')[-1].split(' ')[0].split('_')[-2::-1], TrainRight[i].split('/')[-1].split(' ')[0].split('_')[-2::-1]) != 0:
                print i

        for i in range(len(ValLeft)):
            if cmp(ValLeft[i].split('/')[-2::-1], ValRight[i].split('/')[-2::-1]) != 0:
                print i

            if cmp(ValLeft[i].split('/')[-1].split(' ')[0].split('_')[-2::-1], ValRight[i].split('/')[-1].split(' ')[0].split('_')[-2::-1]) != 0:
                print i

    def GenerateTrainAndVal(self):
        import random
        import numpy as np
        TrainPosi = []
        TrainPosi_r = []
        TrainNega = []
        TrainNega_r = []
        with open ('/data/RAIDataset/ResNetDataset/Labels/Original/trainPosi.txt', 'r') as f:
            TrainPosi = f.readlines()

        with open ('/data/RAIDataset/ResNetDataset/Labels/Original/trainPosi_r.txt', 'r') as f:
            TrainPosi_r = f.readlines()

        with open('/data/RAIDataset/ResNetDataset/Labels/Original/trainNega.txt', 'r') as f:
            TrainNega = f.readlines()

        with open('/data/RAIDataset/ResNetDataset/Labels/Original/trainNega_r.txt', 'r') as f:
            TrainNega_r = f.readlines()

        ExamplesAllLeft =[]
        ExamplesAllRight = []

        ExamplesAllLeft.extend(TrainPosi)
        ExamplesAllLeft.extend(TrainNega)
        ExamplesAllRight.extend(TrainPosi_r)
        ExamplesAllRight.extend(TrainNega_r)

        ExamplesAll = list(zip(ExamplesAllLeft, ExamplesAllRight))
        np.random.shuffle(ExamplesAll)

        ExamplesAllLeft[:], ExamplesAllRight[:] = zip(*ExamplesAll)

        TrainShuffleLeft = []
        TrainShuffleRight = []
        ValShuffleLeft = []
        ValShuffleRight = []

        for i in range(len(ExamplesAllLeft)):
            if random.random() > 0.15:
                TrainShuffleLeft.append(ExamplesAllLeft[i])
                TrainShuffleRight.append(ExamplesAllRight[i])
            else:
                ValShuffleLeft.append(ExamplesAllLeft[i])
                ValShuffleRight.append(ExamplesAllRight[i])

        with open('/data/RAIDataset/ResNetDataset/Labels/NoDataAugmentation/TrainLeft.txt', 'w') as f:
            f.writelines(TrainShuffleLeft)
        with open('/data/RAIDataset/ResNetDataset/Labels/NoDataAugmentation/TrainRight.txt', 'w') as f:
            f.writelines(TrainShuffleRight)
        with open('/data/RAIDataset/ResNetDataset/Labels/NoDataAugmentation/ValLeft.txt', 'w') as f:
            f.writelines(ValShuffleLeft)
        with open('/data/RAIDataset/ResNetDataset/Labels/NoDataAugmentation/ValRight.txt', 'w') as f:
            f.writelines(ValShuffleRight)

    # Shuffle ResNetTrainExample and ResNetValExamples
    def ShuffExamples3(self):
        import os
        import numpy as np


        os.chdir('/data/RAIDataset/ResNetDataset/Labels')


        with open('./NoDataAugmentation/TrainLeft.txt', 'r') as f:
            TrainLeft = f.readlines()
        with open('./NoDataAugmentation/TrainRight.txt', 'r') as f:
            TrainRight = f.readlines()
        with open('./NoDataAugmentation/ValLeft.txt', 'r') as f:
            ValLeft = f.readlines()
        with open('./NoDataAugmentation/ValRight.txt', 'r') as f:
            ValRight = f.readlines()


        with open('./DataAugmentation/HorTrainLeft.txt', 'r') as f:
            HorTrainLeft = f.readlines()
        with open('./DataAugmentation/HorTrainRight.txt', 'r') as f:
            HorTrainRight = f.readlines()
        with open('./DataAugmentation/HorValLeft.txt', 'r') as f:
            HorValLeft = f.readlines()
        with open('./DataAugmentation/HorValRight.txt', 'r') as f:
            HorValRight = f.readlines()

        with open('./DataAugmentation/VerTrainLeft.txt', 'r') as f:
            VerTrainLeft = f.readlines()
        with open('./DataAugmentation/VerTrainRight.txt', 'r') as f:
            VerTrainRight = f.readlines()
        with open('./DataAugmentation/VerValLeft.txt','r') as f:
            VerValLeft = f.readlines()
        with open('./DataAugmentation/VerValRight.txt', 'r') as f:
            VerValRight = f.readlines()

        with open('./DataAugmentation/VerAndHorTrainLeft.txt', 'r') as f:
            VerAndHorTrainLeft = f.readlines()
        with open('./DataAugmentation/VerAndHorTrainRight.txt', 'r') as f:
            VerAndHorTrainRight = f.readlines()
        with open('./DataAugmentation/VerAndHorValLeft.txt', 'r') as f:
            VerAndHorValLeft = f.readlines()
        with open('./DataAugmentation/VerAndHorValRight.txt', 'r') as f:
            VerAndHorValRight = f.readlines()

        with open('./NoDataAugmentation/CropTrainLeft.txt', 'r') as f:
            CropTrainLeft = f.readlines()
        with open('./NoDataAugmentation/CropTrainRight.txt', 'r') as f:
            CropTrainRight = f.readlines()
        with open('./NoDataAugmentation/CropValLeft.txt', 'r') as f:
            CropValLeft = f.readlines()
        with open('./NoDataAugmentation/CropValRight.txt', 'r') as f:
            CropValRight = f.readlines()


        with open('./DataAugmentation/CropHorTrainLeft.txt', 'r') as f:
            CropHorTrainLeft = f.readlines()
        with open('./DataAugmentation/CropHorTrainRight.txt', 'r') as f:
            CropHorTrainRight = f.readlines()
        with open('./DataAugmentation/CropHorValLeft.txt', 'r') as f:
            CropHorValLeft = f.readlines()
        with open('./DataAugmentation/CropHorValRight.txt', 'r') as f:
            CropHorValRight = f.readlines()

        with open('./DataAugmentation/CropVerTrainLeft.txt', 'r') as f:
            CropVerTrainLeft = f.readlines()
        with open('./DataAugmentation/CropVerTrainRight.txt', 'r') as f:
            CropVerTrainRight = f.readlines()
        with open('./DataAugmentation/CropVerValLeft.txt','r') as f:
            CropVerValLeft = f.readlines()
        with open('./DataAugmentation/CropVerValRight.txt', 'r') as f:
            CropVerValRight = f.readlines()

        with open('./DataAugmentation/CropVerAndHorTrainLeft.txt', 'r') as f:
            CropVerAndHorTrainLeft = f.readlines()
        with open('./DataAugmentation/CropVerAndHorTrainRight.txt', 'r') as f:
            CropVerAndHorTrainRight = f.readlines()
        with open('./DataAugmentation/CropVerAndHorValLeft.txt', 'r') as f:
            CropVerAndHorValLeft = f.readlines()
        with open('./DataAugmentation/CropVerAndHorValRight.txt', 'r') as f:
            CropVerAndHorValRight = f.readlines()

        Left = []
        Right = []
        ExamplesAll = []

        Left.extend(TrainLeft)
        Left.extend(HorTrainLeft)
        Left.extend(VerTrainLeft)
        Left.extend(VerAndHorTrainLeft)
        Left.extend(CropTrainLeft)
        Left.extend(CropHorTrainLeft)
        Left.extend(CropVerTrainLeft)
        Left.extend(CropVerAndHorTrainLeft)

        Right.extend(TrainRight)
        Right.extend(HorTrainRight)
        Right.extend(VerTrainRight)
        Right.extend(VerAndHorTrainRight)
        Right.extend(CropTrainRight)
        Right.extend(CropHorTrainRight)
        Right.extend(CropVerTrainRight)
        Right.extend(CropVerAndHorTrainRight)

        Val_Left.extend(ValLeft)
        Val_Left.extend(HorValLeft)
        Val_Left.extend(VerValLeft)
        Val_Left.extend(VerAndHorValLeft)
        Val_Left.extend(CropValLeft)
        Val_Left.extend(CropHorValLeft)
        Val_Left.extend(CropVerValLeft)
        Val_Left.extend(CropVerAndHorValLeft)

        Val_Right.extend(ValRight)
        Val_Right.extend(HorValRight)
        Val_Right.extend(VerValRight)
        Val_Right.extend(VerAndHorValRight)
        Val_Right.extend(CropValRight)
        Val_Right.extend(CropHorValRight)
        Val_Right.extend(CropVerValRight)
        Val_Right.extend(CropVerAndHorValRight)

        TrainExamplesAll = list(zip(Left, Right))

        np.random.shuffle(TrainExamplesAll)
        Left[:], Right[:] = zip(*TrainExamplesAll)


        ValExamplesAll = list(zip(Left, Right))
        with open('../left.txt', 'a') as f:
            for i in Left:
                f.write(i)

        with open('../right.txt', 'a') as f:
            for i in Right:
                f.write(i)

if __name__ == '__main__':
    test1 = RAI()
    #test1.ResizeDataset()
    #test1.DetectError()
    #test1.ImgaugDataAugmentation()
    #test1.GenerateTrainAndVal()

    #test1.ShuffExamples()
    #test1.CheckLabels()
    #test1.ShuffExamples2()
    #test1.MergeAllExamples()
    #test1.CheckNewLabels()
