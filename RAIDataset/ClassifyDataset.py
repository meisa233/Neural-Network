# -*- coding: utf-8 -*-
import os
import numpy as np
import string
# 实现文件复制
import shutil

class ClassifyImage():
    """
    将数据集按照标签的类别分别复制到相应的文件夹里
    """

    def Generate(self):
        # 数据集(图片)的路径
        os.chdir('C:\\caffe\\'+'caffe\\data\\flickr_style\\NewFlickr_Style\\newimages')
        data = []
        Image = []
        ImageType = []
        with open('../newtrain2.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                Image.append(line.split(' ')[0])
                ImageType.append(line.split(' ')[1])
        Num = list(set(ImageType))
        for i in Num:
            os.makedirs('./'+i)#注意，如果已存在文件夹会报错
        for i in range(len(Image)):
            # shutil.copy(source, destination)
            shutil.copyfile(Image[i], './' + ImageType[i] + '/'+ Image[i].split('\\')[-1])    
            #labels_file = f.read()
        #labels = np.loadtxt('./flickr_style/train.txt', str, delimiter = '\t')
        #print labels

if __name__ == "__main__":
    test1 = ClassifyImage().Generate()
