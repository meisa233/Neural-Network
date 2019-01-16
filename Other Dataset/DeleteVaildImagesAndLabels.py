# 删除数据集中下载失败的图片（以及在标签txt中的数据条目）

import os

os.chdir('C:\\caffe\\caffe\\data\\flickr_style\\newimages')
with open('../newtrain.txt', 'r') as f:
    train = f.readlines()
with open('../newtest.txt', 'r') as f:
    test = f.readlines()
#for dirpath, dirnames, filenames in os.walk('.\\'):
#    for filepath in filenames:
#        if os.path.getsize(os.path.join(dirpath, filepath)) / 1024. <= 10.0:
#            #print os.path.join(dirpath, filepath), os.path.getsize(os.path.join(dirpath, filepath)) / 1024.
#            print train.index(filepath)
#            print test.index(filepath)

#print train
# 删除训练集中的失效图片
for i in train:
    # 获取图片的文件大小, os.path.getsize获得的应该是B（文件大小的单位），除以1024.（注意这里是浮点数运算）之后，得到的是KB，在该数据集中
    # 小于10 KB的一般都不是被下载成功的图片
    if os.path.getsize(i.split()[0]) / 1024. <= 10.0:
        os.remove(i.split()[0])
        #print os.path.getsize(i.split()[0]) / 1024
        print i
    else:
    # 合法的图片将会写入一个新的标签文件
        with open('../newtrain2.txt', 'a') as f:
            f.write(i)
# 删除测试集中的失效图片
for i in test:
    if os.path.getsize(i.split()[0]) / 1024. <= 10.0:
        os.remove(i.split()[0])
        #print os.path.getsize(i.split()[0]) / 1024
        print i
    else:
        with open('../newtest2.txt', 'a') as f:
            f.write(i)
