if __name__ == '__main__':
    import json
    # 这里存放的是ClipShots数据集的标签，是以.json形式存储的
    TrainJSONpath = 'D:\\ClipShots\\ClipShots\\ClipShots\\annotations\\train.json'
    # 读取标签
    annotations = json.load(open(TrainJSONpath))
    
    # 用于记录影片的数目
    Count = 0
    # 准备写入的文本文档地址
    TheFirst1000VideoNameTXTpath = 'D:\\ClipShots\\TheFirst2000.txt'
    
    # 遍历所有标签（标签形式是【videoname.mp4:[2,3],[4,9]以及framenum:2000】其中videoname.mp4代表是影片的名字，后面代表的是切换，包含了软切换[4,9]
    # 和硬切换[2,3]，framenum是该影片的总帧数
    for videoname, labels in annotations.items():
        # 进行计数
        Count += 1
        # 打开要写入的文件，进行写入
        with open(TheFirst1000VideoNameTXTpath, 'a') as f:
            f.write(str(videoname)+'\n')
        # 满足个数要求就结束
        if Count == 2000:
            break

