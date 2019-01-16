if __name__ == '__main__':
    import json
    TrainJSONpath = 'D:\\ClipShots\\ClipShots\\ClipShots\\annotations\\train.json'
    annotations = json.load(open(TrainJSONpath))

    Count = 0
    TheFirst1000VideoNameTXTpath = 'D:\\ClipShots\\TheFirst2000.txt'
    for videoname, labels in annotations.items():
        Count += 1
        with open(TheFirst1000VideoNameTXTpath, 'a') as f:
            f.write(str(videoname)+'\n')
        if Count == 2000:
            break

