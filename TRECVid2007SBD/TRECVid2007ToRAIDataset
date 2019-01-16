import os

# Change the annotation of TRECVid2007 to RAI
if __name__ == '__main__':
    os.chdir('E:\\Meisa_SiameseNetwork\\sv2007sbtest1\\sbref071\\ref')
    with open('./ref_BG_2408.xml', 'r') as f:
        Trans = f.readlines()

    TrueTrans = []
    for i in range(len(Trans)):
        if 'CUT' in Trans[i]:
            if len(TrueTrans) == 0:
                #TrueTrans.append(0)
                TrueTrans.append([0,int(Trans[i].split('"')[-4])])
                TrueTrans.append([int(Trans[i].split('"')[-2])])
            else:
                TrueTrans[-1].extend([int(Trans[i].split('"')[-4])])
                TrueTrans.append([int(Trans[i].split('"')[-2])])
        elif 'DIS' in Trans[i] or 'OTH' in Trans[i]:
            TrueTrans[-1].extend([int(Trans[i].split('"')[-4])])
            TrueTrans.append([int(Trans[i].split('"')[-2])])

        if i == (len(Trans) - 2):
            TrueTrans[-1].append(35892)
    #print Trans
    with open('./BG_2048.txt', 'a') as f:
        for i in TrueTrans:
            f.write(str(i[0])+'\t'+str(i[1])+'\n')
```
TRECVid标签示例[文件格式为.xml]
<!DOCTYPE refSeg SYSTEM "shotBoundaryReferenceSegmentation.dtd">
<refSeg src="BG_2408.mpg" creationMethod="MANUAL" totalFNum="35892">
<trans type="CUT" preFNum="258" postFNum="259"/>
<trans type="CUT" preFNum="412" postFNum="413"/>
<trans type="CUT" preFNum="478" postFNum="479"/>
<trans type="CUT" preFNum="567" postFNum="568"/>
<trans type="CUT" preFNum="673" postFNum="674"/>
<trans type="OTH" preFNum="714" postFNum="716"/>
<trans type="CUT" preFNum="812" postFNum="813"/>
<trans type="CUT" preFNum="879" postFNum="880"/>
<trans type="OTH" preFNum="913" postFNum="936"/>
<trans type="CUT" preFNum="1365" postFNum="1366"/>
<trans type="CUT" preFNum="2523" postFNum="2524"/>
<trans type="DIS" preFNum="2593" postFNum="2603"/>
<trans type="DIS" preFNum="3263" postFNum="3282"/>
<trans type="DIS" preFNum="3395" postFNum="3414"/>
/refSeg>

RAIDataset标签示例[文件格式为.txt]
0 15
16 100
105 600
601 870
...
```
