import csv

Frame = [[1.2, 3., 4.], [2.5, 6., 6.2]]

with open('/data/Meisa/hybridCNN/out.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter = ',')
    for item in Frame:
        csvwriter.writerow(item)

with open('/data/Meisa/hybridCNN/out.csv', 'r') as csvfile:
    Framereader = []
    rows = csv.reader((csvfile))
    for row in rows:
        Framereader.append(row)

New = []
for i in Framereader:
    New.append([float(j) for j in i])

print New
#
# print Framereader
# print Framereader

