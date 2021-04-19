from imageNetLabelDict import labelDict
import csv


from tempfile import NamedTemporaryFile
import shutil
import csv

filename = "D:\\Dataset\\part-of-imagenet-master\\ncodes.csv"
tempfile = NamedTemporaryFile('w+t', delete=False)

with open(filename, 'r', newline="\r\n") as csvFile, tempfile:
    reader = csv.reader(csvFile, delimiter=',')
    writer = csv.writer(tempfile, delimiter=',',)

    first = True
    for row in reader:
        if first:
            first = False
            writer.writerow(row)
            continue
        if row[1] in labelDict.values():
            row[2] = "True"
            row[3] = row[3].replace("-1","100")
        writer.writerow(row)


shutil.move(tempfile.name, filename)

