import os
import sys
import xml.etree.ElementTree as ET
import glob



classes = ['coke', 'orange']
# classes_1 = ['car','watcher','base']


def xml_to_txt(indir, outdir):
    os.chdir(indir)
    annotations = os.listdir(".")
    annotations = glob.glob(str(annotations) + '*.xml')
    #count = 0
    for i, file in enumerate(annotations):

        file_save = file.split('.')[0] + '.txt'
        file_txt = os.path.join(outdir, file_save)
        # os.chdir(outdir)
        # if not os.path.exists(file_txt):
            # print("no")
        f_w = open(file_txt, mode='w')

        # actual parsing
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        # print(x for x in root.iter('width'))
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        # print(width)

        for obj in root.iter('object'):
            name = obj.find('name')
            if name is None:
                continue
            cla = name.text
            if cla in classes:
                label = str(classes.index(cla))

                xmlbox = obj.find('bndbox')
                xn = float(xmlbox.find('xmin').text)
                xx = float(xmlbox.find('xmax').text)
                yn = float(xmlbox.find('ymin').text)
                yx = float(xmlbox.find('ymax').text)
                x_center = str(keep_one((xn + xx) / (2.0 * width)))
                y_center = str(keep_one((yn + yx) / (2.0 * height)))
                norm_width = str(keep_one((xx - xn) / width))
                norm_height = str(keep_one((yx - yn) / height))
                # print xn
                f_w.write(label + ' ' + x_center + ' ' + y_center + ' ' + norm_width + ' ' + norm_height + '\n')
            else:
                continue


def keep_one(num):
    if (num > 1):
        num=1
    return num


if __name__ == "__main__":
    indir = "../demo/datasets/xml"  # xml目录
    outdir = '../txt'  # txt目录
    xml_to_txt(indir, outdir)
