import csv
import os
import random
from collections import Counter

from PIL import Image


class ImageRotate:
    """
    图像旋转
    """
    def __init__(
        self,
        img_path,
        label_file,
        save_file_name,
        img_type='png',
        save_path='./new_img',
    ):
        self.img_path = img_path
        self.label_file = label_file
        self.img_type = img_type
        self.save_path = save_path
        self.save_file_name = save_file_name

    def get_imlist(self):
        """
        返回目录中所有图像的文件名列表
        """
        image_list = [
            os.path.join(self.img_path, f) for f in os.listdir(self.img_path)
            if f.endswith(self.img_type)
        ]
        return image_list

    def get_img_label(self):
        """
        获取csv文件中图片对应的标签
        """
        label_dict = {}
        with open(self.label_file, 'r') as fp:
            data = csv.reader(fp)
            for row in data:
                try:
                    label_dict[row[0]] = row[1]
                except:
                    label_dict[row[0]] = ''
        label_num_dict = dict(Counter(label_dict.values()))
        return label_dict, label_num_dict

    def rotate_image(self):
        """
        对图片进行旋转操作
        """
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        new_label_dict = {}
        image_list = self.get_imlist()
        label_dict, label_num_dict = self.get_img_label()
        for img in image_list:
            pil_im = Image.open(img)
            img_name = img.split('\\')[-1].strip()
            if img_name not in label_dict:
                continue
            img_label = label_dict[img_name]
            label_num = label_num_dict[img_label]
            if 1 <= label_num <= 5:
                n = 10
            elif 5 < label_num <= 20:
                n = 30
            elif 20 <= label_num <= 50:
                n = 90
            else:
                n = 360
            for i, angle in enumerate(range(0, 360, n)):
                mode = f'_{i}.'
                img_copy = img_name.replace('.', mode)
                try:
                    label = label_dict[img_name]
                    if len(label) == 1:
                        new_label_dict[img_copy] = label
                        out_file = f'./{self.save_path}/{img_copy}'
                        pil_im.rotate(angle,
                                      fillcolor=(255, 255, 255)).save(out_file)
                        print(out_file)
                except:
                    # new_label_dict[img_copy] = ''
                    pass
        return new_label_dict

    def save_new_label(self):
        """
        打乱并保存标签文件
        """
        label_dict = self.rotate_image()
        label_list = [list(item) for item in label_dict.items()]
        random.shuffle(label_list)
        save_file = self.save_path + '/' + self.save_file_name
        with open(save_file, 'wt', encoding='utf-8', newline='') as fp:
            csvout = csv.writer(fp, delimiter=',')
            csvout.writerows(label_list)

    def __call__(self):
        return self.save_new_label()


if __name__ == "__main__":
    ImageRotate(img_path='../test',
                label_file='../test/traindata_label.csv',
                save_file_name='data.csv')()
