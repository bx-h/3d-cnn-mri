# coding: utf-8
'''
        2019/3/29

    读取6_ROI中的347个病人的mri数据：sub1 ~ sub364
    数据集：(肿瘤医院，中山六院)
    T2: 512*512*N (N为帧数，不同病人的MRi帧数不同)

'''

import os
import sys

sys.path.append('../')

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image
import scipy.stats
import json

# import files of mine
from logger import log, ImProgressBar

# [39人小数据集, 651人CC_ROI, 363人6_ROI]
dataset_paths = [{'xlsx_path': '/data/weishizheng/hbx/data/information.xlsx',
                  'sheet_name': 'Sheet1',
                  'data_path': '/data/weishizheng/hbx/data/',
                  'json_path': '/data/weishizheng/hbx/data/split.json'},
                 {'xlsx_path': '/data/weishizheng/hbx/2019_rect_pcr_data/information.xlsx',
                  'sheet_name': 0,
                  'data_path': '/data/weishizheng/hbx/2019_rect_pcr_data/CC_ROI/',
                  'json_path': '/data/weishizheng/hbx/2019_rect_pcr_data/CC_ROI/split.json'},
                 {'xlsx_path': '/data/weishizheng/hbx/2019_rect_pcr_data/information.xlsx',
                  'sheet_name': 1,
                  'data_path': '/data/weishizheng/hbx/2019_rect_pcr_data/6_ROI/',
                  'json_path': '/data/weishizheng/hbx/2019_rect_pcr_data/6_ROI/split.json'}]

# json_path = '/data/weishizheng/hbx/2019_rect_pcr_data/split.json'

# dataset_paths = [{'xlsx_path': '../../Datasets/data/information.xlsx',
#                   'sheet_name': 'Sheet1',
#                   'data_path': '../../Datasets/data/',
#                   'json_path': '../../Datasets/data/split.json'},
#                  {'xlsx_path': '../../Datasets/2019_rect_pcr_data/information.xlsx',
#                   'sheet_name': 0,
#                   'data_path': '../../Datasets/2019_rect_pcr_data/CC_ROI/',
#                   'json_path': '../../Datasets/2019_rect_pcr_data/CC_ROI/split.json'},
#                  {'xlsx_path': '../../Datasets/2019_rect_pcr_data/information.xlsx',
#                   'sheet_name': 1,
#                   'data_path': '../../Datasets/2019_rect_pcr_data/6_ROI/',
#                   'json_path': '../../Datasets/2019_rect_pcr_data/6_ROI/split.json'}]

json_path = './process/split.json'

mri_post_path = "/MRI/T2"
tumor_post_path = "/MRI/T2tumor.mha"


def _load_tumor(img_path):
    '''
        @ param: p_id为str类型，表示病人的编号
        @ return: 返回这个病人的tumor标注的mri图像的(N,512,512)
        读取一个病人的T2的mri图像的肿瘤标记图像，图像文件格式为mha
        mha为三维数组，元素为0/1 int
    '''
    image_array = None
    if os.path.exists(img_path):
        image = sitk.ReadImage(img_path)
        image_array = sitk.GetArrayFromImage(image)  # (c, h, w)
    else:
        log.logger.critical('Cannot find file \"' + img_path + '\"')

    return image_array


def _load_inf(data_choose):
    '''
        @ return: dict类型 p_id: label
        读取information.xlsx文件，加载病人编号，标签
    '''
    xlsx_path = dataset_paths[data_choose]['xlsx_path']
    sheet_name = dataset_paths[data_choose]['sheet_name']
    data_path = dataset_paths[data_choose]['data_path']

    # patients = {} # {p_id:label }
    df = pd.read_excel(io=xlsx_path, sheet_name=sheet_name)  # 打开excel文件

    patient_ids = df.iloc[:, 0].values  # 读取编号列
    patient_labels = df[u'结局'].values  # 读取结局列
    patient_T2_resolution = df[u'T2序列分辨率'].values  # 读取T2序列分辨率

    patient_T2_resolution = patient_T2_resolution[~np.isnan(patient_labels)]  # 删掉 nan
    patient_ids = patient_ids[~np.isnan(patient_labels)]  # 删掉 nan
    patient_labels = patient_labels[~np.isnan(patient_labels)].astype(np.long)  # 删掉 nan

    # print("2", len(patient_ids), len(patient_labels), len(patient_T2_resolution))

    patient_labels = patient_labels - 1  # 1/2/3 -> 0/1/2

    output = {}

    for i in range(len(patient_ids)):  # 使得所有编号长度相同
        patient_ids[i] = patient_ids[i][0:6]
        patient_T2_resolution[i] = patient_T2_resolution[i][0:3] + \
                                   patient_T2_resolution[i][4:7]

        if patient_T2_resolution[i] != "512512":
            continue
        # 一个病人的路径
        if os.path.exists(data_path + patient_ids[i] + mri_post_path):
            output[patient_ids[i]] = patient_labels[i]

    return output  # eg: {sub001: 0, sub002: 0}


def _get_image_paths(data_choose, patient_id):
    """
    获取一个病人的所有slides的路径, 除去没有肿瘤区域的slides
    return a `list`, the paths of the images (slides) in `img_dir`\\
    Args: \\
        img_dir: the directory of the images (slides)\\
    """
    # if there is no such directory, return an empty list.
    # Notice: Some directories are in the excel file, while not exist in the dataset folder.
    data_path = dataset_paths[data_choose]['data_path']
    img_dir = data_path + patient_id + mri_post_path
    tumor_dir = data_path + patient_id + tumor_post_path

    if not os.path.exists(img_dir):
        return []

    tumor_img = _load_tumor(tumor_dir)  # 获取当前病人的肿瘤标记图像 3维 (c, h, w)
    tumor_index = []  # 储存有肿瘤标记的slide的下标
    tumor_hw_min_max = []  # 每张有肿瘤标记的肿瘤区域，h和w坐标的最大最小值 (h_min, h_max, w_min, w_max)
    for i in range(len(tumor_img)):
        if np.max(tumor_img[i]) > 0:
            """
            np.nonzero()
            取出矩阵中的非零元素的坐标, 用法见: https://blog.csdn.net/u011361880/article/details/73611740
            indices 数据格式格式如：(array([305, 305, 306, ...], dtype=int64), array([241, 242, 243, ...], dtype=int64))
            第一个array是h数据，第二个array是w数据，并且两个array一一对应组成数对，为该非0像素的的坐标
            """
            indices = tumor_img[i].nonzero()
            tumor_hw_min_max.append([
                int(np.min(indices[0])),
                int(np.max(indices[0])),
                int(np.min(indices[1])),
                int(np.max(indices[1]))
            ])
            tumor_index.append(i)

    # 获取当前病人的所有mri图片的路径
    paths = []
    for img_name in os.listdir(img_dir):
        # check the file type
        file_type = os.path.splitext(img_name)[1]
        if (not (file_type == '.dcm')):
            log.logger.critical("File is not .dcm")
            continue
        paths.append(os.path.join(img_dir, img_name))
    paths = sorted(paths)  # 路径从小到大排序

    # 从`paths`里面，筛选有肿瘤标记的slides
    mri_img_paths = []  # 有肿瘤标记的slides的paths
    for i in range(len(tumor_index)):
        mri_img_paths.append(paths[tumor_index[i]])
        # if tumor_index[i] + 1 != int(paths[tumor_index[i]][-6:-4]):
        #     log.logger.critical("tumor and slide don't match! {} {} {} {}".format(tumor_index[i], paths[tumor_index[i]], int(paths[tumor_index[i]][-6:-4]), tumor_index[i] + 1 == int(paths[tumor_index[i]][-6:-4])))

    return mri_img_paths, tumor_index, tumor_hw_min_max


def init_dataset(data_chooses=[0], test_size=0.2, std_spacing_method="global_std_spacing_mode", new_init=False):
    """
    初始化数据集, 按`test_size`来随机按病人和类别 (每个类别都按`test_size`划分) 划分训练集和测试集, 并将各种信息写到json文件里\\
    Args: \\
        `data_chooses`: 选择数据集\\
        `test_size`: 测试集病人数所占数据集病人数的比例\\
        `std_spacing_method`: 选择标准spacing (其他slides都将resize到这个spacing) 的方法, 有
            `global_std_spacing_mode` (全局spacing众数),
            `global_std_spacing_mean` (全局spacing均值),
            `train_std_spacing_mode` (训练集spacing众数),
            `train_std_spacing_mean` (训练集spacing均值)\\
        `new_init`: 是否要重新初始化数据集 (将信息重新写入json)\\
    Return: \\
        `mean_std`: 训练集和测试集的mean, std
    """
    if os.path.exists(json_path) and (new_init is False):
        with open(json_path) as json_file:
            data_info = json.load(json_file)
            return data_info["mean_std"], data_info["max_size_spc"], data_info["global_hw_min_max_spc_world"]

    json_dataset = {"train": [], "test": []}  # 储存路径、label和id

    ## 划分数据集 ################################################################################################
    for data_choose in data_chooses:
        log.logger.info("Initializing dataset {} ...".format(data_choose))

        patients = _load_inf(data_choose)  # 读取病人的编号及结局（标签）{sub001:0, sub002:0}
        dataset = {"train": {}, "test": {}}  # 储存id和label

        class_patients = [[], [], []]

        # 划分训练集和测试集
        for patient_id, patient_label in patients.items():
            class_patients[patient_label].append(patient_id)

        # 随机选择测试集、训练集
        for i in range(len(class_patients)):
            rand_index = np.arange(len(class_patients[i]))  # 获取随机下标
            np.random.shuffle(rand_index)

            for j in range(len(rand_index)):
                if j < test_size * len(rand_index):  # 前test_size部分为测试集
                    dataset["test"][class_patients[i][rand_index[j]]] = i  # id:label
                else:
                    dataset["train"][class_patients[i][rand_index[j]]] = i

        # 在json_dataset里记录信息
        for dataset_type in ["train", "test"]:
            hw_min_max = []
            for patient_id, patient_label in dataset[dataset_type].items():
                mri_img_paths, tumor_index, tumor_hw_min_max = _get_image_paths(data_choose, patient_id)
                for i in range(len(mri_img_paths)):  # 对于每个病人的每张slide
                    json_dataset[dataset_type].append({
                        'path': mri_img_paths[i],
                        'label': patient_label,
                        'id': str(data_choose) + '_' + patient_id,
                        'tumor_index': tumor_index[i],
                        'tumor_hw_min_max': tumor_hw_min_max[i]
                    })
                # 列表拼接，如 [[273, 300, 268, 288], [270, 312, 264, 290]] + [265, 312, 261, 290] = [[273, 300, 268, 288], [270, 312, 264, 290], [265, 312, 261, 290]]
                hw_min_max += tumor_hw_min_max

            col_min = np.array(hw_min_max).min(axis=0)  # 列最小值
            col_max = np.array(hw_min_max).max(axis=0)  # 列最大值
            json_dataset[dataset_type + "_hw_min_max"] = [int(round(col_min[0])), int(round(col_max[1])),
                                                          int(round(col_min[2])), int(round(col_max[3]))]

        json_dataset["global_hw_min_max"] = [
            min(json_dataset["test_hw_min_max"][0], json_dataset["train_hw_min_max"][0]),
            max(json_dataset["test_hw_min_max"][1], json_dataset["train_hw_min_max"][1]),
            min(json_dataset["test_hw_min_max"][2], json_dataset["train_hw_min_max"][2]),
            max(json_dataset["test_hw_min_max"][3], json_dataset["train_hw_min_max"][3])
        ]
    ############################################################################################################

    ## 训练集和测试集分别统计均值方差，并记录一些信息 ##############################################################
    mean_std = {}
    global_spacing_list = [[], [], []]
    for dataset_type in ["train", "test"]:
        log.logger.info("Calculating {} dataset {} info (mean, std, spacing, hw_min_max, etc.)...".format(dataset_type,
                                                                                                          data_chooses))
        pbar = ImProgressBar(total_iter=len(json_dataset[dataset_type]))  # 进度条

        # 仅需统计下面这三个变量，就可以计算均值和方差
        pixel_num = 0
        sum_x = 0
        sum_x_square = 0

        spacing_list = [[], [], []]

        for i, patient_info in enumerate(json_dataset[dataset_type]):
            mri_img_path = patient_info["path"]
            sitk_image = sitk.ReadImage(mri_img_path)
            origin = sitk_image.GetOrigin()  # x, y, z
            spacing = sitk_image.GetSpacing()  # x, y, z
            image = sitk.GetArrayFromImage(sitk_image)  # numpy (c, h, w)
            image = image[0]

            json_dataset[dataset_type][i]["origin"] = [float(x) for x in origin]
            json_dataset[dataset_type][i]["spacing"] = [float(x) for x in spacing]
            json_dataset[dataset_type][i]["shape"] = [int(x) for x in image.shape]

            if spacing[0] != spacing[1]:
                log.logger.critical(
                    "spacing[0] != spacing[1]: {} != {} in {}".format(spacing[0], spacing[1], mri_img_path))

            for ispc, spc in enumerate(spacing):
                spacing_list[ispc].append(spc)

            image = image.astype(np.int64)
            pixel_num += image.shape[0] * image.shape[1]
            sum_x += np.sum(image)
            sum_x_square += np.sum(image ** 2)

            pbar.update(i)
        pbar.finish()

        # 计算众数和均值std_spacing
        for i in range(len(global_spacing_list)):
            global_spacing_list[i] += spacing_list[i]  # 列表拼接
        std_spacing_mode = [float(scipy.stats.mode(x)[0][0]) for x in spacing_list]
        std_spacing_mean = [float(np.mean(x)) for x in spacing_list]
        json_dataset[dataset_type + "_std_spacing_mode"] = std_spacing_mode
        json_dataset[dataset_type + "_std_spacing_mean"] = std_spacing_mean

        # 计算均值方差
        dataset_mean = sum_x / pixel_num
        dataset_std = ((sum_x_square / pixel_num) - dataset_mean ** 2) ** 0.5
        mean_std[dataset_type] = [dataset_mean, dataset_std]

    json_dataset["global_std_spacing_mode"] = [float(scipy.stats.mode(x)[0][0]) for x in global_spacing_list]
    json_dataset["global_std_spacing_mean"] = [float(np.mean(x)) for x in global_spacing_list]
    json_dataset["mean_std"] = mean_std

    ############################################################################################################

    ## 计算spacing之后的MRI和tumor的size, 以及h_w_min_max字段 ####################################################
    std_spacing = json_dataset[std_spacing_method]
    resize_coefs = []
    for dataset_type in ["train", "test"]:
        for i, patient_info in enumerate(json_dataset[dataset_type]):
            cur_spacing = json_dataset[dataset_type][i]["spacing"]
            resize_coef = [float(std_spacing[i] / cur_spacing[i]) for i in range(len(cur_spacing))]
            resize_coefs.append(resize_coef[0])
            json_dataset[dataset_type][i]["resize_coef"] = resize_coef
            json_dataset[dataset_type][i]["shape_spc"] = [int(round(x * resize_coef[0])) for x in
                                                          json_dataset[dataset_type][i]["shape"]]
            json_dataset[dataset_type][i]["tumor_hw_min_max_spc"] = [int(round(x * resize_coef[0])) for x in
                                                                     json_dataset[dataset_type][i]["tumor_hw_min_max"]]

    max_resize_coef = max(resize_coefs)
    max_size_spc = int(round(max_resize_coef * 512))  # spc后最大图片的大小
    for dataset_type in ["train", "test"]:
        hw_min_max = []
        for i, patient_info in enumerate(json_dataset[dataset_type]):
            cur_size_spc = json_dataset[dataset_type][i]["shape_spc"][0]
            json_dataset[dataset_type][i]["tumor_hw_min_max_spc_world"] = [
                int(round((max_size_spc - cur_size_spc) / 2 + x)) for x in
                json_dataset[dataset_type][i]["tumor_hw_min_max_spc"]
            ]
            hw_min_max.append(json_dataset[dataset_type][i]["tumor_hw_min_max_spc_world"])

        col_min = np.array(hw_min_max).min(axis=0)  # 列最小值
        col_max = np.array(hw_min_max).max(axis=0)  # 列最大值
        json_dataset[dataset_type + "_hw_min_max_spc_world"] = [int(round(col_min[0])), int(round(col_max[1])),
                                                                int(round(col_min[2])), int(round(col_max[3]))]

    json_dataset["global_hw_min_max_spc_world"] = [
        min(json_dataset["test_hw_min_max_spc_world"][0], json_dataset["train_hw_min_max_spc_world"][0]),
        max(json_dataset["test_hw_min_max_spc_world"][1], json_dataset["train_hw_min_max_spc_world"][1]),
        min(json_dataset["test_hw_min_max_spc_world"][2], json_dataset["train_hw_min_max_spc_world"][2]),
        max(json_dataset["test_hw_min_max_spc_world"][3], json_dataset["train_hw_min_max_spc_world"][3])
    ]
    json_dataset["max_resize_coef"] = max_resize_coef
    json_dataset["max_size_spc"] = max_size_spc

    ############################################################################################################

    with open(json_path, 'w') as json_file:
        json_file.write(json.dumps(json_dataset))  # 写入json

    return mean_std, max_size_spc, json_dataset["global_hw_min_max_spc_world"]


def read_image(img_path):
    """
    读取对应dcm路径的一张slide\\
    注意：sitk读取进来的图像的数据格式是(c, h, w)，
    而后面进行的ToTensor操作，要求数据格式必须是(h, w, c)，因此要交换一下通道再返回\\
    read an image from `img_path` and return its numpy image\\
    Args: \\
        img_path: the path of the image\\
    """
    sitk_image = sitk.ReadImage(img_path)
    image = sitk.GetArrayFromImage(sitk_image)
    image = image.swapaxes(0, 1).swapaxes(1, 2)  # (c, h, w) -> (h, w, c)
    return image


class MriDataset(Data.Dataset):
    def __init__(self, train=True, transform=None, normalize=True, is_spacing=False):
        """
        return dataset\\
        Args:
            train: is train dataset?
            transform: data augmentation transforms
            normalize: 是否要进行normalize
            is_spacing: is perform spacing resize?
        """

        with open(json_path) as json_file:
            self.data_type = "train" if train else "test"
            self.dataset_info = json.load(json_file)
            self.max_size_spc = self.dataset_info["max_size_spc"]
            self.global_hw_min_max_spc_world = self.dataset_info["global_hw_min_max_spc_world"]

            mean_std = self.dataset_info["mean_std"][self.data_type]
            self.normalize = transforms.Compose([transforms.Normalize(mean=[mean_std[0]], std=[mean_std[1]])])

            self.dataset = self.dataset_info[self.data_type]

            # self.hw_min_max = self.dataset_info["global_hw_min_max_spc"]

        self.transform = transform
        self.is_spacing = is_spacing

    def __getitem__(self, index):
        img = read_image(self.dataset[index]['path'])  # img, 数据格式(h, w, c), 类型numpy int16
        img = Image.fromarray(img, mode="I;16")  # img, 数据格式(h, w, c), 类型PIL.Image.Image image mode=I;16

        # -1. spacing resize
        if self.is_spacing is True:
            shape = self.dataset[index]["shape"]
            shape_spc = self.dataset[index]["shape_spc"]
            if shape[0] != shape_spc[0]:
                img = img.resize(size=(shape_spc[0], shape_spc[1]), resample=Image.NEAREST)  # spacing resize
            if shape_spc[0] != self.max_size_spc:
                img = transforms.Compose([transforms.CenterCrop(size=self.max_size_spc)])(
                    img)  # centercrop to the max_img size

            center_w = (self.global_hw_min_max_spc_world[3] + self.global_hw_min_max_spc_world[2]) / 2
            llength = self.global_hw_min_max_spc_world[1] - self.global_hw_min_max_spc_world[0] + 1
            left = int(round(center_w - llength / 2))
            img = img.crop((
                left,  # left
                self.global_hw_min_max_spc_world[0],  # upper
                left + llength,  # right
                self.global_hw_min_max_spc_world[1] + 1  # lower
            ))
            # img = img.crop((
            #     self.global_hw_min_max_spc_world[2],    # left
            #     self.global_hw_min_max_spc_world[0],    # upper
            #     self.global_hw_min_max_spc_world[3]+1,    # right
            #     self.global_hw_min_max_spc_world[1]+1     # lower
            # ))

        img = img.resize(size=(224, 224), resample=Image.NEAREST)

        # 0. 在这里做数据增广
        if self.transform != None:
            img = self.transform(img)

        # 1. 转成tensor; img, 数据格式(c, h, w), 类型tensor torch.int16; 注意这里的ToTensor不会将像素值scale到0~1
        img = transforms.Compose([transforms.ToTensor()])(img)

        # 2. 类型转换; img, 数据格式(c, h, w), 类型tensor torch.float32
        img = img.float()

        # 3. 归一化
        if self.normalize != None:
            img = self.normalize(img)

        label = self.dataset[index]['label']
        id = self.dataset[index]['id']

        # print(self.dataset[index]["resize_coef"])
        return img, label, id

    def __len__(self):
        return len(self.dataset)

    def get_class_weight(self):
        class_num = [0, 0, 0]  # 统计每类样本的数量
        for data in self.dataset:
            class_num[data['label']] += 1

        class_weight = [1.0 / x for x in class_num]
        """
        class_weight normalization, class_weihght: $$/frac{1/n_i}{1/n1 + 1/n2 + 1/n3}$$
        after class_weight normalization, the sum of class_weight is 1
        """
        sum_class_weight = sum(class_weight)
        class_weight = [x / sum_class_weight for x in class_weight]

        return class_weight

    def get_sampler(self):
        class_weight = self.get_class_weight()
        weights = []
        for i in range(self.__len__()):
            target = self.dataset[i]['label']
            weights.append(class_weight[target])

        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=weights,
            num_samples=self.__len__(),
            replacement=True
        )
        return sampler


if __name__ == "__main__":
    pass
