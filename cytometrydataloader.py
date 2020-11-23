import json
import pickle
import os
import copy
import cv2
import glob
import random
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch

import pandas as pd
import tifffile


from torchvision import transforms




class DataPreparation(object):
    root_path = '/n/pfister_lab2/Lab/wdjang/single_cell_clustering/dataset/imaging_mass_cytometry'
    tiff_rootPath = os.path.join(root_path, 'OMEnMasks/ome')
    def __init__(self, gt_type='survival'):
        self.gt_type = gt_type
        self.valid_channelList = []
        self.split_list = dict()
        self.percentile_minmax = dict()
        self.tiffPath_gtLabelList = []

    def __get_survival_gt__(self, survival_month):
        '''
        bin 0 = [0,50)
        bin 1 = [50,100)
        bin 2 = [100,150)
        bin 3 = [150,200)
        bin 4 = [200-
        '''
        survival_month = np.uint16(survival_month)
        bins = [0, 50, 100, 150, 200]
        if survival_month in bins:
            bin_num = bins.index(survival_month)
        else:
            data = bins + [survival_month]
            bin_num = sorted(data, reverse=False).index(survival_month) - 1
        return bin_num

    def split_dataset(self):
        print("Computing new split...")
        # Basel
        meta_path = os.path.join(self.root_path, 
                                 'Data_publication/BaselTMA/Basel_PatientMetadata.csv')
        meta_cols = pd.read_csv(meta_path, names=None, usecols=['core', 'grade', 'OSmonth'])
        meta_coreList = np.asarray(meta_cols['core'])
        meta_gradeList = np.asarray(meta_cols['grade'])
        meta_survivalList = np.asarray(meta_cols['OSmonth'])
        for meta_id, meta_core in enumerate(meta_coreList):
            split_names = meta_core.split('_')
            head_name = split_names[0] + '_' + split_names[1]
            mid_name = split_names[2] + '_' + split_names[3]
            tiff_path = glob.glob(os.path.join(self.tiff_rootPath, 
                                               '{}*{}*.tiff'.format(head_name, mid_name)))[0]
            if meta_gradeList[meta_id] == 'METASTASIS':
                continue
            tiff_path = tiff_path.split(self.tiff_rootPath + '/')[1]
            if self.gt_type == "survival":
                survival_month = meta_survivalList[meta_id]
                survival_gt = self.__get_survival_gt__(survival_month)
                self.tiffPath_gtLabelList.append({'tiff_path': tiff_path,
                                                  'class_label': survival_gt})
            elif self.gt_type == "grade":
                grade_gt = np.uint16(meta_gradeList[meta_id])-1
                self.tiffPath_gtLabelList.append({'tiff_path': tiff_path,
                                                  'class_label': grade_gt})
        if self.gt_type == "grade":
            # Zurich
            meta_path = os.path.join(self.root_path, 
                                     'Data_publication/ZurichTMA/Zuri_PatientMetadata.csv')
            meta_cols = pd.read_csv(meta_path, names=None, usecols=['core', 'grade'])
            meta_coreList = np.asarray(meta_cols['core'])
            meta_gradeList = np.asarray(meta_cols['grade'])
            for meta_id, meta_core in enumerate(meta_coreList):
                split_names = meta_core.split('_')
                head_name = split_names[0] + '_' + split_names[1]
                mid_name = split_names[2] + '_' + split_names[3]
                tiff_path = glob.glob(os.path.join(self.tiff_rootPath, 
                                                   '{}*{}*.tiff'.format(head_name, mid_name)))[0]
                if meta_gradeList[meta_id] == 'METASTASIS':
                    continue
                tiff_path = tiff_path.split(self.tiff_rootPath + '/')[1]
                grade_gt = np.uint16(meta_gradeList[meta_id])-1
                self.tiffPath_gtLabelList.append({'tiff_path': tiff_path,
                                             'class_label': grade_gt})
        total_num_imgs = len(self.tiffPath_gtLabelList)
        num_trainData = int(total_num_imgs*0.7)
        self.split_list['train'] = self.tiffPath_gtLabelList[:num_trainData]
        self.split_list['test'] = self.tiffPath_gtLabelList[num_trainData:]
        print("Num training imgs: ", len(self.split_list['train']))
        print("Num testing imgs: ", len(self.split_list['test']))


    def get_channelList(self, channels_toExclude=[]):
        csv_path = os.path.join(self.root_path, 'IMC_Nature_TargetList.xlsx')
        df = pd.read_excel(csv_path, header=None, names=['Channel Names'])
        listChannels = df['Channel Names'].values.tolist()
        print('Channels used for training and test')
        for channel_id, channel_name in enumerate(listChannels):
            if channel_name not in channels_toExclude:
                print(channel_name)
                self.valid_channelList.append({'channel_id': channel_id, 
                                               'channel_name': channel_name})
        print('')


    def find_percentile(self):
        print('Finding max intensities across all images for each channel (min is always 0)')
        channel_maxvals = np.zeros(len(self.valid_channelList))
        for img_id, tiffPath_gtLabel in enumerate(self.tiffPath_gtLabelList):
            print('{}/{}'.format(img_id, len(self.tiffPath_gtLabelList)))
#             tiff_reader = bioformats.ImageReader(os.path.join(self.tiff_rootPath, 
#                                                               tiffPath_gtLabel['tiff_path']))
            for channel_id, valid_channel in enumerate(self.valid_channelList):
#                 channel_img = tiff_reader.read(index=valid_channel['channel_id'], 
#                                                rescale=False)
                channel_img = tifffile.imread(os.path.join(self.tiff_rootPath, 
                                                           tiffPath_gtLabel['tiff_path']), 
                                              key=valid_channel['channel_id'])
                max_val = np.max(channel_img)
                if channel_maxvals[channel_id] < max_val:
                    channel_maxvals[channel_id] = max_val
        print(channel_maxvals)
        print('')
        print('Finding min and max percentiles across all images for each channel')
        num_bins = 10000
        channel_histCum = np.zeros((len(self.valid_channelList), num_bins-2), np.int64)
        for img_id, tiffPath_gtLabel in enumerate(self.tiffPath_gtLabelList):
            print('{}/{}'.format(img_id, len(self.tiffPath_gtLabelList)))
#             tiff_reader = bioformats.ImageReader(os.path.join(self.tiff_rootPath, 
#                                                               tiffPath_gtLabel['tiff_path']))
            for channel_id, valid_channel in enumerate(self.valid_channelList):
#                 channel_img = tiff_reader.read(index=valid_channel['channel_id'], 
#                                                rescale=False)
                channel_img = tifffile.imread(os.path.join(self.tiff_rootPath, 
                                                           tiffPath_gtLabel['tiff_path']), 
                                              key=valid_channel['channel_id'])
                bin_list = np.arange(0, channel_maxvals[channel_id]-1e-6, 
                                     channel_maxvals[channel_id]/num_bins)
                channel_hist = np.histogram(channel_img.flatten(), bin_list)[0]
                # excluding bin 0 because they are too many background pixels
                channel_histCum[channel_id] += channel_hist[1:]
        for channel_id, channel_hist in enumerate(channel_histCum):
            cum_sum = channel_hist.cumsum()
            one_percent = cum_sum[-1]*0.01
            max_id = np.where(cum_sum > (cum_sum[-1] - one_percent))[0][0]
            bin_list = np.arange(0, channel_maxvals[channel_id], 
                                 channel_maxvals[channel_id]/num_bins)
            max_val = bin_list[max_id] + channel_maxvals[channel_id]/num_bins
            self.valid_channelList[channel_id]['max_percentile'] = max_val
            print(max_val)
        print('')
    

class BioformatTiffreader(object):
    root_path = '/n/pfister_lab2/Lab/wdjang/single_cell_clustering/dataset/imaging_mass_cytometry'
    tiff_rootPath = os.path.join(root_path, 'OMEnMasks/ome')

    def __init__(self, prepared_data, img_size=224, phase_str='train'):
        print('Creating tiff readers...')
        self.tiffPath_gtLabels = prepared_data.split_list[phase_str]
        self.valid_channelList = prepared_data.valid_channelList
        self.gt_type = prepared_data.gt_type
        print("Ground truth will be: ", self.gt_type)
        # Input size for a network
        self.img_size = img_size
        # self.tiffPath_gtLabels = self.tiffPath_gtLabels[:16]
        self.normalize = transforms.Normalize(mean=[0.5],
                                              std=[0.25])
        if phase_str == 'train':
            # Training Set
            self.transforms = transforms.Compose([
#                 transforms.RandomRotation((-90, 90)), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop(img_size, pad_if_needed=True),
#                 transforms.Resize(img_size),
                transforms.ToTensor(),
#                 transforms.ColorJitter(brightness=0.13, contrast=0.13, 
#                                        saturation=0.13, hue=0.13), 
#                 transforms.RandomErasing(p=0.15, scale=(0.02, 0.25), ratio=(0.5, 25), 
#                                          value=0.5, inplace=False)
            ])
        else: 
            # Test Set
            # if img size is larger than 224
            self.transforms = transforms.Compose([transforms.CenterCrop(img_size),
                                                  transforms.ToTensor(),])
#             # Test Set
#             # if img size is smaller than 224
#             self.transforms = transforms.Compose([transforms.Resize(img_size, img_size),
# #                                                   transforms.CenterCrop(img_size),
#                                                   transforms.ToTensor(),])
    
    def __getitem__(self, idx):
        tiff_path = self.tiffPath_gtLabels[idx]['tiff_path']
        class_label = self.tiffPath_gtLabels[idx]['class_label']
        in_img = np.zeros((len(self.valid_channelList), self.img_size, self.img_size), np.float32)
#         tiff_reader = bioformats.ImageReader(os.path.join(self.tiff_rootPath, tiff_path))
        tensor_imgList = []
        seed = np.random.randint(541474) # make a seed with numpy generator 
        for chId_iter, valid_channel in enumerate(self.valid_channelList):
            in_channel = tifffile.imread(os.path.join(self.tiff_rootPath, tiff_path), key=valid_channel['channel_id'])
#             in_channel = tiff_reader.read(index=valid_channel['channel_id'], rescale=False)
            min_val = 0
            max_val = valid_channel['max_percentile']
            in_channel = (np.clip(in_channel, min_val, max_val) - min_val) / (max_val - min_val)
            if in_channel.shape[0] < self.img_size:
                pad_len = int(self.img_size-in_channel.shape[0])
                in_channel = np.pad(in_channel, ((0, 0), (pad_len, 0)), 'constant', constant_values=(0, 0))
            if in_channel.shape[1] < self.img_size:
                pad_len = int(self.img_size-in_channel.shape[1])
                in_channel = np.pad(in_channel, ((0, 0), (0, pad_len)), 'constant', constant_values=(0, 0))
            pil_img = Image.fromarray((255*in_channel).astype(np.uint8))
            random.seed(seed) # apply this seed to img tranfsorms
            tensor_img = self.transforms(pil_img)
            tensor_img = self.normalize(tensor_img)
            tensor_imgList.append(tensor_img)
        in_img = torch.cat(tensor_imgList)
        return in_img, class_label

    def __len__(self):
        return len(self.tiffPath_gtLabels)
