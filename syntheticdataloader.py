#!/usr/bin/env python
# coding: utf-8

from skimage import data, io
import matplotlib
from matplotlib import pyplot as plt
from skimage.draw import circle
import numpy as np
import random 
import pdb
import copy
import torch
from torch.utils import data
random.seed(4)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SyntheticData(data.Dataset):
    'Characterizes a dataset for PyTorch'
    
    def __init__(self, 
                 num_ground_truths = 3,
                 num_channels = 30,
                 bNoCommonChannels = True,
                 num_images = 600,
                 phase = "none",
                 low_interval = (0.1,0.4),
                 high_interval = (0.6,0.9),
                 bHardCode = False,
                 hard_code_type = "none",
                 bDropChannels = False,
                 num_ch_to_drop = 0,
                 drop_val = "uniform"):
        
     
        self.num_channels = num_channels
        self.num_ground_truths = num_ground_truths
        self.bNoCommonChannels = bNoCommonChannels
        self.num_images = num_images
        self.phase = phase
        self.low_interval = low_interval
        self.high_interval = high_interval
        self.bDropChannels = bDropChannels
        self.num_ch_to_drop = num_ch_to_drop
        
        self.bHardCode = bHardCode
        self.hard_code_type = hard_code_type
        self.drop_val = drop_val
        self.mode_heights = {'right': [0.2, 0.8], 'left': [0.8, 0.2], 'mid': [0.5, 0.5]}
        
       ## seed vals for the actual ds creation random num generators.
        seed_vals = {
            "train": 39,
            "valid": 350,
            "test": 144,
            "none": 2
        }
        
        self.random_seed_val = seed_vals[self.phase]
        self.same_seed_val = 100031
        
        listChannels = range(num_channels)
        sampledAlready = []
        
        
        self.dict_gts_channels = {}
        
        for gt in range(self.num_ground_truths):
            
            np.random.seed(self.same_seed_val+gt)
            self.num_gt_channels = np.random.randint(2,3) ## 2 or 3 tbd ask wd

            random.seed(1255518)
            if bNoCommonChannels:
                listChannels = list(set(listChannels)-set(sampledAlready))
            random.shuffle(listChannels)
            
            ch_combos = random.sample(listChannels,self.num_gt_channels)
            sampledAlready += ch_combos
            
            
    
            mode_heights_choices = random.choices(["left","right"],k=self.num_gt_channels)
            
            all_channels_mode_centers = []
            random.seed(self.same_seed_val+gt)
            randomizers = [random.randint(self.num_gt_channels+1,999) for x in range(self.num_gt_channels)]
            ## tbd ask won dong, should each channel have its own set of mode_centers?
            for i in range(self.num_gt_channels):
                random.seed(self.same_seed_val+gt+randomizers[i])
#                 print(self.same_seed_val+gt+randomizers[i])
                mode_centers = {"low": round(random.uniform(*self.low_interval), 2), 
                                "high": round(random.uniform(*self.high_interval), 2)}
                
                all_channels_mode_centers.append(mode_centers)
            
           
            
            ch_gts_modes = self.merge(ch_combos, mode_heights_choices, all_channels_mode_centers)

            print("GT: {} associated with : {}".format(gt,ch_gts_modes))
            self.dict_gts_channels[gt] = ch_gts_modes 
            
        self.init_ds(self.num_images)
        
        ## Auxillary stuff 
        random.seed(self.random_seed_val) ## tbd ask wd about this, will have diff noise vals in other channels
        #self.channel_intensity = {k: round(random.uniform(0,1), 2) for k in range(self.num_channels)}
        
        ## RESET THE HARD CODE GTS HERE
        self.get_hard_code_gts()
            
#         print("\n\nNEW GT CHANNEL ASSOCIATIONS: ",self.dict_gts_channels)
    
        print("loaded dataset.")
        
        self.normalize = transforms.Normalize(mean=[0.5],
                                              std=[0.25])
        if phase == 'train':
            # Training Set
            self.transforms = transforms.Compose([
#                 transforms.RandomRotation((-90, 90)), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop(224, pad_if_needed=True),
#                 transforms.Resize(img_size),
                transforms.ToTensor(),
#                 transforms.ColorJitter(brightness=0.13, contrast=0.13, 
#                                        saturation=0.13, hue=0.13), 
#                 transforms.RandomErasing(p=0.15, scale=(0.02, 0.25), ratio=(0.5, 25), 
#                                          value=0.5, inplace=False)
            ])
        else: 
            # Valid / Test Set
            # if img size is larger than 224
            self.transforms = transforms.Compose([transforms.CenterCrop(224),
                                                  transforms.ToTensor(),])

        
    def merge(self, list1, list2,list3): 
        merged_list = [(list1[i], list2[i],list3[i]) for i in range(0, len(list1))] 
        
        return merged_list
    
    
    def init_ds(self,num_images=None):
        if num_images % self.num_ground_truths != 0:
              pass
    
        frac_each_gt = int(num_images / self.num_ground_truths)
       
        
        self.ds_gts = []
        for gt in range(self.num_ground_truths):
            self.ds_gts += [gt] * frac_each_gt
        
        
    def add_gauss_noise(self, img, img_id,  mean = 0.0, std  = 10.0):
        ''' add gauss noise to the image, then clip'''
        # we might get out of bounds due to noise
        # we pass img_id so that we get the same noise values for a given image
        # when instantiating this class
        np.random.seed(self.random_seed_val+img_id)
        noisy_img = img + np.random.normal(mean, std, img.shape)
        noisy_img_clipped = np.clip(noisy_img, 0, 255)
        return noisy_img_clipped
        
    def get_image(self,
                  img_id = None,
                  radius = 5,
                  max_circles = 150,
                  min_circles = 20,
                  bRandIntense = True,
                  img_size_w = 224,
                  img_size_h = 224,
                  bShowImages = False,
                  bAddGauss = True,
                  bDeltaNoise = True):
        
        
        
    
        img = np.zeros((img_size_w, img_size_h,self.num_channels), 'uint8')
        img_gt = self.ds_gts[img_id]
        
        random.seed(self.random_seed_val+img_id)#to generate diff number of circles for each image
        num_circles = random.randrange(min_circles, max_circles)
        
        random.seed(self.random_seed_val+img_id)#to generate diff circle centers for each image
        list_centers = [(random.randrange(0, img_size_w), random.randrange(0, img_size_h)) for i in range(num_circles)]

        
        
        random.seed(self.random_seed_val+img_id)
        #randomizers = [np.random.randint(self.num_gt_channels+1,999) for x in range(self.num_gt_channels)]
        
        
        for channel in range(self.num_channels):
            circle_intensities = self.get_ch_dist_vals(channel,gt=img_gt,N=num_circles, img_id=img_id)
            #for circle_num,center_coords in enumerate(list_centers):
            for circle_num in range(len(circle_intensities)):
                center_coords = list_centers[circle_num]
                r_center,c_center = center_coords 
                rr, cc = circle(r_center, c_center, radius, img.shape)
            
                img[rr,cc,channel] = circle_intensities[circle_num] * 255 
            
                
                
            
            if bShowImages:
                io.imshow(img[:,:,channel])
                plt.show()
                
        if bAddGauss:
            img = self.add_gauss_noise(img, img_id)
        
        if self.bDropChannels:
            
            listChannels = list(range(self.num_channels))
            random.seed(self.random_seed_val+img_id)
            channels_to_drop = random.sample(listChannels,k=self.num_ch_to_drop) 
            
#             print(img_id, channels_to_drop)
            for dropped_ch in channels_to_drop:
                if self.drop_val == "uniform":
                    np.random.seed(self.random_seed_val+img_id)
                    img[:,:,dropped_ch] = np.random.uniform(low=0,high=255,size=(224,224))
                
                elif self.drop_val == "zero":
                    img[:,:,dropped_ch] = 0

                elif self.drop_val == "128":
                    img[:,:,dropped_ch] = 128

        return img, img_gt
    
    
    
    def get_hard_code_gts(self):
        if self.bHardCode:
         
            
        
            print("\n\nOVERRIDING ABOVE! WILL HARD CODE GT CHANNEL ASSOCIATIONS")
            print("Hard code type: {}".format(self.hard_code_type))
            if self.hard_code_type == "easy":
                
                #self.dict_gts_channels[0] = [(19,"left",{'low': 0.12, 'high': 0.89}),(27,"right",{'low': 0.12, 'high': 0.89})]
                #self.dict_gts_channels[1] = [(14,"right",{'low': 0.12, 'high': 0.89}), (21,"left",{'low': 0.12, 'high': 0.89})]
                #self.dict_gts_channels[2] = [(29,"left",{'low': 0.12, 'high': 0.89}),(34,"left",{'low': 0.12, 'high': 0.89})]
                
                self.dict_gts_channels = {0: [(19, 'left', {'low': 0.16, 'high': 0.85}),
                                              (27, 'right', {'low': 0.11, 'high': 0.89})],
                                          1: [(14, 'left', {'low': 0.19, 'high': 0.73}),
                                              (21, 'right', {'low': 0.12, 'high': 0.78})],
                                          2: [(29, 'left', {'low': 0.21, 'high': 0.81}),
                                              (34, 'left', {'low': 0.24, 'high': 0.77})]}
                

                
                #self.dict_gts_channels = {0: [(19, 'left', {'low': 0.16, 'high': 0.85}),
                #                              (27, 'right', {'low': 0.16, 'high': 0.85})],
                #                          1: [(14, 'left', {'low': 0.16, 'high': 0.85}),
                #                              (21, 'right', {'low': 0.16, 'high': 0.85})],
                #                          2: [(29, 'left', {'low': 0.16, 'high': 0.85}),
                #                              (34, 'left', {'low': 0.16, 'high': 0.85})]}
                
                
                
                
            elif self.hard_code_type == "hard":
                
                
                #self.dict_gts_channels[0] = [(19,"left",),(27,"right"),(21,"left")]
                #self.dict_gts_channels[1] = [(14,"right"), (21,"left"), (34,"right")]
                #self.dict_gts_channels[2] = [(29,"left"),(34,"left"),(8,"right")]
                
                
                self.dict_gts_channels = {0: [(19, 'left', {'low': 0.16, 'high': 0.85}),
                                              (27, 'right', {'low': 0.11, 'high': 0.84}),
                                              (21, 'left' , {'low': 0.11, 'high': 0.90})],
                                          1: [(14, 'left', {'low': 0.16, 'high': 0.73}),
                                              (21, 'right', {'low': 0.11, 'high': 0.81}),
                                              (34, 'right', {'low': 0.20, 'high': 0.88})],
                                          2: [(29, 'left', {'low': 0.21, 'high': 0.89}),
                                              (34, 'left', {'low': 0.24, 'high': 0.78}),
                                              (8,  'right', {'low': 0.30, 'high': 0.90})]}
                
                #self.dict_gts_channels = {0: [(19, 'left', {'low': 0.16, 'high': 0.85}),
                #                              (27, 'right', {'low': 0.16, 'high': 0.85}),
                #                              (21, 'left' , {'low': 0.16, 'high': 0.85})],
                #                          1: [(14, 'left', {'low': 0.16, 'high': 0.85}),
                #                              (21, 'right', {'low': 0.16, 'high': 0.85}),
                #                              (34, 'right', {'low': 0.16, 'high': 0.85})],
                #                          2: [(29, 'left', {'low': 0.16, 'high': 0.85}),
                #                              (34, 'left', {'low': 0.16, 'high': 0.85}),
                #                              (8,  'right', {'low': 0.16, 'high': 0.85})]}
                
                self.num_gt_channels = 3
            else:
                print("error!! hard code type not specified but flag is set. Is this what you intended?")
        
    
    def get_ch_dist_vals(self,channel,gt,N, img_id):

        #if channel in gt_channels:
        random.seed(self.random_seed_val+img_id)

        ch_combo_data = [elt for elt in self.dict_gts_channels[gt] if elt[0]==channel ]
        if ch_combo_data:
            ch_dist_shape = ch_combo_data[0][1]
            ch_mode_centers = ch_combo_data[0][2]
        
        else:
            ch_dist_shape = "mid"
            ch_mode_centers = {"low": round(random.uniform(*self.low_interval), 2), 
                                "high": round(random.uniform(*self.high_interval), 2)}
            

        #print("Channel distribution shape: {}, mode_center for ch: {}".format(ch_dist_shape,mode_center) )
#         sig_list = [0.1, 0.15]
        sig_list = [0.4, 0.4]#, 0.3]
        
        # Sampling
        np.random.seed(self.random_seed_val+img_id)
#         print('ch_mode_centers', ch_mode_centers['low'])
#         print('sig_list', sig_list)
#         print('self.mode_heights[ch_dist_shape][0] * N', self.mode_heights[ch_dist_shape][0] * N)
        low_rand = np.random.normal(ch_mode_centers['low'], sig_list[0], int(self.mode_heights[ch_dist_shape][0] * N))
#         print('low_rand', low_rand)
#         print('')
        np.random.seed(self.random_seed_val+img_id)
        high_rand = np.random.normal(ch_mode_centers['high'], sig_list[1], int(self.mode_heights[ch_dist_shape][1] * N))
        X = np.concatenate((low_rand,
                            high_rand))[:, np.newaxis]
        # Clipping outliers (x < 0 or x > 1)
        X = np.clip(X, 0, 1)
        np.random.seed(self.random_seed_val+img_id)
        np.random.shuffle(X)
        X = list(X[:,0])
        
#         print(np.mean(X))
        
        return X
    
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X, class_label = self.get_image(index)
        tensor_imgList = []
        np.random.seed()
        seed = np.random.randint(541474) # make a seed with numpy generator 
        for chId_iter in range(X.shape[2]):
            pil_img = Image.fromarray(X[..., chId_iter].astype(np.uint8))
            random.seed(seed) # apply this seed to img tranfsorms
            tensor_img = self.transforms(pil_img)
            tensor_img = self.normalize(tensor_img)
            tensor_imgList.append(tensor_img)
        in_img = torch.cat(tensor_imgList)
        return in_img, class_label

        
        return X, y 
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ds_gts)
        
    

    