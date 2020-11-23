import torch
import os
import pickle


def create_dataloader(dataset_name, gt_type, img_size, 
                      batch_size=8, recompute_dataset=False):
    
    if dataset_name == "cytometry":
        return create_cytometry(gt_type, img_size, batch_size, recompute_dataset)
    elif dataset_name == 'synthetic':
        return create_synthetic(batch_size)
    else:
        print('wrong dataset_name')        



def create_cytometry(gt_type, img_size, batch_size, recompute_dataset):
    from cytometrydataloader import BioformatTiffreader, DataPreparation
    num_classes = 3 if gt_type == "grade" else 5
    # Path to the prepared data
    prepared_dataPath = 'prepared_data_files/prepared_data_{}.pkl'.format(gt_type)
    channels_toExclude = ['not_an_antigen']
    if recompute_dataset | (os.path.exists(prepared_dataPath) == False):
        # define a class for data preparation
        prepared_data = DataPreparation(gt_type=gt_type_arg)
        # resplit the dataset into train/test
        prepared_data.split_dataset()
        # recompute list of channels
        prepared_data.get_channelList(channels_toExclude)
        # recompute percentile
        prepared_data.find_percentile()
        # Save the data preparation obj
        with open(prepared_dataPath, 'wb') as pkl_file:
            pickle.dump(prepared_data, pkl_file)
    else:
        # load the data preparation obj
        with open(prepared_dataPath, 'rb') as pkl_file:
            prepared_data = pickle.load(pkl_file)
    dataset_train = BioformatTiffreader(prepared_data,
                                        img_size=img_size,
                                        phase_str='train')
    dataset_test = BioformatTiffreader(prepared_data,
                                       img_size=img_size,
                                       phase_str='test')
    dataset_loader = dict()
    dataset_loader['train'] = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=8)
    dataset_loader['test'] = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=8)
    num_channels = len(dataset_train.valid_channelList)
    phase_list = ['train', 'test']
    return dataset_loader, num_channels, num_classes, phase_list

def create_synthetic(batch_size):
    from syntheticdataloader import SyntheticData
    num_classes = 3
    num_channels = 30
    dataset_train = SyntheticData(num_images=600, phase='train')
    dataset_valid = SyntheticData(num_images=300, phase='valid')
    dataset_test = SyntheticData(num_images=300, phase='test')
    dataset_loader = dict()
    dataset_loader['train'] = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=batch_size, shuffle=True, 
        drop_last=True, num_workers=8)
    dataset_loader['valid'] = torch.utils.data.DataLoader(
        dataset=dataset_valid, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=8)
    dataset_loader['test'] = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=8)
    phase_list = ['train', 'valid', 'test']
    return dataset_loader, num_channels, num_classes, phase_list
