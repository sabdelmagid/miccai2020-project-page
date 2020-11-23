import torch
import torch.nn as nn
import time
import os
import copy
import sys
import numpy as np
import random
import pdb
import cv2
import pickle
from dataloadersetup import create_dataloader


dataset_name = str(sys.argv[1]) #"cycif" "cytometry" 'synthetic'
method = str(sys.argv[2]) #ours-{2d/3d}-{indep/shared}
gt_type = str(sys.argv[3]) #"treatstatus", "grade", "class"

# "cytometry" "ours-2d-shared" "grade" "simplefc"


# Hyperparameters for training
learning_rate = 0.0001
lrrate_str = 'lr_{}'.format(str(learning_rate).split('.')[-1])#"lr_0001"
print("learning_rate: {}".format(learning_rate))
img_size = 224
print("img_size: ", img_size)
num_epochs = 100
print("num_epochs: ", num_epochs)
batch_size = 8*torch.cuda.device_count()
print("Training {} on {} using {}".format(gt_type, dataset_name, method))
# Dataloader
dataset_loader, num_channels, num_classes, phase_list = create_dataloader(
    dataset_name, gt_type, img_size, 
    batch_size=batch_size, recompute_dataset=False)

from ablationmodels import ImageClassificationModel
backbone_arch = method.split('-')[1]
embedding_method = method.split('-')[2]
classifier = "simplefc" if sys.argv[-1] == "simplefc" else "fc"
model_ft = ImageClassificationModel(num_channels, 
                                    num_classes, 
                                    backbone=backbone_arch,
                                    embedding=embedding_method,
                                    classifier=classifier)
print(model_ft)
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.device_count() > 1:
    print("-----------\n\n\n-------------")
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    print("-----------\n\n\n-------------")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model_ft = nn.DataParallel(model_ft)
# Send the model to GPU
model_ft = model_ft.to(device)
# Define loss function
criterion = nn.CrossEntropyLoss()
params_optimize = model_ft.parameters()
optimizer = torch.optim.Adam(
    params_optimize, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, 
    weight_decay=1e-7, amsgrad=True)
# Allows dynamic learning rate reducing based on some validation measurements
exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.65, patience=6, verbose=False, 
    threshold=0.005, threshold_mode='abs', cooldown=3, min_lr=1e-7, eps=1e-08)
# Start training
print(" START TIME: ", time.ctime())
since = time.time()
best_acc = 0
epoch_acc = 0
method_str = "_{}_method_{}_gttype_{}".format(dataset_name, method, gt_type)
bestModel_path = './best_models_roi/temp_model' + method_str + "_" + lrrate_str
bestState_path = './best_models_roi/temp_state' + method_str + "_" + lrrate_str
# Iterate until predefined epoches
for epoch in range(num_epochs):
    # dataset_loader
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    # Each epoch has a training and validation phase
    for phase in phase_list[:2]:
        print(phase)
        running_loss = 0.0
        running_corrects = 0.0
        running_num = 0.0
        if phase == 'train':
            exp_lr_scheduler.step(epoch_acc)
            model_ft.train()  # Set model to training mode
        else:
            model_ft.eval()   # Set model to evaluate mode
        # Iterate over data.
        print("Iterate over data")
        dataloader= iter(dataset_loader[phase])
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model_ft(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            running_num += inputs.size(0)
        print('epoch!')
        epoch_loss = running_loss / running_num
        epoch_acc = running_corrects.double() / running_num
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
        dict_key = "epoch:{}-phase:{}".format(epoch, phase)
        if 'valid' in phase_list:
            test_phase = 'valid'
        else:
            test_phase = 'test'
        if phase == test_phase and epoch_acc > best_acc:
            best_acc = epoch_acc
            if os.path.exists('./best_models_roi/') == False:
                os.mkdir('./best_models_roi/')
            if os.path.exists(bestModel_path):
                os.remove(bestModel_path)
            bestModel_path = './best_models_roi/best_model_' + str(epoch) + method_str+ "_" + lrrate_str + '.pt'
            if torch.cuda.device_count() > 1:
                torch.save(model_ft.module, bestModel_path)
            else:
                torch.save(model_ft, bestModel_path)
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
# Inference on the test dataset
model_ft = torch.load(bestModel_path)
phase = 'test'
print(phase)
running_loss = 0.0
running_corrects = 0.0
running_num = 0.0
model_ft.eval()   # Set model to evaluate mode
# Iterate over data.
print("Iterate over data")
dataloader= iter(dataset_loader[phase])
for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    # zero the parameter gradients
    optimizer.zero_grad()
    # track history if only in train
    with torch.set_grad_enabled(phase == 'train'):
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
    # statistics
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)
    running_num += inputs.size(0)
epoch_loss = running_loss / running_num
epoch_acc = running_corrects.double() / running_num
print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
