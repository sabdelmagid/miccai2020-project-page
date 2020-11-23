import torch
import torch.nn as nn
import torchvision.models as models
from module_interpretable_classifier import InterpretableClassifier


class EmbeddingEncoder(nn.Module):
    def __init__(self, num_channels, num_classes, backbone, embedding):
        super(EmbeddingEncoder, self).__init__()
        self.backbone = backbone
        self.embedding = embedding
        out_dim = 1
        # Backbone encoder
        if self.backbone == '2d':
            # ResNet18
            base_layers = models.resnet18(pretrained=True)
            base_layers.conv1 = nn.Conv2d(1, 64, 
                                          kernel_size=(7, 7), 
                                          stride=(2, 2), 
                                          padding=(3, 3), 
                                          bias=False)
            backbone_list = list(base_layers.children())[:-2]
            self.backbone_convs = nn.Sequential(*backbone_list)
            # Embedding encoder
            if self.embedding == 'shared':
                layer_name = 'shared_layers'
                chwise_layers = []
                chwise_layers.append(nn.Conv2d(512, 64, kernel_size=(3, 3), 
                                               padding=(1, 1)))
                chwise_layers.append(nn.BatchNorm2d(64))
                chwise_layers.append(nn.ReLU())
                chwise_layers.append(nn.Conv2d(64, 64, kernel_size=(3, 3), 
                                               padding=(1, 1)))
                chwise_layers.append(nn.BatchNorm2d(64))
                chwise_layers.append(nn.ReLU())
                chwise_layers.append(nn.Conv2d(64, 64, kernel_size=(3, 3), 
                                               padding=(1, 1)))
                chwise_layers.append(nn.BatchNorm2d(64))
                chwise_layers.append(nn.ReLU())
                chwise_layers.append(nn.Conv2d(64, out_dim, 
                                               kernel_size=(7, 7)))
                #equivalent to: self.varname= 'something'
                setattr(self, layer_name, nn.Sequential(*chwise_layers)) 
            elif self.embedding == 'indep':
                for channel_id in range(num_channels):
                    layer_name = 'channelwise_layers{}'.format(channel_id)
                    chwise_layers = []
                    chwise_layers.append(nn.Conv2d(512, 64, kernel_size=(3, 3),
                                                   padding=(1, 1)))
                    chwise_layers.append(nn.BatchNorm2d(64))
                    chwise_layers.append(nn.ReLU())
                    chwise_layers.append(nn.Conv2d(64, 64, kernel_size=(3, 3),
                                                   padding=(1, 1)))
                    chwise_layers.append(nn.BatchNorm2d(64))
                    chwise_layers.append(nn.ReLU())
                    chwise_layers.append(nn.Conv2d(64, 64, kernel_size=(3, 3),
                                                   padding=(1, 1)))
                    chwise_layers.append(nn.BatchNorm2d(64))
                    chwise_layers.append(nn.ReLU())
                    chwise_layers.append(nn.Conv2d(64, out_dim, 
                                                   kernel_size=(7, 7)))
                    #equivalent to: self.varname= 'something'
                    setattr(self, layer_name, nn.Sequential(*chwise_layers)) 
        elif self.backbone == '3d':
            # ResNet3D-18
            base_layers = models.video.r3d_18(pretrained=True)
            base_layers.stem[0] = nn.Conv3d(1, 64, 
                                            kernel_size=(3, 7, 7), 
                                            stride=(1, 2, 2), 
                                            padding=(1, 3, 3), 
                                            bias=False)
            backbone_list = list(base_layers.children())[:-5]
            self.backbone_convs = nn.Sequential(*backbone_list)
            # Embedding encoder
            if self.embedding == 'shared':
                layer_name = 'shared_layers'
                chwise_layers = []
                chwise_layers = chwise_layers + \
                                self.__create_convBlock__(64, 64) # 112 -> 56
                chwise_layers = chwise_layers + \
                                self.__create_convBlock__(64, 64) # 56 -> 28
                chwise_layers = chwise_layers + \
                                self.__create_convBlock__(64, 64) # 28 -> 14
                chwise_layers = chwise_layers + \
                                self.__create_convBlock__(64, 64) # 14 -> 7
                chwise_layers.append(nn.Conv2d(64, out_dim, 
                                               kernel_size=(7, 7)))
                #equivalent to: self.varname= 'something'
                setattr(self, layer_name, nn.Sequential(*chwise_layers))
            elif self.embedding == 'indep':
                for channel_id in range(num_channels):
                    layer_name = 'channelwise_layers{}'.format(channel_id)
                    chwise_layers = []
                    chwise_layers = chwise_layers + \
                                    self.__create_convBlock__(64, 64) 
                                    # 112 -> 56
                    chwise_layers = chwise_layers + \
                                    self.__create_convBlock__(64, 64) 
                                    # 56 -> 28
                    chwise_layers = chwise_layers + \
                                    self.__create_convBlock__(64, 64) 
                                    # 28 -> 14
                    chwise_layers = chwise_layers + \
                                    self.__create_convBlock__(64, 64) 
                                    # 14 -> 7
                    chwise_layers.append(nn.Conv2d(64, out_dim, 
                                                   kernel_size=(7, 7)))
                    #equivalent to: self.varname= 'something'
                    setattr(self, layer_name, nn.Sequential(*chwise_layers)) 
        else:
            print('please define the correct backbone')
    
    def __create_convBlock__(self, in_dim, out_dim):
        conv_block = []
        conv_block.append(nn.MaxPool2d(2))
        conv_block.append(nn.Conv2d(in_dim, out_dim, 
                                    kernel_size=(3, 3), padding=(1, 1)))
        conv_block.append(nn.BatchNorm2d(out_dim))
        conv_block.append(nn.ReLU())
        conv_block.append(nn.Conv2d(out_dim, out_dim, 
                                    kernel_size=(3, 3), padding=(1, 1)))
        conv_block.append(nn.BatchNorm2d(out_dim))
        conv_block.append(nn.ReLU())
        return conv_block

    def forward(self, x):
        embedding_list = []
        if self.backbone == '3d':
            backbone_out = self.backbone_convs(
                x.view(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]))
                # B, 64, C, 112, 112
            for channel_id in range(backbone_out.shape[2]):
                if self.embedding == 'indep':
                    layer_name = 'channelwise_layers{}'.format(channel_id)
                else:
                    layer_name = 'shared_layers'
                embedding_convs = getattr(self, layer_name)
                embedding_out = embedding_convs(backbone_out[:, :, channel_id])
                embedding_out = embedding_out.view(embedding_out.shape[0], 1)
                embedding_list.append(embedding_out)
        else:
            for channel_id in range(x.shape[1]):
                backbone_out = self.backbone_convs(
                    x[:, channel_id].view(x.shape[0], 1, 
                                          x.shape[2], x.shape[3]))
                if self.embedding == 'indep':
                    layer_name = 'channelwise_layers{}'.format(channel_id)
                else:
                    layer_name = 'shared_layers'
                embedding_convs = getattr(self, layer_name)
                embedding_out = embedding_convs(backbone_out)
                embedding_out = embedding_out.view(embedding_out.shape[0], 1)
                embedding_list.append(embedding_out)
        ipret_embed = torch.cat(embedding_list, dim=1)
        return ipret_embed
    
class ImageClassificationModel(nn.Module):
    def __init__(self, num_channels, num_classes, 
                 backbone='2d', embedding='shared',classifier='fc'):
        super(ImageClassificationModel, self).__init__()
        self.embedding_encoder = EmbeddingEncoder(
            num_channels, num_classes, backbone, embedding)
        self.ipret_clasifier = InterpretableClassifier(
            num_channels=num_channels, num_classes=num_classes,classifier=classifier)
    def forward(self, input_data):
        ipret_rep = self.embedding_encoder(input_data)
        pred_out = self.ipret_clasifier(ipret_rep)
        return pred_out
