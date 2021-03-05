import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel =3):
        super(Encoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        
        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )
        
        self.moduleConv1 = Basic(n_channel*(t_length-1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)
        
    def forward(self, x):

        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        
        return tensorConv4, tensorConv1, tensorConv2, tensorConv3


class Decoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel =3):
        super(Decoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
                
        
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels = nc, out_channels=intOutput, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
      
        self.moduleConv = Basic(1024, 512)
        # self.moduleConv = Basic(512, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(128,n_channel,64)

    def forward(self, x, skip1, skip2, skip3):
        
        tensorConv = self.moduleConv(x)
        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = torch.cat((skip3, tensorUpsample4), dim = 1)
        
        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim = 1)
        
        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim = 1)
        
        output = self.moduleDeconv1(cat2)

        return output
    

class PreAE(torch.nn.Module):
    def __init__(self, n_channel =3,  t_length = 5, memory_size = 10, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1):
        super(PreAE, self).__init__()

        self.encoder = Encoder(t_length, n_channel)
        self.decoder = Decoder(t_length, n_channel)

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        fea, skip1, skip2, skip3 = self.encoder(x)
        output = self.decoder(fea, skip1, skip2, skip3)

        return output


class PreAEMemory(torch.nn.Module):
    def __init__(self, image_channel =3, t_length = 5, image_height = 256, image_width = 256, item_num_w = 8, item_num_h = 8):
        super(PreAEMemory, self).__init__()

        self.encoder = Encoder(t_length, image_channel)
        self.decoder = Decoder(t_length, image_channel)
        self.image_height = image_height
        self.image_width = image_width
        # self.items = None
        self.item_num_w = item_num_w
        self.item_num_h = item_num_h

    def forward(self, x, bbox, items):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        fea, skip1, skip2, skip3 = self.encoder(x)

        # if self.items is None:
        #     self.items = F.normalize(
        #         torch.rand((self.item_num_w * self.item_num_h, fea.shape[1], fea.shape[2], fea.shape[3]), dtype=torch.float),
        #         dim=1).cuda()  # Initialize the memory items

        fea = F.normalize(fea, dim=1)

        index_w = (bbox[0] + bbox[2]) // (2 * self.image_width / self.item_num_w)
        index_h = (bbox[1] + bbox[3]) // (2 * self.image_height / self.item_num_h)

        index = int(index_h * self.item_num_w + index_w)

        new_fea = self.read(fea,items, index)
        items = self.update(fea,items.clone(), index)
        output = self.decoder(new_fea, skip1, skip2, skip3)
        # if isTrain:
        #     self.items[index] = new_item[0]
        return output, items

    def read(self, fea, items, index):
        item = items[index].unsqueeze(0)
        new_fea = torch.cat((fea, item), dim=1)
        return new_fea

    def update(self, fea, items, index):
        cof = 0.01
        new_item = F.normalize(cof * items[index] + (1 - cof) * fea, dim=1)
        items[index] = new_item[0]
        return items
