import torch
import torch.nn as nn
import torch.optim as optim

from structure.pp2p_generator import PP2PGenerator
from structure.pp2p_discriminator import PP2PDiscriminator
from helper.config import DEVICE

class Trainer:
    def __init__(self, ngpu):
        self.pp2p_gen = PP2PGenerator(ngpu).to(DEVICE)
        self.pp2p_des = PP2PDiscriminator(ngpu).to(DEVICE)
        
        # If use multi gpu
        if (DEVICE.type == 'cuda') and (ngpu > 1):
            self.pp2p_gen = nn.DataParallel(self.pp2p_gen, list(range(ngpu))) 
            self.pp2p_des = nn.DataParallel(self.pp2p_des, list(range(ngpu)))
        
        self.pp2p_gen.apply(self.__weights_init__)
        self.pp2p_des.apply(self.__weights_init__)
    
    def __weights_init__(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0) 
            
    def start(self):
        loss = nn.BCELoss()
        fixed_noise = torch.randn(64, 3, 1, 1, device=DEVICE)
        
        real_label = 1.
        fake_label = 0.

        optimizerD = optim.Adam(self.pp2p_gen.parameters(), lr=0.0001, betas=(beta1, 0.999))
        optimizerG = optim.Adam(self.pp2p_des.parameters(), lr=0.0001, betas=(beta1, 0.999))