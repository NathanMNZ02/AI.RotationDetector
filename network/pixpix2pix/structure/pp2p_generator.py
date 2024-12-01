import torch.nn as nn

class PP2PGenerator(nn.Module):
    NGF = 128 #For high resolution image
    
    def __init__(self, ngpu):
        super(PP2PGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3,
                               out_channels= self.NGF * 8, #Moltiplicazione per 8 usata tipicamente nelle DCGan
                               kernel_size=4, #Dimensione del filtro
                               stride=1,
                               padding=0, 
                               bias=False
                               ),     
            nn.BatchNorm2d(self.NGF * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels = self.NGF * 8,
                    out_channels = self.NGF * 4, 
                    kernel_size = 4, 
                    stride = 2,
                    padding = 1, 
                    bias=False
                    ),     
            nn.BatchNorm2d(self.NGF * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels = self.NGF * 4,
                    out_channels = self.NGF * 2, 
                    kernel_size = 4, 
                    stride = 2,
                    padding = 1, 
                    bias=False
                    ),     
            nn.BatchNorm2d(self.NGF * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels = self.NGF * 2,
                    out_channels = self.NGF, 
                    kernel_size = 4, 
                    stride = 2,
                    padding = 1, 
                    bias=False
                    ),     
            nn.BatchNorm2d(self.NGF),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels = self.NGF,
                    out_channels = 3, #RGB
                    kernel_size = 4, 
                    stride = 2,
                    padding = 1, 
                    bias=False
                    ),     
            nn.Tanh(),
        )
        
    def forward(self, input):
        return self.main(input)