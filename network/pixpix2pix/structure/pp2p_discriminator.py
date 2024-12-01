import torch.nn as nn

class PP2PDiscriminator(nn.Module):
    NDF = 128
    
    def __init__(self, ngpu):
        super(PP2PDiscriminator).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=self.NDF, #Numero di filtri
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False             
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.NDF,
                      out_channels=self.NDF * 2, #Numero di filtri
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False             
            ),
            nn.BatchNorm2d(self.NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.NDF * 2,
                      out_channels=self.NDF * 4, #Numero di filtri
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False             
            ),
            nn.BatchNorm2d(self.NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.NDF * 4,
                      out_channels=self.NDF * 8, #Numero di filtri
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False             
            ),
            nn.BatchNorm2d(self.NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.NDF * 8,
                      out_channels=1, #Numero di filtri
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False             
            ),
            nn.Sigmoid()        
        )

    def forward(self, input):
        return self.main(input)