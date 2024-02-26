import torch 
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            #batch_size , channel, height, width -> batch_size, 128, height, width
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            
            #batch_size, 128, height, width -> batch_size, 128, height, width
            VAE_ResidualBlock(128, 128),

            #batch_size, 128, height, width -> batch_size, 128, height/2, width/2
            VAE_ResidualBlock(128, 128),

            #batch_size, 128, height, width -> batch_size, 128, height/2, width/2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding = 0),

            #batch_size, 128, height/2, width/2 -> batch_size, 256, height/4, width/4
            #just increase the feature of the image, each pixel represent more information
            #but the number of pixel is reduced
            VAE_ResidualBlock(128, 256),

            VAE_ResidualBlock(256,256),

            #batch_size, 256, height/2, width/2 -> batch_size, 256, height/4, width/4
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding = 0),

            #batch_size, 256, height/4, width/4 -> batch_size, 512, height/4, width/4
            VAE_ResidualBlock(256, 512),

            #batch_size, 512, height/4, width/4 -> batch_size, 512, height/4, width/4
            VAE_ResidualBlock(512, 512),

            #batch_size, 512, height/4, width/4 -> batch_size, 512, height/8, width/8
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding = 0),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            #self-attention: pixel attend to each other globally
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(num_groups=32, num_channels=512),

            nn.SiLU(),

            #bottle beck of the encoder
            #batch_size, 512, height/8, width/8 -> batch_size, 8, height/8, width/8
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        #x: batch_size, channel, height, width 
        #noise: batch_size, out_channel, height/8, width/8
        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                #padding_left, padding_right, padding_top, padding_bottom
                x = F.pad(x, (0,1,0,1))
            x = module(x)

        # batch_size, 8, height/8, width/8 -> 2 tensors of size
        # batch_size, 4, height/8, width/8
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        #before exp^log_variace, clamp log_variance to avoid numerical instability
        log_variance = torch.clamp(log_variance, min=-30, max=20)

        variance = log_variance.exp()

        stdev = variance.sqrt()

        #sample from the normal distribution 
        #Z = N(0, 1) -> N(mean, variance) = X ?
        # X = mean + Z*stdev
        x = mean + stdev * noise

        #scale the output by a constant 
        x *= 0.18215
        return x
