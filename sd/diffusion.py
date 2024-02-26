import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4*n_embed)
        self.linear_2 = nn.Linear(4*n_embed, 4*n_embed)

    def forward(self, x):
        #x: (1, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        #x: (1, 1280)
        return x
    
class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                #compute cross attention between image and prompt
                x = layer(x, context)

            elif isinstance(layer, UNET_residualBlock):
                x = layer(x, time)
            
            else: 
                x = layer(x)
        return x
    
class UNET_residualBlock(nn.Module):
    #relate the time embedding with the latent

    def __init__(self, in_channels, out_channels, n_time = 1280):
        super().__init__()

        self.grounorm_features = nn.GroupNorm(32, in_channels)
        self.conv_features = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        #feature: latent (batch_size, in_channels, height, width)
        #time (1, 1280)

        residual = feature
        feature = self.grounorm_features(feature)
        feature = F.silu(feature)

        feature = self.conv_features(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        return merged + self.residual_layer(residual)
    

class UNET_AttentionBlock(nn.Module):

    def __init__(self, n_head, n_embed, d_context=768):
        super().__init__()
        channels = n_head * n_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_gelu_1 = nn.Linear(channels, 4*channels*2)
        self.linear_gelu_2 = nn.Linear(4*channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x: batch_size, features, height, width
        # context: batch_size, seq_len, dim

        residual_long = x
        x = self.groupnorm(x)
        #does not change the size
        x = self.conv_input(x)

        b, c, h, w = x.shape 

        #Normailization + self attention with skip connection

        x = x.view((b, c, h*w))

        #batch_size, height*width, features
        x = x.transpose(-1, -2)

        residual_short = x

        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residual_short

        #normalization + cross attention with skip connection
        x = self.layernorm_2(x)

        #cross attention
        self.attention_2(x, context)
        x += residual_short

        residual_short = x

        #noramailization + ff + skip connection

        x = self.layernorm_3(x)

        x, gate = self.attention_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_gelu_2(x)
        x += residual_short

        #batch_size, feature, height*width
        x = x.transpose(-1, -2)

        x = x.view((b, c, h, w))

        return self.conv_output(x) + residual_long






class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        #x: bacth_size, features, height, width --> batch_size, features, height*2, width*2
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            #batch_size, 4, height /8, width/8
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_residualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_residualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            #reduce the size of the image
            #batch_size, 320, height /16, width/16
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_residualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_residualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            #batch_size, 640, height /32, width/32
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            #increase the features 
            SwitchSequential(UNET_residualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_residualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            #batch_size, 1280, height /64, width/64
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_residualBlock(1280, 1280)),
            SwitchSequential(UNET_residualBlock(1280, 1280)),
        ])

        self.bottle_neck = SwitchSequential(
            UNET_residualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_residualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            #skip connection -> must be 1280 * 2 = 2560
            #batch_size, 2560, height /64, width/64
            SwitchSequential(UNET_residualBlock(2560, 1280)),
            SwitchSequential(UNET_residualBlock(2560, 1280)),
            SwitchSequential(UNET_residualBlock(2560, 1280), UpSample(1280)),

            SwitchSequential(UNET_residualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_residualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_residualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNET_residualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_residualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_residualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNET_residualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_residualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_residualBlock(640, 320), UNET_AttentionBlock(8, 40)),

        ])
    def forward(self, x, context, time):
        #x : batch_size, 4, height/8, width/8
        #context: batch_size, seq_len, dim
        #time: (1, 1280)
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottle_neck(x, context, time)

        for layers in self.decoders:
            #since we always concat with the skip connection of the encoder , the 
            #number of features increase before being set to the decoder's layer 
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = x.layers(x, context, time)

        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        #batch_size, 320, height/8, width/8
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)

        #batch_size, 4, height/8, width/8
        return x

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent, context, time):
        #latent: batch_size, 4, height/8, width/8
        #context: batch_size, seq_len, dim 
        #time: (1, 320)

        #like positional encodding in transformer
        #(1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        #batch_size, 320, height/8, width/8 
        output = self.unet(latent, context, time)

        #batch_size, 4, height/8, width/8
        output = self.final(output)

        return output
        
