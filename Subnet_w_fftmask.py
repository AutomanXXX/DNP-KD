import torch.nn as nn
import torch
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels):
        super(Residual, self).__init__()
        reduced_channels = in_channels // 4  
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=reduced_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=reduced_channels,
                      out_channels=in_channels,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_residual_layers):
        super(ResidualStack, self).__init__()
        self._layers = nn.ModuleList([Residual(in_channels)
                                      for _ in range(num_residual_layers)])
        self.gamma = nn.Parameter(1e-6 * torch.ones((1,in_channels,1,1)), requires_grad=True)
    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        x = self.gamma * x
        return F.relu(x)

class SimpleUnet(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers):
        super(SimpleUnet, self).__init__()
        norm_layer = nn.InstanceNorm2d
        reduced_channels = in_channels // 2

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1),
            norm_layer(reduced_channels),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
            norm_layer(reduced_channels),
            nn.ReLU()
        )

        self.mp1 = nn.Sequential(nn.AvgPool2d(2))

        self.block2 = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
            norm_layer(reduced_channels),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, reduced_channels * 2, kernel_size=3, padding=1),
            norm_layer(reduced_channels * 2),
            nn.ReLU()
        )

        self.mp2 = nn.Sequential(nn.AvgPool2d(2))

        self._residual_stack = ResidualStack(reduced_channels * 2, num_residual_layers)



        self.upblock1 = nn.ConvTranspose2d(in_channels=reduced_channels * 4,
                                           out_channels=reduced_channels,
                                           kernel_size=4,
                                           stride=2, padding=1)

        self.upblock2 = nn.ConvTranspose2d(in_channels=reduced_channels * 2,
                                           out_channels=out_channels,
                                           kernel_size=4,
                                           stride=2, padding=1)




    def forward(self, inputs):
        x = self.block1(inputs)
        b1 = self.mp1(x)
        x = self.block2(b1)
        b2 = self.mp2(x)
        x = self._residual_stack(b2)
        
        x = self.upblock1(torch.cat([x, b2], dim=1))
        x = F.relu(x)
        x = self.upblock2(torch.cat([x, b1], dim=1))
        return x

class MultiSimpleUnet(nn.Module):
    def __init__(self, num_residual_layers=1):
        super(MultiSimpleUnet, self).__init__()
        self.net1 = SimpleUnet(in_channels=256, out_channels=256, num_residual_layers=num_residual_layers)
        self.net2 = SimpleUnet(in_channels=512, out_channels=512, num_residual_layers=num_residual_layers)
        self.net3 = SimpleUnet(in_channels=1024, out_channels=1024, num_residual_layers=num_residual_layers)

    def forward(self, x):
        x1, x2, x3 = x[0], x[1], x[2]
        out1 = self.net1(x1)
        out2 = self.net2(x2)
        out3 = self.net3(x3)
        return [out1, out2, out3]

class KeyValModule(nn.Module):
    def __init__(self, input_dim=1024, num_classe=15):
        super(KeyValModule, self).__init__()
        self.num_classes = num_classe
        self.fc = nn.Linear(input_dim*16*16, num_classe)
        self.keys = torch.eye(num_classe)  # shape: (num_classes, num_classes)
        
    def forward(self, f):
        batch_size = f.size(0)
        f_flat = f.view(batch_size, -1)  # shape: (batch_size, input_dim * 16 * 16)
        predict = self.fc(f_flat)
        key_idx = torch.argmax(predict, dim=1)  

        return predict,key_idx

class FFTmask(nn.Module):

    def __init__(self, num_masks=15):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.mask0 = nn.ParameterList([nn.Parameter(torch.zeros((256, 64, 64)), requires_grad=True) for _ in range(num_masks)])
        self.mask1 = nn.ParameterList([nn.Parameter(torch.zeros((512, 32, 32)), requires_grad=True) for _ in range(num_masks)])
        self.mask2 = nn.ParameterList([nn.Parameter(torch.zeros((1024, 16, 16)), requires_grad=True) for _ in range(num_masks)])


    def forward(self, input, class_label):
        output = []
        last_mask = []
        for j in range(3):
            x = input[j]

            fft_im_ = torch.fft.fft2(x, dim=(-2, -1))
            fft_x_imag = torch.imag(fft_im_)
            fft_x_real = torch.real(fft_im_)
            n,c,h,w = fft_im_.shape
            result = torch.zeros_like(x)
            
            for i in range(n):
                if j ==0:
                    mask = self.mask0[class_label[i]]
                if j ==1:
                    mask = self.mask1[class_label[i]]
                if j ==2:
                    mask = self.mask2[class_label[i]]

                last_tensor_mask = torch.sigmoid(mask)
                x_real = fft_x_real[i] * last_tensor_mask 
                x_imag = fft_x_imag[i] * last_tensor_mask 
                result[i] = torch.complex(x_real, x_imag)
                last_mask.append(last_tensor_mask)

            ifft_x = torch.fft.ifft2(result)
            ifft_x = (torch.real(ifft_x)).float()
            
            output.append(ifft_x)

        return output, last_mask
    
class SubNet_w_FFTMask(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.fftmask =  FFTmask(num_classes)
        self.subnet = MultiSimpleUnet()
        self.key_value = KeyValModule(num_classe=num_classes)

    def forward(self, inputs, train=False):
        if train == True:
            predicts, key_idx = self.key_value(inputs[2].detach())
            inputs_L, masks = self.fftmask(inputs,key_idx.detach())
            out_subnet = self.subnet(inputs_L)
            out_subnet_con = self.subnet(inputs)

            return predicts, masks, out_subnet, out_subnet_con
        
        if train == False:
            _, key_idx = self.key_value(inputs[2].detach())
            inputs_L,_= self.fftmask(inputs,key_idx.detach())
            out_subnet = self.subnet(inputs_L)

            return out_subnet

