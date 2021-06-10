import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable



class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.q = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.K_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )


        self.out = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        q = self.q(g)
        vx = self.W_x(x)
        keyx = self.K_x(x)
        out = F.softmax(q*keyx)
        out = self.out(out)

        return x*out




def conv_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model



def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model




class conv_block2(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block2,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x





def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool





def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model





class Conv_residual_conv(nn.Module):

    def __init__(self,in_dim,out_dim,act_fn):
        super(Conv_residual_conv,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim,self.out_dim,act_fn)
        self.conv_2 = conv_block_3(self.out_dim,self.out_dim,act_fn)
        self.conv_3 = conv_block(self.out_dim,self.out_dim,act_fn)

    def forward(self,input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class Att_Res_Unet(nn.Module):

    def __init__(self,input_nc, output_nc, ngf):
        super(Att_Res_Unet,self).__init__()
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        print("\n------Initiating FusionNet------\n")

        # encoder

        self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()

        # bridge

        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        # decoder

        self.deconv_1 = conv_trans_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
      #  self.Att1 = Attention_block(F_g=self.out_dim*8, F_l=self.out_dim*8,F_int=self.out_dim*4)
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        #self.up_1 = conv_block2(ch_in=self.out_dim*16, ch_out=self.out_dim*8)

        self.deconv_2 = conv_trans_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
   #     self.Att2 = Attention_block(F_g=self.out_dim*4, F_l=self.out_dim*4,F_int=self.out_dim*2)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        #self.up_2 = conv_block2(ch_in=self.out_dim*8, ch_out=self.out_dim*4)


        self.deconv_3 = conv_trans_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
  #      self.Att3 = Attention_block(F_g=self.out_dim*2, F_l=self.out_dim*2,F_int=self.out_dim)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
      #  self.up_3 = conv_block2(ch_in=self.out_dim*4, ch_out=self.out_dim*2)


        self.deconv_4 = conv_trans_block(self.out_dim * 2, self.out_dim, act_fn_2)
    #    self.Att4 = Attention_block(F_g=self.out_dim, F_l=self.out_dim,F_int=self.out_dim//2)
       self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)
   #     self.up_4 = conv_block2(ch_in=self.out_dim*2, ch_out=self.out_dim)

        # output

        self.out = nn.Conv2d(self.out_dim,self.final_out_dim, kernel_size=1, stride=1, padding=0)


        # initialization

    #    for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            m.weight.data.normal_(0.0, 0.02)
    #            m.bias.data.fill_(0)
            
  #          elif isinstance(m, nn.BatchNorm2d):
   #             m.weight.data.normal_(1.0, 0.02)
    #            m.bias.data.fill_(0)


    def forward(self,input):

        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        down_4 = self.Att1(g=deconv_1, x=down_4)
   #     skip_1 = (deconv_1 + down_4)/2
        skip_1 = torch.cat((down_4,deconv_1),dim=1)
        up_1 = self.up_1(skip_1)

        deconv_2 = self.deconv_2(up_1)
        down_3 = self.Att2(g=deconv_2, x=down_3)
#        skip_2 = (deconv_2 + down_3)/2
        skip_2 = torch.cat((down_3,deconv_2),dim=1)
        up_2 = self.up_2(skip_2)

        deconv_3 = self.deconv_3(up_2)
        down_2 = self.Att3(g=deconv_3, x=down_2)
#        skip_3 = (deconv_3 + down_2)/2
        skip_3 = torch.cat((down_2,deconv_3),dim=1)
        up_3 = self.up_3(skip_3)

        deconv_4 = self.deconv_4(up_3)
        down_1 = self.Att4(g=deconv_4, x=down_1)
    #    skip_4 = (deconv_4 + down_1)/2
        skip_4 = torch.cat((down_1,deconv_4),dim=1)
        up_4 = self.up_4(skip_4)
        out = self.out(up_4)
   #     out = self.out_2(out)

        return out
