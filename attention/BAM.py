import torch
import torch.nn.functional as F
from torch import nn

'''
Global context feature aggregation module.
'''
class BAM(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in // 8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        # Q K V
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, input):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # todo 图片上没有AvgPool2d
        x = self.pool(input)
        m_batchsize, C, width, height = x.size()

        '''
        # 创造二维数据x，dim=0时候2，dim=1时候3
        x = torch.randn(2,3)       'x.shape  →  [2,3]'
        # 创造三维数据y，dim=0时候2，dim=1时候3，dim=2时候4
        y = torch.randn(2,3,4)   'y.shape  →  [2,3,4]'
        # 对于transpose
        x.transpose(0,1)     'shape→[3,2] '  
        x.transpose(1,0)     'shape→[3,2] '  
        y.transpose(0,1)     'shape→[3,2,4]' 
        y.transpose(0,2,1)  'error，操作不了多维'
        
        # 对于permute()
        x.permute(0,1)     'shape→[2,3]'
        x.permute(1,0)     'shape→[3,2], 注意返回的shape不同于x.transpose(1,0) '
        y.permute(0,1)     "error 没有传入所有维度数"
        y.permute(1,0,2)  'shape→[3,2,4]'
        '''
        # B X C X (N)/(ds*ds) permute(0, 2, 1)
        # 转置第二三维度
        # N = width * height
        # this proj is Q' in the paper
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # B X C x (*W*H)/(ds*ds)
        # this proj is K' in the paper
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        '''
        transpose check
        torch.bmm(input, mat2, out=None) → Tensor
        torch.bmm()是tensor中的一个相乘操作，类似于矩阵中的A*B。
        '''
        energy = torch.bmm(proj_query, proj_key)
        # todo ?????? self.key_channel = self.chanel_in // 8
        energy = (self.key_channel ** -.5) * energy
        # BX (N) X (N)/(ds*ds)/(ds*ds)
        attention = self.softmax(energy)
        # this proj is V' in the paper
        # B X C X N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        # F in the paper
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # F' in the paper
        out = out.view(m_batchsize, C, width, height)
        # interpolate 上采样
        """
        input(Tensor)：需要进行采样处理的数组。
        size(int或序列)：输出空间的大小
        scale_factor(float或序列)：空间大小的乘数
        mode(str)：用于采样的算法。'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'。默认：'nearest'
        align_corners(bool)：在几何上，我们将输入和输出的像素视为正方形而不是点。如果设置为True，则输入和输出张量按其角像素的中心点对齐，保留角像素处的值。
            如果设置为False，则输入和输出张量通过其角像素的角点对齐，并且插值使用边缘值填充用于边界外值，使此操作在保持不变时独立于输入大小scale_factor。
        recompute_scale_facto(bool)：重新计算用于插值计算的 scale_factor。当scale_factor作为参数传递时，它用于计算output_size。如果recompute_scale_factor的False或没有指定，
            传入的scale_factor将在插值计算中使用。否则，将根据用于插值计算的输出和输入大小计算新的scale_factor（即，如果计算的output_size显式传入，则计算将相同 ）。
            注意当scale_factor 是浮点数，由于舍入和精度问题，重新计算的 scale_factor 可能与传入的不同。
        """
        # 这里应该是进行的reshape    但用的是上采样的方法 N= width * height
        out = F.interpolate(out, [width * self.ds, height * self.ds])
        out = out + input

        return out
