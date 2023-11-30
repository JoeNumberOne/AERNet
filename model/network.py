import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader

from attention.BAM import BAM
from attention.coordatt import CoordAtt
from model.DWConv import DWConv
from model.MyDataset import MyDataSet
from model.SelfAdaptiveWeightedBCE import SelfAdaptiveWeightedBCE


class zh_net(nn.Module):
    def __init__(self, freeze_bn=False):
        super(zh_net, self).__init__()
        self.encoder = resnet34()  # 在此处可切换backbone
        self.decoder = Decoder()

        if freeze_bn:
            self.freeze_bn()

    def forward(self, A, B):
        # 此处用的同一个encoder来共享权重
        output1 = self.encoder(A)
        output2 = self.encoder(B)
        # 解码块
        result = self.decoder(output1, output2)
        print("al1:{}".format(result[0].shape))
        print("al2:{}".format(result[1].shape))
        print("al3:{}".format(result[2].shape))
        print("al4:{}".format(result[3].shape))
        print("result:{}".format(result[4].shape))
        print("seg:{}".format(result[5].shape))
        return result

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', }


# 基础块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


# 解释一下这段代码的作用
#
class ResNet(nn.Module):
    def __init__(self, block, layers):
        #
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # 64 代表的是feature的数量
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, model_path):
        pretrain_dict = model_zoo.load_url(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 记录下来的feature之后往解码块传递
        feature = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feature.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feature.append(x)
        x = self.layer2(x)
        feature.append(x)
        x = self.layer3(x)
        feature.append(x)
        x = self.layer4(x)
        feature.append(x)
        return feature


def resnet34(pretrained=True):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model._load_pretrained_model(model_urls['resnet34'])
    return model


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block, self).__init__()

        self.de_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.de_block2 = DWConv(out_channels, out_channels)

        self.att = CoordAtt(out_channels, out_channels)

        self.de_block3 = DWConv(out_channels, out_channels)

        self.de_block4 = nn.Conv2d(out_channels, 1, 1)

        # 上采样
        self.de_block5 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, input1, input, input2):
        # 将三个输入的特征图进行拼接
        x0 = torch.cat((input1, input, input2), dim=1)
        # Enhanced Coordinate attention (ECA) Attention decoding block.
        # 1*1卷积
        x0 = self.de_block1(x0)
        # todo 可能是DSConv
        x = self.de_block2(x0)
        # CA Coordinate attention.
        x = self.att(x)
        # todo 可能是DSConv
        x = self.de_block3(x)
        # Elemeny-wise sum
        x = x + x0
        # 1*1卷积 al为Deep Supervision Flow( DS )用于辅助反向传播
        #  todo 图中并没有标注要进行1*1卷积进行通道的变换,
        al = self.de_block4(x)
        # 上采样
        result = self.de_block5(x)
        print("al:{}".format(al.shape))
        print("result:{}".format(result.shape))

        return al, result


class ref_seg(nn.Module):
    def __init__(self):
        super(ref_seg, self).__init__()
        # this is the damn D method in the paper. but no softmax
        self.dir_head = nn.Sequential(nn.Conv2d(32, 32, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 8, 1, 1))
        self.conv0 = nn.Conv2d(1, 8, 3, 1, 1, bias=False)
        self.conv0.weight = nn.Parameter(torch.tensor([[[[0, 0, 0], [1, 0, 0], [0, 0, 0]]],
                                                       [[[1, 0, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 1, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 0, 1], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 0, 0], [0, 0, 1], [0, 0, 0]]],
                                                       [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]],
                                                       [[[0, 0, 0], [0, 0, 0], [0, 1, 0]]],
                                                       [[[0, 0, 0], [0, 0, 0], [1, 0, 0]]]]).float())

    # (x, S, E)
    def forward(self, x, masks_pred, edge_pred):
        # this is the damn D method in the paper (Direction Prediction). but no softmax
        direc_pred = self.dir_head(x)
        # this is the D ,D = ρ(F_classifier2(X))
        direc_pred = direc_pred.softmax(1)
        # this is the E ,E = σ(F_classifier1(X)) > 0.5
        edge_mask = 1 * (torch.sigmoid(edge_pred).detach() > 0.5)
        '''
        x.unsqueeze(dim=a)
        用途：进行维度扩充，在指定位置加上维数为1的维度
        
        参数设置：如果设置dim=a，就是在维度为a的位置进行扩充
            import torch
            x = torch.tensor([1,2,3,4])
            print(x)
            x1 = x.unsqueeze(dim=0)
            print(x1)
            x2 = x.unsqueeze(dim=1)
            print(x2)
             
            y = torch.tensor([[1,2,3,4],[9,8,7,6]])
            print(y)
            y1 = y.unsqueeze(dim=0)
            print(y1)
            y2 = y.unsqueeze(dim=1)
            print(y2)
            
            
            output:
            x: tensor([1, 2, 3, 4])
            x1: tensor([[1, 2, 3, 4]])
            x2: tensor([[1],
                    [2],
                    [3],
                    [4]])
            y: tensor([[1, 2, 3, 4],
                    [9, 8, 7, 6]])
            y1: tensor([[[1, 2, 3, 4],
                     [9, 8, 7, 6]]])
            y2: tensor([[[1, 2, 3, 4]],
             
                    [[9, 8, 7, 6]]])
        '''
        # unsqueeze(1)在第二维度上增加一个维度
        # S 经过一个固定的卷积层 * D ,然后在第一维度相加(Element-wise sum) 然后乘以E，最后加上S*(1-E)
        # Z = R × E + S × (1 − E).
        refined_mask_pred = (self.conv0(masks_pred) * direc_pred).sum(1).unsqueeze(1) * edge_mask + masks_pred * (
                1 - edge_mask)
        return refined_mask_pred


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # todo 1024 = encoder（A）+ encoder（B）???
        # BAM is used to fuse the features of two images
        # Global context feature aggregation module.
        self.bam = BAM(1024)

        self.db1 = nn.Sequential(
            # 1*1卷积
            nn.Conv2d(1024, 512, 1), nn.BatchNorm2d(512), nn.ReLU(),
            # todo DWConv是什么 图上没有 可能也是它上采样的一部分
            DWConv(512, 512),
            # ConvTranspose2d是反卷积 上采样
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        )
        # CDN-1
        self.db2 = decoder_block(1024, 256)
        # CDN-2
        self.db3 = decoder_block(512, 128)
        # CDN-3
        self.db4 = decoder_block(256, 64)
        # CDN-4
        self.db5 = decoder_block(192, 32)

        # classifier1 For E but no softmax
        self.classifier1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))
        # classifier2 For D but no softmax , +1 is because the concat
        self.classifier2 = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))
        self.interpo = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine = ref_seg()
        self._init_weight()

    def forward(self, input1, input2):
        input1_1, input2_1, input3_1, input4_1, input5_1 = input1[0], input1[1], input1[2], input1[3], input1[4]
        input1_2, input2_2, input3_2, input4_2, input5_2 = input2[0], input2[1], input2[2], input2[3], input2[4]

        # GCFAM之前的特征融合
        x = torch.cat((input5_1, input5_2), dim=1)
        # GCFAM
        x = self.bam(x)
        x = self.db1(x)

        # 512*16*16
        # Enhanced Coordinate attention (ECA) Attention decoding block.
        # CDN-1 256*32*32
        al1, x = self.db2(input4_1, x, input4_2)
        # CDN-2 128*64*64
        al2, x = self.db3(input3_1, x, input3_2)
        # CDN-3 64*128*128
        al3, x = self.db4(input2_1, x, input2_2)
        # CDN-4 32*256*256
        al4, x = self.db5(input1_1, x, input1_2)

        # Edge refinement module.
        # edge is referred to E in the paper (Edge Prediction). but no sigmoid
        edge = self.classifier1(x)
        # DS4 and x concat, this classifier is the classifier3 in the paper(Coarse Segmentation). and seg is referred to S in the paper.
        seg = self.classifier2(torch.cat((x, self.interpo(al4)), 1))
        # the result is Z and refine is Edge refinement module
        result = self.refine(x, seg, edge)
        # ADB1, ADB2, ADB3, ADB4, Z, S
        return al1, al2, al3, al4, result, seg

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# if __name__ == '__main__':
#     test_data1 = torch.rand(2, 3, 256, 256).cuda()
#     test_data2 = torch.rand(2, 3, 256, 256).cuda()
#     test_label = torch.randint(0, 2, (2, 1, 256, 256)).cuda()
#     model = zh_net()
#     model = model.cuda()
#     output = model(test_data1, test_data2)
#     print(output.shape)


# def manageAltoDS(output):
#     # 用来处理al(ADBi) 1-4 生成 DS 1-4 用于辅助反向传播
#     al1, al2, al3, al4, result, seg = output
#
#     pass

class ManageAltoDS(nn.Module):

    def __init__(self):
        super(ManageAltoDS, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # al1:torch.Size([24, 1, 16, 16])
        # al2:torch.Size([24, 1, 32, 32])
        # al3:torch.Size([24, 1, 64, 64])
        # al4:torch.Size([24, 1, 128, 128])
        # result:torch.Size([24, 1, 256, 256])
        # seg:torch.Size([24, 1, 256, 256])
        self.conv1 = nn.Conv2d(16, 256, 1)
        self.conv2 = nn.Conv2d(32, 256, 1)
        self.conv3 = nn.Conv2d(64, 256, 1)
        self.conv4 = nn.Conv2d(128, 256, 1)

    def forward(self, output):
        # 用来处理al(ADBi) 1-4 生成 DS 1-4 用于辅助反向传播
        al1, al2, al3, al4, result, seg = output
        print("al1:{}".format(al1.shape))
        al1 = self.sigmoid(self.conv1(al1))
        al2 = self.sigmoid(self.conv2(al2))
        al3 = self.sigmoid(self.conv3(al3))
        al4 = self.sigmoid(self.conv4(al4))
        return al1, al2, al3, al4, result, seg


if __name__ == '__main__':
    root_dir = "../HRCUS-CD/train"
    children_dir = "A"
    my_dataset = MyDataSet(root_dir, children_dir)
    train_data_size = len(my_dataset)
    train_data_loader = DataLoader(my_dataset, batch_size=24, shuffle=False, num_workers=2, drop_last=False)
    model = zh_net()
    manageAltoDS = ManageAltoDS()
    loss_fun = SelfAdaptiveWeightedBCE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)

    total_train_step = 0
    total_test_step = 0
    epoch = 125

    for i in range(epoch):
        print("----------------------第 {} 轮训练开始----------------------".format(i + 1))

        model.train()
        for data in train_data_loader:
            img_A, img_B, targets = data
            output = model(img_A, img_B)
            # 用来处理al(ADBi) 1-4 生成 DS 1-4 用于辅助反向传播
            output = manageAltoDS(output)
            al1, al2, al3, al4, result, seg = output
            # 计算loss1 loss2 loss3 loss4 loss5(result) lossTotal
            loss1 = loss_fun(al1, targets)
            loss2 = loss_fun(al2, targets)
            loss3 = loss_fun(al3, targets)
            loss4 = loss_fun(al4, targets)
            loss5 = loss_fun(result, targets)
            lossTotal = loss1 + loss2 + loss3 + loss4 + loss5

            # loss = loss_fun(output, targets)
            optimizer.zero_grad()
            lossTotal.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数:{},Loss:{}".format(total_train_step, lossTotal.item()))
