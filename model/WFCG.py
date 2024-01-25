import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)  # b*hw*hw
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # b*128*hw
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.spectral_attention = Spectral_attention(in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.cab = CAB(in_dim)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        # print("x.size()",x.size()) #pa(torch.Size([1, 128, 610, 340]))
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy  # 作用是什么？
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        spe_att = self.spectral_attention(x)
        out = x + self.gamma * out  # spe_att  # self.gamma2 * spe_attself.gamma * out
        return out


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


# 尝试使用HAT中的通道注意力
class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=4, squeeze_factor=16):  # 30
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


# 这里尝试使用max_pool和average_pool的光谱注意力机制试一下会不会模型性能会不会提升

class Spectral_attention(nn.Module):
    def __init__(self, C):
        super(Spectral_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(C, 256)
        self.fc2 = nn.Linear(256, C)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        # print("x.shape",x.shape)
        max_C = self.max_pool(x)
        avg_C = self.avg_pool(x)
        max_fc1 = self.fc1(max_C.view(b, c))
        avg_fc1 = self.fc1(avg_C.view(b, c))
        max_fc2 = self.fc2(max_fc1)
        avg_fc2 = self.fc2(avg_fc1)
        add_max_avg = self.relu(max_fc2 + avg_fc2).view(b, c, 1, 1)
        return x * add_max_avg


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.alpha = alpha

        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # 计算concat
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # e存储满足条件的值，zero_vec存储不满足条件的值 未归一化
        attention = F.softmax(attention, dim=1)  # 得到归一化的注意力系数
        attention = F.dropout(attention, self.dropout, training=self.training)  # 防止过拟合
        h_prime = torch.matmul(attention, Wh)  # 得到由周围邻居节点通过注意力系数更新的表示

        if self.concat:
            return F.elu(h_prime)  # 激活函数
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes 即超像素的数量
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# source from https://github.com/PetarV-/GAT
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, adj, nout, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.adj = adj
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return x


class SSConv(nn.Module):  # 光谱-空间卷积的两个关键部分：逐点卷积和深度卷积
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class WFCG(nn.Module):
    def __init__(self, height: int, width: int, channel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor,
                 model='normal'):
        super(WFCG, self).__init__()
        self.class_count = class_count  # 类别数
        self.channel = channel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model = model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q 为什么要对列归一化，作用是什么？
        layers_count = 2

        self.WH = 0
        self.M = 2
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:

                self.CNN_Branch.add_module('Attention' + str(i), PAM_Module(128))
                self.CNN_Branch.add_module('Attention' + str(i), CAM_Module(128))
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 128, kernel_size=3))

            else:
                self.CNN_Branch.add_module('Attention' + str(i), PAM_Module(128))
                self.CNN_Branch.add_module('Attention' + str(i), CAM_Module(128))
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))

        self.GAT_Branch = nn.Sequential()
        self.GAT_Branch.add_module('GAT_Branch' + str(i),
                                   GAT(nfeat=128, nhid=30, adj=A, nout=64, dropout=0.4, nheads=4, alpha=0.2))

        self.linear1 = nn.Linear(64, 64)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(64)

        self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))

    def forward(self, x: torch.Tensor):
        (h, w, c) = x.shape
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))  # 1,c,h,w
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])  # h,w,c
        clean_x = noise

        clean_x_flatten = clean_x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # 超像素数量*c 论文中的encoder过程
        hx = clean_x

        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))  # 1,c,h,w
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])

        H = superpixels_flatten
        H = self.GAT_Branch(H)

        GAT_result = torch.matmul(self.Q, H)  # 论文中的decoder过程
        GAT_result = self.linear1(GAT_result)
        GAT_result = self.act1(self.bn1(GAT_result))
        Y = 0.05 * CNN_result + 0.95 * GAT_result
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y
