# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED
from core.model.mca import FFN

import torch.nn as nn
import torch.nn.functional as F
import torch
import tensorly as tl
from tensorly.decomposition import tucker


class AbstractFusion(nn.Module):

    def __init__(self, __C):
        super(AbstractFusion, self).__init__()
        self.__C = __C

    def forward(self, input_v, input_q):
        raise NotImplementedError




class MLBFusion(AbstractFusion):

    def __init__(self, __C, visual_embedding=True, question_embedding=True):
        super(MLBFusion, self).__init__(__C)
        self.visual_embedding = visual_embedding
        self.question_embedding = question_embedding
        # Modules
        if self.visual_embedding:
            self.linear_v = nn.Linear(__C.IMG_FEAT_SIZE, 1024)
        else:
            print('no visual embedding!')
        if self.question_embedding:
            self.linear_q = nn.Linear(1024, 1024)
        else:
            print('no question embedding!')

    def forward(self, input_v, input_q):

        # visual (cnn features)
        x_v = F.dropout(input_v, p=0.5, training=self.training)
        x_v = self.linear_v(x_v)
        x_v = getattr(torch, "tanh")(x_v)

        # question (rnn features)
        x_q = F.dropout(input_q, p=0.5, training=self.training)
        x_q = self.linear_q(x_q)
        x_q = getattr(torch, "tanh")(x_q)

        #  hadamard product
        x_mm = torch.mul(x_q, x_v)
        return x_mm




class MutanFusion(AbstractFusion):

    def __init__(self, __C, visual_embedding=True, question_embedding=True):
        super(MutanFusion, self).__init__(__C)
        self.visual_embedding = visual_embedding
        self.question_embedding = question_embedding
        if self.visual_embedding:
            self.linear_v = nn.Linear(__C.IMG_FEAT_SIZE, 512)
        else:
            print('no visual embedding!')

        if self.question_embedding:
            self.linear_q = nn.Linear(1024, 512)
        else:
            print('no question embedding!')

        self.list_linear_hv = nn.ModuleList([
            nn.Linear(512, 1024)
            for i in range(5)
        ])

        self.list_linear_hq = nn.ModuleList([
            nn.Linear(512, 1024)
            for i in range(5)
        ])

    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 2:
            raise ValueError
        batch_size = input_v.size(0)

        if self.visual_embedding:
            x_v = F.dropout(input_v, p = 0.5, training=self.training)
            x_v = self.linear_v(x_v)
            x_v = getattr(torch, "tanh")(x_v)
        else:
            x_v = input_v

        if self.question_embedding:
            x_q = F.dropout(input_q, p = 0.5, training=self.training)
            x_q = self.linear_q(x_q)
            x_q = getattr(torch, "tanh")(x_q)
        else:
            x_q = input_q

        x_mm = []
        for i in range(5):

            x_hv = F.dropout(x_v, p=0, training=self.training)
            x_hv = self.list_linear_hv[i](x_hv)
            x_hv = getattr(torch, "tanh")(x_hv)

            x_hq = F.dropout(x_q, p=0, training=self.training)
            x_hq = self.list_linear_hq[i](x_hq)
            x_hq = getattr(torch, "tanh")(x_hq)

            x_mm.append(torch.mul(x_hq, x_hv))    #矩阵点乘

        x_mm = torch.stack(x_mm, dim=1)    #拼接
        x_mm = x_mm.sum(1).view(batch_size, 1024)

        x_mm = getattr(torch, "tanh")(x_mm)

        return x_mm


# ------------------------------
# ----     跳过注意力机制     ----
# ------------------------------

class pass_att_layer(nn.Module):
    def __init__(self, __C):
        super(pass_att_layer, self).__init__()
        self.__C = __C

        self.ffn = FFN(__C)

        # self.no_att_drop_mask = nn.Dropout(__C.DROPOUT_R)
        # self.no_att_norm_mask = nn.LayerNorm(__C.HIDDEN_SIZE)

        self.no_att_drop = nn.Dropout(__C.DROPOUT_R)
        self.no_att_norm = nn.LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x):
        # x = self.no_att_norm(x + self.no_att_drop(
        #     self.ffn(x)
        #
        x = self.no_att_norm(x + self.no_att_drop(
            self.ffn(x)
        ))

        return x



# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted



# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net_vqa(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(Net_vqa, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.backbone = MCA_ED(__C)
        self.jump_att = pass_att_layer(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        self.noattflat_img = AttFlat(__C)
        self.noattflat_lang = AttFlat(__C)
        self.mutanfusion = MutanFusion(__C, visual_embedding=True, question_embedding=True)
        self.mlbfusion = MLBFusion(__C, visual_embedding=True, question_embedding=True)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        # self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # 跳过注意力
        noatt_lang_feat = self.jump_att(lang_feat)
        noatt_img_feat = self.jump_att(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )
        # vs = img_feat
        # 在这里拿到img_feat的注意力图像，在attflat之前
        # 也可以看融合之后的注意力热力图

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        noatt_lang_feat = self.noattflat_lang(
            noatt_lang_feat,
            lang_feat_mask
        )

        noatt_img_feat = self.noattflat_img(
            noatt_img_feat,
            img_feat_mask
        )

        # mask = torch.sigmoid()

        # proj_feat = lang_feat + img_feat
        # proj_feat = self.mlbfusion(img_feat, lang_feat)
        proj_feat = self.mutanfusion(img_feat, lang_feat)
        # proj_feat = img_feat * lang_feat
        proj_feat = self.proj_norm(proj_feat)
        # proj_feats = torch.sigmoid(self.proj(proj_feat))

        return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)



class NET_rubi(nn.Module):
    def __init__(self, __C, Net_vqa, token_size, pretrained_emb, answer_size):
        super(NET_rubi, self).__init__()

        self.net = Net_vqa

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )
        self.linear1 = nn.Linear(6144, 1024)
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.softmax = torch.nn.Softmax(dim = 0)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

    def forward(self,img_feat, ques_ix):

        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)
        lang_feat = lang_feat.reshape(-1, 6144)
        lang_feat = self.linear1(lang_feat)

        mask = torch.sigmoid(lang_feat)
        mask = mask * mask
        # mask = self.make_mask(lang_feat)

        base_out = self.net(img_feat,ques_ix)

        feats = mask * base_out
        feats = self.proj_norm(feats)
        # feats = self.softmax(feats)
        # feats = torch.tensor(feats)

        feats_rubi = self.proj(feats)
        feats_rubi = torch.sigmoid(feats_rubi)
        feats_rubi = torch.nn.functional.normalize(feats_rubi, dim = -1)

        feats_q = self.proj_norm(lang_feat)
        # feats_q = self.softmax(feats_q)
        # feats_q = torch.tensor(feats_q)
        feats_q = self.proj(feats_q)
        feats_q = torch.sigmoid(feats_q)
        feats_q = torch.nn.functional.normalize(feats_q, dim = -1)

        return (feats_rubi, feats_q)


    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1)

def data_normal(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data += torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    dst = d_max - d_min
    norm_data = (orign_data - d_min).true_divide(dst)
    return norm_data




