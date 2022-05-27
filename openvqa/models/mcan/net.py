# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.mcan.mca import MCA_ED, MCA_ocr
from openvqa.models.mcan.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
import torch


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

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

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

        self.adapter = Adapter(__C)

        self.backbone = MCA_ED(__C)
        self.backbone2 = MCA_ocr(__C)

        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        self.attflat_ocr = AttFlat(__C)

        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

        self.ocr_glove_lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )
        self.ocr_fasttext = nn.Linear(__C.WORD_EMBED_SIZE, __C.HIDDEN_SIZE)
        self.ocr_phoc = nn.Linear(604, __C.HIDDEN_SIZE)
        self.ocr_location = nn.Linear(4, __C.HIDDEN_SIZE)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)



    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix,
            ocr_glove_ix_iter,
            ocr_fast_text,
            ocr_phoc_frn,
            ocr_bbox_iter,
            keep_1):

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        ocr_feat_mask = make_mask(ocr_glove_ix_iter.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)
        ocr_feat = torch.zeros(1)


        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # deal ocr frn
        if ocr_glove_ix_iter.shape[1] > 1:
            # ocr_lang_feat = self.embedding(ocr_glove_ix_iter)
            # ocr_lang_feat, _ = self.ocr_glove_lstm(ocr_lang_feat)
            # ocr_phoc = self.ocr_phoc(ocr_phoc_frn)
            ocr_fast_text = self.ocr_fasttext(ocr_fast_text)
            # ocr_bbox_iter = self.ocr_location(ocr_bbox_iter)
            # ocr_feat = self.norm1(ocr_lang_feat + ocr_fast_text + ocr_phoc) + self.norm2(ocr_bbox_iter)
            ocr_feat = self.norm1( ocr_fast_text )

            # y 是 question,  x是字幕信息
            lang_feat, ocr_feat = self.backbone2(
                lang_feat,
                ocr_feat,  # x是字幕信息
                lang_feat_mask,
                ocr_feat_mask
            )

            ocr_feat = self.attflat_ocr(
                ocr_feat,
                ocr_feat_mask
            )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        # Classification layers
        proj_feat = lang_feat + img_feat
        if ocr_glove_ix_iter.shape[1] > 1:
            proj_feat = proj_feat + ocr_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat

