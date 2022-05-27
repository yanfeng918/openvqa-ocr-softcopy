# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
import os
from collections import Counter

import numpy as np
import glob, json, re, en_vectors_web_lg

import torch

from openvqa.core.base_dataset import BaseDataSet
from openvqa.utils.ans_punct import prep_ans



from fasttext import load_model

class WordToVectorDict:
    def __init__(self, model):
        self.model = model

    def __getitem__(self, word):
        # Check if mean for word split needs to be done here
        return np.mean([self.model.get_word_vector(w) for w in word.split(" ")], axis=0)

class DataSet(BaseDataSet):
    def __init__(self, __C):
        super(DataSet, self).__init__()
        self.__C = __C

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        # Loading all image paths
        frcn_feat_path_list = \
            glob.glob(__C.FEATS_PATH[__C.DATASET]['train'] + '/*.npz') + \
            glob.glob(__C.FEATS_PATH[__C.DATASET]['val'] + '/*.npz') + \
            glob.glob(__C.FEATS_PATH[__C.DATASET]['test'] + '/*.npz')

        # Loading question word list
        stat_ques_list = \
            json.load(open(__C.RAW_PATH[__C.DATASET]['train'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['val'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['test'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['vg'], 'r'))['questions']

        # Loading answer word list
        # stat_ans_list = \
        #     json.load(open(__C.RAW_PATH[__C.DATASET]['train-anno'], 'r'))['annotations'] + \
        #     json.load(open(__C.RAW_PATH[__C.DATASET]['val-anno'], 'r'))['annotations']

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.RAW_PATH[__C.DATASET][split], 'r'))['questions']
            if __C.RUN_MODE in ['train']:
                self.ans_list += json.load(open(__C.RAW_PATH[__C.DATASET][split + '-anno'], 'r'))['annotations']

        # Define run data size
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print(' ========== Dataset size:', self.data_size)


        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        self.iid_to_frcn_feat_path = self.img_feat_path_load(frcn_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = self.ques_load(self.ques_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = self.tokenize(stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)

        # Answers statistic
        self.ans_to_ix, self.ix_to_ans = self.ans_stat('openvqa/datasets/vqa/answer_dict.json')
        # self.ans_to_ix, self.ix_to_ans = self.ans_stat(stat_ans_list, ans_freq=8)
        self.ans_size = self.ans_to_ix.__len__() + 50
        print(' ========== Answer token vocab size (occur more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')
        # 加载ocr信息

        ocr_path = '/media/yanfeng/9801/vqa_ocr_paddle/valid_token_of_ocr/ocr_tokens_vqa.json'
        ocr_path = '/media/yanfeng/9801/vqa_ocr_Rosetta/ocr_Rosetta_vqa_valid_token.json'
        ocr_path = '/media/yanfeng/9801/vqa_ocr_Rosetta/ocr_Rosetta_vqa.json'

        self.ocr_dict = json.load(open(os.path.join(ocr_path), 'r'))

        # voc_dir2 = '/media/yanfeng/9801/VOC_tokens/ocrofvqa_and_caption.json'
        # voc2 = json.load(open(os.path.join(voc_dir2), 'r'))
        # self.voc = voc2['voc']

        self.PHOC_frn = torch.load(
            '/media/yanfeng/9801/vqa_ocr_paddle/vqa_ocr_paddle_PHOC_frn/PHOC_frn.npz')

        model_file = '/media/yanfeng/colorful1/mmf/wiki.en.bin'

        # logger.info(f"Loading fasttext model now from {model_file}")

        self.model = load_model(model_file)
        # String to Vector
        self.stov = WordToVectorDict(self.model)
        self.max_length = 50



    def img_feat_path_load(self, path_list):
        iid_to_path = {}

        for ix, path in enumerate(path_list):
            iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
            # print(iid)
            iid_to_path[iid] = path

        return iid_to_path

    def ques_load(self, ques_list):
        qid_to_ques = {}

        for ques in ques_list:
            qid = str(ques['question_id'])
            qid_to_ques[qid] = ques

        return qid_to_ques

    def tokenize(self, stat_ques_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        for ques in stat_ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques['question'].lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        voc_dir2 = '/media/yanfeng/9801/VOC_tokens/ocrofvqa_and_caption.json'
        voc2 = json.load(open(os.path.join(voc_dir2), 'r'))
        voc = voc2['voc']

        for word in voc:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)


        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb

    # def ans_stat(self, stat_ans_list, ans_freq):
    #     ans_to_ix = {}
    #     ix_to_ans = {}
    #     ans_freq_dict = {}
    #
    #     for ans in stat_ans_list:
    #         ans_proc = prep_ans(ans['multiple_choice_answer'])
    #         if ans_proc not in ans_freq_dict:
    #             ans_freq_dict[ans_proc] = 1
    #         else:
    #             ans_freq_dict[ans_proc] += 1
    #
    #     ans_freq_filter = ans_freq_dict.copy()
    #     for ans in ans_freq_dict:
    #         if ans_freq_dict[ans] <= ans_freq:
    #             ans_freq_filter.pop(ans)
    #
    #     for ans in ans_freq_filter:
    #         ix_to_ans[ans_to_ix.__len__()] = ans
    #         ans_to_ix[ans] = ans_to_ix.__len__()
    #
    #     return ans_to_ix, ix_to_ans

    def ans_stat(self, json_file):
        ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))

        return ans_to_ix, ix_to_ans

    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_ques_ans(self, idx):
        if self.__C.RUN_MODE in ['train']:
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]
            iid = str(ans['image_id'])

            frcn_feat_path = self.iid_to_frcn_feat_path[iid]
            image_name = (frcn_feat_path).split('/')[-1].replace('.npz', '')
            max_length = 50
            ocr_words = []
            if self.ocr_dict.__contains__(image_name):
                ocr_info = self.ocr_dict[image_name]
                if len(ocr_info) > 0:
                    # get phoc_frn

                    # for item in ocr_info:
                    #     if item['confidence'] > 0.0:
                    #         ocr_words = ocr_words + (item['text'])
                    ocr_words = ocr_info[0]['text']
                    # print(ocr_words)

                    length = min(len(ocr_words), max_length)
                    ocr_words = ocr_words[:length]


            # Process question
            ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)

            # Process answer
            ans_iter = self.proc_ans(ans, self.ans_to_ix, ocr_words)
            return ques_ix_iter, ans_iter, iid

        else:
            ques = self.ques_list[idx]
            iid = str(ques['image_id'])

            ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)

            return ques_ix_iter, np.zeros(1), iid


    def load_img_feats(self, idx, iid):
        frcn_feat = np.load(self.iid_to_frcn_feat_path[iid])
        frcn_feat_x = frcn_feat['x'].transpose((1, 0))
        frcn_feat_iter = self.proc_img_feat(frcn_feat_x, img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])

        # Process OCR 得到ocr token，glove(token),fasttext(text),phoc,location
        frcn_feat_path = self.iid_to_frcn_feat_path[iid]
        image_name = (frcn_feat_path).split('/')[-1].replace('.npz', '')
        max_length = 50

        ocr_glove_ix_iter = np.zeros(max_length, np.int64)
        ocr_fast_text = torch.zeros([50, 300])
        ocr_phoc_frn = torch.zeros([50, 604])

        ocr_bbox_iter = np.zeros((50, 4), dtype=np.float32)

        if self.ocr_dict.__contains__(image_name):
            ocr_info = self.ocr_dict[image_name]
            if len(ocr_info) > 0:
                # get phoc_frn
                # ocr_phoc_frn = self.PHOC_frn[image_name]
                # get fastText
                ocr_words = []
                # ocr_bbox = []
                for item in ocr_info:
                    if item['confidence'] > 0.0:
                        ocr_words = ocr_words + (item['text'])
                        # ocr_bbox = ocr_bbox + [(item['text_box_position'][0])+ (item['text_box_position'][2])]
                # print(ocr_words)
                # ocr_words =ocr_info[0]['text']
                # get glove ix_iter
                ocr_glove_ix_iter = self.proc_ques2(ocr_words, self.token_to_ix, max_token=max_length)


                length = min(len(ocr_words), max_length)
                ocr_words = ocr_words[:length]

                ocr_fast_text = torch.full(
                    (max_length, self.model.get_dimension()),
                    fill_value=0,
                    dtype=torch.float,
                )

                for idx, token in enumerate(ocr_words):
                    ocr_fast_text[idx] = torch.from_numpy(self.stov[token])


                #get ocr location
                # ocr_bbox_iter = self.proc_img_feat(
                #     self.proc_bbox_feat(
                #         np.array(ocr_bbox),
                #         (frcn_feat['image_h'], frcn_feat['image_w'])
                #     ),
                #     img_feat_pad_size=max_length
                # )


        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['image_h'], frcn_feat['image_w'])
            ),
            img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['BBOX_FEAT_SIZE'][0]
        )
        grid_feat_iter = np.zeros(1)
        keep_1 = np.zeros(1)
        ocr_bbox_iter = np.zeros(1)

        return frcn_feat_iter, grid_feat_iter, bbox_feat_iter,ocr_glove_ix_iter, ocr_fast_text,ocr_phoc_frn, ocr_bbox_iter, keep_1



    # ------------------------------------
    # ---- Real-Time Processing Utils ----
    # ------------------------------------

    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat


    def proc_bbox_feat(self, bbox, img_shape):
        if self.__C.BBOX_NORMALIZE:
            bbox_nm = np.zeros((bbox.shape[0], 4), dtype=np.float32)

            bbox_nm[:, 0] = bbox[:, 0] / float(img_shape[1])
            bbox_nm[:, 1] = bbox[:, 1] / float(img_shape[0])
            bbox_nm[:, 2] = bbox[:, 2] / float(img_shape[1])
            bbox_nm[:, 3] = bbox[:, 3] / float(img_shape[0])
            return bbox_nm
        # bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox


    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix

    def proc_ques2(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        # words = re.sub(
        #     r"([.,'!?\"()*#:;])",
        #     '',
        #     ques.lower()
        # ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(ques):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix

    def get_score(self, occur):
        if occur == 0:
            return .0
        elif occur == 1:
            return .3
        elif occur == 2:
            return .6
        elif occur == 3:
            return .9
        else:
            return 1.


    def proc_ans(self, ans, ans_to_ix, tokens):
        # ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
        ans_score = torch.zeros(ans_to_ix.__len__(),dtype=float)
        ans_prob_dict = {}
        answers=[]
        for ans_ in ans['answers']:
            answers.append(ans_['answer'])
            ans_proc = prep_ans(ans_['answer'])
            if ans_proc not in ans_prob_dict:
                ans_prob_dict[ans_proc] = 1
            else:
                ans_prob_dict[ans_proc] += 1

        if self.__C.LOSS_FUNC in ['kld']:
            for ans_ in ans_prob_dict:
                if ans_ in ans_to_ix:
                    ans_score[ans_to_ix[ans_]] = ans_prob_dict[ans_] / 10.
        else:
            for ans_ in ans_prob_dict:
                # 判断了
                if ans_ in ans_to_ix:
                    ans_score[ans_to_ix[ans_]] = self.get_score(ans_prob_dict[ans_])
                #todo else: unkown
                # answers_indices = torch.zeros(self.DEFAULT_NUM_ANSWERS, dtype=torch.long)
                # answers_indices.fill_(self.answer_vocab.get_unk_index())

        gt_answers = list(enumerate(answers))
        answer_counter = Counter(answers)
        length = min(len(tokens), self.max_length)
        tokens_scores = ans_score.new_zeros(self.max_length)

        for idx, token in enumerate(tokens[:length]):
            if answer_counter[token] == 0:
                continue
            accs = []

            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [item for item in other_answers if item[1] == token]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)

            tokens_scores[idx] = sum(accs) / len(accs)
        # ans_score[-len(tokens_scores):] = tokens_scores
        ans_score = torch.cat((ans_score,tokens_scores),-1)

        return ans_score
