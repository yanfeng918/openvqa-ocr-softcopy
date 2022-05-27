import glob
import os

from openvqa.datasets.vqa.eval.vqa import VQA
from openvqa.datasets.vqa.eval.vqaEval import VQAEval
import json, pickle
import numpy as np


def eval(__C, dataset, ans_ix_list, pred_list, result_eval_file, ensemble_file, log_file, valid=False):
    # Loading all image paths
    frcn_feat_path_list = \
        glob.glob(__C.FEATS_PATH[__C.DATASET]['train'] + '/*.npz') + \
        glob.glob(__C.FEATS_PATH[__C.DATASET]['val'] + '/*.npz') + \
        glob.glob(__C.FEATS_PATH[__C.DATASET]['test'] + '/*.npz')

    # Loading question and answer list
    ques_list = dataset.ques_list
    # {image id} -> {image feature absolutely path}
    iid_to_frcn_feat_path = dataset.iid_to_frcn_feat_path
    # iid_to_frcn_feat_path = img_feat_path_load(frcn_feat_path_list)
    # 加载ocr信息

    ocr_path = '/media/yanfeng/9801/vqa_ocr_paddle/valid_token_of_ocr/ocr_tokens_vqa.json'
    ocr_path = '/media/yanfeng/9801/vqa_ocr_Rosetta/ocr_Rosetta_vqa_valid_token.json'
    ocr_path = '/media/yanfeng/9801/vqa_ocr_Rosetta/ocr_Rosetta_vqa.json'
    ocr_dict = json.load(open(os.path.join(ocr_path), 'r'))


    result_eval_file = result_eval_file + '.json'

    qid_list = [ques['question_id'] for ques in dataset.ques_list]
    ans_size = dataset.ans_size

    # result = [{
    #     'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],
    #     # 'answer': dataset.ix_to_ans[ans_ix_list[qix]],
    #     'question_id': int(qid_list[qix])
    # } for qix in range(qid_list.__len__())]

    # todo 如果，答案是超过3129的，需要根据问题获得图片的id，然后获取图片对应的ocr的token
    result = []
    for qix in range(qid_list.__len__()):
        ques = ques_list[qix]
        iid = str(ques['image_id'])

        frcn_feat_path = iid_to_frcn_feat_path[iid]
        image_name = (frcn_feat_path).split('/')[-1].replace('.npz', '')

        if ans_ix_list[qix]<=3128:
           result.append({
                'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],
                'question_id': int(qid_list[qix])
            })
        else:
            print(f'qix:{qix},ques:{ques},image_name:{image_name}')
            ocr_words = []
            if ocr_dict.__contains__(image_name):
                ocr_info = ocr_dict[image_name]
                if len(ocr_info) > 0:
                    for item in ocr_info:
                        if item['confidence'] > 0.0:
                            ocr_words = ocr_words + (item['text'])
                    # ocr_words = ocr_info
                    print(f"ocr_words:{ocr_words}")
                    # length = min(len(ocr_words), 50)
                    # ocr_words = ocr_words[:length]
            print(ans_ix_list[qix])
            #TODO index out
            if ans_ix_list[qix]  > len(ocr_words) + 3128:
                if len(ocr_words)>0:
                    result.append({
                        'answer': str(ocr_words[0]),
                        'question_id': int(qid_list[qix])
                    })
                else:
                    print('unk')
                    result.append({
                        'answer': 'UNK',
                        'question_id': int(qid_list[qix])
                    })

            else:
                result.append({
                    'answer': str(ocr_words[ans_ix_list[qix] - 3128 - 1]),
                    'question_id': int(qid_list[qix])
                })

    print('Save the result to file: {}'.format(result_eval_file))
    json.dump(result, open(result_eval_file, 'w'))


    if __C.TEST_SAVE_PRED:
        print('Save the prediction vector to file: {}'.format(ensemble_file))

        pred_list = np.array(pred_list).reshape(-1, ans_size)
        result_pred = [{
            'pred': pred_list[qix],
            'qid': int(qid_list[qix])
        } for qix in range(qid_list.__len__())]

        pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)


    if valid:
        # create vqa object and vqaRes object
        ques_file_path = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val']]
        ans_file_path = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val'] + '-anno']

        vqa = VQA(ans_file_path, ques_file_path)
        vqaRes = vqa.loadRes(result_eval_file, ques_file_path)

        # create vqaEval object by taking vqa and vqaRes
        vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

        # evaluate results
        """
        If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
        By default it uses all the question ids in annotation file
        """
        vqaEval.evaluate()

        # print accuracies
        print("\n")
        print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        # print("Per Question Type Accuracy is the following:")
        # for quesType in vqaEval.accuracy['perQuestionType']:
        #     print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
        # print("\n")
        print("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        print("\n")

        print('Write to log file: {}'.format(log_file))
        logfile = open(log_file, 'a+')

        logfile.write("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        for ansType in vqaEval.accuracy['perAnswerType']:
            logfile.write("%s : %.02f " % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        logfile.write("\n\n")
        logfile.close()


