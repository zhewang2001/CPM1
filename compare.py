import json

from rouge import Rouge
import numpy as np
import string
from collections import Counter
import re
import jieba
jieba.enable_paddle()

def get_rouge_over_list(prediction, groundtruth):
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    if len(remove_punc(prediction)) == 0:
        return 0.0 # during early stages, it might generate nothin?
    # print(prediction)
    rouge = Rouge()
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([rouge.get_scores(prediction, gt, avg=True)["rouge-l"]["f"] for gt in groundtruth])
    # print([prediction, groundtruth])
    prediction = ' '.join(jieba.cut(prediction,use_paddle=True))
    groundtruth = ' '.join(jieba.cut(groundtruth,use_paddle=True))
    # print([prediction, groundtruth])
    # print('\n')
    return rouge.get_scores(prediction, groundtruth, avg=True)["rouge-l"]["f"]

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def qa_f1_score(prediction, ground_truth):
    prediction = ' '.join(jieba.cut(prediction,use_paddle=True))
    ground_truth = ' '.join(jieba.cut(ground_truth,use_paddle=True))
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_f1_over_list(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([qa_f1_score(prediction, gt) for gt in groundtruth])
    return qa_f1_score(prediction, groundtruth)

def get_answer(s):
    for i in range(0, 100):
        if s[-i] == '；' and s[-i-1] == 'J':
            return s[-i+1: ].rstrip('<eo')

def get_query(s):
    for i in range(5, 100):
        if s[i] == '\n':
            return s[4: i]

def main():
    file_name = 'generated_query/generated_query_topk_webgpt_newdata.json'
    a1 = json.load(open(file_name, 'r'))
    # a1 = json.load(open('generated_query_beam_webgpt_bs32_epoch5_queryonly_2-best.json', 'r'))
    b1 = {}
    for k in a1:
        answer = get_answer(k['query'])
        query = get_query(k['query'])
        b1[k['query']] = [k['pred'], answer, query]

    # for k in b1:
    #     print(k)
    #     print('beam')
    #     print(b1[k][0])
    #     print('topk')
    #     print(b2[k][0][2:])
    #     input()

    rouges = []
    rouges_new = []
    last_query = None
    for k in b1:
        # print((b1[k][0], b1[k][1]))
        r = get_rouge_over_list(b1[k][0], b1[k][1])
        rouges.append(r)
        if last_query == None or last_query != b1[k][2]:
            pass
        else:
            rouges_new.append(r)
        last_query = b1[k][2]
    print('rouges: ' + str(np.mean(rouges)))
    print('rouges_new: ' + str(np.mean(rouges_new)))

    f1s = []
    f1s_new = []
    last_query = None
    for k in b1:
        # print((b1[k][0], b1[k][1]))
        f = get_f1_over_list(b1[k][0], b1[k][1])
        f1s.append(f)
        if last_query == None or last_query != b1[k][2]:
            pass
        else:
            f1s_new.append(f)
        last_query = b1[k][2]
    print('f1: ' + str(np.mean(f1s)))
    print('f1_new: ' + str(np.mean(f1s_new)))
    print(file_name)

if __name__ == "__main__":
    main()