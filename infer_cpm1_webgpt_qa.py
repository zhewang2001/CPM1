#coding:utf-8

import time
import random
import torch
import bmtrain as bmp
import numpy as np
import os

from model_center.model import CPM1Config, CPM1
from model_center.tokenizer import CPM1Tokenizer

from model_center.arguments import get_args
from generation import generate_no_beam, generate_no_beam_qa
import json
from tqdm import tqdm

def get_tokenizer(args):
    tokenizer = CPM1Tokenizer.from_pretrained(args.model_config)
    return tokenizer

def get_model(args):
    model = CPM1.from_pretrained(args.model_config)
    bmp.load(model, args.load, strict=False)
    return model

def setup_model(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args)
    bmp.synchronize()
    bmp.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmp.synchronize()
    return tokenizer, model

def initialize():
    # get arguments
    args = get_args()
    # init bmp 
    bmp.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args

def round_up(x, d):
    return (x + d - 1) // d * d

def make_input(lef_tokens, rig_tokens, spans):
    input = lef_tokens + [0 for i in range(spans)] + rig_tokens
    length = len(input)

    rounded_length = round_up(length, 4)

    input_tokens = torch.zeros(1, rounded_length, dtype=torch.int32)
    input_span = torch.zeros(1, rounded_length, dtype=torch.int32)
    
    context = np.arange((rounded_length))
    context = (context < len(lef_tokens)) | (context >= len(lef_tokens) + spans)
    context = torch.from_numpy(context).view(1, -1).bool()

    input_length = torch.zeros(1, dtype=torch.int32)
    input_tokens[0, :length] = torch.tensor(input).int()
    input_length[0] = length

    return input_tokens.cuda(), input_length.cuda(), input_span.cuda(), context.cuda()

def tokenize(tokenizer, sentence):
    return [1] + tokenizer.encode(sentence)

def generate(lef_sentence, rig_sentence, spans, tokenizer, model, topk=1):

    lef_tokens = tokenizer.encode(lef_sentence)
    rig_tokens = tokenizer.encode(rig_sentence)
    lef_tokens = [1] + lef_tokens

    input_tokens, input_length, input_span, context = make_input(lef_tokens, rig_tokens, spans)
    yield lef_sentence

    with torch.inference_mode():
        for i in range(len(lef_tokens) - 1, len(lef_tokens) + spans - 2):
            logits = model(input_tokens, input_length, context, input_span)
            # assert input_tokens[0][i+1] == 0
            # assert context[0][i] == True and  context[0][i+1] == False
            logits = logits[0, i, :].view(-1)
            # print (torch.topk(logits, topk, sorted=True))
            vocab_idx = logits.argmax().cpu().item()
            input_tokens[0][i + 1] = vocab_idx
            # context[0][i+1] = True
            yield tokenizer.decode([vocab_idx])
    yield rig_sentence


# def get_ppl(sentA : str, results : List[str], tokenizer : T5Tokenizer, model : T5):
#     with torch.inference_mode():
#         enc_tensor, enc_len = make_input( tokenize(tokenizer, sentA) )
#         dec_input = []
#         dec_target = []
#         for i, r in enumerate(results):
#             tokens = tokenizer.encode(r)
#             span_idx = tokenizer.get_span(i)
#             dec_input.append( span_idx )
#             dec_target.append( span_idx )
#             dec_input.extend( tokens )
#             dec_target.extend( tokens )
        
#         dec_target.append( tokenizer.eod_id )
#         dec_target = dec_target[1:]

        
#         dec_tensor, dec_len = make_input(dec_input)
#         while len(dec_target) < dec_tensor.size(1):
#             dec_target.append(-100)
#         target_tensor = torch.tensor([dec_target]).long().cuda()
    
#         logits = model(enc_tensor, enc_len, dec_tensor, dec_len)
#         loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
#         batch, seq_len, vocab_out_size = logits.size()
#         loss = loss_func(logits.view(batch * seq_len, vocab_out_size), target_tensor.view(batch * seq_len))
#         loss = loss.cpu().item()
#         print(enc_tensor.size(), dec_tensor.size())
#         print("Loss: %lf" % loss)
#         print("PPL: %lf" % math.exp(loss))

def demo():
    args = initialize()
    tokenizer, model = setup_model(args)

    dataset = [
                {
                    "context_id": "DEV_0",
                    "context_text": "《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任天堂游戏谜之村雨城改编的新增模式。本作中共有20张战场地图（不含村雨城），后来发行的猛将传再新增3张战场地图。但游戏内战役数量繁多，部分地图会有兼用的状况，战役虚实则是以光荣发行的2本「战国无双3 人物真书」内容为主，以下是相关介绍。（注：前方加☆者为猛将传新增关卡及地图。）合并本篇和猛将传的内容，村雨城模式剔除，战国史模式可直接游玩。主打两大模式「战史演武」&「争霸演武」。系列作品外传作品",
                    "qas": [
                        {
                            "query_text": "《战国无双3》是由哪两个公司合作开发的？",
                            "query_id": "DEV_0_QUERY_0",
                            "answers": [
                                "光荣和ω-force",
                                "光荣和ω-force",
                                "光荣和ω-force"
                            ]
                        },
                        {
                            "query_text": "男女主角亦有专属声优这一模式是由谁改编的？",
                            "query_id": "DEV_0_QUERY_1",
                            "answers": [
                                "村雨城",
                                "村雨城",
                                "任天堂游戏谜之村雨城"
                            ]
                        },
                        {
                            "query_text": "战国史模式主打哪两个模式？",
                            "query_id": "DEV_0_QUERY_2",
                            "answers": [
                                "「战史演武」&「争霸演武」",
                                "「战史演武」&「争霸演武」",
                                "「战史演武」&「争霸演武」"
                            ]
                        }
                    ],
                    "title": "战国无双3"
                },
                {
                    "context_id": "TRAIN_186",
                    "context_text": "范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世。范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生；童年时接受良好教育后，被一位越南神父带到河内继续其学业。范廷颂于1940年在河内大修道院完成神学学业。范廷颂于1949年6月6日在河内的主教座堂晋铎；及后被派到圣女小德兰孤儿院服务。1950年代，范廷颂在河内堂区创建移民接待中心以收容到河内避战的难民。1954年，法越战争结束，越南民主共和国建都河内，当时很多天主教神职人员逃至越南的南方，但范廷颂仍然留在河内。翌年管理圣若望小修院；惟在1960年因捍卫修院的自由、自治及拒绝政府在修院设政治课的要求而被捕。1963年4月5日，教宗任命范廷颂为天主教北宁教区主教，同年8月15日就任；其牧铭为「我信天主的爱」。由于范廷颂被越南政府软禁差不多30年，因此他无法到所属堂区进行牧灵工作而专注研读等工作。范廷颂除了面对战争、贫困、被当局迫害天主教会等问题外，也秘密恢复修院、创建女修会团体等。1990年，教宗若望保禄二世在同年6月18日擢升范廷颂为天主教河内总教区宗座署理以填补该教区总主教的空缺。1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山教区宗座署理；同年11月26日，若望保禄二世擢升范廷颂为枢机。范廷颂在1995年至2001年期间出任天主教越南主教团主席。2003年4月26日，教宗若望保禄二世任命天主教谅山教区兼天主教高平教区吴光杰主教为天主教河内总教区署理主教；及至2005年2月19日，范廷颂因获批辞去总主教职务而荣休；吴光杰同日真除天主教河内总教区总主教职务。范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。",
                    "qas": [
                        {
                            "query_id": "TRAIN_186_QUERY_0",
                            "query_text": "范廷颂是什么时候被任为主教的？",
                            "answers": [
                                "1963年"
                            ]
                        },
                        {
                            "query_id": "TRAIN_186_QUERY_1",
                            "query_text": "1990年，范廷颂担任什么职务？",
                            "answers": [
                                "1990年被擢升为天主教河内总教区宗座署理"
                            ]
                        },
                        {
                            "query_id": "TRAIN_186_QUERY_2",
                            "query_text": "范廷颂是于何时何地出生的？",
                            "answers": [
                                "范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生"
                            ]
                        },
                        {
                            "query_id": "TRAIN_186_QUERY_3",
                            "query_text": "1994年3月，范廷颂担任什么职务？",
                            "answers": [
                                "1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山教区宗座署理"
                            ]
                        },
                        {
                            "query_id": "TRAIN_186_QUERY_4",
                            "query_text": "范廷颂是何时去世的？",
                            "answers": [
                                "范廷颂于2009年2月22日清晨在河内离世"
                            ]
                        }
                    ],
                    "title": "范廷颂"
                }
            ]

    # dataset = [
    #     {
    #         "lef_sentence": "爱因斯坦出生于",
    #         "rig_sentence": "",
    #         "spans": 40,
    #     },
    #     {
    #         "lef_sentence": "据北京环球度假区介绍，园区开园前的各项准备工作已",
    #         "rig_sentence": "",
    #         "spans": 40,
    #     },
    #     {
    #         "lef_sentence": "北京四合院是一种中国传统高档合院式建筑",
    #         "rig_sentence": "",
    #         "spans": 40,
    #     },
    #     {
    #         "lef_sentence": "北上广深是指",
    #         "rig_sentence": "",
    #         "spans": 40,
    #     },
    #     {
    #         "lef_sentence": "据北京环球度假区介绍,园区开园前的各项准备工作已",
    #         "rig_sentence": "已经盛大开业",
    #         "spans": 16,
    #     },
    #     {
    #         "lef_sentence": "苏联的一次大会上，主持人突然说到：",
    #         "rig_sentence": "主持人慌忙说：那请您赶快坐到主席台上来",
    #         "spans": 32,
    #     },
    #     {
    #         "lef_sentence": "兔子有",
    #         "rig_sentence": "条腿",
    #         "spans": 1,
    #     },
    #     {
    #         "lef_sentence": "青蛙有",
    #         "rig_sentence": "条腿",
    #         "spans": 1,
    #     },
    #     {
    #         "lef_sentence": "人有",
    #         "rig_sentence": "条腿",
    #         "spans": 1,
    #     },
    #     {
    #         "lef_sentence": "天空是蔚蓝色",
    #         "rig_sentence": "千纸鹤",
    #         "spans": 5,
    #     },
    #     {
    #         "lef_sentence": "清华大学自然语言处理实验室推出了新的大规模预训练语言模型，他完善了CPM-1模型中已知的几个主要问题。新一代的模型被命名为“",
    #         "rig_sentence": "”，受到了业界的广泛好评。",
    #         "spans": 5,
    #     },
    #     {
    #         "lef_sentence": "近日，经十三届全国人大五次会议审议通过的《政府工作报告》正式发布",
    #         "rig_sentence": "。",
    #         "spans": 35,
    #     },
    #     {
    #         "lef_sentence": "近日，经十三届全国人大五次会议审议通过的《政府工作报告》正式发布",
    #         "rig_sentence": "",
    #         "spans": 35,
    #     },
    #     {
    #         "lef_sentence": """平儿进入厅中，她姊妹三人正议论些家务，说的便是年内赖大家请吃酒，""",
    #         "rig_sentence":"""你奶奶怎么就没想到这个？”""",
    #         "spans": 100
    #     }
    # ]

    fout = open(f"{args.base_path}/cmrc_ep5.out", "w", encoding="utf-8")
    span = 50
    for data in dataset:
        doc = data['context_text']
        qas = data['qas']
        for qa in qas:
            pred = []
            for it in generate_no_beam_qa(model, tokenizer, doc, qa['query_text'], span, no_repeat_ngram_size=0, temperature=1.0, top_k=1):
                if it == '<eod>':
                    break
                pred.append(it)
                # fout.write(it)
                # fout.flush()
            format_res = {
                'context': doc,
                'question': qa['query_text'],
                'answer': ''.join(pred)
            }

            fout.write(json.dumps(format_res, ensure_ascii=False, indent=4))
            fout.write('\n')
    fout.close()

def main():
    os.environ["MASTER_PORT"] = (str)((int)(os.environ["MASTER_PORT"]) + 1123)
    args = initialize()
    tokenizer, model = setup_model(args)

    # if args.split == 'trial':
    #     dataset = json.load(open(os.path.join(args.data_dir, 'cmrc2018_trial.json'), 'r'))
    #     fout = open(os.path.join(args.save_dir, f'cmrc2018_trial_pred_{args.output_name}.json'), 'w')
    dataset = json.load(open(os.path.join('/home/qinyujia/ModelCenter/down_data/webgpt_qa', 'dev.json'), 'r'))
    fout = open(os.path.join('/home/qinyujia/ModelCenter/output', f'webgpt_qa_dev_pred_output.json'), 'w')

    span = 500
    all_pred = {}
    # dataset = dataset[:5]
    for data in tqdm(dataset):
        # qid = data['context_id']
        new_doc = ""
        for c in data['context']:
            new_doc += "摘要：" + c + "\n"
        doc = new_doc

        question = data['question']
        pred = []
        doc = "".join(doc)
        for it in generate_no_beam_qa(model, tokenizer, doc, question, span, no_repeat_ngram_size=0, temperature=1.0, top_k=1):
            if it == '<eod>':
                break
            pred.append(it)
            # fout.write(it)
            # fout.flush()
        qid = data['id']
        all_pred[qid] = ''.join(pred)
        print('question')
        print(question)
        print('context')
        print(doc)
        print('prediction')
        print(''.join(pred))
        print('answer')
        print(data['answer'])
        input()

    json.dump(all_pred, fout)
    fout.close()

if __name__ == "__main__":
    main()
