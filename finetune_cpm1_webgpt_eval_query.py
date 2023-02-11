import time
import random
import torch
import sys
import os
from interact import platformctrl
if not os.path.exists('/home/qinyujia') and not os.path.exists('/liuzyai04/qinyujia') and not os.path.exists('/zhipuai06/qinyujia'):
    import GPUtil
    gpu_name = GPUtil.getGPUs()[0].name
    if 'A100' in gpu_name:
        # sys.path.append('/apdcephfs/share_47076/weizechen/BMTrain-a100/install/lib/python3.8/site-packages/bmtrain-0.1.5-py3.8-linux-x86_64.egg')
        sys.path.append('/apdcephfs/share_47076/weizechen/BMTrain-a100/install/lib/python3.8/site-packages/bmtrain-0.1.5-py3.8-linux-x86_64.egg')
        print("append /apdcephfs/share_47076/weizechen/BMTrain-a100/")
    elif 'V100' in gpu_name:
        sys.path.append('/apdcephfs/share_47076/weizechen/BMTrain-v100/install/lib/python3.8/site-packages/bmtrain-0.1.5-py3.8-linux-x86_64.egg')
        print("append /apdcephfs/share_47076/weizechen/BMTrain-v100/")
import bmtrain as bmt
import numpy as np
import csv
import json
import re

from model_center import get_args
from model_center.model import CPM1
from model_center.tokenizer import CPM1Tokenizer
from model_center.dataset.cpm1dataset import DATASET
from model_center.utils import print_inspect
from model_center.dataset import DistributedDataLoader
import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, top_k_accuracy_score
import edit_distance

from generation import generate_no_beam_qa, generate_beam
from compare import get_rouge_over_list

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-50000):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
    batch_size = logits.size()[0]

    if top_p > 0.0:
        logits=logits.view(batch_size, -1).contiguous()
        for index in range(len(logits)):

            sorted_logits, sorted_indices = torch.sort(logits[index].view(-1), descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[index][indices_to_remove] = filter_value

        logits=logits.view(batch_size, -1).contiguous()
    
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    return logits

def get_tokenizer(args):
    tokenizer = CPM1Tokenizer.from_pretrained(args.model_config)
    return tokenizer

def get_model(args):
    model = CPM1.from_pretrained(args.model_config)
    return model

def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), 
                                               weight_decay=args.weight_decay, 
                                               scale=args.loss_scale)
    return optimizer

def get_learning_rate_scheduler(args, optimizer, dataloader):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = int(len(dataloader["train"]) / args.accumulation_steps * args.epochs)
        # args.lr_decay_iters = args.train_iters * args.epochs
    args.warmup_iters = int(len(dataloader["train"]) / args.accumulation_steps * args.epochs * args.warmup_ratio)
    if args.lr_decay_style == "noam":
        lr_scheduler = bmt.lr_scheduler.Noam(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "constant":
        lr_scheduler = bmt.lr_scheduler.NoDecay(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = -1,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "exponential":
        lr_scheduler = bmt.lr_scheduler.Exponential(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = bmt.lr_scheduler.Cosine(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler

def setup_model_and_optimizer(args):
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    # get the model
    model = get_model(args)
    bmt.synchronize()
    # get the optimizer
    optimizer = get_optimizer(args, model)
    bmt.synchronize()
    # get the memory usage
    bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmt.synchronize()
    return tokenizer, model, optimizer

def initialize():
    # get arguments
    args = get_args()
    # init bmt 
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 123)
    bmt.init_distributed(seed = 28321, loss_scale_factor = 2, loss_scale_steps = 1024)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def prepare_dataset(args, tokenizer, base_path, dataset_name, rank, world_size, query_only, abstract_only, inference):
    if inference == 1:
        splits = ['inference']
    else:
        splits = ['train', 'dev', 'test']
    dataset = {}
    for split in splits:
        dataset[split] = DATASET[dataset_name](base_path, split, rank, world_size, tokenizer, args.max_length, debug = args.debug, do_infer = False if split == 'train' else True, query_only = query_only, abstract_only = abstract_only) # ratio = 0.005)

    verbalizer = torch.LongTensor(dataset[split].action_verbalizer).cuda()
    action2idx = dataset[split].action2idx
    all_actions = list(dataset[split].action2idx.keys())

    return dataset, verbalizer, all_actions, action2idx

def inference_mode(args, tokenizer, model, optimizer, dataset, verbalizer, all_actions, query_only, abstract_only, action2idx, inference, base_path, rank, world_size, loss_func, inference_json):
    op = platformctrl.Operator()
    op.execute(platformctrl.Operation.START)

    past_href = None
    in_page = False
    last_page = None
    ban_down = False
    ban_up = False
    ban_abstract = False
    asked_questions = []
    for _ in range(100):
        inference_dataset = DATASET["webgpt"](base_path, 'inference_new', rank, world_size, tokenizer, args.max_length, debug = args.debug, do_infer = True, query_only = query_only, abstract_only = abstract_only)
        # max length overflow, fix later
        if inference_dataset.idx_drop > 0:
            print('abstract too long, early exit')
            break
        inference_dataloader = DistributedDataLoader(inference_dataset, batch_size=1, shuffle=False, is_eval = False)
        model.eval()
        with torch.no_grad():
            for it, data in enumerate(inference_dataloader):
                if it == len(inference_dataloader) - 1:
                    break
            input_tokens = data["input_tokens"]
            input_length = data["input_length"]
            input_context = data["input_context"]
            input_span = data["input_span"]
            output_ids = data["output_ids"]
            index = data["index"]
            feasible_state = data["feasible_state"]

            a = input_tokens[0].cpu().numpy().tolist()
            print(tokenizer.decode(a).rstrip("<pad>"))
            # b = output_ids[0].cpu().numpy().tolist()
            # print(tokenizer.decode(b).rstrip("<pad>"))
            
            method = 'top1'
            if method == 'top1':
                logits = model(input_tokens, input_length, input_context, input_span)
                loss = loss_func(logits.view(-1, logits.size(-1)), output_ids.view(-1))
                logits = logits.index_select(dim=-1, index=verbalizer)
                logits = logits[torch.where(index==1)]
                probs = F.softmax(logits, dim=-1)
                logits[feasible_state == 0] = -100
                if ban_down:
                    logits[0, 0] = -100
                if ban_abstract:
                    logits[0, 3] = -100
                    ban_abstract = False
                if ban_up:
                    logits[0, 1] = -100
                topk = torch.topk(logits, 5)[1].cpu().numpy().tolist()[0]
                topk_prob = torch.topk(probs, 5)[0].cpu().numpy().tolist()[0]
                next_action_id = logits.argmax(dim=-1).cpu().numpy().tolist()[0]
                print(next_action_id)
                print(topk)
                print(topk_prob)
                if in_page == False and next_action_id == 2:
                    next_action_id = topk[1]
                # input()
                if next_action_id == 0:
                    action_name = "TRIGGER_SCROLL_DOWN"
                    op.page_down()
                    if in_page:
                        current_page = op.get_page_detail()
                        action = {"action": action_name, "pageContentInViewport": current_page}
                    else:
                        current_page = op.get_page_detail()
                        # print(current_page)
                        action = {"action": action_name, "pageContentInViewport": current_page}
                        past_href = [k["href"] for k in current_page]
                elif next_action_id == 1:
                    action_name = "TRIGGER_SCROLL_UP"
                    op.page_up()
                    if in_page:
                        current_page = op.get_page_detail()
                        action = {"action": action_name, "pageContentInViewport": current_page}
                    else:
                        current_page = op.get_page_detail()
                        # print(current_page)
                        action = {"action": action_name, "pageContentInViewport": current_page}
                        past_href = [k["href"] for k in current_page]
                elif next_action_id == 2:
                    action_name = "PAGE_GO_BACK"
                    op.execute(platformctrl.Operation.GO_BACK)
                    current_page = op.get_page_detail()
                    # print(current_page)
                    action = {"action": action_name, "pageContentInViewport": current_page}
                    past_href = [k["href"] for k in current_page]
                    in_page = False
                elif next_action_id == 3:
                    action_name = "ADD_DIGEST"
                    context_length = torch.sum(input_context, dim = 1).cpu().tolist()[0]
                    input_tokens[0][context_length] = inference_dataset.action_start2idx[action_name][0]
                    input_tokens[0][context_length + 1] = tokenizer.encode("；")[0]
                    input_length += 2
                    decode_abstract, final_abstract_truth = get_abstract(model, tokenizer, input_tokens, output_ids, input_length, input_span, input_context)
                    if decode_abstract in ["Did not find the quote!", "ending before starting"]:
                        ban_abstract = True
                        print('abstract makes faults, continue')
                        continue
                    decode_abstract = re.sub('\n', '', decode_abstract)
                    print(decode_abstract)
                    op.execute(platformctrl.Operation.ADD_DIGEST, decode_abstract)
                    action = {"action": action_name, "details": {"text": decode_abstract}, "digests": op.get_digests()}
                elif next_action_id == 4:
                    action_name = "MERGE_DIGEST"
                    op.merge([-2,-1])
                    action = {"action": action_name, "digests": op.get_digests()}
                elif next_action_id in [5, 6, 7]:
                    if next_action_id == 5:
                        op.execute(platformctrl.Operation.LOAD_PAGE_1)
                    elif next_action_id == 6:
                        op.execute(platformctrl.Operation.LOAD_PAGE_2)
                    elif next_action_id == 7:
                        op.execute(platformctrl.Operation.LOAD_PAGE_3)
                    current_page = op.get_page_detail()
                    action_name = "LOAD_PAGE_DETAIL"
                    action = {"action": action_name, "pageContentInViewport": current_page, "details": {"href": past_href[next_action_id - 5]}}
                    in_page = True
                elif next_action_id == 8:
                    break
                elif next_action_id == 9:
                    action_name = "PRESS_SEARCH"
                    context_length = torch.sum(input_context, dim = 1).cpu().tolist()[0]
                    input_tokens[0][context_length] = inference_dataset.action_start2idx[action_name][0]
                    input_tokens[0][context_length + 1] = tokenizer.encode("；")[0]
                    input_length += 2
                    decode_query = get_topk_query(model, tokenizer, input_tokens, input_length, input_span, input_context)
                    # print(decode_query)
                    if decode_query in asked_questions:
                        decode_query_beam_best = get_beam_query(model, tokenizer, input_tokens, input_length, input_span, input_context, return_best = True)
                        decode_query_beam_worst = get_beam_query(model, tokenizer, input_tokens, input_length, input_span, input_context, return_best = False)
                        if decode_query_beam_best not in asked_questions:
                            print('use best beam search')
                            decode_query = decode_query_beam_best
                        elif decode_query_beam_worst not in asked_questions:
                            print('use worst beam search')
                            decode_query = decode_query_beam_worst
                        else:
                            break
                    op.execute(platformctrl.Operation.SEARCH, decode_query)
                    current_page = op.get_page_detail()
                    # print(current_page)
                    action = {"action": action_name, "details": {"keyword": decode_query, "result": []}, "pageContentInViewport": current_page}
                    past_href = [k["href"] for k in current_page]

                    asked_questions.append(decode_query)

                if next_action_id == 0 and current_page == last_page:
                    ban_down = True
                    last_page = current_page
                    continue
                elif next_action_id == 1 and current_page == last_page:
                    ban_up = True
                    last_page = current_page
                    continue
                else:
                    ban_down = False
                    ban_up = False
                    last_page = current_page
                    inference_json["data"][0]["actions"].insert(-1, action)
                    json.dump(inference_json, open(base_path + '/webgpt/inference_new.json', 'w', encoding='utf8'), ensure_ascii=False)

    # load the QA model
    digests = op.get_digests()
    question = json.load(open(base_path + 'webgpt/inference_new.json', "r", encoding='utf-8'))["data"][0]["question"]
    if len(digests) == 0:
        answer = "没有摘要"
    else:
        checkpoint = torch.load('/home/qinyujia/ModelCenter/results/finetune-cpm1-webgpt-newdata_qa-best.pt')
        checkpoint = {k: v.cuda() for k,v in checkpoint.items()}
        model.load_state_dict(checkpoint)

        new_digests = []
        for digest in digests:
            new_digests.append("".join([d['desc'] for d in digest]))

        doc = ""
        for c in new_digests:
            doc += "摘要：" + c + "\n"
        # doc = "".join(new_digests)
        
        pred = []
        span = 500
        for it in generate_no_beam_qa(model, tokenizer, doc, question, span, no_repeat_ngram_size=0, temperature=1.0, top_k=1):
            if it == '<eod>':
                break
            pred.append(it)
        answer = "".join(pred)  
    print('the question is')
    print(question)
    print('the answer is')
    print(answer)
    return {'question': question, 'abstract': digests, 'answer': answer}

def finetune(args, tokenizer, model, optimizer, dataset, verbalizer, all_actions, query_only, abstract_only, action2idx, inference, base_path = "", rank = "", world_size = ""):
    # torch.set_printoptions(profile="full")
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    if inference:
        dump_pred = []
        for question in ["为什么大多数甲壳类动物或海鲜在加热时会从浅灰色变为红色或橙色？"]:
            inference_json = {"data": [{'_id': 'test_001', 'actions': [{"action": "RECORD_START", "details": {}, "digests": [], "keyword": "", "triggerAt": 1650969909468, "stackLength": 0, "step": 0, "traceId": "718dd8557f9bf83d80468c954c88", "currentPageInfo": {"title": "", "type": "", "href": "", "scrollTop": 0}}, {"action": "RECORD_CLOSE", "details": {"digests": []}, "digests": [], "keyword": "", "triggerAt": 1650969912433, "stackLength": 0, "step": 0, "traceId": "718dd8557f9bf83d80468c954c88", "currentPageInfo": {"title": "", "type": "", "href": "", "scrollTop": 0}}], 'digests': [], 'question': '你去健身房的动力是什么', 'answer': '未知'}]}
            inference_json["data"][0]["question"] = question
            json.dump(inference_json, open(base_path + '/webgpt/inference_new.json', 'w', encoding='utf8'), ensure_ascii=False)

            checkpoint = torch.load(os.path.join(args.save, args.load))
            checkpoint = {k: v.cuda() for k,v in checkpoint.items()}
            model.load_state_dict(checkpoint)

            pred_dict = inference_mode(args, tokenizer, model, optimizer, dataset, verbalizer, all_actions, query_only, abstract_only, action2idx, inference, base_path, rank, world_size, loss_func, inference_json)
            dump_pred.append(pred_dict)
        json.dump(dump_pred, open('generated_answer/dump_pred.json', 'w', encoding='utf8'), indent = 4, ensure_ascii=False)
    else:
        dataloader = {
            # shuffle of train data set to False to facilitate in-batch contrastive learning
            "train": DistributedDataLoader(dataset['train'], batch_size=args.train_batch_size, shuffle=False, is_eval = False),
            "dev": DistributedDataLoader(dataset['dev'], batch_size=args.eval_batch_size, shuffle=False, is_eval = True),
            "test": DistributedDataLoader(dataset['test'], batch_size=args.eval_batch_size, shuffle=False, is_eval = True),
        }

        lr_scheduler = get_learning_rate_scheduler(args, optimizer, dataloader)

        best_dev_metric = None

        bmt.load(model, os.path.join(args.save, args.load), strict=False)

        if query_only:
            dev_metric = evaluate_query(args, model, dataloader['dev'], tokenizer, loss_func, verbalizer, all_actions, decoding_method = 'topk')
        elif abstract_only:
            dev_metric = evaluate_abstract(args, model, dataloader['dev'], tokenizer, loss_func, verbalizer, all_actions)
        else:
            dev_metric = evaluate_action(args, model, dataloader['dev'], tokenizer, loss_func, verbalizer, all_actions, action2idx)

            bmt.print_rank(
                "dev | micro_f1: {:6f} | macro_f1: {:6f}  | micro_precision: {:6f} | macro_precision: {:6f}  | macro_recall: {:6f} | micro_recall: {:6f}  |avg_loss: {:6f} |".format(
                    dev_metric['micro_f1'],
                    dev_metric['macro_f1'],
                    dev_metric['micro_precision'],
                    dev_metric['macro_precision'],
                    dev_metric['micro_recall'],
                    dev_metric['macro_recall'],
                    dev_metric['avg_loss'],
                )
            )

            bmt.print_rank(dev_metric['all_f1'])

def evaluate_action(args, model, dataloader, tokenizer, loss_func, verbalizer, all_actions, action2idx):
    model.eval()
    with torch.no_grad():
        total_num = 0
        total_loss = 0
        acc = 0
        target_list = []
        pred_list = []
        all_topk = []

        action2idx = [k[2: ] for k in list(action2idx.values())]
        for it, data in enumerate(dataloader):
            idx = data["idx"]
            input_tokens = data["input_tokens"]
            input_length = data["input_length"]
            input_context = data["input_context"]
            input_span = data["input_span"]
            output_ids = data["output_ids"]
            target = data["target"]
            index = data["index"]
            # index = data["index"]
            # truth = data["truth"]
            feasible_state = data["feasible_state"]

            # a = input_tokens[0].cpu().numpy().tolist()
            # print(tokenizer.decode(a).rstrip("<pad>"))
            # b = output_ids[0].cpu().numpy().tolist()
            # print(tokenizer.decode(b).rstrip("<pad>"))

            # c = 0
            # for item in b:
            #     if item == -100:
            #         c += 1
            #     else:
            #         break
            # print(tokenizer.decode(a[c: ]))

            method = 'top1'
            if method == 'top1':
                logits = model(input_tokens, input_length, input_context, input_span)
                loss = loss_func(logits.view(-1, logits.size(-1)), output_ids.view(-1))
                logits = logits.index_select(dim=-1, index=verbalizer)
                logits = logits[torch.where(index==1)]
                # logits[feasible_state == 0] = -100
                topk = torch.topk(logits, 3)[1].cpu().numpy().tolist()[0]
                all_topk.append(topk)
                logits = logits.argmax(dim=-1)
                acc += torch.sum(logits == target).item()
                loss = loss.sum()
                total_loss += loss.item()
                total_num += logits.shape[0]
                target_list += target.cpu().numpy().tolist()
                pred_list += logits.cpu().numpy().tolist()
            elif method == 'ppl':
                target_probs = torch.ones(input_tokens.size()[0], len(action2idx), dtype = torch.float).cuda()
                for i, target_action in enumerate(action2idx):
                    context_length = torch.sum(input_context, dim = 1)
                    for j in range(len(target_action)):
                        input_tokens[0][context_length + j] = target_action[j]
                        output_ids[0][context_length + j-1] = target_action[j]
                    output_ids[0][context_length + len(target_action) - 1: ] = -100

                    # a = input_tokens[0].cpu().numpy().tolist()
                    # print(tokenizer.decode(a).rstrip("<pad>"))
                    # b = output_ids[0].cpu().numpy().tolist()
                    # print(tokenizer.decode(b).rstrip("<pad>"))

                    logits = model(input_tokens, input_length, input_context, input_span)
                    loss = loss_func(logits.view(-1, logits.size(-1)), output_ids.view(-1))
                    target_probs[:, i] = - loss.item()
                
                # print(target_probs)

                topk = torch.topk(target_probs, 3)[1].cpu().numpy().tolist()[0]
                all_topk.append(topk)

                target_probs = target_probs.argmax(dim=-1)
                acc += torch.sum(target_probs == target).item()
                # print(acc)
                loss = loss.sum()
                total_loss += loss.item()
                total_num += target_probs.shape[0]
                target_list += target.cpu().numpy().tolist()
                pred_list += target_probs.cpu().numpy().tolist()

                # print('pred')
                # print(all_actions[pred_list[-1]])
                # print('\n')
                # print('topk')
                # for k in topk:
                #     print(all_actions[k])
                # print('\n')
                # print('target')
                # print(all_actions[target_list[-1]])
                # print('\n')
                # input()

    topk_dic = {v: 0 for v in all_actions}
    num_dic = {v: 1e-6 for v in all_actions}
    for idx, k in enumerate(target_list):
        num_dic[all_actions[k]] += 1
        if k in all_topk[idx]:
            topk_dic[all_actions[k]] += 1
    topk_dic = {k: float(v) / num_dic[k] for k,v in topk_dic.items()}
    print(topk_dic)
    print(np.mean(list(topk_dic.values())))

    macro_f1 = f1_score(target_list, pred_list, average='macro')
    micro_f1 = f1_score(target_list, pred_list, average='micro')
    all_f1 = f1_score(target_list, pred_list, average=None).tolist()
    all_f1 = {all_actions[idx]: v for idx, v in enumerate(all_f1)}
    macro_precision = precision_score(target_list, pred_list, average='macro')
    micro_precision = precision_score(target_list, pred_list, average='micro')
    macro_recall = recall_score(target_list, pred_list, average='macro')
    micro_recall = recall_score(target_list, pred_list, average='micro')

    return {'micro_f1': micro_f1, 'macro_f1': macro_f1, 'micro_precision': micro_precision, 'macro_precision': macro_precision, 'micro_recall': micro_recall, 'macro_recall': macro_recall, 'avg_loss': total_loss / total_num, 'all_f1': all_f1}

def get_topk_query(model, tokenizer, input_tokens, input_length, input_span, input_context, max_length):
    context_length = torch.sum(input_context, dim = 1).cpu().tolist()[0] + 2
    # print(tokenizer.decode(a[context_length: ]).rstrip("<pad>"))
    spans = 50
    temperature, top_k, top_p = 1, 1, 0.9
    decode_query = ""
    with torch.inference_mode():
        for i in range(context_length - 1, context_length + spans - 1):
            if i >= max_length-1:
                continue
            # print('\n')
            # a = input_tokens[0].cpu().numpy().tolist()[: i+1]
            # print(tokenizer.decode(a[context_length: ]).rstrip("<pad>"))
            logits = model(input_tokens, input_length, input_context, input_span)
            logits = logits[:, i, :] / temperature
            logits = top_k_logits(logits, top_k = top_k, top_p = top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            # more_tokens = torch.multinomial(probs, num_samples=10).squeeze(1)
            # print(more_tokens[0].cpu().numpy().tolist())
            # print(tokenizer.decode(more_tokens[0].cpu().numpy().tolist()))

            # input()
            input_tokens[0][i + 1] = next_token
            input_length += 1
            # input_context[0][i+1] = True
            decoded_token = tokenizer.decode([next_token.cpu().item()])
            if decoded_token in ["<eod>", "<pad>", "蟪", "蟥"]:
                break
            decode_query += decoded_token

    return decode_query

def get_beam_query(model, tokenizer, input_tokens, input_length, input_span, input_context, return_best = True):
    decode_query = ''
    span = 50
    for it in generate_beam(model, tokenizer, input_tokens, input_length, input_span, input_context, span, no_repeat_ngram_size=0, temperature=1.0, top_k=1, return_best = return_best):
        if it in ['<eod>', "<pad>", "蟪", "蟥"]:
            break
        decode_query += it

    return decode_query


def evaluate_query(args, model, dataloader, tokenizer, loss_func, verbalizer, all_actions, is_save=True, decoding_method = 'topk'):
    model.eval()
    with torch.no_grad():
        total_num = 0
        total_loss = 0
        acc = 0
        target_list = []
        pred_list = []
        all_topk = []

        dump_answer = []
        rouges = []

        def get_answer(s):
            for i in range(0, 100):
                if s[-i] == '；' and s[-i-1] == 'J':
                    return s[-i+1: ].rstrip('<eo')

        for it, data in enumerate(dataloader):
            idx = data["idx"]
            input_tokens = data["input_tokens"]
            input_length = data["input_length"]
            input_context = data["input_context"]
            input_span = data["input_span"]
            output_ids = data["output_ids"]
            target = data["target"]
            index = data["index"]
            # index = data["index"]
            # truth = data["truth"]
            feasible_state = data["feasible_state"]

            a = input_tokens[0].cpu().numpy().tolist()
            # print(tokenizer.decode(a).rstrip("<pad>"))
            # b = output_ids[0].cpu().numpy().tolist()
            # print(tokenizer.decode(b).rstrip("<pad>"))

            # c = 0
            # for item in b:
            #     if item == -100:
            #         c += 1
            #     else:
            #         break
            # print(tokenizer.decode(a[c: ]))

            if decoding_method == 'beam':
                decode_query = get_beam_query(model, tokenizer, input_tokens, input_length, input_span, input_context)
            elif decoding_method == 'topk':
                decode_query = get_topk_query(model, tokenizer, input_tokens, input_length, input_span, input_context, args.max_length)
            
            input_converted = tokenizer.decode(a).rstrip("<pad>")
            dump_answer.append({'query': input_converted, 'pred': decode_query})
            ground_truth_query = get_answer(input_converted)
            try:
                r = get_rouge_over_list(decode_query, ground_truth_query)
            except:
                r = 0
            rouges.append(r)
            # print(r)
            # print(decode_query)
            # input()

    if is_save:
        json.dump(dump_answer, open('generated_query_topk_webgpt_newdata.json', 'w', encoding='utf8'), indent = 4, ensure_ascii=False)
        exit()
    else:
        return {'rouge': np.mean(rouges)}

def get_abstract(model, tokenizer, input_tokens, output_ids, input_length, input_span, input_context):
    a = input_tokens[0].cpu().numpy().tolist()
    # print(tokenizer.decode(a).rstrip("<pad>"))
    b = output_ids[0].cpu().numpy().tolist()
    # print(tokenizer.decode(b).rstrip("<pad>"))

    for idx in range(len(b) - 5):
        if b[idx: idx + 5] == [367, 16, 26992, 10459, 15]:
            start_truth = b[idx + 5: idx + 15]
        if b[idx: idx + 3] == [23156, 10459, 15]:
            end_truth = b[idx + 3: idx + 13]

    # c = 0
    # for item in b:
    #     if item == -100:
    #         c += 1
    #     else:
    #         break
    # print(tokenizer.decode(a[c: ]))

    abstract_content = a[: torch.sum(input_context, dim = 1).cpu().tolist()[0]]
    start = -1
    end = -1
    for idx in range(2000):
        if abstract_content[idx: idx + 3] == [12585, 20730, 15]:
            start = idx
            break
    for idx in range(start, 2000):
        if abstract_content[idx: idx + 5] == [6145, 15046, 17606, 15359, 15]:
            end = idx
            break
    abstract_content = abstract_content[start+3: end]

    method = 'topk'
    if method == 'beam':
        decode_query = ''
        span = 50
        for it in generate_beam(model, tokenizer, input_tokens, input_length, input_span, input_context, span, no_repeat_ngram_size=0, temperature=1.0, top_k=1):
            if it in ['<eod>', "<pad>", "蟪", "蟥"]:
                break
            decode_query += it
    elif method == 'topk':
        start_num = 5
        generated_token_num = 0
        for idx in [0, 1]:
            if idx == 0:
                context_length = torch.sum(input_context, dim = 1).cpu().tolist()[0] + start_num
            elif idx == 1:
                context_length = torch.sum(input_context, dim = 1).cpu().tolist()[0] + start_num + generated_token_num
            # print(tokenizer.decode(a[context_length: ]).rstrip("<pad>"))
            spans = 10
            temperature, top_k, top_p = 1, 1, 0.9
            decode_query = ""
            decode_query_idx = []
            with torch.inference_mode():
                for i in range(context_length - 1, context_length + spans - 1):
                    # print('\n')
                    # a = input_tokens[0].cpu().numpy().tolist()
                    # print(tokenizer.decode(a[torch.sum(input_context, dim = 1).cpu().tolist()[0]: ]).rstrip("<pad>"))
                    logits = model(input_tokens, input_length, input_context, input_span)
                    logits = logits[:, i, :] / temperature
                    logits = top_k_logits(logits, top_k = top_k, top_p = top_p)

                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

                    # more_tokens = torch.multinomial(probs, num_samples=10).squeeze(1)
                    # print(more_tokens[0].cpu().numpy().tolist())
                    # print(tokenizer.decode(more_tokens[0].cpu().numpy().tolist()))
                    input_tokens[0][i + 1] = next_token
                    generated_token_num += 1
                    # input_length += 1
                    # input_context[0][i+1] = True
                    decoded_token = tokenizer.decode([next_token.cpu().item()])
                    decode_query_idx.append(next_token.cpu().item())
                    if decoded_token in ["<eod>", "<pad>", "蟪", "蟥"]:
                        break
                    decode_query += decoded_token

            if idx == 0:
                for next_token in [71, 23156, 10459, 15]:
                    input_tokens[0][context_length + generated_token_num] = next_token
                    generated_token_num += 1

            def find_position(abstract_content, decode_query_idx, tokenizer, if_end):
                min_leven = 10
                min_abs_idx = -1
                for abs_idx in range(len(abstract_content) - len(decode_query_idx)):
                    sm = edit_distance.SequenceMatcher(a=tokenizer.decode(abstract_content[abs_idx: abs_idx + len(decode_query_idx)]), b=tokenizer.decode(decode_query_idx))
                    leven = sm.distance()

                    if if_end == 0:
                        if min_leven > leven:
                            min_abs_idx = abs_idx
                            min_leven = leven
                    elif if_end == 1:
                        if min_leven >= leven:
                            min_abs_idx = abs_idx + len(decode_query_idx)
                            min_leven = leven
                return min_abs_idx

            if idx == 0:
                min_abs_idx_start = find_position(abstract_content, decode_query_idx, tokenizer, idx)
                min_abs_idx_start_truth = find_position(abstract_content, start_truth, tokenizer, idx)
            elif idx == 1:
                min_abs_idx_end = find_position(abstract_content, decode_query_idx, tokenizer, idx)
                min_abs_idx_end_truth = find_position(abstract_content, end_truth, tokenizer, idx)

    def get_final_abstract(start, end, abstract_content, tokenizer):
        if start == -1 or end == -1:
            final_abstract = "Did not find the quote!"
        elif start < end:
            final_abstract = tokenizer.decode(abstract_content[start: end])
        else:
            final_abstract = "ending before starting"
        return final_abstract

    final_abstract = get_final_abstract(min_abs_idx_start, min_abs_idx_end, abstract_content, tokenizer)
    final_abstract_truth = get_final_abstract(min_abs_idx_start_truth, min_abs_idx_end_truth, abstract_content, tokenizer)

    return final_abstract, final_abstract_truth

def evaluate_abstract(args, model, dataloader, tokenizer, loss_func, verbalizer, all_actions, is_save = True):
    model.eval()
    with torch.no_grad():
        total_num = 0
        total_loss = 0
        acc = 0
        target_list = []
        pred_list = []
        all_topk = []

        rouges = []
        dump_answer = []

        for it, data in enumerate(dataloader):
            idx = data["idx"]
            input_tokens = data["input_tokens"]
            input_length = data["input_length"]
            input_context = data["input_context"]
            input_span = data["input_span"]
            output_ids = data["output_ids"]
            target = data["target"]
            index = data["index"]
            # index = data["index"]
            # truth = data["truth"]
            feasible_state = data["feasible_state"]

            a = input_tokens[0].cpu().numpy().tolist()
            final_abstract, final_abstract_truth = get_abstract(model, tokenizer, input_tokens, output_ids, input_length, input_span, input_context)

            if final_abstract in ["Did not find the quote!", "ending before starting"]:
                r = 0
            else:
                r = get_rouge_over_list(final_abstract, final_abstract_truth)
            rouges.append(r)

            # print(final_abstract)
            dump_answer.append({'query': tokenizer.decode(a).rstrip("<pad>"), 'pred': final_abstract})
            # input()
    if is_save:
        json.dump(dump_answer, open('generated_abstract_topk_webgpt_newdata_new.json', 'w', encoding='utf8'), indent = 4, ensure_ascii=False)
        exit()
    else:
        return {'rouge': np.mean(rouges)}

def main():
    args = initialize()
    tokenizer, model, optimizer = setup_model_and_optimizer(args)
    inference = 1
    abstract_only = 0
    query_only = 0
    dataset, verbalizer, all_actions, action2idx = prepare_dataset(
        args,
        tokenizer,
        f"{args.base_path}/down_data/",
        args.dataset_name,
        bmt.rank(), bmt.world_size(),
        query_only,
        abstract_only,
        inference,
    )

    finetune(args, tokenizer, model, optimizer, dataset, verbalizer, all_actions, query_only, abstract_only, action2idx, inference, base_path = f"{args.base_path}/down_data/", rank = bmt.rank(), world_size = bmt.world_size())

if __name__ == "__main__":
    main()
