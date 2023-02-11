import time
import random
import torch
import sys
import os
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

from model_center import get_args
from model_center.model import CPM1
from model_center.tokenizer import CPM1Tokenizer
from model_center.dataset.cpm1dataset import DATASET
from model_center.utils import print_inspect
from model_center.dataset import DistributedDataLoader
import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, top_k_accuracy_score

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
    bmt.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def prepare_dataset(args, tokenizer, base_path, dataset_name, rank, world_size):
    splits = ['train', 'dev', 'test']
    dataset = {}
    for split in splits:
        dataset[split] = DATASET[dataset_name](base_path, split, rank, world_size, tokenizer, args.max_length, debug = args.debug, do_infer = False if split == 'train' else True) # ratio = 0.005)
    verbalizer = torch.LongTensor(dataset[split].action_verbalizer).cuda()
    action2idx = dataset[split].action2idx
    all_actions = list(dataset[split].action2idx.keys())

    return dataset, verbalizer, all_actions, action2idx

def finetune(args, tokenizer, model, optimizer, dataset, verbalizer, all_actions, action2idx):
    # torch.set_printoptions(profile="full")
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    
    dataloader = {
        # shuffle of train data set to False to facilitate in-batch contrastive learning
        "train": DistributedDataLoader(dataset['train'], batch_size=args.train_batch_size, shuffle=False, is_eval = False),
        "dev": DistributedDataLoader(dataset['dev'], batch_size=args.eval_batch_size, shuffle=False, is_eval = True),
        "test": DistributedDataLoader(dataset['test'], batch_size=args.eval_batch_size, shuffle=False, is_eval = True),
    }

    lr_scheduler = get_learning_rate_scheduler(args, optimizer, dataloader)

    # # load the checkpoint if terminated:
    # if os.path.exists(os.path.join(args.save, args.save_name + "-last" + ".pt")):
    #     bmt.print_rank('found existing checkpoints, loading it now...')
    #     misc_to_load = torch.load(os.path.join(args.save, args.save_name + "-last" + ".misc"))
    #     global_step = misc_to_load['global_step']
    #     lr_scheduler.load_state_dict(misc_to_load['lr_scheduler'])
    #     optimizer.load_state_dict(misc_to_load['optimizer'])
    #     best_dev_metric = misc_to_load['best_dev_metric']
    #     last_epoch = misc_to_load['epoch']
    #     checkpoint = torch.load(os.path.join(args.save, args.save_name + "-last" + ".pt"))
    #     checkpoint = {k: v.cuda() for k,v in checkpoint.items()}
    #     model.load_state_dict(checkpoint)
    # else:
    #     best_dev_metric = None
    #     global_step = 0
    #     last_epoch = 0

    global_step = 0
    best_dev_metric = None
    # for epoch in range(last_epoch+1, args.epochs):
    for epoch in range(args.epochs):
        dev_metric = evaluate(args, model, dataloader['dev'], tokenizer, loss_func, verbalizer, all_actions, action2idx)
 
        bmt.print_rank(
            "dev | epoch {:3d} | micro_f1: {:6f} | macro_f1: {:6f}  | micro_precision: {:6f} | macro_precision: {:6f}  | macro_recall: {:6f} | micro_recall: {:6f}  |avg_loss: {:6f} |".format(
                epoch,
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
        exit()

        # if args.save != None:
        #     if best_dev_metric == None:
        #         best_dev_metric = dev_metric
        #         save_ckpt(args, model, global_step, lr_scheduler, optimizer, best_dev_metric, "-best", epoch)
        #     elif dev_metric['macro_f1'] > best_dev_metric['macro_f1']:
        #         best_dev_metric = dev_metric
        #         save_ckpt(args, model, global_step, lr_scheduler, optimizer, best_dev_metric, "-best", epoch)
        #     # save_ckpt(args, model, global_step, lr_scheduler, optimizer, best_dev_metric, "-last", epoch)
    bmt.print_rank(best_dev_metric)
    bmt.print_rank("The training is done.")

def save_ckpt(args, model, global_step, lr_scheduler, optimizer, best_dev_metric, prefix, epoch):
    bmt.save(model, os.path.join(args.save, args.save_name + prefix + ".pt"))
    # misc_to_save = {
    #     'global_step': global_step,
    #     'lr_scheduler': lr_scheduler.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'best_dev_metric': best_dev_metric,
    #     'epoch': epoch,
    # }
    # torch.save(misc_to_save, os.path.join(args.save, args.save_name + prefix + ".misc"))

def evaluate(args, model, dataloader, tokenizer, loss_func, verbalizer, all_actions, action2idx):
    model.eval()
    with torch.no_grad():
        total_num = 0
        total_loss = 0
        acc = 0
        target_list = []
        pred_list = []
        all_topk = []

        checkpoint = torch.load(os.path.join(args.save, args.load))
        checkpoint = {k: v.cuda() for k,v in checkpoint.items()}
        model.load_state_dict(checkpoint)

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

            method = 'ppl'
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
    # return {'micro_acc': acc / total_num, 'macro_acc': np.mean(list(acc_dic.values())), 'micro_f1': micro_f1, 'macro_f1': macro_f1, 'micro_precision': micro_precision, 'macro_precision': macro_precision, 'micro_recall': micro_recall, 'macro_recall': macro_recall, 'avg_loss': total_loss / total_num, 'acc_dic': acc_dic}

def main():
    args = initialize()
    tokenizer, model, optimizer = setup_model_and_optimizer(args)
    dataset, verbalizer, all_actions, action2idx = prepare_dataset(
        args,
        tokenizer,
        f"{args.base_path}/down_data/",
        args.dataset_name,
        bmt.rank(), bmt.world_size(),
    )

    finetune(args, tokenizer, model, optimizer, dataset, verbalizer, all_actions, action2idx)

if __name__ == "__main__":
    main()