import time
import random
import torch
import bmtrain as bmt
import numpy as np
import os
import csv
import json

from model_center import get_args
from model_center.model import CPM1
from model_center.tokenizer import CPM1Tokenizer
from model_center.dataset.cpm1dataset import DATASET
from model_center.utils import print_inspect
from model_center.dataset import DistributedDataLoader
from generation import generate_no_beam, generate_no_beam_qa
from compare import get_rouge_over_list

# added by Yujia
import torch.nn.functional as F

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
    # get the optimizer and lr_scheduler
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
    os.environ["MASTER_PORT"] = (str)((int)(os.environ["MASTER_PORT"]) + 123)
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
    verbalizer = torch.LongTensor(DATASET[dataset_name].get_verbalizer(tokenizer)).cuda()
    return dataset, verbalizer

def finetune(args, tokenizer, model, optimizer, dataset, verbalizer):
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    dataloader = {
        "train": DistributedDataLoader(dataset['train'], batch_size=args.train_batch_size, shuffle=True),
        "dev": DistributedDataLoader(dataset['dev'], batch_size=args.eval_batch_size, shuffle=False),
        "test": DistributedDataLoader(dataset['test'], batch_size=args.eval_batch_size, shuffle=False),
    }

    lr_scheduler = get_learning_rate_scheduler(args, optimizer, dataloader)

    best_dev_metric = 0

    for epoch in range(args.epochs):
        model.train()
        global_loss = 0
        for it, data in enumerate(dataloader['train']):
            idx = data["idx"]
            input_tokens = data["input_tokens"]
            input_length = data["input_length"]
            input_context = data["input_context"]
            input_span = data["input_span"]
            output_ids = data["output_ids"]
            truth = data["truth"]

            # a = input_tokens[0].cpu().numpy().tolist()
            # print(tokenizer.decode(a))
            # b = output_ids[0].cpu().numpy().tolist()
            # print(tokenizer.decode(b))
            # c = 0
            # for item in b:
            #     if item == 0:
            #         c += 1
            #     else:
            #         break
            # print(tokenizer.decode(a[c: ]))
            # print(truth[0])
            # input()

            logits = model(input_tokens, input_length, input_context, input_span)

            loss = loss_func(logits.view(-1, logits.size(-1)), output_ids.view(-1))
            # lprobs = F.log_softmax(logits, dim=-1)

            # output_ids = output_ids.unsqueeze(-1)
            # loss = -lprobs.gather(dim=-1, index=output_ids)
            # pad_mask = output_ids.eq(tokenizer.pad_id)

            # loss.masked_fill_(pad_mask, 0.0)
            
            # loss = torch.sum(loss, dim = 1) / torch.sum(pad_mask, dim = 1)
            loss = loss.sum() / args.accumulation_steps

            global_loss += loss.item()

            loss = optimizer.loss_scale(loss)
            loss.backward()

            if it % args.accumulation_steps == 0:
                grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, args.clip_grad, scale = optimizer.scale, norm_type = 2)
                bmt.optim_step(optimizer, lr_scheduler)
                optimizer.zero_grad()

                if it % args.log_interval == 0:
                    global_loss = global_loss / args.log_interval / args.accumulation_steps
                    bmt.print_rank(
                        "train | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                            epoch,
                            it,
                            len(dataloader["train"]),
                            global_loss,
                            lr_scheduler.current_lr,
                            int(optimizer.scale),
                            grad_norm
                        )
                    )
                    global_loss = 0

        dev_metric = evaluate_generate(args, model, tokenizer) 
        # dev_metric = evaluate_loss(args, model, dataloader['dev'], tokenizer, loss_func)            

        bmt.print_rank(
            "dev | epoch {:3d} | rouge: {:6f} |".format(
                epoch,
                dev_metric,
            )
        )

        # to-do
        if args.save != None:
            if dev_metric > best_dev_metric:
                bmt.save(model, os.path.join(args.save, args.save_name+("-best.pt")))
                best_dev_metric = dev_metric
            # bmt.save(model, os.path.join(args.save, args.save_name+(f"-{epoch}.pt")))

        bmt.print_rank("resampling the dataloader")
        dataloader = {
            "train": DistributedDataLoader(dataset['train'], batch_size=args.train_batch_size, shuffle=True),
            "dev": DistributedDataLoader(dataset['dev'], batch_size=args.eval_batch_size, shuffle=False),
            "test": DistributedDataLoader(dataset['test'], batch_size=args.eval_batch_size, shuffle=False),
        }
    bmt.print_rank("best dev metric: " + str(best_dev_metric))

def evaluate_generate(args, model, tokenizer):
    dataset = json.load(open(args.base_path + "/down_data/webgpt_qa/dev.json", 'r'))
    span = 500
    all_pred = {}

    # todo
    dataset = dataset[:5]
    rouges = []
    model.eval()
    with torch.no_grad():
        for data in dataset:
            doc = data['context']
            question = data['question']
            answer = data['answer']
            pred = []
            doc = "".join(doc)
            for it in generate_no_beam_qa(model, tokenizer, doc, question, span, no_repeat_ngram_size=0, temperature=1.0, top_k=1):
                if it == '<eod>':
                    break
                pred.append(it)
            pred = ''.join(pred)
            try:
                r = get_rouge_over_list(pred, answer)
            except:
                r = 0
                print('r=0 wrong here!')
                print('pred')
                print(pred)
                print('answer')
                print(answer)
            rouges.append(r)

    return np.mean(rouges)

def evaluate_loss(args, model, dataloader, tokenizer, loss_func):
    model.eval()
    with torch.no_grad():
        total_num = 0
        total_loss = 0

        # checkpoint = torch.load(os.path.join(args.save, args.save_name+("-%d.pt" % 2)))
        # checkpoint = {k: v.cuda() for k,v in checkpoint.items()}
        # model.load_state_dict(checkpoint)

        for it, data in enumerate(dataloader):
            input_tokens = data["input_tokens"]
            input_length = data["input_length"]
            input_context = data["input_context"]
            input_span = data["input_span"]
            output_ids = data["output_ids"]
            truth = data["truth"]

            logits = model(input_tokens, input_length, input_context, input_span)
            loss = loss_func(logits.view(-1, logits.size(-1)), output_ids.view(-1))
            # lprobs = F.log_softmax(logits, dim=-1)

            # output_ids = output_ids.unsqueeze(-1)
            # loss = -lprobs.gather(dim=-1, index=output_ids)
            # pad_mask = output_ids.eq(tokenizer.pad_id)
            # loss.masked_fill_(pad_mask, 0.0)
            
            # loss = torch.sum(loss, dim = 1) / torch.sum(pad_mask, dim = 1)
            loss = loss.sum()

            total_num += logits.shape[0]
            total_loss += loss.item()

            # temperature = 1.0
            # answer_list = []
            # for i in range(input_length.cpu().numpy().tolist()[0] - 1, args.max_length):
            #     logits = model(input_tokens, input_length, input_context, input_span)
            #     logits = logits[:, i, :] / 1.0
            #     logits = top_k_logits(logits, top_k = 10, top_p = 0.9)
            #     probs = F.softmax(logits, dim=-1)
            #     next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            #     answer_list.append(next_token)
            #     if i < args.max_length - 1:
            #         input_tokens[0][i + 1] = next_token
            #     if next_token == 4:
            #         break
            #     # context[0][i+1] = True
            # answer = torch.cat(answer_list).cpu().numpy().tolist()
            # print(tokenizer.decode(answer))
            # print(truth)
            # print('\n')
            # input()
            
            # for i in input_tokens[0].cpu().numpy():
            #     yield tokenizer.decode([i])

    return {'loss': total_loss / total_num}

def main():
    args = initialize()
    tokenizer, model, optimizer = setup_model_and_optimizer(args)
    dataset, verbalizer = prepare_dataset(
        args,
        tokenizer,
        f"{args.base_path}/down_data/",
        args.dataset_name,
        bmt.rank(), bmt.world_size(),
    )
    finetune(args, tokenizer, model, optimizer, dataset, verbalizer)

if __name__ == "__main__":
    main()
