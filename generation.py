import torch
import torch.nn.functional as F
import numpy as np
import bmtrain as bmt
import math

# class BeamHypotheses(object):
#     def __init__(self, num_beams, max_length, length_penalty, early_stopping, tokenizer=None):
#         """
#         Initialize n-best list of hypotheses.
#         """
#         self.max_length = max_length - 1  # ignoring bos_token
#         self.length_penalty = length_penalty
#         self.early_stopping = early_stopping
#         self.num_beams = num_beams
#         self.length_fact = []
#         self.beams = []
#         self.worst_score = 1e9
#         self.raw_worst_score = 1e9

#         self.tokenizer = tokenizer

#     def __len__(self):
#         """
#         Number of hypotheses in the list.
#         """
#         return len(self.beams)

#     def add(self, hyp, sum_logprobs):
#         """
#         Add a new hypothesis to the list.
#         """
#         score = sum_logprobs / len(hyp) ** self.length_penalty
#         # print(f'add hyp = {self.tokenizer.decode(hyp.cpu().tolist())}, score = {score}')
#         if len(self) < self.num_beams or score > self.worst_score:
#             self.beams.append((score, hyp))
#             self.length_fact.append(len(hyp) ** self.length_penalty)
#             if len(self) > self.num_beams:
#                 sorted_scores = sorted([(s, idx, _) for idx, (s, _) in enumerate(self.beams)])
#                 del self.beams[sorted_scores[0][1]]
#                 self.worst_score = sorted_scores[1][0]
#                 self.raw_worst_score = self.worst_score * (len(sorted_scores[1][2]) ** self.length_penalty)
#             else:
#                 self.worst_score = min(score, self.worst_score)
#                 self.raw_worst_score = sum_logprobs
        
#         # print('maintained hypothesis: ')
#         # for score, hyp in self.beams:
#         #     print(f'raw_score = {score * (len(hyp) ** self.length_penalty)}, score = {score}, hyp = {self.tokenizer.decode(hyp.cpu().tolist())}')

#     def is_done(self, best_sum_logprobs, cur_len):
#         """
#         If there are enough hypotheses and that none of the hypotheses being generated
#         can become better than the worst one in the heap, then we are done with this sentence.
#         """

#         if len(self) < self.num_beams:
#             return False
#         elif self.early_stopping:
#             return True
#         else:
#             cur_score = best_sum_logprobs / cur_len ** self.length_penalty
#             # print(f'cur best score = {cur_score}, cur worst score = {self.worst_score}, cur raw worst score = {self.raw_worst_score}')
#             ret = self.worst_score >= cur_score

#             # print("in beam")
#             # for x in self.beams:
#             #     print(x[0], self.tokenizer.decode(x[1].cpu().tolist()))
#             # print("end beam")

#             return ret


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping, tokenizer=None):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        # try to penalize repetation (fail)
        # score = sum_logprobs / len(set(hyp.cpu().tolist())) ** self.length_penalty
        
        # bmp.print_rank(sum_logprobs, len(hyp))
        # bmp.print_rank(f'score = {score}, hyp = {self.tokenizer.decode(hyp.cpu().tolist())}')
        # bmp.print_rank('============================')

        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
        
    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / cur_len ** self.length_penalty


# def calc_banned_ngram_tokens(prev_input_ids, num_hypos: int, no_repeat_ngram_size: int, tokenizer):
#     """Copied from fairseq for no_repeat_ngram in beam_search"""
#     # cur_len = prev_input_ids.size(-1)
#     # # prev_input_words = tokenizer.decode(prev)
#     # if cur_len + 1 < no_repeat_ngram_size:
#     #     # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
#     #     return [[] for _ in range(num_hypos)]
#     generated_ngrams = [{} for _ in range(num_hypos)]
#     prev_input_words = []
#     for ids in prev_input_ids:
#         tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
#         words = []
#         for token in tokens:
#             token = token.replace("</_>", "") # NOTE: "▁" is different from "_"
#             if len(token) > 0:
#                 if token in ['<sep>', "<unk>", "<s>", "</s>", "<eod>", "<mask>"]:
#                     words.append(token)
#                 else:
#                     words += list(token)
#         prev_input_words.append(words)
#     # print(prev_input_words)
#     for idx in range(num_hypos):
#         gen_words = prev_input_words[idx]
#         # print('gen_words = ', gen_words)
#         # gen_tokens = prev_input_ids[idx].tolist()
#         # gen_words = tokenizer.decode(gen_tokens)
#         generated_ngram = generated_ngrams[idx]
#         for ngram in zip(*[gen_words[i:] for i in range(no_repeat_ngram_size)]):
#             for prefix_len in range(no_repeat_ngram_size):
#                 prev_ngram = ''.join(ngram[:prefix_len])
#                 if "</n>" not in prev_ngram:
#                     suffix_ngram = ''.join(ngram[prefix_len:])
#                     suffix_ngram_2 = "▁" + suffix_ngram
#                     if tokenizer.check(suffix_ngram): # 在词表中
#                         generated_ngram[prev_ngram] = generated_ngram.get(prev_ngram, set()) | set([suffix_ngram])
#                     if tokenizer.check(suffix_ngram_2): # 在词表中
#                         generated_ngram[prev_ngram] = generated_ngram.get(prev_ngram, set()) | set([suffix_ngram_2])
#             # prev_ngram_tuple = ''.join(ngram[:-1])
#             # generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, set()) | set([ngram[-1]])
#     # for g in generated_ngrams:
#     #     print(g)
#     # print('generated_ngrams = ', generated_ngrams)

#     def _get_generated_ngrams(hypo_idx):
#         # Before decoding the next token, prevent decoding of ngrams that have already appeared

#         cur_len = len(prev_input_words[hypo_idx])
        
#         generated_ngram_idx = []
#         for prefix_len in range(no_repeat_ngram_size):
#             # print('')
#             ngram_words = ''.join(prev_input_words[hypo_idx][cur_len-prefix_len:])
#             # print('prev_input = ', prev_input_words[hypo_idx])
#             # print('ngram_words = ', ngram_words)
#             generated_ngram_words = generated_ngrams[hypo_idx].get(ngram_words, [])
#             # print('generated_ngram_words = ', generated_ngram_words)
#             # print('all generated_ngrams = ', generated_ngrams[hypo_idx])
#             generated_ngram_idx += tokenizer.convert_tokens_to_ids(generated_ngram_words)
#             # generated_ngram_idx += [x for word in generated_ngram_words for x in tokenizer.get_prefix_id_list(word)]
#             # print('generated_ngram_idx = ', generated_ngram_idx)
#             # print('='*100)
#         prev_input_str = "".join(prev_input_words[hypo_idx])
#         # print("prev input str", prev_input_str)
#         if prev_input_str[-1] in ['，', ',']:
#             generated_ngram_idx.append(tokenizer.convert_token_to_id('但'))
#             generated_ngram_idx.append(tokenizer.convert_token_to_id('▁但'))
#             generated_ngram_idx.append(tokenizer.convert_token_to_id(','))
#         if prev_input_str[-2:] in ["我是", "我叫"]:
#             generated_ngram_idx.append(tokenizer.convert_token_to_id(','))
#             generated_ngram_idx.append(tokenizer.convert_token_to_id('▁.'))
#             generated_ngram_idx.append(tokenizer.convert_token_to_id('.'))
#             generated_ngram_idx.append(tokenizer.convert_token_to_id('。'))
#             generated_ngram_idx.append(tokenizer.convert_token_to_id('▃'))
#             generated_ngram_idx.append(tokenizer.convert_token_to_id('—'))


#         return generated_ngram_idx

#     banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
#     return banned_tokens

def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    # 拿到n-gram中需要ban的最后一个token的list
    # return banned_ngrams.get(ngram_idx, []), ngram_idx # for debug
    return banned_ngrams.get(ngram_idx, [])

def calc_banned_ngram_tokens(
    prev_input_ids: torch.Tensor, num_hypos: int, ngram_size: int, start_idx=None, end_idx=None, window_size=None, tokenizer=None):
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if start_idx is not None and end_idx is not None:
        # 可能end_idx < start_idx，但符合逻辑
        if window_size:
            prev_input_ids = prev_input_ids[:, max(start_idx, end_idx + 1 - window_size): end_idx+1]
        else:
            prev_input_ids = prev_input_ids[:, start_idx: end_idx+1]
        
    cur_len = prev_input_ids.size(1)
    
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]

    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)

    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    # for hypo_idx in range(num_hypos):
    #     bmp.print_rank(tokenizer.decode(list(banned_tokens[hypo_idx][1])) + "|" + "/".join([tokenizer.decode([x]) for x in banned_tokens[hypo_idx][0]]))
    return banned_tokens
    # return [x[0] for x in banned_tokens]

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


def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids, start_idx=None,  end_idx=None):
    if start_idx is not None and end_idx is not None:
        # 可能end_idx < start_idx，但符合逻辑
        prev_input_ids = prev_input_ids[:, start_idx: end_idx+1]
        
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue
            # 如果最后一个token之前的token都match上了，那就把最后一个token禁掉
            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens

# def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
#     banned_tokens = []

#     def _tokens_match(prev_tokens, tokens):
#         if len(tokens) == 0:
#             # if bad word tokens is just one token always ban it
#             return True
#         if len(tokens) > len(prev_input_ids):
#             # if bad word tokens are longer then prev input_ids they can't be equal
#             return False

#         if prev_tokens[-len(tokens) :] == tokens:
#             # if tokens match
#             return True
#         else:
#             return False

#     for prev_input_ids_slice in prev_input_ids:
#         banned_tokens_slice = []

#         for banned_token_seq in bad_words_ids:
#             assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
#                 bad_words_ids
#             )

#             if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
#                 # if tokens do not match continue
#                 continue

#             banned_tokens_slice.append(banned_token_seq[-1])

#         banned_tokens.append(banned_tokens_slice)

#     return banned_tokens


def enforce_repetition_penalty_(tokenizer, 
                                lprobs, 
                                batch_size, 
                                num_beams, 
                                prev_output_tokens, 
                                repetition_penalty):

    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def postprocess_next_token_scores(tokenizer,
                                  scores,
                                  input_ids,
                                  no_repeat_ngram_size,
                                  bad_words_ids,
                                  repetition_penalty,
                                  batch_size,
                                  num_beams,
                                  start_idx=None,
                                  end_idx=None,
                                  window_size=None,
                                  min_len=None):

    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        enforce_repetition_penalty_(
            tokenizer, scores, batch_size, num_beams, input_ids, repetition_penalty, start_idx, end_idx, window_size
        )

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(input_ids, num_batch_hypotheses, no_repeat_ngram_size, start_idx, end_idx, window_size, tokenizer)
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    if bad_words_ids is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids, start_idx, end_idx)

        for i, banned_tokens in enumerate(banned_tokens):
            scores[i, banned_tokens] = -float("inf")

    # 允许生成eos和bos，以及换行
    scores[:, [0, 1, 2, 3, 4, 5, 6]] = -float("inf")

    # if start_idx is not None and end_idx is not None and min_len is not None:
    #     min_length_constraint(scores, end_idx - start_idx + 2, min_len, tokenizer)

    return scores

# def postprocess_next_token_scores(tokenizer,
#                                   scores,
#                                   input_ids,
#                                   no_repeat_ngram_size,
#                                   bad_words_ids,
#                                   repetition_penalty,
#                                   batch_size,
#                                   num_beams):

#     # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
#     if repetition_penalty != 1.0:
#         enforce_repetition_penalty_(
#             tokenizer, scores, batch_size, num_beams, input_ids, repetition_penalty,
#         )

#     # set eos token prob to zero if min_length is not reached
#     # if eos_token_id is not None and cur_len < min_length:
#     #     scores[:, eos_token_id] = -10000

#     if no_repeat_ngram_size > 0:
#         # calculate a list of banned tokens to prevent repetitively generating the same ngrams
#         num_batch_hypotheses = batch_size * num_beams
#         # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
#         banned_batch_tokens = calc_banned_ngram_tokens(input_ids, num_batch_hypotheses, no_repeat_ngram_size, tokenizer=tokenizer)
#         for i, banned_tokens in enumerate(banned_batch_tokens):
#             scores[i, banned_tokens] = -10000

#     if bad_words_ids is not None:
#         # calculate a list of banned tokens according to bad words
#         banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

#         for i, banned_tokens in enumerate(banned_tokens):
#             scores[i, banned_tokens] = -10000

#     scores[:, 0] = -50000
#     scores[:, 1] = -50000
#     scores[:, 2] = -50000
#     scores[:, 3] = -50000
#     scores[:, 4] = -50000
#     scores[:, 5] = -50000
#     scores[:, 6] = -50000

#     return scores


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


def generate_no_beam(model, tokenizer, lef_sentence, rig_sentence, spans, 
                     temperature = .9, top_k = 0, top_p = 0.9,
                     no_repeat_ngram_size = 3, repetition_penalty = 1):

    lef_tokens = tokenizer.encode(lef_sentence)
    rig_tokens = tokenizer.encode(rig_sentence)
    lef_tokens = [1] + lef_tokens
    if len(rig_tokens) > 0:
        rig_tokens = rig_tokens + [4]

    input_tokens, input_length, input_span, context = make_input(lef_tokens, rig_tokens, spans)

    for i in range(len(lef_tokens) - 1, len(lef_tokens) + spans - 1):
        logits = model(input_tokens, input_length, context, input_span)
        logits = logits[:, i, :] / temperature
        logits = top_k_logits(logits, top_k = top_k, top_p = top_p)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_tokens[0][i + 1] = next_token
        # context[0][i+1] = True
    for i in input_tokens[0].cpu().numpy():
        yield tokenizer.decode([i])

def generate_no_beam_qa(model, tokenizer, doc, question, spans, 
                     temperature = .9, top_k = 0, top_p = 0.9,
                     no_repeat_ngram_size = 3, repetition_penalty = 1):

    context_tokens = tokenizer.encode(doc)
    question_tokens = tokenizer.encode(question)
    lef_tokens = [1] + tokenizer.encode("文章：") + context_tokens + tokenizer.encode("问题：") + question_tokens + tokenizer.encode("答案：")
    # lef_tokens = tokenizer.encode(lef_sentence)
    rig_tokens = tokenizer.encode("")
    # import pdb;pdb.set_trace()
    if len(rig_tokens) > 0:
        rig_tokens = rig_tokens + [4]

    input_tokens, input_length, input_span, context = make_input(lef_tokens, rig_tokens, spans)

    # print(input_tokens)
    # print(input_length)
    # print(input_span)
    # print(context)
    for i in range(len(lef_tokens) - 1, len(lef_tokens) + spans - 1):
        logits = model(input_tokens, input_length, context, input_span)
        logits = logits[:, i, :] / temperature # (bs, vocab)
        # mask = torch.empty_like(logits, device=logits.device, dtype=torch.bool).fill_(False)
        # mask[:, context_tokens] = True
        # mask[:, tokenizer.eod_id] = True
        # logits = torch.where(mask, logits, torch.empty_like(logits, device=logits.device, dtype=logits.dtype).fill_(-1000))
        logits = top_k_logits(logits, top_k = top_k, top_p = top_p)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_tokens[0][i + 1] = next_token

        yield tokenizer.decode([next_token.cpu().item()])
        # context[0][i+1] = True
    # for i in input_tokens[0].cpu().numpy():
    #     yield tokenizer.decode([i])


def generate_beam(model, tokenizer, input_tokens, 
                     input_length, input_span, context,
                     spans, beam_size = 3,
                     temperature = .9, top_k = 0, top_p = 0.9,
                     no_repeat_ngram_size = 0, repetition_penalty = 1, random_sample=False, min_len=None, return_best = True):
    
    vocab_size = tokenizer.vocab_size
    # (batch, max_length)
    # context_tokens = tokenizer.encode(doc)
    # question_tokens = tokenizer.encode(question)
    # batch_size = 1 # unsupport batch infer
    # lef_tokens = [1] + tokenizer.encode("文章：") + context_tokens + tokenizer.encode("问题：") + question_tokens + tokenizer.encode("答案：")
    # # lef_tokens = tokenizer.encode(lef_sentence)
    # rig_tokens = tokenizer.encode("")
    # if len(rig_tokens) > 0:
    #     rig_tokens = rig_tokens + [4]

    # input_tokens, input_length, input_span, context = make_input(lef_tokens, rig_tokens, spans)

    batch_size = 1
    max_length = input_tokens.size(-1)
    input_tokens = input_tokens.unsqueeze(1).expand(batch_size, beam_size, max_length)
    input_length = input_length.unsqueeze(1).expand(batch_size, beam_size)
    context = context.unsqueeze(1).expand(batch_size, beam_size, max_length)
    input_span = input_span.unsqueeze(1).expand(batch_size, beam_size, max_length)

    input_tokens = input_tokens.contiguous().view(batch_size*beam_size, max_length)
    input_length = input_length.contiguous().view(batch_size*beam_size, )
    context = context.contiguous().view(batch_size*beam_size, max_length)
    input_span = input_span.contiguous().view(batch_size*beam_size, max_length)
    done = [False for _ in range(batch_size)]
    # (batch_size * beam_size, 0)
    
    beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=input_tokens.device)
    beam_scores[:, 1:] = -1e9 # 确保第一次只在一个vocab大小里选取
    beam_scores = beam_scores.view(-1)

    # current position
    cur_len = 0

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(beam_size, spans, length_penalty=1 , early_stopping=False, tokenizer=tokenizer)
        for _ in range(batch_size)
    ]

    lef = torch.sum(context, dim = 1).cpu().tolist()[0] + 2
    # lef = len(lef_tokens)
    rig = lef + spans
    # bmt.print_rank(lef_tokens, rig_tokens+1)
    with torch.inference_mode():
        for i in range(lef-1, rig):
            logits = model(input_tokens, input_length, context, input_span)

            # skip all steps when we are done with each sentence
            if all(done):
                break # Note: break not supports multi-GPUs
    
            # (batch * beam, seqlen, model_dim)
            logits = logits[:, i, :] / temperature
            # logits = postprocess_next_token_scores(
            #     tokenizer=tokenizer,
            #     scores=logits,
            #     input_ids=input_tokens,
            #     no_repeat_ngram_size=no_repeat_ngram_size,
            #     bad_words_ids=[[0]],
            #     repetition_penalty=repetition_penalty,
            #     batch_size=batch_size,
            #     num_beams=beam_size,
            #     start_idx=lef,
            #     end_idx=i,
            #     window_size=None,
            #     min_len=min_len
            # )
            scores = F.log_softmax(logits, dim=-1)

            if random_sample:
                # TODO: need to check this part
                assert temperature != 0, "temperature should not be zero!"
                scores = scores - math.log(temperature)
                _scores = scores + beam_scores[:, None].expand_as(scores)
                             
                _scores = top_k_logits(_scores, top_k=top_k, top_p=top_p)
                _scores = _scores.contiguous().view(batch_size, beam_size * vocab_size)
                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_words = torch.multinomial(probs, num_samples=2 * beam_size)  # (batch_size, beam_size * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_words)  # (batch_size, beam_size * 2)
                # sort the sampled vector to make sure that the first beam_size samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_words = torch.gather(next_words, -1, next_scores_indices)  # (batch_size, beam_size * 2)            
            else:
                # import pdb; pdb.set_trace()
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * beam_size, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, beam_size * vocab_size
                )  # (batch_size, beam_size * vocab_size)

                next_scores, next_words = torch.topk(next_scores, 2 * beam_size, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, 2 * beam_size)
            # next batch beam content
            next_batch_beam = []

            for sent_id in range(batch_size):
                 # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item(), cur_len)
                if done[sent_id]:
                    next_batch_beam.extend([(0, tokenizer.pad_id, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    # print(word_id, tokenizer.eod_id, cur_len, spans)
                    if word_id == tokenizer.eod_id or cur_len == spans:
                        if cur_len > 0:
                            generated_hyps[sent_id].add(input_tokens[sent_id * beam_size + beam_id, lef:lef+cur_len].clone(), value.item())
                    # elif cur_len + 1 == span_length:
                    #     # 没有正常结束，指定为很低的分数
                    #     generated_hyps[sent_id].add(input_tokens[sent_id * beam_size + beam_id, lef:lef+cur_len].clone(), -50000)
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len == spans else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, tokenizer.pad_id, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # At the last step, we should not add the token to the next position
            if i == rig - 1:
                break
            
            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_tokens.new([x[1] for x in next_batch_beam])
            beam_idx = input_length.new([x[2] for x in next_batch_beam]).long()

            # re-order batch and internal states
            input_tokens = input_tokens[beam_idx, :]
            input_tokens[:, lef + cur_len] = beam_words

            # update current length
            cur_len = cur_len + 1

        # select the best hypotheses
        best = []
        worst = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            best.append(best_hyp)
            worst_hyp = min(hypotheses.hyp, key=lambda x: x[0])[1]
            worst.append(worst_hyp)

        # because batch_size = 1        
        if return_best:
            for id in best[0].cpu().numpy():
                token = tokenizer.decode([id])

                yield token
        else:
            for id in worst[0].cpu().numpy():
                token = tokenizer.decode([id])

                yield token