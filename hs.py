import os
import sys
import random
import numpy as np
import argparse
import torch
import fitlog
from dataloader import DiffusionLoader
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from models.modeling_roberta import RobertaForMaskedLM
import diffusion_word_freq
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import fastNLP
from tqdm import tqdm
from sampling import Categorical, WholeWordMasking
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import datetime

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, required=False)
    parser.add_argument("--task_name", default='lm1b', type=str, required=False)
    parser.add_argument("--lr", default=5e-4, type=float, required=False)
    parser.add_argument("--epochs", default=3, type=int, required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False)
    parser.add_argument("--word_freq_lambda", default=0.3, type=float, required=False)
    parser.add_argument("--num_steps", default=2048, type=int, required=False)
    parser.add_argument("--eval_step_size", default=4, type=int, required=False)
    parser.add_argument("--dev_size", default=5e-4, type=float, required=False)
    parser.add_argument("--hybrid_lambda", default=1e-2, type=float, required=False)
    parser.add_argument("--eval_steps", default=15000, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    # parser.add_argument("--device", default='cuda:0', type=str, required=False)
    parser.add_argument("--logging_steps", default=1000, type=int, required=False)
    parser.add_argument('--predict_x0', default=True, type=bool, required=False)
    parser.add_argument("--load_step", default=-1, type=int, required=False)
    parser.add_argument("--sample_strategy", default='Categorical', type=str, required=False)
    parser.add_argument("--schedule", default='mutual', type=str, required=False)
    parser.add_argument("--from_scratch", default=False, type=bool, required=False)
    parser.add_argument("--timestep", default='none', type=str, required=False)
    # parser.add_argument("--local_rank", default=-1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("os.environ :",os.environ['LOCAL_RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    print("local_rank :",local_rank)
    device = torch.device("cuda", local_rank)
    print('device :',device)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=9600))

    set_seed(args)
    if args.timestep in ['none', 'token']:
        from models.modeling_bert import BertForMaskedLM
    elif args.timestep == 'layerwise':
        from models.modeling_bert_new_timestep import BertForMaskedLM
    else:
        raise NotImplementedError

    if dist.get_rank() == 0:
        log_dir = './logs'
        fitlog.set_log_dir(log_dir)
        # fitlog.commit(__file__)
        # fitlog.add_hyper(args)
        # fitlog.add_hyper_in_file(__file__)

        save_path = f'./model_name_{args.model_name_or_path}_lr_{args.lr}_seed_{args.seed}_numsteps_{args.num_steps}_sample_{args.sample_strategy}_schedule_{args.schedule}_hybridlambda_{args.hybrid_lambda}_wordfreqlambda_{args.word_freq_lambda}_fromscratch_{args.from_scratch}_timestep_{args.timestep}_ckpts'
    if args.model_name_or_path in ['bert-base-uncased', 'bert-large-uncased']:
        model_cls = BertForMaskedLM
        cfg_cls = BertConfig
        tok_cls = BertTokenizer
    elif args.model_name_or_path in ['roberta-base']:
        model_cls = RobertaForMaskedLM
        cfg_cls = RobertaConfig
        tok_cls = RobertaTokenizer
    else:
        raise NotImplementedError


    tokenizer = tok_cls.from_pretrained(args.model_name_or_path)
    # word_freq = torch.load(f'./word_freq/{args.model_name_or_path}_{args.task_name}.pt')
    # assert word_freq.size(0) == tokenizer.vocab_size


    # def word_freq_preprocess_fn(wf):
    #     wf = wf + 1
    #     wf = wf.log()
    #     wf = wf / wf.max()

    #     # range: 0 - 1
    #     return wf

    def process_fn_in_collate(wf):
        return wf - wf.mean()

    # word_freq = word_freq_preprocess_fn(word_freq)

    # word_freq[tokenizer.pad_token_id] = 0.  # stable training

    if args.sample_strategy == 'Categorical':
        sample_cls = Categorical()
    elif args.sample_strategy == 'wwm':
        sample_cls = WholeWordMasking(tokenizer)
    else:
        raise ValueError

    diffusion_schedule = diffusion_word_freq.create_discrete_diffusion_schedule(args.schedule, num_steps=args.num_steps)
    diffusion_instance = diffusion_word_freq.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        word_freq_lambda=args.word_freq_lambda,
        device=device
    )

    if args.load_step > 0:
        ckpt = torch.load(os.path.join(save_path, f'{args.load_step}.th'))
    cfg = cfg_cls.from_pretrained(args.model_name_or_path)
    cfg.overall_timestep = diffusion_instance.num_steps

    if args.from_scratch:
        model = model_cls(cfg).to(device)
    elif args.load_step <= 0:
        model = model_cls.from_pretrained(args.model_name_or_path, config=cfg).to(device)
    else:
        model = model_cls(cfg).to(device)
        model.load_state_dict(ckpt['model'])

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda n: n / 10000. + 1e-3 if n < 10000 else 100. / math.sqrt(n))

    train_data, test_data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name='lm1b', splits=['train', 'test'])
    train_data, dev_data = train_data.train_test_split(test_size=args.dev_size).values()