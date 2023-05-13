import argparse
import os
import math
import sys
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from peft.tuners.lora import LoraConfig 
from peft.mapping import get_peft_model
from peft import (prepare_model_for_int8_training,
                  LoraConfig,
                  get_peft_model)
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_model
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--dataset_name',
                        nargs='*',
                        default=['unicorn'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument(
        '--data_input_path',
        nargs='*',
        default=['/data/Workspace/unicorn/step1_supervised_finetuning/sft/MyData/data_demo.jsonl'],
        help='')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="do eval")
    parser.add_argument(
        "--eval_split",
        type=float,
        default=1000.0,
        help="eval split",
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=10000,
        help="save step",
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=2,
        help="save step",
    )
    parser.add_argument(
        "--log_step",
        type=int,
        default=2,
        help="logging step",
    )
    parser.add_argument(
        "--ds_log_step",
        type=int,
        default=10,
        help="ds log step",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## FP16
    parser.add_argument("--use_fp16",
                        action='store_true',
                        help="use fp16")
    ## 8bit
    parser.add_argument("--use_8bit",
                        action='store_true',
                        help="use 8bit")
    ## LoRA for efficient training setting
    parser.add_argument("--use_lora",
                        action='store_true',
                        help="use lora")
    parser.add_argument("--lora_dim",
                        type=int,
                        default=8,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_alpha",
                        type=int,
                        default=32,
                        help="lora alpha")
    parser.add_argument("--lora_dropout",
                        type=float,
                        default=0.05,
                        help="lora_dropout")
    parser.add_argument('--lora_target_modules',
                        nargs='+', 
                        default=["q_proj", "v_proj",], 
                        help='list lora_target_modules')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Convert args type
    args.eval_split = int(args.eval_split) if args.eval_split >= 1 else args.eval_split

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert not args.gradient_checkpointing, '--gradient_checkpointing and --only_optimizer_lora cannot be enabled at the same time.'
    if args.zero_stage == 3:
        assert False, 'ZeRO-3 is not support now.'

    tqdm.write(str(args))

    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    ds_config['fp16']["enabled"] = args.use_fp16
    if args.use_8bit and args.use_fp16:
        assert False, ' args.use_8bit==True and args.use_fp16==True not support'
    if args.use_lora and args.zero_stage==3:
        assert False, ' args.use_lora==True and args.zero_stage==3 not support now'
    if args.ds_log_step != -1:
        ds_config['steps_per_print'] = args.ds_log_step
    if args.ds_log_step == -1:
        ds_config['steps_per_print'] = math.inf

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    assert not args.offload, "zero-offload is not currently supported but coming soon!"

    torch.distributed.barrier()
    
    if "llama" in args.model_name_or_path.lower():
        if args.local_rank == 0:
            print_rank_0("Use LlamaTokenizer", args.global_rank)
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)  
        try:
            tokenizer.pad_token_id = 0  
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
        except:
            pass
        tokenizer.padding_side = "left"
        tokenizer.bos_token = "<s>"
        tokenizer.eor_token = "</s>"
        tokenizer.pad_token = '<unk>'
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, fast_tokenizer=True)
        tokenizer.pad_token = tokenizer.eos_token


    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            args.use_8bit,
                            disable_dropout=args.disable_dropout)
    vocab_size = len(tokenizer.get_vocab())
    model.resize_token_embeddings(vocab_size)

    if args.use_lora:
        if args.use_8bit:
            model = prepare_model_for_int8_training(model)
        config = LoraConfig(r=args.lora_dim,
                            lora_alpha=args.lora_alpha,
                            target_modules=args.lora_target_modules,
                            lora_dropout=args.lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",)
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    # Prepare the data
    train_phase = 1
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.dataset_name,
        args.data_input_path,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len,
        eval_split=args.eval_split)

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    start_log_time = time.time()
    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    if args.do_eval:
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
            args.global_rank)
        perplexity = evaluation(model, eval_dataloader)
        print_rank_0(f"ppl: {perplexity}", args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        # step = 0
        training_step_losses = []
        model.train()
        # for batch in tqdm(train_dataloader):
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)
            model.step()
            # step = step+1
            training_step_losses.append(loss)
            if step%args.log_step == 0:
                end_log_time = time.time()
                log_time = end_log_time-start_log_time
                _loss = sum(training_step_losses)/len(training_step_losses)
                _log_step = (epoch*len(train_dataloader))+step+1
                _speed = (log_time)/((epoch*len(train_dataloader))+step+1)
                _train_schedule = ((epoch*len(train_dataloader))+step+1)/(args.num_train_epochs*len(train_dataloader))
                _all_to_consume = (log_time)/(((epoch*len(train_dataloader))+step+1)/(args.num_train_epochs*len(train_dataloader)))
                _estimated_to_consume = ((log_time)/(((epoch*len(train_dataloader))+step+1)/(args.num_train_epochs*len(train_dataloader))))*(1-(((epoch*len(train_dataloader))+step+1)/(args.num_train_epochs*len(train_dataloader))))
                print_rank_0(f"epoch {epoch} step {step} train loss {_loss}, log_step {_log_step}, speed {_speed}, train schedule {_train_schedule}, all to consume {_all_to_consume}, estimated to consume {_estimated_to_consume}", args.global_rank)
                training_step_losses = []
            if step%args.save_step == 0:
                save_model(model, args)
            if args.do_eval:
                if step%args.eval_step == 0:
                    print_rank_0(
                        f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} , step {step}/{len(train_dataloader)}*****",
                        args.global_rank)
                    perplexity = evaluation(model, eval_dataloader)
                    print_rank_0(f"ppl: {perplexity}", args.global_rank)
        
        # Evaluate perplexity on the validation set.
        if args.do_eval:
            print_rank_0(
                f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
                args.global_rank)
            perplexity = evaluation(model, eval_dataloader)
            print_rank_0(f"ppl: {perplexity}", args.global_rank)
        
        model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        save_model(model, args)

if __name__ == "__main__":
    main()
