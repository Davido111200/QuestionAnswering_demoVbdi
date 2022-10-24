import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
import torch.nn.init as init
from transformers import WEIGHTS_NAME, RobertaConfig, RobertaForSequenceClassification, PhobertTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import print_model_report
from data_utils import convert_examples_to_cls_features, get_examples
import time
import argparse
import logging
import os
import random
import json
import glob

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from constant import MODEL_FILE
from model_cls import PhobertMixLayer
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'phobert': (RobertaConfig, RobertaForSequenceClassification, PhobertTokenizer),
    'phobert_large': (RobertaConfig, RobertaForSequenceClassification, PhobertTokenizer),
    'xlm_roberta': (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    'xlm_roberta_large': (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    'phobert_mixlayer_large': (RobertaConfig, PhobertMixLayer, PhobertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, model, tokenizer):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset, id_map = load_and_cache_examples(args, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    print("="*100)


    if args.max_steps > 0:
        t_total = args.max_steps
        args.epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("****************************")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for e in range(args.epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}

            outputs = model(input_ids= batch[0], attention_mask= batch[1], labels= batch[3])
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = 'eval_{}'.format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)

                    logger.info("  loss = {:.3f}".format(loss_scalar))
                    logger.info("  step = %d", global_step)
                    # print(json.dumps({**logs, **{'step': global_step}}, indent= 4))

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    if "mixlayer" in args.model_type:
                        torch.save(model.state_dict(), os.path.join(output_dir,'pytorch_model.bin'))
                    else:
                        model.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    results = {}
    eval_output_dir = args.output_dir

    eval_dataset, id_map = load_and_cache_examples(args, tokenizer, evaluate=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    num_id = 0
    preds = None
    out_label_ids = None
    key_map = {}
    cnt_map = {}
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
           
            outputs = model(input_ids= batch[0], attention_mask= batch[1], labels= batch[3])
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        
        nb_eval_steps += 1
        logits = logits.detach().cpu().numpy()

        for logit in logits:
            qas_id = id_map[num_id]
            if qas_id in key_map:
                logit_list = key_map[qas_id]
                logit_list[0] += logit[0]
                logit_list[1] += logit[1]
                cnt_map[qas_id] += 1
            else:
                cnt_map[qas_id] = 1
                key_map[qas_id] = [logit[0], logit[1]]
            num_id += 1

        if preds is None:
            preds = logits
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits, axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
    print(len(preds))
    eval_loss = eval_loss / nb_eval_steps

    preds = np.argmax(preds, axis=1)

    result = {'acc': (preds == out_label_ids).mean()}
    results.update(result)

    final_map = {}
    for idx, key in enumerate(key_map):
        key_list = key_map[key]
        key_list[0] = key_list[0] / cnt_map[key]
        key_list[1] = key_list[1] / cnt_map[key]
        final_map[key] = key_list[1] - key_list[0]

    with open(os.path.join(eval_output_dir, prefix, "cls_score.json"), "w") as writer:
        writer.write(json.dumps(final_map, indent=4) + "\n")

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write("***** Eval results %s *****\n" % (str(prefix)))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, predict=False):
    if predict:
        examples = get_examples(args.predict_file, is_training = False)
    else:
        if evaluate:
            examples = get_examples(args.dev_file, is_training= False)
        else: 
            examples = []
            for train_file in args.train_file.split(","):
                examples.extend(get_examples(train_file, is_training= True))

    dataset, id_map = convert_examples_to_cls_features(examples,
                                            tokenizer = tokenizer,
                                            max_length = args.max_seq_len,
                                            return_dataset = 'pt'
    )
    return dataset,id_map


def get_args():
    parser = argparse.ArgumentParser(description='sketchy reading')
    # Arguments
    parser.add_argument('--model_type', type=str, default='phobert', 
                        help='Model type')
    parser.add_argument("--train_file", default=None, type=str, required=False,
                        help="The input training data file.")
    parser.add_argument("--dev_file", default=None, type=str, required=False,
                        help="The input training data file.")
    parser.add_argument("--predict_file", default=None, type=str, required=False,
                        help="The input test data file")
    parser.add_argument("--do_train", action='store_true',
                        help="Activate training")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")    
    parser.add_argument('--seed', type=int, default=21, 
                        help='Random seed for initalization')
    parser.add_argument('--num_labels', type=int, default=2, 
                        help='Number of classes to classify')
    parser.add_argument('--max_seq_len', type=int, default=256, 
                        help='max sequence lenght')
    parser.add_argument('--epochs', type=int, default=4, 
                        help='Number of epochs')
    parser.add_argument('--max_steps', type=int, default=0, 
                        help='Maximum training steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help='accumulated gradient')
    parser.add_argument('--optim', type=str, default="AdamW", 
                        help='Optimization method')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                        help='weight_decay for optimizer')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--mix_type", default=None, type=str, choices= ["HSUM", "PSUM"],
                        help="Mix type for mix layer method")
    parser.add_argument("--mix_count", default=None, type=int,
                        help="Number of mix layers")
    parser.add_argument('--n_gpu', type=int, default=1, 
                        help='Number of gpu to use')
    parser.add_argument('--local_rank', type=int, default= -1, 
                        help='local_rank for distributed')
    parser.add_argument("--output_dir", default= "cls_result", type=str,
                        help="Output dir")
    parser.add_argument("--overwrite_output_dir", action='store_true',
                        help="Overwrite the content of the output directory")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    # parser.add_argument('--train_batch_size', type=int, default=32, 
    #                     help='(default=%(default)d)')
    # parser.add_argument('--test_batch_size', type=int, default=32, 
    #                     help='(default=%(default)d)')

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    args = parser.parse_args()
    return args



def main():
    args = get_args()
    args.output_dir = "../result/cls_{}_{}_lr{}_len{}_bs{}_ep{}_wm{}".format(args.dev_file.split("/")[-1].replace(".json", ""), args.model_type, args.lr, args.max_seq_len, args.per_gpu_train_batch_size, args.epochs, args.warmup_steps)
    if "mixlayer" in args.model_type:
        args.output_dir += "_mixcount{}_mixtype{}".format(args.mix_count, args.mix_type)
    for key, val in args._get_kwargs():
        print("\t{}: {}".format(key, val))

    print("="*100)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1 #torch.cuda.device_count()

    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    set_seed(args)

    # set up model
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model_files = MODEL_FILE[args.model_type]
    config = config_class.from_pretrained(model_files['config_file'], num_labels= args.num_labels)
    if model_files['merges_file'] is not None:
        tokenizer = tokenizer_class(vocab_file= model_files['vocab_file'], merges_file= model_files['merges_file'], do_lower_case= args.do_lower_case)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_files['model_file'], do_lower_case= args.do_lower_case)
    tokenizer.do_lower_case = args.do_lower_case
    if "mixlayer" in args.model_type:
        model = model_class(model_files['model_file'], config=config, count = args.mix_count, mix_type= args.mix_type)
    else:
        model = model_class.from_pretrained(model_files['model_file'], config=config)


    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        print("Loading dataset")
        global_step, tr_loss = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        if not "mixlayer" in args.model_type:
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        else:
            torch.save(model.state_dict(), os.path.join(args.output_dir,'pytorch_model.bin'))
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model.bin')))
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))   
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            
            if "mixlayer" in args.model_type:
                model = model_class(model_files['model_file'], config=config, count = args.mix_count, mix_type= args.mix_type)
                model.load_state_dict(torch.load(os.path.join(checkpoint, "pytorch_model.bin"), map_location= torch.device(args.device)))
                tokenizer = tokenizer_class.from_pretrained(model_files['model_file'], do_lower_case= args.do_lower_case)
            else:
                model = model_class.from_pretrained(checkpoint)
                tokenizer = tokenizer_class.from_pretrained(args.output_dir)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_predict:
        # checkpoint = args.model_name_or_path
        # model = model_class.from_pretrained(checkpoint, force_download=True)
        # model.to(args.device)

        print("Loading test dataset")
        args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_dataset, id_map = load_and_cache_examples(args, tokenizer, predict= True)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.test_batch_size)

        num_id = 0
        key_map = {}
        cnt_map = {}
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1]}

                logits = model(**inputs)
                logits = logits[0].detach().cpu().numpy()
                for logit in logits:
                    qas_id = id_map[num_id]
                    if qas_id in key_map:
                        logit_list = key_map[qas_id]
                        logit_list[0] += logit[0]
                        logit_list[1] += logit[1]
                        cnt_map[qas_id] += 1
                    else:
                        cnt_map[qas_id] = 1
                        key_map[qas_id] = [logit[0], logit[1]]
                    num_id += 1

        final_map = {}
        for idx, key in enumerate(key_map):
            key_list = key_map[key]
            key_list[0] = key_list[0] / cnt_map[key]
            key_list[1] = key_list[1] / cnt_map[key]
            # key_list[0] = key_list[0]
            # key_list[1] = key_list[1]
            # final_map[key] = key_list[1]
            # final_map[key] = key_list[1]*2
            final_map[key] = key_list[1] - key_list[0]

        with open(os.path.join(args.output_dir, "test_cls_score.json"), "w") as writer:
            writer.write(json.dumps(final_map, indent=4) + "\n")

if __name__ == "__main__":
    main()
