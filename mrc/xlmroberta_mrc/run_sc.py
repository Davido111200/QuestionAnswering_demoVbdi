# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor, SquadResult
# from transformers.data.metrics.squad_metrics import compute_predictions_logits, compute_predictions_log_probs, compute_predictions_log_probs
from evaluate_squad import squad_evaluate, compute_predictions_logits, compute_predictions_log_probs
import argparse
import logging
import os
import random
import glob
import timeit
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from data_utils import get_examples, convert_examples_to_features
import pickle
from torch.utils.tensorboard import SummaryWriter


from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, 
                          RobertaConfig, PhobertTokenizer, RobertaForQuestionAnswering,
                          XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer,
                          BertConfig, BertForQuestionAnswering, BertTokenizer)
from model import XLMRobertaForQuestionAnsweringSeqSC, XLMRobertaForQuestionAnsweringSeqSCMixLayer

from transformers import AdamW, get_linear_schedule_with_warmup
from constant import MODEL_FILE
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'xlm_roberta_sc_large': (XLMRobertaConfig, XLMRobertaForQuestionAnsweringSeqSC, XLMRobertaTokenizer),
    'xlm_roberta_sc': (XLMRobertaConfig, XLMRobertaForQuestionAnsweringSeqSC, XLMRobertaTokenizer),
    'xlm_roberta_mixlayer_sc_large': (XLMRobertaConfig, XLMRobertaForQuestionAnsweringSeqSCMixLayer, XLMRobertaTokenizer),
    'xlm_roberta_mixlayer_sc': (XLMRobertaConfig, XLMRobertaForQuestionAnsweringSeqSCMixLayer, XLMRobertaTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

  
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
    for e in range(args.num_train_epochs):
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                'input_ids':       batch[0],
                'attention_mask':  batch[1],
                'start_positions': batch[3],
                'end_positions':   batch[4],
                'is_impossibles':   batch[5],
                'pq_end_pos':   batch[6],
            }
            if 'phobert' not in args.model_type and 'roberta' not in args.model_type:
                inputs['token_type_ids'] = batch[2]

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
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

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                # epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            # train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'pq_end_pos': batch[5],
            }
            if 'phobert' not in args.model_type and 'roberta' not in args.model_type:
                inputs['token_type_ids'] = batch[2]
                
            example_indices = batch[3]

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            start_logits, end_logits = output
            result = SquadResult(
                unique_id, start_logits, end_logits
            )

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(examples, features, all_results, args.n_best_size,
                    args.max_answer_length, args.do_lower_case, output_prediction_file,
                    output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                    args.version_2_with_negative, args.null_score_diff_threshold, tokenizer)

    # with open(os.path.join(args.output_dir, str(prefix) + "_eval_examples.pkl"), 'wb') as f:
    #     pickle.dump(examples, f)
    # with open(os.path.join(args.output_dir, str(prefix) + "_eval_features.pkl"), 'wb') as f:
    #     pickle.dump(features, f)
    # with open(os.path.join(args.output_dir, str(prefix) + "_res.pkl"), 'wb') as f:
    #     pickle.dump(all_results, f)

    import json
    with open(output_null_log_odds_file) as f:
        na_probs = json.load(f)

    results = squad_evaluate(examples, predictions, na_probs,
                            args.null_score_diff_threshold)
    return results

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_dir = "../cache"
    cached_features_file = os.path.join(input_dir, 'intensive_reader_cached_sc{}_{}{}_{}_{}_{}_{}'.format(
        args.predict_file.split("/")[-1].replace(".json", ""),
        'dev' if evaluate else 'train', len(args.train_file.split(",")),
        args.model_type,
        str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
    )

    # Init features and dataset from cache if it exists
    input_file = args.predict_file if evaluate else args.train_file
    
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset = features_and_dataset["features"], features_and_dataset["dataset"]
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        if evaluate:
            examples = get_examples(input_file, is_training= not evaluate)
        else:
            examples = []
            for file in input_file.split(","):
                examples.extend(get_examples(file, is_training= not evaluate))

        features, dataset = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset='pt',
            pq_end=True
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset

def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--padding_side", default="right", type=str,
                        help="right/left, padding_side of passage / question")
    parser.add_argument("--train_file", default=None, type=str,
                        help="The input training file. If a data dir is specified, will look for the file there" +
                             "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="The input evaluation file. If a data dir is specified, will look for the file there" +
                             "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--sc_ques", action='store_true',
                        help="Attention question or context")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=2, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=21,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")

    args = parser.parse_args()
    return args

def main():
    
    args = get_args()
    args.output_dir = "../result/av_single_{}_{}_lr{}_len{}_bs{}_ep{}_wm{}_scques{}".format(args.predict_file.split("/")[-1].replace(".json", ""), args.model_type, args.learning_rate, args.max_seq_length, args.per_gpu_train_batch_size, args.num_train_epochs, args.warmup_steps, args.sc_ques)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))


    # Setup CUDA, GPU & distributed training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1 #torch.cuda.device_count()


    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # Set seed
    set_seed(args)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model_files = MODEL_FILE[args.model_type]
    config = config_class.from_pretrained(model_files['config_file'], num_labels= 2)
    if model_files['merges_file'] is not None:
        tokenizer = tokenizer_class(vocab_file= model_files['vocab_file'], merges_file= model_files['merges_file'], do_lower_case= args.do_lower_case)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_files['model_file'], do_lower_case= args.do_lower_case)
    
    if args.model_type != 'vibert':
        tokenizer.do_lower_case = args.do_lower_case

 
    model = model_class(model_files['model_file'], config= config, args= args, sc_ques= args.sc_ques)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir,'pytorch_model.bin'))
        # tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model.bin')))
        if model_files['merges_file'] is not None:
            tokenizer = tokenizer_class(vocab_file= model_files['vocab_file'], merges_file= model_files['merges_file'], do_lower_case= args.do_lower_case)
        else:
            tokenizer = tokenizer_class.from_pretrained(model_files['model_file'], do_lower_case= args.do_lower_case)
        if args.model_type != 'vibert':
            tokenizer.do_lower_case = args.do_lower_case
        model.to(args.device)


    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:

        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            if args.eval_all_checkpoints:
                checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
                logger.info("Loading checkpoint %s for evaluation", checkpoints)
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
            else:
                logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
                checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            config = config_class.from_pretrained(model_files['config_file'], num_labels= 2)
            # model = model_class.from_pretrained(model_files['model_file'], config= config)
            model.load_state_dict(torch.load(os.path.join(checkpoint, "pytorch_model.bin")))
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))
    with open(os.path.join(args.output_dir, "result.txt"), "a") as writer:
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\t" % (key, str(results[key])))
            writer.write("\t\n")
    return results


if __name__ == "__main__":
    main()
