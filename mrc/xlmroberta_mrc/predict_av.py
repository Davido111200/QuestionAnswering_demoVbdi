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
from model import PhobertForQuestionAnsweringAVPool, XLMRobertaForQuestionAnsweringAVPool, XLM_MIXLAYER_single


from transformers import AdamW, get_linear_schedule_with_warmup
from constant import MODEL_FILE
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'phobert': (RobertaConfig, PhobertForQuestionAnsweringAVPool, PhobertTokenizer),
    'phobert_large_single': (RobertaConfig, RobertaForQuestionAnswering, PhobertTokenizer),
    'xlm_roberta': (XLMRobertaConfig, XLMRobertaForQuestionAnsweringAVPool, XLMRobertaTokenizer),
    'phobert_single': (RobertaConfig, RobertaForQuestionAnswering, PhobertTokenizer),
    'xlm_roberta_single': (XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer),
    'xlm_roberta_large_single': (XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer),
    'xlm_roberta_large': (XLMRobertaConfig, XLMRobertaForQuestionAnsweringAVPool, XLMRobertaTokenizer),
    'xlm_roberta_mixlayer_large_single': (XLMRobertaConfig, XLM_MIXLAYER_single, XLMRobertaTokenizer),
    'vibert': (BertConfig, BertForQuestionAnswering, BertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()


def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Model path")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters

    parser.add_argument("--mix_type", default=None, type=str, choices= ["HSUM", "PSUM"],
                        help="Mix type for mix layer method")
    parser.add_argument("--mix_count", default=None, type=int,
                        help="Number of mix layers")
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
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="")

    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    parser.add_argument('--seed', type=int, default=21,
                        help="random seed for initialization")

    args = parser.parse_args()
    return args

def main():
    
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1 #torch.cuda.device_count()
    
    
    model_files = MODEL_FILE[args.model_type]
    if "single" in args.model_path:
        args.model_type += "_single"

    args.device = device

    # Set seed
    set_seed(args)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  
    config = config_class.from_pretrained(model_files['config_file'], num_labels= 2)
    if model_files['merges_file'] is not None:
        tokenizer = tokenizer_class(vocab_file= model_files['vocab_file'], merges_file= model_files['merges_file'], do_lower_case= args.do_lower_case)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_files['model_file'], do_lower_case= args.do_lower_case)
    
    if args.model_type != 'vibert':
        tokenizer.do_lower_case = args.do_lower_case

    if "mixlayer" in args.model_path:
        model = model_class(model_files['model_file'], config= config, count = args.mix_count, mix_type= args.mix_type)
    elif "single" in args.model_path:
        model = model_class.from_pretrained(model_files['model_file'], config= config)
    else:
        model = model_class(model_files['model_file'], config= config)
    model.load_state_dict(torch.load(args.model_path, map_location = torch.device(args.device)))


    model.to(args.device)



    print("Load data")
    examples = get_examples(args.predict_file, is_training= False)

    features, dataset = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False,
        return_dataset='pt'
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []

    for batch in tqdm(eval_dataloader, desc= "Evaluation"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1]
            }
            if args.model_type not in ['phobert', 'xlm_roberta']:
                inputs['token_type_ids'] = batch[2]
                
            example_indices = batch[3]

            if "single" in args.model_path:
                outputs = model(input_ids= batch[0], attention_mask= batch[1], token_type_ids= None, return_dict= False)
            else:
                outputs = model(input_ids= batch[0], attention_mask= batch[1], token_type_ids= None)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            if "single" in args.model_path:
                start_logits, end_logits = output
                result = SquadResult(
                    unique_id, start_logits, end_logits
                )
            else:
                start_logits, end_logits, _ = output
                result = SquadResult(
                    unique_id, start_logits, end_logits
                )

            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_final.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_final.json")

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_final.json")
    else:
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(examples, features, all_results, args.n_best_size,
                    args.max_answer_length, args.do_lower_case, output_prediction_file,
                    output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                    args.version_2_with_negative, args.null_score_diff_threshold, tokenizer)

if __name__ == "__main__":
    main()