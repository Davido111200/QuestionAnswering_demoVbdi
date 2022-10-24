import argparse
import os
import torch
import random
import numpy as np
from constant import MODEL_FILE
from transformers import WEIGHTS_NAME, RobertaConfig, RobertaForSequenceClassification, PhobertTokenizer, XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import logging
from constant import MODEL_FILE
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data_utils import get_examples, convert_examples_to_cls_features
import json
from model_cls import PhobertMixLayer
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'phobert': (RobertaConfig, RobertaForSequenceClassification, PhobertTokenizer),
    'phobert_large': (RobertaConfig, RobertaForSequenceClassification, PhobertTokenizer),
    'phobert_mixlayer_large': (RobertaConfig, PhobertMixLayer, PhobertTokenizer),
    'xlm_roberta_large': (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_args():
    parser = argparse.ArgumentParser(description='sketchy reading')
    # Arguments
    parser.add_argument('--model_type', type=str, default='phobert', 
                        help='Model type')
    parser.add_argument('--model_path', type=str, default='phobert', 
                        help='Model path')
    parser.add_argument('--max_seq_len', type=int, default=256, 
                        help='max sequence lenght')
    parser.add_argument("--mix_type", default=None, type=str, choices= ["HSUM", "PSUM"],
                        help="Mix type for mix layer method")
    parser.add_argument("--mix_count", default=None, type=int,
                        help="Number of mix layers")
    parser.add_argument("--predict_file", default=None, type=str, required=False,
                        help="The input test data file")
    parser.add_argument('--seed', type=int, default=21, 
                        help='Random seed for initalization')
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--output_dir", default= "cls_result", type=str,
                        help="Output dir")
    parser.add_argument("--overwrite_output_dir", action='store_true',
                        help="Overwrite the content of the output directory")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.n_gpu = 1
    if os.path.exists(os.path.join(args.output_dir, "test_cls_score.json")) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args)

    # set up model
    args.model_type = args.model_type.lower()
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model_files = MODEL_FILE[args.model_type]    
    
    if "mixlayer" in args.model_type:
        config = config_class.from_pretrained(model_files['config_file'])
        model = model_class(model_files['model_file'], config=config, count = args.mix_count, mix_type= args.mix_type)
        model.load_state_dict(torch.load(args.model_path, map_location= torch.device(args.device)))
    else:
        model = model_class.from_pretrained(args.model_path)
    if model_files['vocab_file'] is not None:
        tokenizer = tokenizer_class(vocab_file= model_files['vocab_file'], merges_file= model_files['merges_file'], do_lower_case= True)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_files['model_file'])
    tokenizer.do_lower_case = True   
    model.to(args.device)

    print("Loading test dataset")
    examples = get_examples(args.predict_file, is_training = False)
    eval_dataset, id_map = convert_examples_to_cls_features(examples,
                                            tokenizer = tokenizer,
                                            max_length = args.max_seq_len,
                                            return_dataset = 'pt'
    )
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    # Eval!
    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)

    num_id = 0
    key_map = {}
    cnt_map = {}
    for batch in tqdm(eval_dataloader, desc= "Evaluation"):
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

    print(final_map)
    with open(os.path.join(args.output_dir, "test_cls_score.json"), "w") as writer:
        writer.write(json.dumps(final_map, indent=4) + "\n")
