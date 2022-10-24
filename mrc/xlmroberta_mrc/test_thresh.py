from evaluate_squad import squad_evaluate, compute_predictions_logits, compute_predictions_log_probs, get_raw_scores, make_eval_dict
import argparse
import os
import json
from data_utils import get_examples

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument('--input_null_files', type=str, default="null_odds.json")
parser.add_argument('--input_nbest_files', type=str, default="nbest_predictions.json")
parser.add_argument('--thresh', default=0, type=float)
parser.add_argument("--predict_file", default="data/dev-v2.0.json")
parser.add_argument("--predict_test", action='store_true', help="Whether to test.")
args = parser.parse_args()

with open(args.input_null_files, 'r') as f:
    null_score = json.load(f)

with open(args.input_nbest_files, 'r') as f:
    nbest = json.load(f)

def get_pred(thresh):
    preds = {}
    for id in null_score:
        if id in preds:
          continue

        score = null_score[id]
        if score > thresh:
            preds[id] = ""
        else:
            for answer in nbest[id]:
                if answer['text'] != "":
                    preds[id] = answer['text']
                    break
    return preds

examples = get_examples(args.predict_file, is_training= False)
preds = get_pred(args.thresh)

if not args.predict_test:
    evalrs = squad_evaluate(examples, preds, null_score, args.thresh)
    print(json.dumps(evalrs, indent= 4))
else:
    print(json.dumps(preds, indent= 4, ensure_ascii=False))
    with open("results.json", "w") as f:
        json.dump(preds, f, indent= 4)
  


