import torch
import sys
sys.path.append(r'..\..\AnswerExtraction_demoVbdi')
from mrc.xlmroberta_mrc.data_utils import SquadExample ,convert_examples_to_features
from mrc.xlmroberta_mrc.model import XLM_MIXLAYER_single
from transformers import XLMRobertaConfig, XLMRobertaTokenizer
import numpy as np


PATH = "/content/drive/MyDrive/NLP_Project/result/weights/checkpoint-22000/pytorch_model.bin"

class XlmRobertaQA:
    def __init__(self):

        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large", do_lower_case=True)
        config = XLMRobertaConfig.from_pretrained("xlm-roberta-large", num_labels= 2)
        config.hidden_size =1024
        model_path = "xlm-roberta-large"
        self.model = XLM_MIXLAYER_single(model_path, config, count= 4, mix_type= "HSUM")
        self.model.load_state_dict(torch.load(PATH))
        self.model.cuda()
        self.model.eval()

    def extract(self ,question, passages):
        #passages = context
        def returnSquadExample(question, context):
            return SquadExample(qas_id="A01", question_text = question, context_text=context, answer_text='', start_position_character=None, is_impossible=False)

        examples = [returnSquadExample(question,item) for item in passages]
        feat = convert_examples_to_features(examples, tokenizer=self.tokenizer, max_seq_length=400, doc_stride=128, max_query_length=64, is_training=False, return_dataset="pt")
        input_ids = torch.stack([x[0] for x in feat[1]], dim = 0).cuda()
        attention_mask = torch.stack([x[1] for x in feat[1]], dim = 0).cuda()

        outputs = self.model(input_ids,attention_mask)
        #tuple -> start_tensor, end_tensor : [N,400], [N,400]
        result = self.take_score(input_ids,attention_mask,outputs,self.tokenizer)
        return result

    def take_score(self, input_ids,attention_mask,outputs,tokenizer,max_answer_len=15):

        start = outputs[0].detach().cpu().numpy()
        end = outputs[1].detach().cpu().numpy()
        #print("Start,end: ", start.shape, " ", end.shape)
        undesired_tokens = attention_mask.detach().cpu().numpy()
        undesired_tokens_mask = undesired_tokens == 0.0

        start_ = np.where(undesired_tokens_mask, -10000.0, start)
        end_ = np.where(undesired_tokens_mask, -10000.0, end)

        start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
        end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))

        outer = np.matmul(np.expand_dims(start_, -1), np.expand_dims(end_, 1))
        #print("outer: ", outer.shape)
        candidates = np.tril(np.triu(outer), max_answer_len - 1)
        scores_flat = candidates.reshape(candidates.shape[0],-1)
        #print("scores_flat: ",scores_flat.shape)
        idx_sort = [np.argmax(scores_flat,axis=1)]
        answers = []
        starts, ends = np.unravel_index(idx_sort, candidates.shape)[1:]
        #print(starts)
        for i in range(len(starts[0])):
            start = starts[0][i]
            end = ends[0][i]
            end += 1
            score = candidates[i, start, end-1]
            #print(start,end)
            start, end, score = start.item(), end.item(), score.item()
            answer = tokenizer.decode(input_ids[i][start:end])
            answers.append([answer,score])
        answers.sort(key=lambda x: x[1], reverse=True)
        print(answers)
        result = {}
        result['answer'] = [item[0] for item in answers]
        result['score'] = [item[1] for item in answers]
        return result


