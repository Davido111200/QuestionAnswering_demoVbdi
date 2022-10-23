import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer, RobertaConfig
import numpy as np

class PhoBertForQuestionAnswering(nn.Module):
    def __init__(self, model_path, hidden_size, hidden_dropout_prob):
        super(PhoBertForQuestionAnswering, self).__init__()
        
        #self.bertconfig = get_bertconfig(cfg)
        #self.config_bert = RobertaConfig.from_pretrained("/content/drive/MyDrive/AcMa/QuestionAnswering_demoVbdi/mrc/PhoQA/PhoBERT_base_transformers/config.json",from_tf=False)
        #self.bert = RobertaModel(self.config_bert)
        #self.bert = self.bert.from_pretrained(pretrained_model_name_or_path = "/content/drive/MyDrive/AcMa/QuestionAnswering_demoVbdi/mrc/PhoQA/PhoBERT_base_transformers/model.bin",
        #                                                config = self.config_bert)

        self.bert = AutoModel.from_pretrained(model_path)
        
        self.qa_outputs = nn.Linear(hidden_size, 2) # start, end
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.possible = nn.Linear(hidden_size, 2) # possible, impossible
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.qa_outputs.weight.data)
        init.xavier_uniform_(self.possible.weight.data)
        self.qa_outputs.bias.data.fill_(0.01)
        self.possible.bias.data.fill_(0.01)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        is_impossible_logits = self.possible(sequence_output[:, 0, :])

        outputs = (start_logits, end_logits, is_impossible_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None and is_impossibles is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            is_impossible_loss = loss_fct(is_impossible_logits, is_impossibles)

            total_loss = (start_loss + end_loss + is_impossible_loss) / 3

            outputs = (total_loss,) + outputs

        return outputs

def get_bertconfig(path_bertconfig):
    #path_bertconfig = cfg.MODEL.PHOBERT.CONFIG_PATH
    with open(path_bertconfig, "r", encoding="utf-8") as bert_cfg:
        bert_cfg = json.load(bert_cfg)
    return bert_cfg

def build_model(model_path):
    model = PhoBertForQuestionAnswering(model_path, 768, 0.2)
    return model

if __name__ == '__main__':
    inputs = {'input_ids': torch.tensor([[    0,  1023,   423,  5477,   480,   133,   454,   109,  1737,  8883,
          1612,    60,    17,   284,  1315,    30,  1333,  3705,  3931,  2352,
         13119,    31,  7701, 31866,  1701,   114,     2,     2,  2140,  1333,
          3705,  3931,    23,    11,  1294, 12813,    63,    82,  2140,  6430,
          1111, 14157,  2546,  5717, 10549,  4107,     6,   983,  2340,  4439,
         53781,  1187,    19,     6, 52632, 14157,  2529,  1465,   418,   222,
             6,     9,   325,    45, 34299,  1236, 18020,  1850,    19,   102,
           208,    91,   102,    10,   121,  1766,  2927,    28,  4439,   567,
          4439, 17942,    11,  1820,   181,    12,  1005,  6829,     6,   316,
          5191,  1333,  3705,  3931,    23,   102,   312,   720,    24,  1218,
         14114,     9,   325,   882, 55565, 21573,     2,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
        [    0,  1206,  5294,    84,    67,    64,  9591,  2428, 59588, 32491,
          1877,  4323,  1656,   208,    11,    35, 19695, 18407,  8038, 50381,
           858, 61983, 16186,    29,  1939,     8,  1529, 10143,   114,     2,
             2,  1430,  1945,    84,    18, 21706,   418,    35, 14809,  2423,
         29445,     6,    14,   208,   302,  5294,  3463,  1830,   599,    35,
         19695, 18407,  8038, 50381,   858, 61983, 16186,   101,   129,   624,
            11,  1391,  1766,    35, 46463, 12017,     4,  1219, 20405,  1301,
             4,    29,  1939,  2146, 23250,  5887, 17806, 10838,     2,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1]]), 'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0]]), 'start_positions': torch.tensor([ 0, 74]), 'end_positions': torch.tensor([ 0, 77]), 'is_impossibles': torch.tensor([1, 0])}



    model = build_model('vinai/phobert-base')
    print(model(**inputs))