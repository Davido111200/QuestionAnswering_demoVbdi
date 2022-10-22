import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer


class PhoBertForQuestionAnswering(nn.Module):
    def __init__(self, model_path, hidden_size, hidden_dropout_prob):
        super(PhoBertForQuestionAnswering, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_path)
        self.qa_outputs = nn.Linear(hidden_size, 2) # start, end
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.possible = nn.Linear(hidden_dropout_prob, 2) # possible, impossible
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


def bulid_model(model_path):
    model = PhoBertForQuestionAnswering(model_path, 768, 0.2)
    return model