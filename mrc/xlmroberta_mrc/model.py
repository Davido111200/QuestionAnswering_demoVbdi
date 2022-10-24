from transformers import RobertaModel, XLMRobertaModel
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.init as init


################################################
class PhobertForQuestionAnsweringAVPool(nn.Module):
    def __init__(self, model_path, config):
        super(PhobertForQuestionAnsweringAVPool, self).__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.phobert = RobertaModel.from_pretrained(model_path, config= config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.has_ans = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.qa_outputs.weight.data)
        self.qa_outputs.bias.data.uniform_(0, 0)
        init.xavier_uniform_(self.has_ans.weight.data)
        self.has_ans.bias.data.uniform_(0, 0)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        first_word = sequence_output[:, 0, :]

        has_log = self.has_ans(self.dropout(first_word))

        outputs = (start_logits, end_logits, has_log,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
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
            is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            choice_loss = loss_fct(has_log, is_impossibles)
            total_loss = (start_loss + end_loss + choice_loss) / 3
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

class XLMRobertaForQuestionAnsweringAVPool(nn.Module):
    def __init__(self, model_path, config):
        super(XLMRobertaForQuestionAnsweringAVPool, self).__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.phobert = XLMRobertaModel.from_pretrained(model_path, config= config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.has_ans = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.qa_outputs.weight.data)
        self.qa_outputs.bias.data.uniform_(0, 0)
        init.xavier_uniform_(self.has_ans.weight.data)
        self.has_ans.bias.data.uniform_(0, 0)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        first_word = sequence_output[:, 0, :]

        has_log = self.has_ans(self.dropout(first_word))

        outputs = (start_logits, end_logits, has_log,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
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
            is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            choice_loss = loss_fct(has_log, is_impossibles)
            total_loss = (start_loss + end_loss + choice_loss) / 3
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

####################################################
from modeling import RobertaLayer, ACT2FN
class HSUM(nn.Module):
    def __init__(self, count, config, num_labels):
        super(HSUM, self).__init__()
        self.count = count
        self.num_labels = num_labels
        self.pre_layers = torch.nn.ModuleList()
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.init_weight()
        for i in range(count):
            self.pre_layers.append(RobertaLayer(config))

    def init_weight(self):
        init.xavier_uniform_(self.classifier.weight.data)
        self.classifier.bias.data.uniform_(0, 0)

    def forward(self, layers, attention_mask, return_output = False):
        logitses = []
        output = torch.zeros_like(layers[0])

        for i in range(self.count):
            output = output + layers[-i-1]
            output = self.pre_layers[i](output, attention_mask)[0]
            if not return_output:
                logits = self.classifier(output)
            else:
                logits = output 
            logitses.append(logits)

        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        return avg_logits

class PSUM(nn.Module):
    def __init__(self, count, config, num_labels):
        super(PSUM, self).__init__()
        self.count = count
        self.num_labels = num_labels
        self.pre_layers = torch.nn.ModuleList()
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.init_weight()
        for i in range(count):
            self.pre_layers.append(RobertaLayer(config))

    def init_weight(self):
        init.xavier_uniform_(self.classifier.weight.data)
        self.classifier.bias.data.uniform_(0, 0)

    def forward(self, layers, attention_mask, return_output= False):
        logitses = []

        for i in range(self.count):
            layer = self.pre_layers[i](layers[-i-1], attention_mask)[0]
            if return_output:
                logits = layer
            else:
                logits = self.classifier(layer)
            logitses.append(logits)

        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        return avg_logits

class XLM_MIXLAYER_single(nn.Module):
    def __init__(self, model_path, config, count= 4, mix_type= "HSUM"):
        super(XLM_MIXLAYER_single, self).__init__()
        self.xlmroberta = XLMRobertaModel.from_pretrained(model_path, config=config)
        if mix_type.upper() == "HSUM":
            self.mixlayer = HSUM(count, config, 2)
        elif mix_type.upper() == "PSUM":
            self.mixlayer = PSUM(count, config, 2)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None, return_dict= False):
    
        outputs = self.xlmroberta(input_ids= input_ids, token_type_ids=None, attention_mask=attention_mask, output_hidden_states= True)
        layers = outputs[2]
        # print(len(layers))
        extend_attention_mask = (1.0 - attention_mask[:,None, None, :]) * -10000.0
        logits = self.mixlayer(layers, extend_attention_mask)
        # print(logits.shape)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        outputs = (start_logits, end_logits,)
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            outputs = (total_loss,) + outputs
        return outputs

###############################################
def split_ques_context(sequence_output, pq_end_pos, ques_max_len, seq_max_len):
    ques_max_len = ques_max_len
    context_max_len = seq_max_len
    sep_tok_len = 2
    ques_sequence_output = sequence_output.new(
        torch.Size((sequence_output.size(0), ques_max_len, sequence_output.size(2)))).zero_()
    context_sequence_output = sequence_output.new_zeros(
        (sequence_output.size(0), context_max_len, sequence_output.size(2)))
    context_attention_mask = sequence_output.new_zeros((sequence_output.size(0), context_max_len))
    ques_attention_mask = sequence_output.new_zeros((sequence_output.size(0), ques_max_len))
    for i in range(0, sequence_output.size(0)):
        q_end = pq_end_pos[i][0]
        p_end = pq_end_pos[i][1]
        ques_sequence_output[i, :min(ques_max_len, q_end)] = sequence_output[i,
                                                                   1: 1 + min(ques_max_len, q_end)]
        context_sequence_output[i, :min(context_max_len, p_end - q_end - sep_tok_len)] = sequence_output[i,
                                                                                     q_end + sep_tok_len + 1: q_end + sep_tok_len + 1 + min(
                                                                                         p_end - q_end - sep_tok_len,
                                                                                         context_max_len)]
        context_attention_mask[i, :min(context_max_len, p_end - q_end - sep_tok_len)] = sequence_output.new_ones(
            (1, context_max_len))[0, :min(context_max_len, p_end - q_end - sep_tok_len)]
        ques_attention_mask[i, : min(ques_max_len, q_end)] = sequence_output.new_ones((1, ques_max_len))[0,
                                                                   : min(ques_max_len, q_end)]
    return ques_sequence_output, context_sequence_output, ques_attention_mask, context_attention_mask


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        #mask = mask.half()

        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result

class SCAttention(nn.Module) :
    def __init__(self, input_size, hidden_size) :
        super(SCAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size)
        self.map_linear = nn.Linear(hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.W.weight.data)
        self.W.bias.data.fill_(0.1)

    def forward(self, passage, question, q_mask):
        Wp = passage
        Wq = question
        scores = torch.bmm(Wp, Wq.transpose(2, 1))
        mask = q_mask.unsqueeze(1).repeat(1, passage.size(1), 1)
        # scores.data.masked_fill_(mask.data, -float('inf'))
        alpha = masked_softmax(scores, mask)
        output = torch.bmm(alpha, Wq)
        output = nn.ReLU()(self.map_linear(output))
        #output = self.map_linear(all_con)
        return output

class XLMRobertaForQuestionAnsweringSeqSC(nn.Module):
    def __init__(self, model_path, config, args, sc_ques= True):
        super(XLMRobertaForQuestionAnsweringSeqSC, self).__init__()
        self.args = args
        self.sc_ques = sc_ques
        self.num_labels = config.num_labels
        self.xlm_roberta = XLMRobertaModel.from_pretrained(model_path, config= config)
        self.attention = SCAttention(config.hidden_size, config.hidden_size)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.qa_outputs.weight.data)
        self.qa_outputs.bias.data.fill_(0.1)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pq_end_pos=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        sequence_output = outputs[0]

        query_sequence_output, context_sequence_output, query_attention_mask, context_attention_mask = \
            split_ques_context(sequence_output, pq_end_pos, self.args.max_query_length, self.args.max_seq_length)

        if self.sc_ques:
            sequence_output = self.attention(sequence_output, query_sequence_output, query_attention_mask)
        else:
            sequence_output = self.attention(sequence_output, context_sequence_output, context_attention_mask)

        sequence_output = sequence_output + outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)


        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
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
            is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class XLMRobertaForQuestionAnsweringSeqSCMixLayer(nn.Module):
    def __init__(self, model_path, config, args, count=4, sc_ques= True):
        super(XLMRobertaForQuestionAnsweringSeqSCMixLayer, self).__init__()
        self.args = args
        self.sc_ques = sc_ques
        self.num_labels = config.num_labels
        self.count = count
        self.mixlayer = PSUM(count, config, 2)
        self.xlm_roberta = XLMRobertaModel.from_pretrained(model_path, config= config)
        self.attention = SCAttention(config.hidden_size, config.hidden_size)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.qa_outputs.weight.data)
        self.qa_outputs.bias.data.fill_(0.1)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pq_end_pos=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states= True
        )


        layers = outputs[2]
        # print(len(layers))
        
        extend_attention_mask = (1.0 - attention_mask[:,None, None, :]) * -10000.0
        sequence_output = self.mixlayer(layers, extend_attention_mask, return_output = True)

        query_sequence_output, context_sequence_output, query_attention_mask, context_attention_mask = \
            split_ques_context(sequence_output, pq_end_pos, self.args.max_query_length, self.args.max_seq_length)

        if self.sc_ques:
            sequence_output_ = self.attention(sequence_output, query_sequence_output, query_attention_mask)
        else:
            sequence_output_ = self.attention(sequence_output, context_sequence_output, context_attention_mask)
      
        sequence_output = sequence_output_ + sequence_output

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)


        outputs = (start_logits, end_logits,)
        if start_positions is not None and end_positions is not None:
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
            is_impossibles.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

###################################### Matching attention
class TrmCoAttLayer(nn.Module):
    def __init__(self, config):
        super(TrmCoAttLayer, self).__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()

        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # attention mask 对应 input_ids
    def forward(self, input_ids, input_ids_1, attention_mask=None, head_mask=None):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask = extended_attention_mask

        mixed_query_layer = self.query(input_ids_1)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        reshaped_context_layer = context_layer.view(*new_context_layer_shape)


        # Should find a better way to do this
        w = self.dense.weight.t().view(self.num_attention_heads, self.attention_head_size, self.hidden_size).to(context_layer.dtype)
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_1 + projected_context_layer_dropout)

        ffn_output = self.ffn(layernormed_context_layer)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + layernormed_context_layer)
        return hidden_states

class XLMRobertaForQuestionAnsweringSeqTrm(nn.Module):
    def __init__(self, model_path, config, args, sc_ques= True):
        super(XLMRobertaForQuestionAnsweringSeqTrm, self).__init__()
        self.args = args
        self.config = config
        self.num_labels = config.num_labels
        self.sc_ques = sc_ques
        self.xlm_roberta = XLMRobertaModel.from_pretrained(model_path, config= config)
        self.att_layer = TrmCoAttLayer(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.qa_outputs.weight.data)
        self.qa_outputs.bias.data.fill_(0.1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pq_end_pos=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        sequence_output = outputs[0]

        query_sequence_output, context_sequence_output, query_attention_mask, context_attention_mask = \
            split_ques_context(sequence_output, pq_end_pos, self.args.max_query_length, self.args.max_seq_length)

        if self.sc_ques:
            sequence_output = self.att_layer(query_sequence_output, sequence_output, query_attention_mask)
        else:
            sequence_output = self.att_layer(context_sequence_output, sequence_output, context_attention_mask)

        sequence_output = sequence_output + outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)


        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
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
            is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

class XLMRobertaForQuestionAnsweringSeqTrmMixLayer(nn.Module):
    def __init__(self, model_path, config, args, count=4, mix_type= "HSUM", sc_ques= True):
        super(XLMRobertaForQuestionAnsweringSeqTrmMixLayer, self).__init__()
        self.args = args
        self.config = config
        self.num_labels = config.num_labels
        self.sc_ques = sc_ques
        self.count = count
        if mix_type == "HSUM":     
            self.mixlayer = HSUM(count, config, 2)
        elif mix_type == "PSUM":
            self.mixlayer = PSUM(count, config, 2)
        self.xlm_roberta = XLMRobertaModel.from_pretrained(model_path, config= config)
        self.att_layer = TrmCoAttLayer(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.qa_outputs.weight.data)
        self.qa_outputs.bias.data.fill_(0.1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pq_end_pos=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states= True
        )

        layers = outputs[2]
        print(len(layers))
        
        extend_attention_mask = (1.0 - attention_mask[:,None, None, :]) * -10000.0
        sequence_output = self.mixlayer(layers, extend_attention_mask, return_output = True)

        query_sequence_output, context_sequence_output, query_attention_mask, context_attention_mask = \
            split_ques_context(sequence_output, pq_end_pos, self.args.max_query_length, self.args.max_seq_length)

        if self.sc_ques:
            sequence_output = self.att_layer(query_sequence_output, sequence_output, query_attention_mask)
        else:
            sequence_output = self.att_layer(context_sequence_output, sequence_output, context_attention_mask)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)


        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
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
            is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)