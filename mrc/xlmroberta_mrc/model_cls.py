import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import RobertaModel
from modeling import RobertaLayer, RobertaPooler

class HSUM(nn.Module):
    def __init__(self, count, config, num_labels):
        super(HSUM, self).__init__()
        self.count = count
        self.num_labels = num_labels
        self.pre_layers = torch.nn.ModuleList()
        self.loss_fct = torch.nn.ModuleList()
        self.pooler = RobertaPooler(config)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        for i in range(count):
            self.pre_layers.append(RobertaLayer(config))
            self.loss_fct.append(torch.nn.CrossEntropyLoss(ignore_index=-1))
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.classifier.weight.data)
        self.classifier.bias.data.uniform_(0, 0)

    def forward(self, layers, attention_mask, labels):
        losses = []
        logitses = []
        output = torch.zeros_like(layers[0])
        total_loss = torch.Tensor(0)
        for i in range(self.count):
            output = output + layers[-i-1]
            output = self.pre_layers[i](output, attention_mask)[0]
            out = self.pooler(output)
            logits = self.classifier(out)
            if labels is not None:
                loss = self.loss_fct[i](logits.view(-1, self.num_labels), labels.view(-1))
                losses.append(loss)
            logitses.append(logits)
        if labels is not None:
            total_loss = torch.sum(torch.stack(losses), dim=0)
        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        if labels is None:
            return avg_logits
        return total_loss, avg_logits

class PhobertMixLayer(nn.Module):
    def __init__(self, model_path, config, count, mix_type= "HSUM"):
        super(PhobertMixLayer, self).__init__()
        self.num_labels = config.num_labels
        self.config = config
        self.count = count
        self.mix_type = mix_type

        self.phobert = RobertaModel.from_pretrained(model_path, config= config)
        if mix_type.upper() == "HSUM":
            self.mixlayer = HSUM(count, config, 2)
        elif mix_type.upper() == "PSUM":
            self.mixlayer = PSUM(count, config, 2)

    def forward(self, input_ids= None, token_type_ids=None, attention_mask= None, labels= None):
        outputs = self.phobert(input_ids= input_ids, token_type_ids=None, attention_mask=attention_mask, output_hidden_states= True)
        layers = outputs[2]
        extend_attention_mask = (1.0 - attention_mask[:,None, None,:]) * -10000.0

        if labels is None:
            logits = self.mixlayer(layers, extend_attention_mask, labels= labels)
            return (logits,)

        loss, logits = self.mixlayer(layers, extend_attention_mask, labels= labels)
        return loss, logits

