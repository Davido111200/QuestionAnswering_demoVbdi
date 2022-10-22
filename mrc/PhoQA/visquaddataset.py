import json, os
import re
import os
import json
import torch
from tqdm import tqdm
import collections
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
                tok_text_span = doc_tokens[new_start : (new_end + 1)]
                if 'phobert' in str(type(tokenizer)):
                    string_text_span = " ".join(tok_text_span).replace("@@ ","").replace("@@", "")
                elif 'roberta' in str(type(tokenizer)):
                    string_text_span = tokenizer.convert_tokens_to_string(tok_text_span).strip("_")
                else:
                    string_text_span = " ".join(tok_text_span).replace(" ##","").replace("##", "")

                if text_span == tok_answer_text or string_text_span.lower() == orig_answer_text.lower():
                    return (new_start, new_end)

        return (input_start, input_end)


class ViSquadSample:
    def __init__(self, title, qas_id, question_text, context_text, answer_start_character, answer_text, is_impossible, answers):
        self.title = title
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.answer_start_character = answer_start_character
        self.is_impossible = is_impossible
        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if answer_start_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[answer_start_character]
            self.end_position = char_to_word_offset[
                min(answer_start_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


class ViSquadFeatures(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        pq_end_pos=None,
        tag_seq = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.pq_end_pos = pq_end_pos
        self.tag_seq = tag_seq


class ViSquadDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_seq_length, max_query_length, doc_stride, is_training):
        """_summary_

        Args:
            data_path (_type_): _description_
            tokenizer (_type_): _description_
            max_seq_length (_type_): _description_
            max_query_length (_type_): _description_
            is_training (bool): _description_
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.is_training = is_training

        self.data = self.load_data()
        self.feature = self.create_features()

    def load_data(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)['data']

        samples = []
        for article in tqdm(data):
            title = article['title']
            for paragraph in article['paragraphs']:
                context_text = paragraph['context']

                for qa in paragraph['qas']:
                    id = qa['id']
                    question_text = qa['question']
                    is_impossible = qa['is_impossible']
                    if is_impossible:
                        answer_text = qa['plausible_answers'][0]['text']
                        answer_start_character = qa['plausible_answers'][0]['answer_start']
                        answer = qa['plausible_answers']
                    else:
                        if self.is_training:
                            answer_text = qa['answers'][0]['text']
                            answer_start_character = qa['answers'][0]['answer_start']
                            answer = qa['answers']

                    samples.append(ViSquadSample(title, id, question_text, context_text, answer_start_character, answer_text, is_impossible, answer))

        return samples

    def create_features(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        #features = []
        #for (example_index, example) in enumerate(tqdm(self.data)):
        features = self.convert_examples_to_features()
        #features.append(feature)

        return features

    def __len__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return len(self.data)

    def __getitem__(self, index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_id = torch.tensor(self.feature[index].input_ids, dtype=torch.long)
        attention_mask = torch.tensor(self.feature[index].attention_mask, dtype=torch.long)
        token_type_id = torch.tensor(self.feature[index].token_type_ids, dtype=torch.long)
        cls_index = torch.tensor(self.feature[index].cls_index, dtype=torch.long)
        p_mask = torch.tensor(self.feature[index].p_mask, dtype=torch.float)
        start_position = torch.tensor(self.feature[index].start_position, dtype=torch.long)
        end_position = torch.tensor(self.feature[index].end_position, dtype=torch.long)
        is_impossible = torch.tensor(self.feature[index].is_impossible, dtype=torch.long)
        if self.feature[index].pq_end_pos:
            pq_end_pos = torch.tensor(self.feature[index].pq_end_pos, dtype=torch.long)
        else:
            pq_end_pos = None
        
        return input_id, attention_mask, token_type_id, start_position, end_position, is_impossible, pq_end_pos, cls_index, p_mask

    def convert_examples_to_features(self,
                                 return_dataset=False, regression=False, pq_end=False,
                                 add_prefix_space=False,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0, 
                                 mask_padding_with_zero=True):
        """
        Loads a data file into a list of `InputBatch`s.
        """
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        pad_token_id = self.tokenizer.pad_token_id

        unique_id = 1000000000
        
        features = []
        print("Creating features from dataset file at %s", self.data_path)
        for (example_index, example) in enumerate(tqdm(self.data)):
            
            if add_prefix_space:
                query_tokens = self.tokenizer.tokenize(example.question_text, add_prefix_space= True)
            else:
                query_tokens = self.tokenizer.tokenize(example.question_text)       

            if len(query_tokens) > self.max_query_length:
                query_tokens = query_tokens[0:self.max_query_length]
                
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            
            # `token`s are separated by whitespace; `sub_token`s are separated in a `token` by symbol
            for (i, token) in enumerate(example.doc_tokens):
        
                orig_to_tok_index.append(len(all_doc_tokens))
                if add_prefix_space:
                    sub_tokens = self.tokenizer.tokenize(token)
                else:
                    sub_tokens = self.tokenizer.tokenize(token)

                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
    
            tok_start_position = None
            tok_end_position = None
            
            if self.is_training and example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1

            if self.is_training and not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1

                (tok_start_position, tok_end_position) = _improve_answer_span(all_doc_tokens, 
                                                                            tok_start_position, 
                                                                            tok_end_position, 
                                                                            self.tokenizer, 
                                                                            example.answer_text)
            if 'phobert' in str(type(self.tokenizer)) or 'roberta' in str(type(self.tokenizer)):
                max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 4 # phobert use 2 sep tokens
            else:
                max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3
            
            # We can have documents that are longer than the maximum sequence length. To deal with this we do a 
            # sliding window approach, where we take chunks of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))

                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.doc_stride)
            
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                # `p_mask`: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
                # Original TF implem also keeps the classification token (set to 0) (not sure why...)
                p_mask = []

                # `[CLS]` token at the beginning

                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0
                    
                # Query
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(sequence_a_segment_id)
                    p_mask.append(1)

                # [SEP] token
                if 'phobert' in str(type(self.tokenizer)) or 'roberta' in str(type(self.tokenizer)):
                    tokens.extend([sep_token, sep_token])
                    segment_ids.extend([sequence_a_segment_id, sequence_a_segment_id])
                    p_mask.extend([1,1])
                else:
                    tokens.append(sep_token)
                    segment_ids.append(sequence_a_segment_id)
                    p_mask.append(1)

                # Paragraph built based on `doc_span`
                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                    is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(sequence_b_segment_id)
                    p_mask.append(0)
                paragraph_len = doc_span.length

                # [SEP] token
                tokens.append(sep_token)
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(1)


                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                
                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < self.max_seq_length:
                    input_ids.append(pad_token_id)
                    input_mask.append(0 if mask_padding_with_zero else 1)
                    segment_ids.append(pad_token_segment_id)
                    p_mask.append(1)

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length
                assert len(p_mask) == self.max_seq_length

                span_is_impossible = example.is_impossible
                start_position = None
                end_position = None

                num_special_tokens = 3 if 'phobert' in str(type(self.tokenizer)) or 'roberta' in str(type(self.tokenizer)) else 2

                # Get `start_position` and `end_position`
                if self.is_training and not span_is_impossible:
                    # For training, if our document chunk does not contain an annotation we throw it out, 
                    # since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                        span_is_impossible = True
                    else:
                        doc_offset = len(query_tokens) + num_special_tokens
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
            
                if self.is_training and span_is_impossible:
                    start_position = cls_index
                    end_position = cls_index
            
                question_end_index = len(query_tokens)
                doc_end_index = question_end_index + paragraph_len + num_special_tokens
                pq_end_pos = [question_end_index,doc_end_index]

                # Display some examples
                if example_index < 0:              
                    print("*** Example ***")
                    # *** Example ***
                    print("unique_id: %s" % (unique_id))
                    print("example_index: %s" % (example_index))
                    print("doc_span_index: %s" % (doc_span_index))
                    print("tokens: %s" % " ".join(tokens))
                    print("end_ques: {}, end_text: {}".format(pq_end_pos[0], pq_end_pos[1]))
                    print("token_to_orig_map: %s" % " ".join([
                        "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                    print("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                    ]))
                    print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    print("p_mask: %s" % " ".join([str(x) for x in p_mask]))
                    if self.is_training and span_is_impossible:
                        print("impossible example")
                    if self.is_training and not span_is_impossible:
                        answer_text = " ".join(tokens[start_position:(end_position + 1)])
                        print("start_position: %d" % (start_position))
                        print("end_position: %d" % (end_position))
                        print("answer: %s" % (answer_text))
                        print("original answer: %s" % (example.answer_text))
                    print("="*100)

                if pq_end:
                    features.append(
                        ViSquadFeatures(
                            input_ids= input_ids,
                            attention_mask= input_mask,
                            token_type_ids= segment_ids,
                            cls_index= cls_index,
                            p_mask= p_mask,
                            example_index= example_index,
                            unique_id= unique_id,
                            paragraph_len= paragraph_len,
                            token_is_max_context= token_is_max_context,
                            tokens= tokens,
                            token_to_orig_map= token_to_orig_map,
                            start_position= start_position,
                            end_position= end_position,
                            is_impossible= span_is_impossible,
                            pq_end_pos=pq_end_pos,
                            tag_seq = None,
                        )
                    )
                else:
                    features.append(
                        ViSquadFeatures(
                            input_ids= input_ids,
                            attention_mask= input_mask,
                            token_type_ids= segment_ids,
                            cls_index= cls_index,
                            p_mask= p_mask,
                            example_index= example_index,
                            unique_id= unique_id,
                            paragraph_len= paragraph_len,
                            token_is_max_context= token_is_max_context,
                            tokens= tokens,
                            token_to_orig_map= token_to_orig_map,
                            start_position= start_position,
                            end_position= end_position,
                            is_impossible= span_is_impossible,
                            pq_end_pos=None,
                            tag_seq = None,
                        )
                    )
                unique_id += 1

        if return_dataset == "pt":
            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
            all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

            if not self.is_training:
                all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
                if regression:
                    all_is_impossibles = torch.tensor([int(f.is_impossible) for f in features], dtype=torch.float)
                else:
                    all_is_impossibles = torch.tensor([int(f.is_impossible) for f in features], dtype=torch.long)
                if pq_end:
                    all_pq_end_pos = torch.tensor([f.pq_end_pos for f in features], dtype=torch.long)
                    dataset = TensorDataset(
                        all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_is_impossibles, all_pq_end_pos, all_cls_index, all_p_mask
                    )
                else:
                    dataset = TensorDataset(
                        all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_is_impossibles, all_cls_index, all_p_mask
                    )
            else:
                all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
                all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
                if regression:
                    all_is_impossibles = torch.tensor([int(f.is_impossible) for f in features], dtype=torch.float)
                else:
                    all_is_impossibles = torch.tensor([int(f.is_impossible) for f in features], dtype=torch.long)
                    print("Impossible: {}, Possible: {}".format(sum(all_is_impossibles == 1), sum(all_is_impossibles == 0)))
                if pq_end:
                    all_pq_end_pos = torch.tensor([f.pq_end_pos for f in features], dtype=torch.long)
                    dataset = TensorDataset(
                        all_input_ids,
                        all_attention_masks,
                        all_token_type_ids,
                        all_start_positions,
                        all_end_positions,
                        all_is_impossibles,
                        all_pq_end_pos,
                        all_cls_index,
                        all_p_mask,
                    )
                else:
                    dataset = TensorDataset(
                        all_input_ids,
                        all_attention_masks,
                        all_token_type_ids,
                        all_start_positions,
                        all_end_positions,
                        all_is_impossibles,
                        all_cls_index,
                        all_p_mask,
                    )

            return features, dataset


        return features

def collate_fn(batch):
    """Collate's training data after loading the dataset"""
    # Unzip the batch
    input_ids, attention_mask, token_type_ids, start_positions, end_positions, is_impossibles, pq_end_pos, cls_index, p_mask = zip(*batch)
    batch_input_ids = torch.stack(input_ids, dim=0)
    batch_attention_masks = torch.stack(attention_mask, dim=0)
    batch_token_type_ids = torch.stack(token_type_ids, dim=0)
    batch_start_positions = torch.stack(start_positions, dim=0)
    batch_end_positions = torch.stack(end_positions, dim=0)
    batch_is_impossibles = torch.stack(is_impossibles, dim=0)
    
    batch_cls_index = torch.stack(cls_index, dim=0)
    batch_p_mask = torch.stack(p_mask, dim=0)

    if pq_end_pos[0] is not None:
        batch_pq_end_pos = torch.stack(pq_end_pos, dim=0)
        outputs = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_masks,
            "token_type_ids": batch_token_type_ids,
            "start_positions": batch_start_positions,
            "end_positions": batch_end_positions,
            "is_impossible": batch_is_impossibles,
            "pq_end_pos": batch_pq_end_pos,
            "cls_index": batch_cls_index,
            "p_mask": batch_p_mask
        }
    else:
        outputs = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_masks,
            "token_type_ids": batch_token_type_ids,
            "start_positions": batch_start_positions,
            "end_positions": batch_end_positions,
            "is_impossible": batch_is_impossibles,
            "cls_index": batch_cls_index,
            "p_mask": batch_p_mask
        }
    return outputs

def build_dataloader(train_file, test_file, batch_size, max_seq_length, max_query_length, doc_stride):
    """Builds the dataloader for the model"""
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    #data_path, tokenizer, max_seq_length, max_query_length, doc_stride, is_training
    train_dataset = ViSquadDataset(train_file, tokenizer, max_query_length=max_seq_length, max_query_length=max_query_length, doc_stride=doc_stride , is_training=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = ViSquadDataset(test_file, tokenizer, max_query_length=max_seq_length, max_query_length=max_query_length, doc_stride=doc_stride , is_training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, test_dataloader

if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer
    #phobert = AutoModel.from_pretrained("vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    dataset = ViSquadDataset('/home/huycq/QAVN/AnswerExtraction_Vietnamese/model/answer_extraction/mrc/datasets/UIT-ViQuAD 2.0/train.json', tokenizer, 512, 64, 64 , True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch[1].shape)
        break