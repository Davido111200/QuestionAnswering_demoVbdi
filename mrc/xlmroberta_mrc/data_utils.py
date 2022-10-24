import json, os
import re
import os
import json
import torch
import tqdm
import collections
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def _padding(seq, max_length, pad_id= 0):
  if len(seq) < max_length:
    seq += [pad_id]* (max_length - len(seq))

def load_features_cls(data, max_length, tokenizer, do_lower_case):  
  input_ids = []
  attention_masks = []
  type_ids = []
  impossibles = []
  start_positions = []
  end_positions = []

  vocab = tokenizer.get_vocab()
  cls_id = vocab[tokenizer.cls_token]
  sep_id = vocab[tokenizer.sep_token]
  pad_id = vocab[tokenizer.pad_token]
  id_map = {}
  for i, id in enumerate(data):
    id_map[i] = id
    text = data[id]['context']
    question = data[id]['question']
    is_imposible = int(data[id]['is_impossible'])
    
    if do_lower_case:
        question = question.lower()
        text = text.lower()

    question_token_ids = tokenizer.encode(question)[1:-1]
    text_token_ids = tokenizer.encode(text)[1:-1]
    
    _truncate_seq_pair(question_token_ids, text_token_ids, max_length - 4) #placehold for <s>...</s></s>...</s>
    
    input_id = [cls_id] + question_token_ids + [sep_id, sep_id] + text_token_ids + [sep_id]
    attention_mask = [1] * len(input_id)
    _padding(input_id, max_length, pad_id)
    _padding(attention_mask, max_length, 0)
    
    type_id = [0] * len(input_id)

    assert len(input_id) == max_length, "Error with input length {} vs {}".format(len(input_id), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
    assert len(type_id) == max_length, "Error with input length {} vs {}".format(len(type_id), max_length)

    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    type_ids.append(type_id)
    impossibles.append(is_imposible)
    if not is_imposible:
      answer = data[id]['answers'][0]
      start_positions.append(answer['answer_start'])
      end_positions.append(answer['answer_start'] + len(answer['text']))
    else:
      start_positions.append(-1)
      end_positions.append(-1)

  # for i in range(10):
  #   print(tokenizer.convert_ids_to_tokens(input_ids[i]))
  #   print(attention_masks[i])
  #   print(impossibles[i])
  return id_map, input_ids, attention_masks, type_ids, impossibles, start_positions, end_positions


def getData_cls(data, max_seq_len, tokenizer, batch_size, sampler, do_lower_case):
  def toDataLoader(input_id, attention_mask, type_id, label, batch_size, sampler):
    input_id_ = torch.tensor(input_id, dtype= torch.long)
    attention_mask_ = torch.tensor(attention_mask, dtype= torch.long)
    type_id_ = torch.tensor(type_id, dtype= torch.long)
    label_ = torch.tensor(label, dtype= torch.long)

    TensorData = TensorDataset(input_id_, attention_mask_, type_id_, label_)
    Sampler = sampler(TensorData)
    
    dataloader = DataLoader(TensorData, sampler=Sampler, batch_size= batch_size)
    return dataloader
  
  id_map, input_ids, attention_masks, type_ids, impossibles, start_positions, end_positions = load_features_cls(data, max_seq_len, tokenizer, do_lower_case)
  loader = toDataLoader(input_ids, attention_masks, type_ids, impossibles, batch_size, RandomSampler)
  return loader, id_map


def convert_examples_to_cls_features(examples, tokenizer, max_length, return_dataset):
    id_map = {}
    input_ids = []
    attention_masks = []
    type_ids = []
    impossibles = []

    cls_token= tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    for idx, example in enumerate(examples):
        id_map[idx] = example.qas_id
        text = example.context_text
        question = example.question_text
        is_imposible = int(example.is_impossible)

        question_tokens = tokenizer.tokenize(question)
        text_tokens = tokenizer.tokenize(text)
        
        _truncate_seq_pair(question_tokens, text_tokens, max_length - 4) #placehold for <s>...</s></s>...</s>
        
        input_id = [cls_token] + question_tokens + [sep_token, sep_token] + text_tokens + [sep_token]
        attention_mask = [1] * len(input_id)
        assert len(input_id) == len(attention_mask), f"{len(input_id)} vs {len(attention_mask)}"
        if len(input_id) < max_length:
            input_id = input_id + [pad_token] * (max_length - len(input_id))
            attention_mask = attention_mask + [0] * (max_length - len(attention_mask))
        
        input_id = tokenizer.convert_tokens_to_ids(input_id)
        type_id = [0] * len(input_id)

        assert len(input_id) == max_length, "Error with input length {} vs {}".format(len(input_id), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(type_id) == max_length, "Error with input length {} vs {}".format(len(type_id), max_length)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        type_ids.append(type_id)
        impossibles.append(is_imposible)
        if idx < 0:
            print(" ".join(tokenizer.convert_ids_to_tokens(input_id)))
            print(attention_mask)
            print(type_id)
            print(is_imposible)

    if return_dataset == 'pt':
        input_id_ = torch.tensor(input_ids, dtype= torch.long)
        attention_mask_ = torch.tensor(attention_masks, dtype= torch.long)
        type_id_ = torch.tensor(type_ids, dtype= torch.long)
        label_ = torch.tensor(impossibles, dtype= torch.long)

        TensorData = TensorDataset(input_id_, attention_mask_, type_id_, label_)
        return TensorData, id_map
    return input_ids, attention_masks, type_ids, impossibles, id_map
################SQUAD MRC###########################################################
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False
    
def get_examples(data_file, is_training):
    with open(data_file, 'r') as f:
        input_data = json.load(f)['data']
    
    bads = 0
    examples = []
    for entry in input_data:
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []

                if "is_impossible" in qa:
                    is_impossible = qa["is_impossible"]
                else:
                    is_impossible = False

                if not is_impossible:
                    if is_training:
                        if len(qa["answers"]) == 0:
                            print("empty answer!!!")
                            continue
                        answer = qa["answers"][0]
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"]
                        if answer_text == context_text[start_position_character: start_position_character + len(answer_text)]:
                            start_position_character = start_position_character
                        elif answer_text == context_text[start_position_character + 1: start_position_character + len(answer_text) + 1]:
                            start_position_character = start_position_character + 1
                        elif answer_text == context_text[start_position_character - 1: start_position_character + len(answer_text) - 1]:
                            start_position_character = start_position_character - 1
                        elif answer_text == context_text[start_position_character - 2: start_position_character + len(answer_text) - 2]:
                            start_position_character = start_position_character - 2
                        else:
                            # print(f"'{answer_text}' || '{context_text[start_position_character: start_position_character + len(answer_text)]}'")
                            # print(context_text.index(answer_text))
                            bads += 1
                    else:
                        if "answers" in qa:
                            answers = qa["answers"]

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    is_impossible=is_impossible,
                    answers=answers,
                )

                examples.append(example)
    print("Bad:", bads)
    return examples    


class SquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifierg cụ phụ trợ như trình biên dịch, trình hợp dịch hay trình liên kế does not match các công cụ phụ trợ như trình biên dịch, trình hợp dịch hay trình liên kết
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """
    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.is_impossible = is_impossible
        self.answers = answers

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
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

class SquadFeatures(object):
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


def convert_examples_to_features(examples, 
                                 tokenizer, 
                                 max_seq_length,
                                 doc_stride,
                                 max_query_length,
                                 is_training, 
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
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    unique_id = 1000000000
    
    features = []
    for (example_index, example) in enumerate(examples):
        
        if add_prefix_space:
            query_tokens = tokenizer.tokenize(example.question_text, add_prefix_space= True)
        else:
            query_tokens = tokenizer.tokenize(example.question_text)       

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
            
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        
        # `token`s are separated by whitespace; `sub_token`s are separated in a `token` by symbol
        for (i, token) in enumerate(example.doc_tokens):
    
            orig_to_tok_index.append(len(all_doc_tokens))
            if add_prefix_space:
                sub_tokens = tokenizer.tokenize(token)
            else:
                sub_tokens = tokenizer.tokenize(token)

            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
  
        tok_start_position = None
        tok_end_position = None
        
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(all_doc_tokens, 
                                                                          tok_start_position, 
                                                                          tok_end_position, 
                                                                          tokenizer, 
                                                                          example.answer_text)
        if 'phobert' in str(type(tokenizer)) or 'roberta' in str(type(tokenizer)):
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 4 # phobert use 2 sep tokens
        else:
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        
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
            start_offset += min(length, doc_stride)
        
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
            if 'phobert' in str(type(tokenizer)) or 'roberta' in str(type(tokenizer)):
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


            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token_id)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(p_mask) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None

            num_special_tokens = 3 if 'phobert' in str(type(tokenizer)) or 'roberta' in str(type(tokenizer)) else 2

            # Get `start_position` and `end_position`
            if is_training and not span_is_impossible:
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
           
            if is_training and span_is_impossible:
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
                if is_training and span_is_impossible:
                    print("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    print("start_position: %d" % (start_position))
                    print("end_position: %d" % (end_position))
                    print("answer: %s" % (answer_text))
                    print("original answer: %s" % (example.answer_text))
                print("="*100)

            if pq_end:
                features.append(
                    SquadFeatures(
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
                    SquadFeatures(
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

        if not is_training:
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

def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, choice_logits=None, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id
        if choice_logits:
            self.choice_logits = choice_logits

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
        self.cls_logits = cls_logits
