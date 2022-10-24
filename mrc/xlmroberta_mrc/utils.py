import json
import os
import random
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from copy import deepcopy



def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return
################################################################################

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(list(p.size()), end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

################################################################################
def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def re_annotate(filename):
    """
    Re-annotate .ann files to reconstruct span of target according to line
    :param: 
        filename: .ann original file
    """
    with open(filename, "r") as f:
        raw_lines = f.readlines()
    with open(filename.replace(".txt",".ann"), "r") as f:
        ann_lines = f.readlines()
    
    rs_file = open(filename.replace(".txt",".ann").replace("raw_data", "raw_data/re_annotate/ann"), "w")
    span = [[int(line.split()[2]), int(line.split()[3])] for line in ann_lines]
    sentiments = [line.split()[1] for line in ann_lines]

    targets = []

    golds = [line[find_nth(line, "\t", 2)+1:-1].replace(" ", "_") for line in ann_lines]

    cc_line_len = 0
    span_i = 0
    for idx, line in enumerate(raw_lines):
        len_line = len(line)
        for s in span[span_i:]:
            gold = golds[span_i]
            sentiment = sentiments[span_i]
            if s[1] - cc_line_len > len_line:
                cc_line_len += len_line
                break
            else:
                span_i += 1
                new_s = s[0] - cc_line_len, s[1] - cc_line_len
                target = line[new_s[0] : new_s[1]].replace(" ", "_")
                # print(target, gold)
                assert target == gold
                targets.append(target)
                rs_file.write("{} {} {} {} {}\n".format(idx, sentiment, new_s[0], new_s[1], gold))

    rs_file.close()
    # print(golds)
    # print(targets)
    # print(span)
    assert golds == targets

################################################################################
def process_ae(filename):
    """
    :param: 
        filename (re-annotated file)
    :output: 
        dict with structure

    This function is used to produce json file from .ann and .txt file   
    The structure is:
      { 
        "id": {"label": [[t-/a-][B/I]-[positive/negative/neutral], O], "sentence": [words]}
      }
    This structure can serve for aspect extraction or TABSA problem
    """
    # reading files
    with open(filename, "r") as f:
        raw_lines = [line.replace("\u200b", " ") for line in f.readlines()]
    with open("ann/"+ filename.split("/")[-1].replace(".txt", ".ann"), "r") as f:
        ann_lines = f.readlines()
    
    labeled_data = []

    targets = []
    golds = [line.split()[4:] for line in ann_lines]

    span_i = 0
    for idx, line in enumerate(raw_lines):
        # try:
            # print("====={}=====".format(idx))

            # get real sentence before ### 
            len_line = len(line)
            end_line = line.find("###term")
            if end_line < 0:
                end_line = line.find("### term")

            new_line =  line[:end_line].split() # get tokenized words
            
            label = []
            mark_id = 0 # marked visited word
            cur = 0 # current location
            num_space = 0 # number of space character
            for i in range(len_line):
                if line[cur+i] == " ":
                    num_space += 1
                else:
                    break
            cur += num_space

            for s in ann_lines:
                id, sentiment, t_s, t_e, gold = s.split() # id- line index, sentiment label, target start span, target end span, target text
            
                if "term" in sentiment or "aspect" in sentiment: # target with no labeled sentiment
                    pol = ""
                    ty = ""
                else: # target with labeled sentiment
                    ty, pol = sentiment.split("-")[0].lower().strip() + "-", "-" + sentiment.split("-")[1].upper().strip()
                    

                id, t_s, t_e = int(id), int(t_s), int(t_e)
                if int(id) > idx:
                    break
                elif int(id) == idx:
                    check = []        
                    for word in new_line[mark_id:]:
                        # print(word, cur, t_s)
                        if cur == t_s:
                            label.append(ty + "B" + pol)
                            check.append(word)
                        elif t_s < cur < t_e:
                            label.append(ty + "I" + pol)
                            check.append(word)
                        elif cur >= t_e:
                            break
                        else:
                            label.append("O")

                        
                        cur += len(word)
                        num_space = 0
                        for i in range(len_line):
                            if line[cur+i] == " ":
                                num_space += 1
                            else:
                                break
                        cur += num_space
                        mark_id += 1
                    # print(check, gold, line[t_s:t_e])
                    assert "_".join(check) == gold.replace("__", "_")
                    assert "_".join(check) == line[t_s:t_e].replace(" ", "_").replace("__", "_")

            label += ["O"]*(len(new_line)- len(label))

            assert len(label) == len(new_line)
            labeled_data.append([idx, new_line, label])
        # except:
        #     print(idx, line)
  
    return labeled_data

################################################################################
def split_train_dev_test(data_dir, split_rate= [0.7, 0.1]):
    """
    Split a folder of data files into train/dev/test set
    :param:
        data_dir: contains json structure files
        split_rate: train/dev ratio
    :output:
        split files is saved to three folder train/dev/test
    """

    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(data_dir + file, "r") as f:
                dat = json.load(f)
            
            data = {}
            for id in dat:
                # skip lines do not contain labeled sentiment
                if not ("t-B-POSITIVE" in dat[id]['label'] or
                        "t-B-NEGATIVE" in dat[id]['label'] or
                        "t-B-NEUTRAL" in dat[id]['label'] or
                        "a-B-POSITIVE" in dat[id]['label'] or
                        "a-B-NEGATIVE" in dat[id]['label'] or
                        "a-B-NEUTRAL" in dat[id]['label']):
                    continue
                else:
                    data[id] = dat[id]

            tmp = list(data.items())
            random.shuffle(tmp)

            train_num, valid_num = [int(x*len(tmp)) for x in split_rate]
            with open(data_dir+ "train/" + file, "w") as write_file:
                json.dump(dict(tmp[:train_num]), write_file, indent=4)
            with open(data_dir+ "dev/" + file, "w") as write_file:
                json.dump(dict(tmp[train_num: train_num + valid_num]), write_file, indent=4)
            with open(data_dir+ "test/" + file, "w") as write_file:
                json.dump(dict(tmp[train_num + valid_num:]), write_file, indent=4)

################################################################################
def mi_ma_cro_f1(y_true, y_pred, bin_masks):
    """
    Caculate micro and macro F1 score
    :param:
        y_true: list of ground truth labels
        y_pred: list of predicting labels
        bin_masks: list of binary mask for identify targets
    output:
        micro-F1 and macro-F1 score
    """
    groundtruth = []
    prediction = []
    for i, check in enumerate(bin_masks):
        if check == 1:
            groundtruth.append(y_true[i])
            prediction.append(y_pred[i])
    return f1_score(groundtruth, prediction, average='micro'), f1_score(groundtruth, prediction, average='macro')

def get_confusion_matrix(y_true, y_pred, bin_masks):
    """
    Caculate confusion matrix
    :param:
        y_true: list of ground truth labels
        y_pred: list of predicting labels
        bin_masks: list of binary mask for identify targets
    output:
        confusion matrix
    """
    groundtruth = []
    prediction = []
    for i, check in enumerate(bin_masks):
        if check == 1:
            groundtruth.append(y_true[i])
            prediction.append(y_pred[i])
    return confusion_matrix(groundtruth, prediction, labels= [0,1,2])