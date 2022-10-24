import os 
import json
from data_utils import get_examples
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    data_path = "data/VLSP_data/VLSP_train_split.json"
    examples = get_examples(data_path, True)
    impossibles = []
    for example in examples:
        impossibles.append(int(example.is_impossible))

    print("Possible: {}, Impossible: {}".format(len(impossibles)- np.sum(impossibles), np.sum(impossibles)))
    
    