import pandas as pd
import os
import numpy as np
from pytorch_utilities import get_XY 
from utilities import load_object 
import numpy as np 
from seq2seq_attention.build_dataloaders import (
    build_fields, 
    get_datasets,
    build_vocab,
) 
MIN_FREQ = 2
MAX_VOCAB_SIZE = 8000  

num_props = 1

ws_range = [2, 5, 6, 10, 20, 30, 3, 3, 3, 3]

for filename in os.listdir("actual_train"):
    for ws_use in ws_range:
        varname = filename.replace("actual_train_", "")

        src_field, trg_field = build_fields()
        train_set, val_set, test_set = get_datasets(train_path="tokenized_data/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv", 
                                                    val_path="tokenized_data/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv", 
                                                    test_path="tokenized_data/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv", 
                                                    src_field=src_field, 
                                                    trg_field=trg_field)
        build_vocab(src_field=src_field, trg_field=trg_field, train_set=train_set, min_freq=MIN_FREQ, max_vocab_size=MAX_VOCAB_SIZE)
        # Check vocabulary 
        print(varname, len(src_field.vocab), len(trg_field.vocab))