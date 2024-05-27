import pandas as pd
import os  
from utilities import load_object, save_object
from sacrebleu.metrics import BLEU
import numpy as np

def get_XY(dat, time_steps, len_skip = -1, len_output = -1):
    X = []
    Y = [] 
    if len_skip == -1:
        len_skip = time_steps
    if len_output == -1:
        len_output = time_steps
    for i in range(0, len(dat), len_skip):
        x_vals = dat[i:min(i + time_steps, len(dat))]
        y_vals = dat[i + time_steps:i + time_steps + len_output]
        if len(x_vals) == time_steps and len(y_vals) == len_output:
            X.append(np.array(x_vals))
            Y.append(np.array(y_vals))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

predicted_all = dict()
actual_all = dict()
y_test_all = dict()
ws_all = dict() 
BLEU_all = dict()
num_to_ws = [-1, 5, 6, 5, 6, 5, 6, 5, 6, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 3, 3, 3, 3, 4, 4, 4, 4, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 15, 15, 15, 15, 19, 19, 19, 19, 25, 25, 25, 25, 29, 29, 29, 29, 16, 16, 16, 16, 32, 32, 32, 32]

model_name = "GRU_Att"

for varname in os.listdir("train_attention1"):
    
    print(varname)
    
    predicted_all[varname] = dict()
    actual_all[varname] = dict()
    y_test_all[varname] = dict()
    ws_all[varname] = dict() 
    BLEU_all[varname] = dict()

    for test_num in range(1, 69):
        if not os.path.isdir("train_attention" + str(test_num)):
            continue
        ws_use = num_to_ws[test_num]

        print(test_num)
        
        predicted_all[varname][test_num] = dict()
        actual_all[varname][test_num] = dict()
        y_test_all[varname][test_num] = dict()
        ws_all[varname][test_num] = dict() 
        BLEU_all[varname][test_num] = dict()
        
        for model_name in os.listdir("train_attention" + str(test_num) + "/" + varname + "/predictions/val/"):

            predicted_all[varname][test_num][model_name] = dict()
            actual_all[varname][test_num][model_name] = dict()
            y_test_all[varname][test_num][model_name] = dict() 
            BLEU_all[varname][test_num][model_name] = []

            for filename in os.listdir("train_attention" + str(test_num) + "/" + varname + "/predictions/val/" + model_name):
    
                final_test_data = pd.read_csv("train_attention" + str(test_num) + "/" + varname + "/predictions/val/" + model_name + "/" + filename, sep = ";", index_col = False)
                
                final_test_data_predicted = [str(x).strip() for x in final_test_data["predicted"]]
                final_test_data_actual = [str(x).strip() for x in final_test_data["actual"]]

                final_test_data_predicted_new = []

                for ix_x in range(len(final_test_data_predicted)):

                    value_ix = final_test_data_predicted[ix_x].replace("a", ".")

                    while "  " in value_ix:

                        value_ix = value_ix.replace("  ", " ")

                    value_ix = value_ix.split(" ")

                    while len(value_ix) < ws_use:
                        
                        value_ix.append(value_ix[-1])

                    for vx in range(ws_use):
                        
                        final_test_data_predicted_new.append(value_ix[vx])

                final_test_data_predicted = final_test_data_predicted_new

                final_test_data_actual_new = []

                for ix_x in range(len(final_test_data_actual)):

                    value_ix = final_test_data_actual[ix_x].replace("a", ".")

                    while "  " in value_ix:

                        value_ix = value_ix.replace("  ", " ")

                    for vx in value_ix.split(" "):
                         
                        final_test_data_actual_new.append(float(vx)) 

                final_test_data_actual = final_test_data_actual_new
                
                test_unk = 0
                for i in range(len(final_test_data_predicted)):
                    if str(final_test_data_predicted[i]) == '<unk>' or str(final_test_data_predicted[i]) == '<sos>' or str(final_test_data_predicted[i]) == 'n.n':
                        test_unk += 1
                        if i > 0:
                            final_test_data_predicted[i] = final_test_data_predicted[i - 1]
                        else:
                            final_test_data_predicted[i] = 0
                    else:
                        final_test_data_predicted[i] = float(final_test_data_predicted[i])
                
                file_object_test = load_object("actual_val/actual_val_" + varname)

                ws_use = int(filename.replace(".csv", "").split("_")[-2])
                ws_all[varname][test_num][model_name] = ws_use
    
                len_total = 0

                for k in file_object_test:

                    x_test_part, y_test_part = get_XY(file_object_test[k], ws_use, ws_use, ws_use)
                    
                    y_test_all[varname][test_num][model_name][k] = []
                    
                    for ix1 in range(len(y_test_part)): 
                        for ix2 in range(len(y_test_part[ix1])): 
                            y_test_all[varname][test_num][model_name][k].append(y_test_part[ix1][ix2])

                    predicted_all[varname][test_num][model_name][k] = list(final_test_data_predicted[len_total:len_total + len(y_test_all[varname][test_num][model_name][k])])
                    actual_all[varname][test_num][model_name][k] = list(final_test_data_actual[len_total:len_total + len(y_test_all[varname][test_num][model_name][k])])
                    len_total += len(y_test_all[varname][test_num][model_name][k])

                    bleu_params = dict(effective_order=True, tokenize=None, smooth_method="floor", smooth_value=0.01)
                    bleu = BLEU(**bleu_params)
                    pred_str = ""
                    actual_str = "" 
                    round_val = 10
                    if "dir" in varname or "speed" in varname:
                        round_val = 0
                    if "time" in varname:
                        round_val = 3
                    for val_ix in range(len(predicted_all[varname][test_num][model_name][k])):
                        pred_str += str(np.round(float(predicted_all[varname][test_num][model_name][k][val_ix]), round_val)) + " "
                        actual_str += str(np.round(float(actual_all[varname][test_num][model_name][k][val_ix]), round_val)) + " "
                    pred_str = pred_str[:-1]
                    actual_str = actual_str[:-1]
                    blsc = bleu.sentence_score(hypothesis=pred_str, references=[actual_str]).score
                    BLEU_all[varname][test_num][model_name].append(blsc)
                    print(varname, model_name, k, BLEU_all[varname][test_num][model_name][-1])
                    save_object("attention_result_val/BLEU_all", BLEU_all) 
            print(varname, test_num, model_name, np.mean(BLEU_all[varname][test_num][model_name]))

for varname in BLEU_all:
    
    for test_num in BLEU_all[varname]:

        for model_name in BLEU_all[varname][test_num]:

            print(varname, test_num, model_name, np.mean(BLEU_all[varname][test_num][model_name]))