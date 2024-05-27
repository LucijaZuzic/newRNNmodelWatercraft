import pandas as pd
import os
from utilities import load_object, save_object
import math
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

num_to_ws = [-1, 5, 6, 5, 6, 5, 6, 5, 6, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 3, 3, 3, 3, 4, 4, 4, 4, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 15, 15, 15, 15, 19, 19, 19, 19, 25, 25, 25, 25, 29, 29, 29, 29, 16, 16, 16, 16, 32, 32, 32, 32]

num_to_params = [-1, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
model_name = "GRU_Att"

dicti_to_print = dict()

for varname in os.listdir("train_attention1"):
    
    dicti_to_print[varname] = dict()
    
    print(varname)

    final_test_NRMSE = []
    final_test_RMSE = []
    final_test_R2 = []
    final_test_MAE = []

    test_ix = []
    unk_arr = []
    
    all_mine = load_object("actual/actual_" + varname)
    all_mine_flat = []
    for filename in all_mine: 
        for val in all_mine[filename]:
            all_mine_flat.append(val)

    for test_num in range(1, 69):
        if not os.path.isdir("train_attention" + str(test_num)):
            continue
        ws_use = num_to_ws[test_num] 

        final_test_data = pd.read_csv("train_attention" + str(test_num) + "/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_test.csv", sep = ";", index_col = False)
        
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

        final_test_MAE.append(mean_absolute_error(final_test_data_actual, final_test_data_predicted))
        final_test_R2.append(r2_score(final_test_data_actual, final_test_data_predicted))
        final_test_NRMSE.append(math.sqrt(mean_squared_error(final_test_data_actual, final_test_data_predicted)) / (max(all_mine_flat) - min(all_mine_flat)))
        final_test_RMSE.append(math.sqrt(mean_squared_error(final_test_data_actual, final_test_data_predicted)))

        test_ix.append(test_num)
        unk_arr.append(test_unk / len(final_test_data_predicted))

    for mini_ix_val in range(len(final_test_RMSE)):
        ws_use = num_to_ws[test_ix[mini_ix_val]]
        if model_name + "_" + str(num_to_params[test_ix[mini_ix_val]]) not in dicti_to_print[varname]:
            dicti_to_print[varname][model_name + "_" + str(num_to_params[test_ix[mini_ix_val]])] = dict()
        dicti_to_print[varname][model_name + "_" + str(num_to_params[test_ix[mini_ix_val]])][str(ws_use)] = dict()
        dicti_to_print[varname][model_name + "_" + str(num_to_params[test_ix[mini_ix_val]])][str(ws_use)]["NRMSE"] = final_test_NRMSE[mini_ix_val] 
        dicti_to_print[varname][model_name + "_" + str(num_to_params[test_ix[mini_ix_val]])][str(ws_use)]["RMSE"] = final_test_RMSE[mini_ix_val] 
        dicti_to_print[varname][model_name + "_" + str(num_to_params[test_ix[mini_ix_val]])][str(ws_use)]["R2"] = final_test_R2[mini_ix_val]  
        dicti_to_print[varname][model_name + "_" + str(num_to_params[test_ix[mini_ix_val]])][str(ws_use)]["MAE"] = final_test_MAE[mini_ix_val]  
        print(ws_use, num_to_params[test_ix[mini_ix_val]], np.round(unk_arr[mini_ix_val] * 100, 4), np.round(final_test_NRMSE[mini_ix_val] * 100, 2), np.round(final_test_R2[mini_ix_val] * 100, 2), np.round(final_test_MAE[mini_ix_val], 6), np.round(final_test_RMSE[mini_ix_val], 6))
        
rv_metric = {"R2": 2, "RMSE": 6, "MAE": 6, "NRMSE": 2}
mul_metric = {"R2": 100, "RMSE": 1, "MAE": 1, "NRMSE": 100}
list_ws = sorted([int(x) for x in dicti_to_print["speed"]["GRU_Att_1"]])

dicti_all = dict()
if os.path.isfile("dicti_all"):
    dicti_all = load_object("dicti_all")

for metric_name_use in list(rv_metric.keys()):
    for varname in dicti_to_print:
        str_pr = ""
        first_line = metric_name_use + " " + varname
        for model_name_use in dicti_to_print[varname]:
            for val_ws in list_ws:
                first_line += " & $" + str(val_ws) + "$s"
            break
        print(first_line + " \\\\ \\hline")
        for model_name_use in dicti_to_print[varname]:
            str_pr += varname + " " + metric_name_use + " " + model_name_use
            for val_ws in list_ws: 
                vv = dicti_to_print[varname][model_name_use][str(val_ws)][metric_name_use] 
                if varname not in dicti_all:
                    dicti_all[varname] = dict()
                if model_name_use not in dicti_all[varname]:
                    dicti_all[varname][model_name_use] = dict()
                if str(val_ws) not in dicti_all[varname][model_name_use]:
                    dicti_all[varname][model_name_use][str(val_ws)] = dict()
                if metric_name_use not in dicti_all[varname][model_name_use][str(val_ws)]:
                    dicti_all[varname][model_name_use][str(val_ws)][metric_name_use] = dict()
                dicti_all[varname][model_name_use][str(val_ws)][metric_name_use] = vv
                vv = np.round(vv * mul_metric[metric_name_use], rv_metric[metric_name_use])
                str_pr += " & $" + str(vv) + "$"
            str_pr += " \\\\ \\hline\n"
        print(str_pr)

for metric_name_use in list(rv_metric.keys()):
    for model_name_use in dicti_to_print["speed"]:
        str_pr = ""
        first_line = metric_name_use + " " + model_name_use
        for varname in dicti_to_print:
            for val_ws in list_ws:
                first_line += " & $" + str(val_ws) + "$s"
            break
        print(first_line + " \\\\ \\hline")
        for varname in dicti_to_print:
            str_pr += varname + " " + metric_name_use + " " + model_name_use
            for val_ws in list_ws: 
                vv = dicti_to_print[varname][model_name_use][str(val_ws)][metric_name_use] 
                if varname not in dicti_all:
                    dicti_all[varname] = dict()
                if model_name_use not in dicti_all[varname]:
                    dicti_all[varname][model_name_use] = dict()
                if str(val_ws) not in dicti_all[varname][model_name_use]:
                    dicti_all[varname][model_name_use][str(val_ws)] = dict()
                if metric_name_use not in dicti_all[varname][model_name_use][str(val_ws)]:
                    dicti_all[varname][model_name_use][str(val_ws)][metric_name_use] = dict()
                dicti_all[varname][model_name_use][str(val_ws)][metric_name_use] = vv
                vv = np.round(vv * mul_metric[metric_name_use], rv_metric[metric_name_use])
                str_pr += " & $" + str(vv) + "$"
            str_pr += " \\\\ \\hline\n"
        print(str_pr)

save_object("dicti_all", dicti_all)