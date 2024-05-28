from utilities import load_object
import numpy as np

dicti_all = load_object("dicti_all")
ord_metric = ["GRU_Att_1", "GRU_Att_2", "GRU_Att_3", "GRU_Att_4", "UniTS_longlat_speed_direction", "UniTS_offsets_speed_direction"]
hidden_range = [256]
model_list = ["LSTM", "GRU", "RNN"]
modes = ["Reference", "Third", "Linear", "Twice"]
for mod_use in modes:
    for model_name in model_list:
            ord_metric.append(model_name + "_" + mod_use + "_256")
metric_dicti = {"NRMSE": 2, "R2": 2, "MAE": 0, "RMSE": 0}
translate_metric = {"NRMSE": "NRMSE (\\%)", "R2": "$R^{2}$ (\\%)", "MAE": "MAE", "RMSE": "RMSE"}
translate_model = { 
        "UniTS_longlat_speed_direction": "UniTS model trained without time intervals", 
        "UniTS_offsets_speed_direction": "UniTS model trained with time intervals", 
        "GRU_Att_1": "GRU attention model using the hyperparameters from experiment 1", 
        "GRU_Att_2": "GRU attention model using the hyperparameters from experiment 2", 
        "GRU_Att_3": "GRU attention model using the hyperparameters from experiment 3", 
        "GRU_Att_4": "GRU attention model using the hyperparameters from experiment 4"}
translate_mod = {"Reference": "reference model", "Third": "model with a third dense layer", "Linear": "model with linear activation in the second dense layer", "Twice": "model with two reccurent layers"}
for mod_use in modes:
    for model_name in model_list:
            translate_model[model_name + "_" + mod_use + "_256"] = model_name + " " + translate_mod[mod_use].replace("reccurent", model_name)
translate_varname = {"direction": "heading", "speed": "speed", "longitude_no_abs": "$x$ offset", "latitude_no_abs": "$y$ offset", "time": "time intervals"}
translate_ws = {"2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", "15": "fifteen", "19": "nineteen", "20": "twenty", "25": "twenty-five", "29": "twenty-nine", "30": "thirty"}
list_ws = [2, 3, 4, 5, 10, 20, 30]
 
metrictouse = ["MAE", "R2"]
vartouse = ["direction", "speed", "longitude_no_abs", "latitude_no_abs"]
for metric_name_use in metrictouse:
    for varname in vartouse:
        duplicate_val_all = True
        duplicate_val = True
        too_small = True
        mul_metric = 0
        rv_metric = 2
        while too_small or duplicate_val_all:
            set_values_all = set()
            set_values = dict()
            for val_ws in list_ws:
                set_values[val_ws] = set()
            max_col = dict()
            max_col_str = dict()
            max_col_ix = dict()
            for val_ws in list_ws:
                max_col[val_ws] = -1000000
                max_col_str[val_ws] = -1
                max_col_ix[val_ws] = -1
            min_col = dict()
            min_col_str = dict()
            min_col_ix = dict()
            for val_ws in list_ws:
                min_col[val_ws] = 1000000
                min_col_str[val_ws] = -1
                min_col_ix[val_ws] = -1
            duplicate_val_all = False
            duplicate_val = False
            too_small = False
            str_pr = ""
            first_line = metric_name_use + " " + varname + " 10^{" + str(mul_metric) + "} " + str(rv_metric)
            for model_name_use in ord_metric:
                for val_ws in list_ws:
                    first_line += " & $" + str(val_ws) + "$s"
                break
            for model_name_use in ord_metric:
                if "offsets" in model_name_use:
                    continue
                str_pr += model_name_use.replace("_", " ")
                for val_ws in list_ws: 
                    vv = dicti_all[varname][model_name_use][str(val_ws)][metric_name_use]  
                    vv = np.round(vv * (10 ** metric_dicti[metric_name_use]) * (10 ** mul_metric), rv_metric)
                    str_pr += " & $" + str(vv) + "$"
                    if vv in set_values[val_ws]:
                        duplicate_val = True
                    if vv in set_values_all:
                        duplicate_val_all = True
                    if "$0." in str_pr:
                        too_small = True
                    set_values[val_ws].add(vv)
                    set_values_all.add(vv)
                    vv_new = str(vv)
                    if metric_dicti[metric_name_use] == 2:
                        vv_new += "\%"
                    if mul_metric != 0:
                        vv_new += "\\times 10^{-" + str(mul_metric) + "}"
                    if vv > max_col[val_ws]:
                        max_col[val_ws] = vv
                        max_col_str[val_ws] = vv_new
                        max_col_ix[val_ws] = model_name_use
                    if vv < min_col[val_ws]:
                        min_col[val_ws] = vv
                        min_col_str[val_ws] = vv_new
                        min_col_ix[val_ws] = model_name_use
                str_pr += " \\\\ \\hline\n"
            if "R2" not in metric_name_use and "NRMSE" not in metric_name_use:
                if too_small:
                    mul_metric += 1
                    rv_metric = 2
                elif duplicate_val_all:
                    rv_metric += 1
            else: 
                rv_metric += 1
            if ("R2" in metric_name_use or "NRMSE" in metric_name_use) and (rv_metric > 3 or mul_metric > 3):
                break
            if rv_metric > 3 or mul_metric > 6:
                break
        if "R2" in metric_name_use:
            max_max = -1000000
            max_max_ix = -1
            max_max_str = -1
            max_max_ws = -1
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(max_col[val_ws]) + "$", "$\\mathbf{" + str(max_col[val_ws]) + "}$") 
                if max_col[val_ws] > max_max:
                    max_max = max_col[val_ws]
                    max_max_ix = max_col_ix[val_ws]
                    max_max_str = max_col_str[val_ws]
                    max_max_ws = val_ws
                strnew = "The highest " + translate_metric[metric_name_use] + " value of $" + max_col_str[val_ws] + "$ for the " + translate_varname[varname] + " estimated using testing data, a window size of " + translate_ws[str(val_ws)] + " seconds was achieved with the " + translate_model[max_col_ix[val_ws]] + "."
                #print(strnew)
            strnew = "The highest " + translate_metric[metric_name_use] + " value of $" + max_max_str + "$ for the estimated testing data " + translate_varname[varname] + " was achieved using a window size of " + translate_ws[str(max_max_ws)] + " seconds and the " + translate_model[max_max_ix] + "."
            print(strnew)
        else:
            min_min = 1000000
            min_min_ix = -1
            min_min_str = -1
            min_min_ws = -1
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(min_col[val_ws]) + "$", "$\\mathbf{" + str(min_col[val_ws]) + "}$")
                if min_col[val_ws] < min_min:
                    min_min = min_col[val_ws]
                    min_min_ix = min_col_ix[val_ws]
                    min_min_str = min_col_str[val_ws]
                    min_min_ws = val_ws
                strnew = "The lowest " + translate_metric[metric_name_use] + " value of $" + min_col_str[val_ws] + "$ for the " + translate_varname[varname] + " estimated using testing data, a window size of " + translate_ws[str(val_ws)] + " seconds was achieved with the " + translate_model[min_col_ix[val_ws]] + "."
                #print(strnew)
            strnew = "The lowest " + translate_metric[metric_name_use] + " value of $" + min_min_str + "$ for the estimated testing data " + translate_varname[varname] + " was achieved using a window size of " + translate_ws[str(min_min_ws)] + " seconds and the " + translate_model[min_min_ix] + "."
            print(strnew)