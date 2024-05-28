from utilities import load_object, save_object
import numpy as np
attuse = True
unitsuse = True
pytorchuse = True
dicti_all = dict()

num_to_ws = [-1, 5, 6, 5, 6, 5, 6, 5, 6, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 3, 3, 3, 3, 4, 4, 4, 4, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 15, 15, 15, 15, 19, 19, 19, 19, 25, 25, 25, 25, 29, 29, 29, 29, 16, 16, 16, 16, 32, 32, 32, 32]

num_to_params = [-1, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
if attuse:
    BLEU_all = load_object("attention_result/BLEU_all")

    for varname in BLEU_all:
        
        for test_num in BLEU_all[varname]:

            for model_name in BLEU_all[varname][test_num]:

                ws_use = num_to_ws[test_num]
                params_use = num_to_params[test_num]

                if varname not in dicti_all:
                    dicti_all[varname] = dict()

                if model_name + "_" + str(params_use) not in dicti_all[varname]:
                    dicti_all[varname][model_name + "_" + str(params_use)] = dict()

                if str(ws_use) not in dicti_all[varname][model_name + "_" + str(params_use)]:
                    dicti_all[varname][model_name + "_" + str(params_use)][str(ws_use)] = dict()
    
                dicti_all[varname][model_name + "_" + str(params_use)][str(ws_use)]["BLEU"] = np.mean(BLEU_all[varname][test_num][model_name])

                #print(varname, test_num, model_name, np.mean(BLEU_all[varname][test_num][model_name]))
if unitsuse:
    BLEU_all = load_object("UniTS_final_result/BLEU_all")
                
    for varname in BLEU_all:
        
        for model_name in BLEU_all[varname]:

            for ws_use in BLEU_all[varname][model_name]:
    
                if varname not in dicti_all:
                    dicti_all[varname] = dict()

                if model_name not in dicti_all[varname]:
                    dicti_all[varname][model_name] = dict()

                if str(ws_use) not in dicti_all[varname][model_name]:
                    dicti_all[varname][model_name][str(ws_use)] = dict()

                dicti_all[varname][model_name][str(ws_use)]["BLEU"] = np.mean(BLEU_all[varname][model_name][ws_use])

                #print(varname, model_name, ws_use, np.mean(BLEU_all[varname][model_name][ws_use]))
if pytorchuse:
    BLEU_all = load_object("pytorch_result/BLEU_all")
                
    for varname in BLEU_all:
        
        for model_name in BLEU_all[varname]:

            for ws_use in BLEU_all[varname][model_name]:

                for hidden_use in BLEU_all[varname][model_name][ws_use]: 

                    if varname not in dicti_all:
                        dicti_all[varname] = dict()

                    if model_name + "_" + str(hidden_use) not in dicti_all[varname]:
                        dicti_all[varname][model_name + "_" + str(hidden_use)] = dict()

                    if str(ws_use) not in dicti_all[varname][model_name + "_" + str(hidden_use)]:
                        dicti_all[varname][model_name + "_" + str(hidden_use)][str(ws_use)] = dict()

                    dicti_all[varname][model_name + "_" + str(hidden_use)][str(ws_use)]["BLEU"] = np.mean(BLEU_all[varname][model_name][ws_use][hidden_use])

                    #print(varname, model_name, ws_use, hidden_use, np.mean(BLEU_all[varname][model_name][ws_use][hidden_use]))

ord_metric = ["GRU_Att_1", "GRU_Att_2", "GRU_Att_3", "GRU_Att_4", "UniTS_longlat_speed_direction", "UniTS_offsets_speed_direction"]
hidden_range = [256]
model_list = ["LSTM", "GRU", "RNN"]
modes = ["Reference", "Third", "Linear", "Twice"]
for mod_use in modes:
    for model_name in model_list:
            ord_metric.append(model_name + "_" + mod_use + "_256")
metric_dicti = {"BLEU": 0}
list_ws = [2, 3, 4, 5, 10, 20, 30]
 
for metric_name_use in list(metric_dicti.keys()):
    for model_name_use in ord_metric:
        duplicate_val_all = True
        duplicate_val = True
        mul_metric = 0
        rv_metric = 2
        while duplicate_val_all:
            set_values_all = set()
            set_values = dict()
            for val_ws in list_ws:
                set_values[val_ws] = set()
            max_col = dict()
            for val_ws in list_ws:
                max_col[val_ws] = -1000000
            duplicate_val_all = False
            duplicate_val = False
            str_pr = ""
            first_line = metric_name_use + " " + model_name_use + " 10^{" + str(mul_metric) + "} " + str(rv_metric)
            for varname in dicti_all:
                for val_ws in list_ws:
                    first_line += " & $" + str(val_ws) + "$s"
                break
            for varname in dicti_all:
                str_pr += varname
                for val_ws in list_ws:
                    print(dicti_all[varname].keys())
                    vv = dicti_all[varname][model_name_use][str(val_ws)][metric_name_use]  
                    vv = np.round(vv * (10 ** metric_dicti[metric_name_use]) * (10 ** mul_metric), rv_metric)
                    str_pr += " & $" + str(vv) + "$"
                    if vv in set_values[val_ws] and str(vv) != "0.0":
                        duplicate_val = True
                    if vv in set_values_all and str(vv) != "0.0":
                        duplicate_val_all = True
                    set_values[val_ws].add(vv)
                    set_values_all.add(vv)
                    if vv > max_col[val_ws]:
                        max_col[val_ws] = vv
                    str_pr += " \\\\ \\hline\n"
            if duplicate_val_all:
                rv_metric += 1
            if rv_metric > 3 or mul_metric > 6:
                break
        for val_ws in list_ws:
            str_pr = str_pr.replace("$" + str(max_col[val_ws]) + "$", "$\\mathbf{" + str(max_col[val_ws]) + "}$") 
        #print(first_line + " \\\\ \\hline")
        #print(str_pr)

metrictouse = ["BLEU"]
vartouse = ["direction", "speed", "longitude_no_abs", "latitude_no_abs"]
translate_varname = {"direction": "heading", "speed": "speed", "longitude_no_abs": "$x$ offset", "latitude_no_abs": "$y$ offset", "time": "time intervals"}
start_of_table = "\\begin{table*}[!t]\n\t\\begin{center}\n\t\t\\caption{The BLEU score for the VARNAME estimated on the testing dataset by different RNN models using different window sizes.}\n\t\t\\label{tab:test_VARNAME_BLEU}\n\t\t\\resizebox{\linewidth}{!}{"
end_of_table = "\t\t\\end{tabular}}\n\t\\end{center}\n\\end{table*}\n"
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
            for val_ws in list_ws:
                max_col[val_ws] = -1000000
            min_col = dict()
            for val_ws in list_ws:
                min_col[val_ws] = 1000000
            duplicate_val_all = False
            duplicate_val = False
            too_small = False
            str_pr = ""
            first_line = metric_name_use + " " + varname + " 10^{" + str(mul_metric) + "} " + str(rv_metric)
            first_line = "\t\t\\begin{tabular}{|c|} \\hline\n\t\t\tModel"
            longc = "c"
            for model_name_use in ord_metric:
                for val_ws in list_ws:
                    first_line += " & $" + str(val_ws) + "$s"
                    longc += "|c"
                break
            first_line = first_line.replace("{|c|}", "{|"+ longc + "|}")
            for model_name_use in ord_metric:
                if "offsets" in model_name_use:
                    continue
                str_pr += "\t\t\t" + model_name_use.replace("_", " ").replace(" 256", "").replace(" longlat speed direction", "")
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
                    if vv > max_col[val_ws]:
                        max_col[val_ws] = vv
                    if vv < min_col[val_ws]:
                        min_col[val_ws] = vv
                str_pr += " \\\\ \\hline\n"
            if duplicate_val_all:
                rv_metric += 1
            if rv_metric > 3 or mul_metric > 6:
                break
        for val_ws in list_ws:
            str_pr = str_pr.replace("$" + str(max_col[val_ws]) + "$", "$\\mathbf{" + str(max_col[val_ws]) + "}$")
        newstart = start_of_table.replace("VARNAME_", varname + "_").replace("VARNAME ", translate_varname[varname] + " ")
        print(newstart)
        print(first_line + " \\\\ \\hline")
        print(str_pr + end_of_table)

save_object("dicti_all_BLEU", dicti_all)