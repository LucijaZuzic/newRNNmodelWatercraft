from utilities import load_object
import numpy as np

dicti_all = load_object("dicti_all_traj")
ord_metric = ["GRU_Att_1", "GRU_Att_2", "GRU_Att_3", "GRU_Att_4", "UniTS_longlat_speed_direction", "UniTS_offsets_speed_direction"]
hidden_range = [256]
model_list = ["LSTM", "GRU", "RNN"]
modes = ["Reference", "Third", "Linear", "Twice"]
for mod_use in modes:
    for model_name in model_list:
            ord_metric.append(model_name + "_" + mod_use + "_256")
metric_dicti = {"Euclid": 0, "R2": 2, "MAE": 0, "RMSE": 0, "R2_wt": 2, "MAE_wt": 0, "RMSE_wt": 0}
list_ws = [2, 3, 4, 5, 10, 20, 30] 
 
for metric_name_use in list(metric_dicti.keys()):
    for model_name_use in ord_metric:
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
            first_line = metric_name_use + " " + model_name_use + " 10^{" + str(mul_metric) + "} " + str(rv_metric)
            for varname in dicti_all:
                for val_ws in list_ws:
                    first_line += " & $" + str(val_ws) + "$s"
                break
            for varname in dicti_all:
                str_pr += varname
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
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(max_col[val_ws]) + "$", "$\\mathbf{" + str(max_col[val_ws]) + "}$") 
        else:
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(min_col[val_ws]) + "$", "$\\mathbf{" + str(min_col[val_ws]) + "}$") 
        #print(first_line + " \\\\ \\hline")
        #print(str_pr) 

metrictouse = ["Euclid", "MAE", "R2"]
vartouse = ["long speed actual dir", "long no abs"]
translate_varname = {"long speed ones dir": "speed, heading, a fixed one-second time interval",
                    "long speed dir": "speed, heading, time intervals",
                    "long speed actual dir": "speed, heading, the actual time interval",
                    "long no abs": "$x$ and $y$ offset"}
start_of_table = "\\begin{table*}[!t]\n\t\\begin{center}\n\t\t\\caption{METRICNAME for the trajectories in the testing dataset estimated using VARNAME, different RNN models, and different window sizes.}\n\t\t\\label{tab:test_VARNAME_METRICNAME}\n\t\t\\resizebox{\linewidth}{!}{"
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
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(max_col[val_ws]) + "$", "$\\mathbf{" + str(max_col[val_ws]) + "}$") 
        else:
            for val_ws in list_ws:
                str_pr = str_pr.replace("$" + str(min_col[val_ws]) + "$", "$\\mathbf{" + str(min_col[val_ws]) + "}$")
        newstart = start_of_table.replace("METRICNAME", metric_name_use).replace("VARNAME_", varname.replace(" ", "_") + "_").replace("NRMSE ", "NRMSE (\%) ").replace("R2 ", "$R^{2}$ (\%) ").replace("VARNAME", translate_varname[varname])
        if "R2" not in metric_name_use and "NRMSE" not in metric_name_use and mul_metric != 0:
            newstart = newstart.replace(metric_name_use + " ", metric_name_use + " ($\\times 10^{-" + str(mul_metric) + "}$) ")
        if "wt" in metric_name_use:
            newstart = newstart.replace("_wt", "").replace("trajectories", "trajectories and time stamps")
        print(newstart.replace("Euclid ", "The Euclidean distance "))
        print(first_line + " \\\\ \\hline")
        print(str_pr + end_of_table)