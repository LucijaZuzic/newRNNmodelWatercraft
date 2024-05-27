import pandas as pd
import os  
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utilities import load_object, save_object
import numpy as np

ws_range = [2, 3, 4, 5, 10, 20, 30]
 
dicti_to_print = dict()

for varname in ["speed", "direction", "longitude_no_abs", "latitude_no_abs", "time"]:
    
    if varname not in dicti_to_print:
        dicti_to_print[varname] = dict()

    print(varname)

    for dirnam in os.listdir("UniTS_final_res"):
        
        final_test_NRMSE = []
        final_test_RMSE = []
        final_test_R2 = []
        final_test_MAE = []
        
        ws_arr = []
        model_arr = []

        all_mine = load_object("actual/actual_" + varname)
        all_mine_flat = []
        for filename in all_mine: 
            for val in all_mine[filename]:
                all_mine_flat.append(val)
                
        for ws_use in ws_range: 

            if varname != "time":
                
                final_test_data = pd.read_csv("UniTS_final_res/" + dirnam + "/" + str(ws_use) + "/" + varname + ".csv", index_col = False)

            else:
            
                final_test_data = pd.read_csv("UniTS_final_res/" + dirnam.replace("longlat", "offsets") + "/" + str(ws_use) + "/" + varname + ".csv", index_col = False)

            is_a_nan = False
            for val in final_test_data["predicted"]:
                if str(val) == 'nan':
                    is_a_nan = True
                    break

            ws_arr.append(ws_use) 
                
            if is_a_nan:
                final_test_MAE.append(1000000)
                final_test_R2.append(1000000)
                final_test_NRMSE.append(1000000)
                final_test_RMSE.append(1000000)
            else:
                final_test_MAE.append(mean_absolute_error(final_test_data["actual"], final_test_data["predicted"]))
                final_test_R2.append(r2_score(final_test_data["actual"], final_test_data["predicted"]))
                final_test_NRMSE.append(math.sqrt(mean_squared_error(final_test_data["actual"], final_test_data["predicted"])) / (max(all_mine_flat) - min(all_mine_flat)))
                final_test_RMSE.append(math.sqrt(mean_squared_error(final_test_data["actual"], final_test_data["predicted"])))
        
        model_name_used = "UniTS_" + dirnam

        for mini_ix_val in range(len(final_test_RMSE)):
            ws_use = ws_arr[mini_ix_val]
            if model_name_used not in dicti_to_print[varname]:
                dicti_to_print[varname][model_name_used] = dict()
            dicti_to_print[varname][model_name_used][str(ws_use)] = dict()
            dicti_to_print[varname][model_name_used][str(ws_use)]["NRMSE"] = final_test_NRMSE[mini_ix_val] 
            dicti_to_print[varname][model_name_used][str(ws_use)]["RMSE"] = final_test_RMSE[mini_ix_val] 
            dicti_to_print[varname][model_name_used][str(ws_use)]["R2"] = final_test_R2[mini_ix_val]  
            dicti_to_print[varname][model_name_used][str(ws_use)]["MAE"] = final_test_MAE[mini_ix_val]  
            print(ws_arr[mini_ix_val], np.round(final_test_NRMSE[mini_ix_val] * 100, 2), np.round(final_test_R2[mini_ix_val] * 100, 2), np.round(final_test_MAE[mini_ix_val], 6), np.round(final_test_RMSE[mini_ix_val], 6))
      
rv_metric = {"R2": 2, "RMSE": 6, "MAE": 6, "NRMSE": 2}
mul_metric = {"R2": 100, "RMSE": 1, "MAE": 1, "NRMSE": 100}
list_ws = sorted([int(x) for x in dicti_to_print["speed"][model_name_used]])
list_ws = [2, 3, 4, 5, 10, 20, 30]

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