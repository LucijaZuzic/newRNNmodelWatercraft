from utilities import load_object
import numpy as np
import matplotlib.pyplot as plt
import os

dicti_all = load_object("dicti_all_traj")
ord_metric1 = ["GRU_Att_1", "GRU_Att_2", "GRU_Att_3", "GRU_Att_4"]
ord_metric2 = ["UniTS_longlat_speed_direction", "UniTS_offsets_speed_direction"]
ord_metric = ["GRU_Att_1", "GRU_Att_2", "GRU_Att_3", "GRU_Att_4", "UniTS_longlat_speed_direction", "UniTS_offsets_speed_direction"]
hidden_range = [256]
model_list = ["LSTM", "GRU", "RNN"]
modes = ["Reference", "Third", "Linear", "Twice"]
ord_metric3 = []
for mod_use in modes:
    for model_name in model_list:
            ord_metric.append(model_name + "_" + mod_use + "_256")
            ord_metric3.append(model_name + "_" + mod_use + "_256")
metric_dicti = {"Euclid": 0, "R2": 2, "MAE": 0, "RMSE": 0, "R2_wt": 2, "MAE_wt": 0, "RMSE_wt": 0}
metric_translate = {"Euclid": "Euclidean distance", "R2": "$R^{2}$ (%)", "MAE": "MAE", "RMSE": "RMSE", "R2_wt": "$R^{2}$ (%) (time)", "MAE_wt": "MAE (time)", "RMSE_wt": "RMSE (time)"}
list_ws = [2, 3, 4, 5, 10, 20, 30] 

metrictouse = ["Euclid", "MAE", "R2"]
vartouse = ["long speed actual dir", "long no abs"]
translate_varname = {"long speed ones dir": "speed, heading, a fixed one-second time interval",
                    "long speed dir": "speed, heading, time intervals",
                    "long speed actual dir": "speed, heading, the actual time interval",
                    "long no abs": "$x$ and $y$ offset"}
start_of_table = "\\begin{figure}[!t]\n\t\\centering\n\t\\includegraphics[width = 0.99\linewidth]{FILENAME}"
end_of_table = "\n\t\\caption{METRICNAME for the trajectories in the testing dataset estimated using VARNAME for different RNN models, and forecasting times.}\n\t\\label{fig:test_VARNAME_METRICNAME}\n\\end{figure}\n"
for metric_name_use in metrictouse:
    for varname in vartouse:
        line_for_model = dict()
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
                line_for_model[model_name_use] = []
                for val_ws in list_ws: 
                    line_for_model[model_name_use].append(dicti_all[varname][model_name_use][str(val_ws)][metric_name_use] * (10 ** metric_dicti[metric_name_use]))
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
        if not os.path.isdir("new_img_traj"):
            os.makedirs("new_img_traj")
        plt.figure(figsize = (7, 9))
        plt.subplot(3, 1, 1)
        plt.title(metric_translate[metric_name_use] + "\n" + translate_varname[varname].capitalize())
        for model_name_use in ord_metric1:
            plt.plot(list_ws, line_for_model[model_name_use], label = model_name_use.replace("_", " ").replace(" 256", "").replace(" longlat speed direction", ""))
        plt.xticks(list_ws)
        plt.legend(ncol = 2)
        plt.subplot(3, 1, 2)
        for model_name_use in ord_metric2:
            if "offsets" in model_name_use:
                continue
            plt.plot(list_ws, line_for_model[model_name_use], label = model_name_use.replace("_", " ").replace(" 256", "").replace(" longlat speed direction", ""))
        plt.xticks(list_ws)
        plt.legend()
        plt.subplot(3, 1, 3)
        for model_name_use in ord_metric3:
            plt.plot(list_ws, line_for_model[model_name_use], label = model_name_use.replace("_", " ").replace(" 256", "").replace(" longlat speed direction", ""))
        plt.xticks(list_ws)
        plt.xlabel("Forecasting time")
        plt.legend(ncol = 4, loc = "lower center", bbox_to_anchor = (0.5, -0.7))
        #plt.show()
        plt.savefig("new_img_traj/" + metric_name_use + "_" + varname + ".png", bbox_inches = "tight")
        plt.close()
        newend = end_of_table.replace("METRICNAME", metric_name_use).replace("VARNAME_", varname + "_").replace("NRMSE ", "NRMSE (\%) ").replace("R2 ", "$R^{2}$ (\%) ").replace("VARNAME", translate_varname[varname])
        newstart = start_of_table.replace("FILENAME", "new_img_traj/" + metric_name_use + "_" + varname + ".png")
        print(newstart + newend.replace("Euclid ", "The Euclidean distance "))