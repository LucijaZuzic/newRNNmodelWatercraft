import os
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_object, save_object, compare_traj_and_sample
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

num_to_ws = [-1, 5, 6, 5, 6, 5, 6, 5, 6, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 3, 3, 3, 3, 4, 4, 4, 4, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 15, 15, 15, 15, 19, 19, 19, 19, 25, 25, 25, 25, 29, 29, 29, 29, 16, 16, 16, 16, 32, 32, 32, 32]

num_to_params = [-1, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]

def str_convert_new(val):
    new_val = val
    power_to = 0
    while abs(new_val) < 1 and new_val != 0.0:
        new_val *= 10
        power_to += 1 
    rounded = "$" + str(np.round(new_val, 2))
    if rounded[-2:] == '.0':
        rounded = rounded[:-2]
    if power_to != 0:  
        rounded += " \\times 10^{-" + str(power_to) + "}"
    return rounded + "$"

def new_metric_translate(metric_name):
    new_metric_name = {"trapz x": "$x$ integration", 
              "trapz y": "$y$ integration",
              "euclidean": "Euclidean distance"}
    if metric_name in new_metric_name:
        return new_metric_name[metric_name]
    else:
        return metric_name
    
def translate_category(long):
    translate_name = {
        "long no abs": "$x$ and $y$ offset",  
        "long speed dir": "Speed, heading, and time", 
        "long speed ones dir": "Speed, heading, and a 1s time interval",
        "long speed actual dir": "Speed, heading, and actual time"
    }
    if long in translate_name:
        return translate_name[long]
    else:
        return long
    
def draw_mosaic(rides_actual, rides_predicted, name):
    
    x_dim_rides = int(np.sqrt(len(rides_actual)))
    y_dim_rides = x_dim_rides
 
    while x_dim_rides * y_dim_rides < len(rides_actual):
        y_dim_rides += 1
    
    plt.figure(figsize = (10, 10 * y_dim_rides / x_dim_rides), dpi = 80)

    for ix_ride in range(len(rides_actual)):
 
        x_actual, y_actual = rides_actual[ix_ride]["long"], rides_actual[ix_ride]["lat"]
        x_predicted, y_predicted = rides_predicted[ix_ride]["long"], rides_predicted[ix_ride]["lat"]
            
        plt.subplot(y_dim_rides, x_dim_rides, ix_ride + 1)
        plt.rcParams.update({'font.size': 28}) 
        plt.rcParams['font.family'] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.axis("equal")
        plt.axis("off")

        plt.plot(x_predicted, y_predicted, c = "b", linewidth = 2, label = "Estimated")
    
        plt.plot(x_actual, y_actual, c = "k", linewidth = 2, label = "Original")

    plt.savefig(name, bbox_inches = "tight")
    plt.close()
    
def draw_mosaic_one(x_actual, y_actual, x_predicted, y_predicted, k, model_name, name, test_num, dist_name):
     
    plt.figure(figsize = (10, 10), dpi = 80)
    plt.rcParams.update({'font.size': 28}) 
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.axis("equal")
  
    plt.plot(x_predicted, y_predicted, c = "b", linewidth = 10, label = "Estimated")
  
    plt.plot(x_actual, y_actual, c = "k", linewidth = 10, label = "Original")

    plt.plot(x_actual[0], x_actual[0], marker = "o", label = "Start", color = "k", mec = "k", mfc = "g", ms = 20, mew = 10, linewidth = 10) 
   
    split_file_veh = k.split("/")
    vehicle = split_file_veh[0].replace("Vehicle_", "")
    ride = split_file_veh[-1].replace("events_", "").replace(".csv", "")

    ws_use = num_to_ws[test_num]

    title_new = "Vehicle " + vehicle + " Ride " + ride + "\n" + model_name + " model\n" + "Window size " + str(ws_use) + "\n" + "Experiment number " + str(num_to_params[test_num]) + "\n" 

    title_new += translate_category(dist_name) + "\n" 
    for metric in distance_predicted_new:
        if "simpson" in metric:
            continue
        title_new += new_metric_translate(metric) + ": " + str_convert_new(distance_predicted_new[metric][model_name][test_num][dist_name][k]) + "\n"
    
    plt.title(title_new.replace("GRU_Att", "GRU Attention"))
    plt.legend()
    plt.savefig(name, bbox_inches = "tight")
    plt.close()
 
predicted_all = load_object("attention_result_train/predicted_all")
y_test_all = load_object("attention_result_train/y_test_all")
ws_all = load_object("attention_result_train/ws_all")

actual_long = load_object("attention_result_train/actual_long")
actual_lat = load_object("attention_result_train/actual_lat")
predicted_long = load_object("attention_result_train/predicted_long")
predicted_lat = load_object("attention_result_train/predicted_lat")

distance_predicted_new = dict()

if os.path.isfile("attention_result_train/distance_predicted_new"):
    distance_predicted_new = load_object("attention_result_train/distance_predicted_new")

metric_names = ["euclidean"] 

if not os.path.isdir("mosaic_attention_train"):
    os.makedirs("mosaic_attention_train")

if not os.path.isdir("mosaic_attention_train_all"):
    os.makedirs("mosaic_attention_train_all")

dicti_to_print = dict()
use_draw = True
for metric in metric_names:

    if metric not in distance_predicted_new:
        distance_predicted_new[metric] = dict()

    for model_name in predicted_long:

        if model_name not in distance_predicted_new[metric]:
            distance_predicted_new[metric][model_name] = dict()

        for test_num in predicted_long[model_name]:

            if test_num not in distance_predicted_new[metric][model_name]:
                distance_predicted_new[metric][model_name][test_num] = dict()

            for dist_name in predicted_long[model_name][test_num]:

                if dist_name not in distance_predicted_new[metric][model_name][test_num]:
                    distance_predicted_new[metric][model_name][test_num][dist_name]  = dict()
                
                all_actual = []
                all_predicted = []

                actual_long_lat = []
                actual_long_lat_time = []
                predicted_long_lat = []
                predicted_long_lat_time = []

                vals_avg = []
                
                int_veh = sorted([int(k.split("/")[0].split("_")[1]) for k in predicted_long[model_name][test_num][dist_name].keys()])

                min_k = ""
                min_dist = 1000000
                max_k = ""
                max_dist = - 1000000

                for v in set(int_veh):

                    all_actual_vehicle = []
                    all_predicted_vehicle = []
                    
                    for k in predicted_long[model_name][test_num][dist_name]:
                        veh_new = int(k.split("/")[0].split("_")[1])
                        
                        if veh_new != v:
                            continue 

                        actual_long_one = actual_long[model_name][test_num][k]
                        actual_lat_one = actual_lat[model_name][test_num][k]

                        predicted_long_one = predicted_long[model_name][test_num][dist_name][k]
                        predicted_lat_one = predicted_lat[model_name][test_num][dist_name.replace("long", "lat")][k]

                        use_len = min(len(actual_long_one), len(predicted_long_one))
                        
                        actual_long_one = actual_long_one[:use_len]
                        actual_lat_one = actual_lat_one[:use_len]

                        predicted_long_one = predicted_long_one[:use_len]
                        predicted_lat_one = predicted_lat_one[:use_len]
                        
                        time_actual = y_test_all["time"][model_name][test_num][k]
                        time_predicted = predicted_all["time"][model_name][test_num][k]

                        time_actual_cumulative = [0]
                        time_predicted_cumulative = [0]
                        
                        for ix in range(len(time_actual)):
                            time_actual_cumulative.append(time_actual_cumulative[-1] + time_actual[ix])
                            time_predicted_cumulative.append(time_predicted_cumulative[-1] + time_predicted[ix])
                            
                        use_len_time = min(use_len, len(time_actual_cumulative))

                        actual_long_one = actual_long_one[:use_len_time]
                        actual_lat_one = actual_lat_one[:use_len_time]

                        predicted_long_one = predicted_long_one[:use_len_time]
                        predicted_lat_one = predicted_lat_one[:use_len_time]
                        
                        time_actual_cumulative = time_actual_cumulative[:use_len_time]
                        time_predicted_cumulative = time_predicted_cumulative[:use_len_time]

                        if k not in distance_predicted_new[metric][model_name][test_num][dist_name]:
                            distance_predicted_new[metric][model_name][test_num][dist_name][k] = compare_traj_and_sample(actual_long_one, actual_lat_one, time_actual_cumulative, {"long": predicted_long_one, "lat": predicted_lat_one, "time": time_predicted_cumulative}, metric)
                        
                        if distance_predicted_new[metric][model_name][test_num][dist_name][k] < min_dist:
                            min_k = k
                            min_dist = distance_predicted_new[metric][model_name][test_num][dist_name][k]

                        if distance_predicted_new[metric][model_name][test_num][dist_name][k] > max_dist:
                            max_k = k
                            max_dist = distance_predicted_new[metric][model_name][test_num][dist_name][k]

                        split_file_veh = k.split("/")
                        vehicle = split_file_veh[0].replace("Vehicle_", "")
                        ride = split_file_veh[-1].replace("events_", "").replace(".csv", "")

                        if use_draw:
                            
                            filename = "mosaic_attention_train_all/Vehicle_" + vehicle + "_events_" + ride + "_" + model_name + "_" + str(test_num) + "_" + dist_name + "_" + dist_name.replace("long", "lat") + "_test_mosaic.png"
                            draw_mosaic_one(actual_long_one, actual_lat_one, predicted_long_one, predicted_lat_one, k, model_name, filename, test_num, dist_name)

                        all_actual.append({"long": actual_long_one, "lat": actual_lat_one})
                        all_predicted.append({"long": predicted_long_one, "lat": predicted_lat_one})

                        vals_avg.append(distance_predicted_new[metric][model_name][test_num][dist_name][k])

                        all_actual_vehicle.append({"long": actual_long_one, "lat": actual_lat_one})
                        all_predicted_vehicle.append({"long": predicted_long_one, "lat": predicted_lat_one})

                        for ix_use_len in range(use_len_time):

                            actual_long_lat.append([actual_long_one[ix_use_len], actual_lat_one[ix_use_len]])
                            actual_long_lat_time.append([actual_long_one[ix_use_len], actual_lat_one[ix_use_len], time_actual_cumulative[ix_use_len]])
                            
                            predicted_long_lat.append([predicted_long_one[ix_use_len], predicted_lat_one[ix_use_len]])
                            predicted_long_lat_time.append([predicted_long_one[ix_use_len], predicted_lat_one[ix_use_len], time_predicted_cumulative[ix_use_len]])
                    
                    if use_draw:

                        filename_veh = "mosaic_attention_train/Vehicle_" + str(v) + "_" + model_name + "_" + str(test_num) + "_" + dist_name.replace("long", "lat") + "_test_mosaic.png"
                        draw_mosaic(all_actual_vehicle, all_predicted, filename_veh)

                if use_draw:

                    filename = "mosaic_attention_train/all_" + model_name + "_" + str(test_num) + "_" + dist_name.replace("long", "lat") + "_test_mosaic.png"
                    draw_mosaic(all_actual, all_predicted, filename)

                print(model_name + "_" + str(test_num) + "_" + dist_name, np.round(np.average(vals_avg), 6))

                print(min_k, min_dist)
                print(max_k, max_dist)

                r2_pred_wt = r2_score(actual_long_lat_time, predicted_long_lat_time)

                mae_pred_wt = mean_absolute_error(actual_long_lat_time, predicted_long_lat_time)

                rmse_pred_wt = math.sqrt(mean_squared_error(actual_long_lat_time, predicted_long_lat_time))

                r2_pred = r2_score(actual_long_lat, predicted_long_lat)
        
                mae_pred = mean_absolute_error(actual_long_lat, predicted_long_lat)

                rmse_pred = math.sqrt(mean_squared_error(actual_long_lat, predicted_long_lat))
                
                print("R2", np.round(r2_pred * 100, 2), "MAE", np.round(mae_pred, 6), "RMSE", np.round(rmse_pred, 6), "R2_wt", np.round(r2_pred_wt * 100, 2), "MAE_wt", np.round(mae_pred_wt, 6), "RMSE_wt", np.round(rmse_pred_wt, 6))

                ws_use = num_to_ws[test_num]

                if dist_name not in dicti_to_print:
                    dicti_to_print[dist_name] = dict()

                if "GRU_Att_" + str(num_to_params[test_num]) not in dicti_to_print[dist_name]:
                    dicti_to_print[dist_name]["GRU_Att_" + str(num_to_params[test_num])] = dict()
                    
                if str(ws_use) not in dicti_to_print[dist_name]["GRU_Att_" + str(num_to_params[test_num])]:
                    dicti_to_print[dist_name]["GRU_Att_" + str(num_to_params[test_num])][str(ws_use)] = dict()

                dicti_to_print[dist_name]["GRU_Att_" + str(num_to_params[test_num])][str(ws_use)]["R2_wt"] = r2_pred_wt
                dicti_to_print[dist_name]["GRU_Att_" + str(num_to_params[test_num])][str(ws_use)]["MAE_wt"] = mae_pred_wt
                dicti_to_print[dist_name]["GRU_Att_" + str(num_to_params[test_num])][str(ws_use)]["RMSE_wt"] = rmse_pred_wt
                dicti_to_print[dist_name]["GRU_Att_" + str(num_to_params[test_num])][str(ws_use)]["Euclid"] = np.average(vals_avg)
                dicti_to_print[dist_name]["GRU_Att_" + str(num_to_params[test_num])][str(ws_use)]["R2"] = r2_pred
                dicti_to_print[dist_name]["GRU_Att_" + str(num_to_params[test_num])][str(ws_use)]["MAE"] = mae_pred
                dicti_to_print[dist_name]["GRU_Att_" + str(num_to_params[test_num])][str(ws_use)]["RMSE"] = rmse_pred

save_object("attention_result_train/distance_predicted_new", distance_predicted_new)

rv_metric = {"R2": 2, "RMSE": 6, "MAE": 6, "R2_wt": 2, "RMSE_wt": 6, "MAE_wt": 6, "Euclid": 6}
mul_metric = {"R2": 100, "RMSE": 1, "MAE": 1, "R2_wt": 100, "RMSE_wt": 1, "MAE_wt": 1, "Euclid": 1}
list_ws = sorted([int(x) for x in dicti_to_print["long no abs"]["GRU_Att_1"]])

dicti_all_traj_train = dict()
if os.path.isfile("dicti_all_traj_train"):
    dicti_all_traj_train = load_object("dicti_all_traj_train")

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
                if varname not in dicti_all_traj_train:
                    dicti_all_traj_train[varname] = dict()
                if model_name_use not in dicti_all_traj_train[varname]:
                    dicti_all_traj_train[varname][model_name_use] = dict()
                if str(val_ws) not in dicti_all_traj_train[varname][model_name_use]:
                    dicti_all_traj_train[varname][model_name_use][str(val_ws)] = dict()
                if metric_name_use not in dicti_all_traj_train[varname][model_name_use][str(val_ws)]:
                    dicti_all_traj_train[varname][model_name_use][str(val_ws)][metric_name_use] = dict()
                dicti_all_traj_train[varname][model_name_use][str(val_ws)][metric_name_use] = vv
                vv = np.round(vv * mul_metric[metric_name_use], rv_metric[metric_name_use])
                str_pr += " & $" + str(vv) + "$"
            str_pr += " \\\\ \\hline\n"
        print(str_pr)

for metric_name_use in list(rv_metric.keys()):
    for model_name_use in dicti_to_print["long no abs"]:
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
                if varname not in dicti_all_traj_train:
                    dicti_all_traj_train[varname] = dict()
                if model_name_use not in dicti_all_traj_train[varname]:
                    dicti_all_traj_train[varname][model_name_use] = dict()
                if str(val_ws) not in dicti_all_traj_train[varname][model_name_use]:
                    dicti_all_traj_train[varname][model_name_use][str(val_ws)] = dict()
                if metric_name_use not in dicti_all_traj_train[varname][model_name_use][str(val_ws)]:
                    dicti_all_traj_train[varname][model_name_use][str(val_ws)][metric_name_use] = dict()
                dicti_all_traj_train[varname][model_name_use][str(val_ws)][metric_name_use] = vv
                vv = np.round(vv * mul_metric[metric_name_use], rv_metric[metric_name_use])
                str_pr += " & $" + str(vv) + "$"
            str_pr += " \\\\ \\hline\n"
        print(str_pr)

save_object("dicti_all_traj_train", dicti_all_traj_train)