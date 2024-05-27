import pandas as pd
import os  
from utilities import load_object, save_object, get_sides_from_angle
from pytorch_utilities import get_XY

dictangle = dict()
def change_angle(angle, name_file):
        
    if name_file not in dictangle:
        file_with_ride = pd.read_csv(name_file) 
        x_dir = list(file_with_ride["fields_longitude"])[0] < list(file_with_ride["fields_longitude"])[-1]
        y_dir = list(file_with_ride["fields_latitude"])[0] < list(file_with_ride["fields_latitude"])[-1]
        dictangle[name_file] = (x_dir, y_dir)
    else:
        x_dir, y_dir = dictangle[name_file]

    new_dir = (90 - angle + 360) % 360 
    if not x_dir: 
        new_dir = (180 - new_dir + 360) % 360
    if not y_dir: 
        new_dir = 360 - new_dir 

    return new_dir

predicted_all = dict()
y_test_all = dict()
ws_all = dict() 
num_to_ws = [-1, 5, 6, 5, 6, 5, 6, 5, 6, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 3, 3, 3, 3, 4, 4, 4, 4, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 15, 15, 15, 15, 19, 19, 19, 19, 25, 25, 25, 25, 29, 29, 29, 29, 16, 16, 16, 16, 32, 32, 32, 32]
model_name = "GRU_Att"

if os.path.isfile("attention_result_val/predicted_all"):
    predicted_all = load_object("attention_result_val/predicted_all")
    
if os.path.isfile("attention_result_val/y_test_all"):
    y_test_all = load_object("attention_result_val/y_test_all")

if os.path.isfile("attention_result_val/ws_all"):
    ws_all = load_object("attention_result_val/ws_all")

for varname in os.listdir("train_attention1"):

    print(varname)
  
    all_mine = load_object("actual_val/actual_val_" + varname)
    all_mine_flat = []
    for filename in all_mine: 
        for val in all_mine[filename]:
            all_mine_flat.append(val)
    
    if varname not in predicted_all:
        predicted_all[varname] = dict()

    if varname not in y_test_all:
        y_test_all[varname] = dict()

    if varname not in ws_all:
        ws_all[varname] = dict()

    if model_name not in predicted_all[varname]:
        predicted_all[varname][model_name] = dict()
        
    if model_name not in y_test_all[varname]:
        y_test_all[varname][model_name] = dict()

    if model_name not in ws_all[varname]:
        ws_all[varname][model_name] = dict()

    for test_num in range(1, 69):
        if not os.path.isdir("train_attention" + str(test_num)):
            continue
        ws_use = num_to_ws[test_num]

        if test_num not in predicted_all[varname][model_name]:
            predicted_all[varname][model_name][test_num] = dict()
            
        if test_num not in y_test_all[varname][model_name]:
            y_test_all[varname][model_name][test_num] = dict()

        if test_num not in ws_all[varname][model_name]:
            ws_all[varname][model_name][test_num] = dict()
           
        final_test_data = pd.read_csv("train_attention" + str(test_num) + "/" + varname + "/predictions/val/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_val.csv", sep = ";", index_col = False)

        final_test_data_predicted = [str(x).strip() for x in final_test_data["predicted"]]

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

        ws_all[varname][model_name][test_num] = ws_use

        len_total = 0

        for k in file_object_test:

            x_test_part, y_test_part = get_XY(file_object_test[k], ws_use, ws_use, ws_use)
            
            y_test_all[varname][model_name][test_num][k] = []
            for ix1 in range(len(y_test_part)): 
                for ix2 in range(len(y_test_part[ix1])): 
                    y_test_all[varname][model_name][test_num][k].append(y_test_part[ix1][ix2])

            predicted_all[varname][model_name][test_num][k] = list(final_test_data_predicted[len_total:len_total + len(y_test_all[varname][model_name][test_num][k])])
            len_total += len(y_test_all[varname][model_name][test_num][k])  

if not os.path.isdir("attention_result_val"):
    os.makedirs("attention_result_val")

save_object("attention_result_val/predicted_all", predicted_all)
save_object("attention_result_val/y_test_all", y_test_all)
save_object("attention_result_val/ws_all", ws_all)

predicted_long = dict()
predicted_lat = dict()

actual_long = dict()
actual_lat = dict()

if os.path.isfile("attention_result_val/predicted_long"):
    predicted_long = load_object("attention_result_val/predicted_long")

if os.path.isfile("attention_result_val/predicted_lat"):
    predicted_lat = load_object("attention_result_val/predicted_lat")

if os.path.isfile("attention_result_val/actual_long"):
    actual_long = load_object("attention_result_val/actual_long")

if os.path.isfile("attention_result_val/actual_lat"):
    actual_lat = load_object("attention_result_val/actual_lat")
    
if model_name not in actual_long:
    actual_long[model_name] = dict()
if model_name not in actual_lat:
    actual_lat[model_name] = dict() 

if model_name not in predicted_long:
    predicted_long[model_name] = dict()
if model_name not in predicted_lat:
    predicted_lat[model_name] = dict()  

for test_num in range(1, 69):
    if not os.path.isdir("train_attention" + str(test_num)):
        continue
    if test_num not in actual_long[model_name]:
        actual_long[model_name][test_num] = dict()
    if test_num not in actual_lat[model_name]:
        actual_lat[model_name][test_num] = dict()  

    for k in y_test_all["longitude_no_abs"][model_name][test_num]:
        print(model_name, k, "actual")
        actual_long[model_name][test_num][k] = [0]
        actual_lat[model_name][test_num][k] = [0]
        
        max_offset_long_lat = max(ws_all["longitude_no_abs"][model_name][test_num], ws_all["latitude_no_abs"][model_name][test_num])
        long_offset = max_offset_long_lat - ws_all["longitude_no_abs"][model_name][test_num]
        lat_offset = max_offset_long_lat - ws_all["latitude_no_abs"][model_name][test_num]
        range_long = len(y_test_all["longitude_no_abs"][model_name][test_num][k]) - long_offset
        range_lat = len(y_test_all["latitude_no_abs"][model_name][test_num][k]) - lat_offset
        min_range_long_lat = min(range_long, range_lat)

        for ix in range(min_range_long_lat):
            actual_long[model_name][test_num][k].append(actual_long[model_name][test_num][k][-1] + y_test_all["longitude_no_abs"][model_name][test_num][k][ix + long_offset])
            actual_lat[model_name][test_num][k].append(actual_lat[model_name][test_num][k][-1] + y_test_all["latitude_no_abs"][model_name][test_num][k][ix + lat_offset])

    if test_num not in predicted_long[model_name]:
        predicted_long[model_name][test_num] = dict()
    if test_num not in predicted_lat[model_name]:
        predicted_lat[model_name][test_num] = dict()   
        
    if "long no abs" not in predicted_long[model_name][test_num]:
        predicted_long[model_name][test_num]["long no abs"] = dict()
    if "lat no abs" not in predicted_lat[model_name][test_num]:
        predicted_lat[model_name][test_num]["lat no abs"] = dict()   

    for k in predicted_all["longitude_no_abs"][model_name][test_num]:
        print(model_name, k, "long no abs")
        predicted_long[model_name][test_num]["long no abs"][k] = [0]
        predicted_lat[model_name][test_num]["lat no abs"][k] = [0]
        
        max_offset_long_lat = max(ws_all["longitude_no_abs"][model_name][test_num], ws_all["latitude_no_abs"][model_name][test_num])
        long_offset = max_offset_long_lat - ws_all["longitude_no_abs"][model_name][test_num]
        lat_offset = max_offset_long_lat - ws_all["latitude_no_abs"][model_name][test_num]
        range_long = len(y_test_all["longitude_no_abs"][model_name][test_num][k]) - long_offset
        range_lat = len(y_test_all["latitude_no_abs"][model_name][test_num][k]) - lat_offset
        min_range_long_lat = min(range_long, range_lat)

        for ix in range(min_range_long_lat):
            predicted_long[model_name][test_num]["long no abs"][k].append(predicted_long[model_name][test_num]["long no abs"][k][-1] + predicted_all["longitude_no_abs"][model_name][test_num][k][ix + long_offset])
            predicted_lat[model_name][test_num]["lat no abs"][k].append(predicted_lat[model_name][test_num]["lat no abs"][k][-1] + predicted_all["latitude_no_abs"][model_name][test_num][k][ix + lat_offset])

    if "long speed dir" not in predicted_long[model_name][test_num]:
        predicted_long[model_name][test_num]["long speed dir"] = dict()
    if "lat speed dir" not in predicted_lat[model_name][test_num]:
        predicted_lat[model_name][test_num]["lat speed dir"] = dict()    

    for k in predicted_all["speed"][model_name][test_num]:
        print(model_name, k, "long speed dir")
        predicted_long[model_name][test_num]["long speed dir"][k] = [0]
        predicted_lat[model_name][test_num]["lat speed dir"][k] = [0]
    
        max_offset_speed_dir_time = max(max(ws_all["speed"][model_name][test_num], ws_all["direction"][model_name][test_num]), ws_all["time"][model_name][test_num])
        speed_offset_time = max_offset_speed_dir_time - ws_all["speed"][model_name][test_num]
        dir_offset_time = max_offset_speed_dir_time - ws_all["direction"][model_name][test_num]
        time_offset_time = max_offset_speed_dir_time - ws_all["time"][model_name][test_num]
        range_speed_time = len(y_test_all["speed"][model_name][test_num][k]) - speed_offset_time
        range_dir_time = len(y_test_all["direction"][model_name][test_num][k]) - dir_offset_time
        range_time_time = len(y_test_all["time"][model_name][test_num][k]) - time_offset_time
        min_range_speed_dir_time = min(min(range_speed_time, range_dir_time), range_time_time)

        for ix in range(min_range_speed_dir_time):
            new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][test_num][k][ix + speed_offset_time] / 111 / 0.1 / 3600 * predicted_all["time"][model_name][test_num][k][ix + time_offset_time], change_angle(predicted_all["direction"][model_name][test_num][k][ix + dir_offset_time], k))
            predicted_long[model_name][test_num]["long speed dir"][k].append(predicted_long[model_name][test_num]["long speed dir"][k][-1] + new_long)
            predicted_lat[model_name][test_num]["lat speed dir"][k].append(predicted_lat[model_name][test_num]["lat speed dir"][k][-1] + new_lat)
            
    if "long speed actual dir" not in predicted_long[model_name][test_num]:
        predicted_long[model_name][test_num]["long speed actual dir"] = dict()
    if "lat speed actual dir" not in predicted_lat[model_name][test_num]:
        predicted_lat[model_name][test_num]["lat speed actual dir"] = dict()    

    for k in predicted_all["speed"][model_name][test_num]:
        print(model_name, k, "long speed actual dir")
        predicted_long[model_name][test_num]["long speed actual dir"][k] = [0]
        predicted_lat[model_name][test_num]["lat speed actual dir"][k] = [0]
    
        max_offset_speed_dir_time = max(max(ws_all["speed"][model_name][test_num], ws_all["direction"][model_name][test_num]), ws_all["time"][model_name][test_num])
        speed_offset_time = max_offset_speed_dir_time - ws_all["speed"][model_name][test_num]
        dir_offset_time = max_offset_speed_dir_time - ws_all["direction"][model_name][test_num]
        time_offset_time = max_offset_speed_dir_time - ws_all["time"][model_name][test_num]
        range_speed_time = len(y_test_all["speed"][model_name][test_num][k]) - speed_offset_time
        range_dir_time = len(y_test_all["direction"][model_name][test_num][k]) - dir_offset_time
        range_time_time = len(y_test_all["time"][model_name][test_num][k]) - time_offset_time
        min_range_speed_dir_time = min(min(range_speed_time, range_dir_time), range_time_time)

        for ix in range(min_range_speed_dir_time):
            new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][test_num][k][ix + speed_offset_time] / 111 / 0.1 / 3600 * y_test_all["time"][model_name][test_num][k][ix + time_offset_time], change_angle(predicted_all["direction"][model_name][test_num][k][ix + dir_offset_time], k))
            predicted_long[model_name][test_num]["long speed actual dir"][k].append(predicted_long[model_name][test_num]["long speed actual dir"][k][-1] + new_long)
            predicted_lat[model_name][test_num]["lat speed actual dir"][k].append(predicted_lat[model_name][test_num]["lat speed actual dir"][k][-1] + new_lat)
            
    if "long speed ones dir" not in predicted_long[model_name][test_num]:
        predicted_long[model_name][test_num]["long speed ones dir"] = dict()
    if "lat speed ones dir" not in predicted_lat[model_name][test_num]:
        predicted_lat[model_name][test_num]["lat speed ones dir"] = dict()     

    for k in predicted_all["speed"][model_name][test_num]:
        print(model_name, k, "long speed ones dir")
        predicted_long[model_name][test_num]["long speed ones dir"][k] = [0]
        predicted_lat[model_name][test_num]["lat speed ones dir"][k] = [0]
    
        max_offset_speed_dir = max(ws_all["speed"][model_name][test_num], ws_all["direction"][model_name][test_num])
        speed_offset = max_offset_speed_dir - ws_all["speed"][model_name][test_num]
        dir_offset = max_offset_speed_dir - ws_all["direction"][model_name][test_num]
        range_speed = len(y_test_all["speed"][model_name][test_num][k]) - speed_offset
        range_dir = len(y_test_all["direction"][model_name][test_num][k]) - dir_offset
        min_range_speed_dir = min(range_speed, range_dir)

        for ix in range(min_range_speed_dir):
            new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][test_num][k][ix + speed_offset] / 111 / 0.1 / 3600, change_angle(predicted_all["direction"][model_name][test_num][k][ix + dir_offset], k))
            predicted_long[model_name][test_num]["long speed ones dir"][k].append(predicted_long[model_name][test_num]["long speed ones dir"][k][-1] + new_long)
            predicted_lat[model_name][test_num]["lat speed ones dir"][k].append(predicted_lat[model_name][test_num]["lat speed ones dir"][k][-1] + new_lat)

if not os.path.isdir("attention_result_val"):
    os.makedirs("attention_result_val")

save_object("attention_result_val/actual_long", actual_long)
save_object("attention_result_val/actual_lat", actual_lat)
save_object("attention_result_val/predicted_long", predicted_long)
save_object("attention_result_val/predicted_lat", predicted_lat)