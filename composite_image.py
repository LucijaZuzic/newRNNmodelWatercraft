from utilities import load_object
import os
import numpy as np
import matplotlib.pyplot as plt 

best_best = dict()
worst_worst = dict()
best_best_ride = dict()
worst_worst_ride = dict()
best_best_score = dict()
worst_worst_score = dict()

best_best_avg_ride = dict()
worst_worst_avg_ride = dict()
best_best_avg_score = dict()
worst_worst_avg_score = dict()

avg_for_ride_longlat = dict()

best_best_xavg_ride = dict()
worst_worst_xavg_ride = dict()
best_best_xavg_score = dict()
worst_worst_xavg_score = dict()

xavg_for_ride_longlat = dict()

best_best_yavg_ride = dict()
worst_worst_yavg_ride = dict()
best_best_yavg_score = dict()
worst_worst_yavg_score = dict()

yavg_for_ride_longlat = dict()
      
set_drawable = set()

for metric in best_best_ride:
    for method_composite in best_best_ride[metric]:
        set_drawable.add(best_best_ride[metric][method_composite].replace("/", "/cleaned_csv/") + method_composite)
         
for method_composite in best_best_avg_ride:
    set_drawable.add(best_best_avg_ride[method_composite].replace("/", "/cleaned_csv/") + method_composite)
 
for method_composite in best_best_xavg_ride:
    set_drawable.add(best_best_xavg_ride[method_composite].replace("/", "/cleaned_csv/") + method_composite)
 
for method_composite in best_best_yavg_ride:
    set_drawable.add(best_best_yavg_ride[method_composite].replace("/", "/cleaned_csv/") + method_composite)

for metric in worst_worst_ride:
    for method_composite in worst_worst_ride[metric]:
        set_drawable.add(worst_worst_ride[metric][method_composite].replace("/", "/cleaned_csv/") + method_composite)

for method_composite in worst_worst_avg_ride:
    set_drawable.add(worst_worst_avg_ride[method_composite].replace("/", "/cleaned_csv/") + method_composite)
 
for method_composite in worst_worst_xavg_ride:
    set_drawable.add(worst_worst_xavg_ride[method_composite].replace("/", "/cleaned_csv/") + method_composite)
 
for method_composite in worst_worst_yavg_ride:
    set_drawable.add(worst_worst_yavg_ride[method_composite].replace("/", "/cleaned_csv/") + method_composite)

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
    }
    if long in translate_name:
        return translate_name[long]
    else:
        return long
 
def mosaic(rides, name, method_long = "", method_lat = ""):
    
    x_dim_rides = int(np.sqrt(len(rides)))
    y_dim_rides = x_dim_rides
 
    while x_dim_rides * y_dim_rides < len(rides):
        y_dim_rides += 1
    
    plt.figure(figsize = (10, 10 * y_dim_rides / x_dim_rides), dpi = 80)

    for ix_ride in range(len(rides)):

        test_ride = rides[ix_ride]
            
        plt.subplot(y_dim_rides, x_dim_rides, ix_ride + 1)
        plt.rcParams.update({'font.size': 28}) 
        plt.rcParams['font.family'] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.axis("equal")
        plt.axis("off")
 
        plt.plot(test_ride[0], test_ride[1], c = "k", linewidth = 2)
        
    plt.savefig(name, bbox_inches = "tight")
    plt.close()
 
if not os.path.isdir("mosaic"):
    os.makedirs("mosaic")

all_longlats = []
 
all_subdirs = os.listdir()
 
test_rides_all = []
only_train_rides_all = []
val_rides_all = []
train_rides_all = []
rides_all = []

actual_traj = load_object("actual/actual_traj")

int_veh = sorted([int(v.split("_")[1]) for v in actual_traj.keys()])

for i in int_veh:  

    subdir_name = "Vehicle_" + str(i)

    val_rides = set()
    if os.path.isfile(subdir_name + "/val_rides"):
        val_rides = load_object(subdir_name + "/val_rides")

    test_rides_veh = []
    only_train_rides_veh = []
    val_rides_veh = []
    train_rides_veh = []
    rides_veh = []
        
    for some_file in actual_traj[subdir_name]:  

        longitudes, latitudes, is_test = actual_traj[subdir_name][some_file]
 
        if is_test == "test": 
            test_rides_all.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])
            test_rides_veh.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])
        else:
            train_rides_all.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])
            train_rides_veh.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])
            if some_file in val_rides:
                val_rides_all.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])
                val_rides_veh.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])
            else:
                only_train_rides_all.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])
                only_train_rides_veh.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])
 
        rides_veh.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])
        rides_all.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])

    if len(test_rides_veh):
        mosaic(test_rides_veh, "mosaic/" + subdir_name + "_test_mosaic.png")

    if len(train_rides_veh):
        mosaic(train_rides_veh, "mosaic/" + subdir_name + "_train_mosaic.png")
        
    if len(only_train_rides_veh):
        mosaic(only_train_rides_veh, "mosaic/" + subdir_name + "_only_train_mosaic.png")

    if len(val_rides_veh):
        mosaic(val_rides_veh, "mosaic/" + subdir_name + "_val_mosaic.png")

    if len(rides_veh):
        mosaic(rides_veh, "mosaic/" + subdir_name + "_all_mosaic.png")

    for ix_longlat in range(len(all_longlats)):

        if len(test_rides_veh):
            mosaic(test_rides_veh, "mosaic/" + subdir_name + "_" + all_longlats[ix_longlat][0] + "_" + all_longlats[ix_longlat][1] + "_test_mosaic.png", all_longlats[ix_longlat][0], all_longlats[ix_longlat][1])

    print(subdir_name, len(test_rides_veh), len(train_rides_veh), len(rides_veh))

if len(test_rides_all):
    mosaic(test_rides_all, "mosaic/all_test_mosaic.png")

if len(train_rides_all):
    mosaic(train_rides_all, "mosaic/all_train_mosaic.png")
        
if len(only_train_rides_all):
    mosaic(only_train_rides_all, "mosaic/all_only_train_mosaic.png")

if len(val_rides_all):
    mosaic(val_rides_all, "mosaic/all_val_mosaic.png")

if len(rides_all):
    mosaic(rides_all, "mosaic/all_all_mosaic.png")

for ix_longlat in range(len(all_longlats)):
        
    if len(test_rides_all):
        mosaic(test_rides_all, "mosaic/all_" + all_longlats[ix_longlat][0] + "_" + all_longlats[ix_longlat][1] + "_test_mosaic.png", all_longlats[ix_longlat][0], all_longlats[ix_longlat][1])

print(len(test_rides_all), len(train_rides_all), len(rides_all))