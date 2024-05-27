from utilities import load_object, save_object, preprocess_long_lat, scale_long_lat
import os
import pandas as pd
        
all_subdirs = os.listdir()

dict_long_lat = dict()

for subdir_name in all_subdirs: 
    if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
        continue
    dict_long_lat[subdir_name] = dict()
    all_files = os.listdir(subdir_name + "/cleaned_csv/") 
    bad_rides_filenames = set()
    if os.path.isfile(subdir_name + "/bad_rides_filenames"):
        bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")
    gap_rides_filenames = set()
    if os.path.isfile(subdir_name + "/gap_rides_filenames"):
        gap_rides_filenames = load_object(subdir_name + "/gap_rides_filenames")
    test_rides = set()
    if os.path.isfile(subdir_name + "/test_rides"):
        test_rides = load_object(subdir_name + "/test_rides")

    test_rides_veh = []
    train_rides_veh = []
    rides_veh = []
        
    for some_file in all_files:  

        if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames:
            continue
        file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)

        longitudes = list(file_with_ride["fields_longitude"]) 
        latitudes = list(file_with_ride["fields_latitude"]) 

        longitudes, latitudes = preprocess_long_lat(longitudes, latitudes)
        longitudes, latitudes = scale_long_lat(longitudes, latitudes, 0.1, 0.1, True)

        if some_file in test_rides:
            dict_long_lat[subdir_name][some_file] = [longitudes, latitudes, "test"]
        else:
            dict_long_lat[subdir_name][some_file] = [longitudes, latitudes, "train"]

if not os.path.isdir("actual/"):
    os.makedirs("actual/")

save_object("actual/actual_traj", dict_long_lat)