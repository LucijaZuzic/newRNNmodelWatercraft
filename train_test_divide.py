from utilities import load_object, save_object
import os
from sklearn.model_selection import train_test_split

all_subdirs = os.listdir() 
num_occurences_of_direction = dict()
num_occurences_of_direction_diff = dict()
num_occurences_of_direction_in_next_step = dict()
num_occurences_of_direction_in_next_next_step = dict()

all_good_rides = dict()
train_all = 0
val_all = 0
test_all = 0
for subdir_name in all_subdirs: 
    if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
        continue
      
    all_files = os.listdir(subdir_name + "/cleaned_csv/") 
    good_rides = dict()  

    bad_rides_filenames = set()
    if os.path.isfile(subdir_name + "/bad_rides_filenames"):
        bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")
    gap_rides_filenames = set()
    if os.path.isfile(subdir_name + "/gap_rides_filenames"):
        gap_rides_filenames = load_object(subdir_name + "/gap_rides_filenames")
        
    for some_file in all_files:  
        if subdir_name + "/cleaned_csv/" + some_file not in bad_rides_filenames and subdir_name + "/cleaned_csv/" + some_file not in gap_rides_filenames:
            good_rides[some_file] = 0 

    if os.path.isfile(subdir_name + "/test_rides"):
        os.remove(subdir_name + "/test_rides")

    if os.path.isfile(subdir_name + "/train_rides"):
        os.remove(subdir_name + "/train_rides")

    if os.path.isfile(subdir_name + "/val_rides"):
        os.remove(subdir_name + "/val_rides")
 
    if len(good_rides) > 1:
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(list(good_rides.keys()), list(good_rides.values()), test_size=0.15, random_state=42)
       
    if len(good_rides) == 1: 
        X_train_val = list(good_rides.keys())
        X_test = []

    if len(good_rides) == 0: 
        X_train_val = []
        X_test = []

    if len(X_train_val) > 1:
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, list(range(len(X_train_val))), test_size=0.15 / 0.85, random_state=42)
       
    if len(X_train_val) == 1: 
        X_train = X_train_val
        X_val = []

    if len(good_rides) == 0: 
        X_train = []
        X_val = []

    print(subdir_name, len(good_rides), len(X_train), len(X_val), len(X_train) + len(X_val), len(X_test), len(X_train) + len(X_val) + len(X_test))
    train_all += len(X_train)
    val_all += len(X_val)
    test_all += len(X_test)
    save_object(subdir_name + "/train_rides", X_train) 
    save_object(subdir_name + "/val_rides", X_val)
    save_object(subdir_name + "/test_rides", X_test)
print(train_all, val_all, train_all + val_all, test_all, train_all + val_all + test_all)
print(train_all / (train_all + val_all + test_all), val_all / (train_all + val_all + test_all), test_all / (train_all + val_all + test_all))