import numpy as np
import pickle
from datetime import datetime
from scipy.integrate import simpson
import pandas as pd
import matplotlib.pyplot as plt
 
def new_metric(metric_name):
    new_metric_name = {"simpson x": "Simpson x", "trapz x": "Trapz x", 
              "simpson y": "Simpson y", "trapz y": "Trapz y",
              "euclidean": "Euclidean"}
    if metric_name in new_metric_name:
        return new_metric_name[metric_name]
    else:
        return metric_name
    
def translate_method(longlat):
    translate_name = {
        "long no abs-lat no abs": "x and y offset",  
        "long speed dir-lat speed dir": "Speed and heading", 
        "long speed ones dir-lat speed ones dir": "Speed and heading, 1s", 
    }
    if longlat in translate_name:
        return translate_name[longlat]
    else:
        return longlat
    
translate_var = {
             "Direction": "Heading",  
             "Latitude no abs": "y offset",  
             "Longitude no abs": "x offset",   
             "Time": "Time",
             "Speed": "Speed", 
             }

def random_colors(num_colors):
    colors_set = []
    for x in range(num_colors):
        string_color = "#"
        while string_color == "#" or string_color in colors_set:
            string_color = "#"
            set_letters = "0123456789ABCDEF"
            for y in range(6):
                string_color += set_letters[np.random.randint(0, 16)]
        colors_set.append(string_color)
    return colors_set
    
def load_object(file_name): 
    with open(file_name, 'rb') as file_object:
        data = pickle.load(file_object) 
        file_object.close()
        return data
    
def save_object(file_name, std1):       
    with open(file_name, 'wb') as file_object:
        pickle.dump(std1, file_object) 
        file_object.close()
    
def preprocess_long_lat(long_list, lat_list):
    x_dir = long_list[0] < long_list[-1]
    y_dir = lat_list[0] < lat_list[-1]
 
    long_list2 = [x - min(long_list) for x in long_list]
    lat_list2 = [y - min(lat_list) for y in lat_list]
    if x_dir == False: 
        long_list2 = [max(long_list2) - x for x in long_list2]
    if y_dir == False:
        lat_list2 = [max(lat_list2) - y for y in lat_list2]

    return long_list2, lat_list2    
      
def process_time(time_as_str):
    milisecond = int(time_as_str.split(".")[1]) / 1000
    time_as_str = time_as_str.split(".")[0]
    epoch = datetime(1970, 1, 1)
    return (datetime.strptime(time_as_str, '%Y-%m-%d %H:%M:%S') - epoch).total_seconds() + milisecond

def scale_long_lat(long_list, lat_list, xmax = 0, ymax = 0, keep_aspect_ratio = True):
    minx = np.min(long_list)
    maxx = np.max(long_list)
    miny = np.min(lat_list)
    maxy = np.max(lat_list)
    x_diff = maxx - minx
    if x_diff == 0:
        x_diff = 1
    y_diff = maxy - miny 
    if y_diff == 0:
        y_diff = 1
    if xmax == 0 and ymax == 0 and keep_aspect_ratio:
        xmax = max(x_diff, y_diff)
        ymax = max(x_diff, y_diff)
    if xmax == 0 and ymax == 0 and not keep_aspect_ratio:
        xmax = x_diff
        ymax = y_diff
    if xmax == 0 and ymax != 0 and keep_aspect_ratio:
        xmax = ymax 
    if xmax == 0 and ymax != 0 and not keep_aspect_ratio:
        xmax = x_diff 
    if xmax != 0 and ymax == 0 and keep_aspect_ratio:
        ymax = xmax 
    if xmax != 0 and ymax == 0 and not keep_aspect_ratio:
        ymax = y_diff 
    if xmax != 0 and ymax != 0 and keep_aspect_ratio and xmax != ymax:
        ymax = xmax  
    long_list2 = [(x - min(long_list)) / xmax for x in long_list]
    lat_list2 = [(y - min(lat_list)) / ymax for y in lat_list]
    return long_list2, lat_list2  
 
def euclidean(longitudes1, latitudes1, longitudes2, latitudes2):
    sum_dist = 0
    for i in range(len(longitudes1)):
        sum_dist += np.sqrt((longitudes1[i] - longitudes2[i]) ** 2 + (latitudes1[i] - latitudes2[i]) ** 2)
    return sum_dist / len(longitudes1)

def load_traj_name(name): 
    file_with_ride = pd.read_csv(name)
    longitudes = list(file_with_ride["fields_longitude"])
    latitudes = list(file_with_ride["fields_latitude"]) 
    times = list(file_with_ride["time"])  
    return longitudes, latitudes, times

def get_sides_from_angle(longest, angle):
    return longest * np.cos(angle / 180 * np.pi), longest * np.sin(angle / 180 * np.pi)
 
def fill_gap(list_gap):
    list_no_gap = []
    last_val = 0
    for index_num in range(len(list_gap)):
        if list_gap[index_num] != "undefined":
            last_val = list_gap[index_num]
        list_no_gap.append(last_val)  
    return list_no_gap

def compare_traj_and_sample(sample_x, sample_y, sample_time, t1, metric_used): 
    if "simpson " in metric_used: 
        sample_time_new = [x for x in sample_time]
        for x in range(1, len(sample_time_new)):
            if sample_time_new[x] == sample_time_new[x - 1]:
                sample_time_new[x] = sample_time_new[x - 1] + 10 ** -20
        t1_new = [x for x in t1["time"]]
        for x in range(1, len(t1_new)):
            if t1_new[x] == t1_new[x - 1]:
                t1_new[x] = t1_new[x - 1] + 10 ** -20     
    if metric_used == "trapz x":
        return abs(np.trapz(t1["long"], t1["time"]) - np.trapz(sample_x, sample_time))
    if metric_used == "simpson x":  
        return abs(simpson(t1["long"], t1_new) - simpson(sample_x, sample_time_new)) 
    if metric_used == "trapz y":
        return abs(np.trapz(t1["lat"], t1["time"]) - np.trapz(sample_y, sample_time))
    if metric_used == "simpson y": 
        return abs(simpson(t1["lat"], t1_new) - simpson(sample_y, sample_time_new))  
    if metric_used == "euclidean":
        return euclidean(t1["long"], t1["lat"], sample_x, sample_y)
    
def predict_prob(probability, probability_in_next_step, probability_in_next_next_step, minval, maxval, stepv):
    roundingval = int(-np.log10(stepv))
    possible_values = (maxval - minval) // stepv + 1 
    x = []
    n = 10000
    prev_distance = 0
    prev_prev_distance = 0
    for i in range(n):
        if i == 0:
            distance = np.random.choice(list(probability.keys()),p=list(probability.values()))  
        if i == 1:
            if prev_distance in probability_in_next_step:
                distance = np.random.choice(list(probability_in_next_step[prev_distance].keys()),p=list(probability_in_next_step[prev_distance].values())) 
            else:
                distance = np.random.choice(list(probability_in_next_step["undefined"].keys()),p=list(probability_in_next_step["undefined"].values())) 
        if i > 1:
            if prev_prev_distance in probability_in_next_next_step and prev_distance in probability_in_next_next_step[prev_prev_distance]:
                distance = np.random.choice(list(probability_in_next_next_step[prev_prev_distance][prev_distance].keys()),p=list(probability_in_next_next_step[prev_prev_distance][prev_distance].values())) 
            else:
                if prev_prev_distance in probability_in_next_next_step:
                    distance = np.random.choice(list(probability_in_next_next_step[prev_prev_distance]["undefined"].keys()),p=list(probability_in_next_next_step[prev_prev_distance]["undefined"].values())) 
                else:
                    if prev_distance in probability_in_next_next_step["undefined"]:
                        distance = np.random.choice(list(probability_in_next_next_step["undefined"][prev_distance].keys()),p=list(probability_in_next_next_step["undefined"][prev_distance].values()))
                    else:
                        distance = np.random.choice(list(probability_in_next_next_step["undefined"]["undefined"].keys()),p=list(probability_in_next_next_step["undefined"]["undefined"].values()))
        if distance == "undefined": 
            distance = np.round(minval + np.random.randint(possible_values) * stepv, roundingval) 
        prev_prev_distance = prev_distance
        prev_distance = distance
        x.append(distance)
    return x

def predict_prob_with_array(probability, probability_in_next_step, probability_in_next_next_step, array_vals, minval, maxval, stepv, isangle = False):
    roundingval = int(-np.log10(stepv))
    possible_values = (maxval - minval) // stepv + 1 
    x = []
    n = len(array_vals)
    prev_distance = 0
    prev_prev_distance = 0
    no_empty = 0
    match_score = 0 
    no_empty = 0
    delta_series = [] 
    for i in range(n):
        if i > 1:
            prev_prev_distance = array_vals[i - 2]
        if i > 0:
            prev_distance = array_vals[i - 1]
        if i == 0:
            distance = np.random.choice(list(probability.keys()),p=list(probability.values()))  
        if i == 1:
            if prev_distance in probability_in_next_step:
                distance = np.random.choice(list(probability_in_next_step[prev_distance].keys()),p=list(probability_in_next_step[prev_distance].values())) 
            else:
                distance = np.random.choice(list(probability_in_next_step["undefined"].keys()),p=list(probability_in_next_step["undefined"].values())) 
        if i > 1:
            if prev_prev_distance in probability_in_next_next_step and prev_distance in probability_in_next_next_step[prev_prev_distance]:
                distance = np.random.choice(list(probability_in_next_next_step[prev_prev_distance][prev_distance].keys()),p=list(probability_in_next_next_step[prev_prev_distance][prev_distance].values())) 
            else:
                if prev_prev_distance in probability_in_next_next_step:
                    distance = np.random.choice(list(probability_in_next_next_step[prev_prev_distance]["undefined"].keys()),p=list(probability_in_next_next_step[prev_prev_distance]["undefined"].values())) 
                else:
                    if prev_distance in probability_in_next_next_step["undefined"]:
                        distance = np.random.choice(list(probability_in_next_next_step["undefined"][prev_distance].keys()),p=list(probability_in_next_next_step["undefined"][prev_distance].values()))
                    else:
                        distance = np.random.choice(list(probability_in_next_next_step["undefined"]["undefined"].keys()),p=list(probability_in_next_next_step["undefined"]["undefined"].values()))
        if distance == "undefined": 
            distance = np.round(minval + np.random.randint(possible_values) * stepv, roundingval)
        else:
            no_empty += 1
        x.append(float(distance)) 
        if float(array_vals[i]) == float(x[i]):
            match_score += 1   
        delta_x = abs(float(array_vals[i]) - float(x[i]))
        if isangle:
            if delta_x > 180:
                delta_x = 360 - delta_x
        delta_series.append(delta_x) 
        
    return x, n, match_score, no_empty, delta_series
 
def fix_prob(num_occurences, num_occurences_in_next_step, num_occurences_in_next_next_step, fix = True):
    min_prob = 10 ** -20  
    if not fix:
        min_prob = 0
    probability = dict()
    for distance in num_occurences:
        probability[distance] = num_occurences[distance] / sum(list(num_occurences.values())) - min_prob
    if fix:
        probability["undefined"] = min_prob
    
    probability_in_next_step = dict()
    for prev_distance in num_occurences_in_next_step:
        probability_in_next_step[prev_distance] = dict()
        for distance in num_occurences_in_next_step[prev_distance]:
            probability_in_next_step[prev_distance][distance] = num_occurences_in_next_step[prev_distance][distance] / sum(list(num_occurences_in_next_step[prev_distance].values())) - min_prob
        if fix:    
            probability_in_next_step[prev_distance]["undefined"] = min_prob

    if fix:
        probability_in_next_step["undefined"] = dict()
        for distance in num_occurences:
            probability_in_next_step["undefined"][distance] = num_occurences[distance] / sum(list(num_occurences.values())) - min_prob
        probability_in_next_step["undefined"]["undefined"] = min_prob

    probability_in_next_next_step = dict()
    for prev_prev_distance in num_occurences_in_next_next_step:
        probability_in_next_next_step[prev_prev_distance] = dict()
        for prev_distance in num_occurences_in_next_next_step[prev_prev_distance]:
            probability_in_next_next_step[prev_prev_distance][prev_distance] = dict()
            for distance in num_occurences_in_next_next_step[prev_prev_distance][prev_distance]:
                probability_in_next_next_step[prev_prev_distance][prev_distance][distance] = num_occurences_in_next_next_step[prev_prev_distance][prev_distance][distance] / sum(list(num_occurences_in_next_next_step[prev_prev_distance][prev_distance].values())) - min_prob
            if fix:    
                probability_in_next_next_step[prev_prev_distance][prev_distance]["undefined"] = min_prob
 
    if fix:
        probability_in_next_next_step["undefined"] = dict()

        probability_in_next_next_step["undefined"]["undefined"] = dict() 
        for distance in num_occurences: 
            probability_in_next_next_step["undefined"]["undefined"][distance] = num_occurences[distance] / sum(list(num_occurences.values())) - min_prob
        probability_in_next_next_step["undefined"]["undefined"]["undefined"] = min_prob

        for prev_distance in num_occurences:
            probability_in_next_next_step["undefined"][prev_distance] = dict()  
            for distance in num_occurences:
                probability_in_next_next_step["undefined"][prev_distance][distance] = num_occurences[distance] / sum(list(num_occurences.values())) - min_prob
            probability_in_next_next_step["undefined"][prev_distance]["undefined"] = min_prob 

        for prev_distance in probability_in_next_next_step: 
            probability_in_next_next_step[prev_distance]["undefined"] = dict() 
            for distance in num_occurences:
                probability_in_next_next_step[prev_distance]["undefined"][distance] = num_occurences[distance] / sum(list(num_occurences.values())) - min_prob
            probability_in_next_next_step[prev_distance]["undefined"]["undefined"] = min_prob

    return probability, probability_in_next_step, probability_in_next_next_step
 
def format_e(n):
    if abs(n) >= 10 ** -2 or n == 0:
        return ("$" + str(np.round(n, 2)) + "$").replace(".0$", "$").replace(".00$", "$")
    a = '%.2E' % n
    return "$" + str(str(a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]) + "}$").replace("E-0", "*10^{-").replace("E+0", "*10^{").replace("$1*", "$")

def format_e2(n):
    if abs(n) >= 10 ** -2 or n == 0:
        return str(np.round(n, 2))  
    else:
        return str(np.round(n, 5))
    
def composite_image(filename, show, long1, lat1, nrow, ncol, long_other = [], lat_other = [], legends = [], mark_start = False, subtitles = []):  
    random_colors_legend = ["green", "blue", "red"]
    plt.rcParams.update({'font.size': 28})
    plt.figure(figsize=(15 * ncol, 15 * nrow)) 
    numseen = 0
    for ix in range(len(show)):  
        if show[ix]:
            numseen += 1
            plt.subplot(nrow, ncol, numseen)  
            new_subtitle = ""
            new_subtitle_separated = subtitles[ix].split("\n")
            segment = ""
            for el in new_subtitle_separated:
                if segment != "": 
                    segment += " "
                segment += el
                if len(segment) > 30:
                    if new_subtitle != "": 
                        new_subtitle += "\n"
                    new_subtitle += segment
                    segment = ""
            if segment != "":
                if new_subtitle != "": 
                    new_subtitle += "\n"
                new_subtitle += segment
                segment = ""
            plt.title(new_subtitle) 
            if len(long_other) >= ix:
                for i in range(len(long_other[ix])): 
                    plt.plot(long_other[ix][i], lat_other[ix][i], label = legends[i], color = random_colors_legend[i + 2], linewidth = 10)  
            plt.plot(long1[ix], lat1[ix], label = "Original", color = random_colors_legend[0], linewidth = 10)     
            if mark_start:
                plt.plot(long1[ix][0], lat1[ix][0], marker = "o", label = "Start", color = random_colors_legend[0], mec = random_colors_legend[0], mfc = random_colors_legend[1], ms = 20, mew = 10, linewidth = 10) 
            if len(legends) > 0:
                plt.legend()  

    plt.savefig(filename, bbox_inches = "tight")
    plt.close() 
    composite_image_reverse(filename, show, long1, lat1, ncol, nrow, long_other, lat_other, legends, mark_start, subtitles)
    
def composite_image_reverse(filename, show, long1, lat1, nrow, ncol, long_other = [], lat_other = [], legends = [], mark_start = False, subtitles = []):  
    random_colors_legend = ["green", "blue", "red"]
    plt.rcParams.update({'font.size': 28})
    plt.figure(figsize=(15 * ncol, 15 * nrow)) 
    numseen = 0 
    for ix in range(len(show)):  
        if show[ix]:
            num_of_row = numseen % nrow
            num_of_col = numseen // nrow
            plt.subplot(nrow, ncol, num_of_row * ncol + num_of_col + 1)  
            numseen += 1
            new_subtitle = ""
            new_subtitle_separated = subtitles[ix].split("\n")
            segment = ""
            for el in new_subtitle_separated:
                if segment != "": 
                    segment += " "
                segment += el
                if len(segment) > 30:
                    if new_subtitle != "": 
                        new_subtitle += "\n"
                    new_subtitle += segment
                    segment = ""
            if segment != "":
                if new_subtitle != "": 
                    new_subtitle += "\n"
                new_subtitle += segment
                segment = ""
            plt.title(new_subtitle) 
            if len(long_other) >= ix:
                for i in range(len(long_other[ix])): 
                    plt.plot(long_other[ix][i], lat_other[ix][i], label = legends[i], color = random_colors_legend[i + 2], linewidth = 10)  
            plt.plot(long1[ix], lat1[ix], label = "Original", color = random_colors_legend[0], linewidth = 10)     
            if mark_start:
                plt.plot(long1[ix][0], lat1[ix][0], marker = "o", label = "Start", color = random_colors_legend[0], mec = random_colors_legend[0], mfc = random_colors_legend[1], ms = 20, mew = 10, linewidth = 10) 
            if len(legends) > 0:
                plt.legend()  

    plt.savefig(filename.replace(".png", "_reverse.png"), bbox_inches = "tight")
    plt.close() 