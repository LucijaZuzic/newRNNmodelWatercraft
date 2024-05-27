from utilities import load_object
import os
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    
ws_range = [2, 3, 4, 5, 10, 20, 30]
setse = {"offsets": ["longitude_no_abs", "latitude_no_abs", "time"], "speed_direction": ["speed", "direction"], "longlat": ["longitude_no_abs", "latitude_no_abs"],
         "xoffsets": ["longitude_no_abs", "time"], "yoffsets": ["latitude_no_abs", "time"]}

for dirname in os.listdir("retry/data_provider/"):

    for ws_use in ws_range: 

        if not os.path.isdir("results/all_array_" + dirname + "_" + str(ws_use) + "_test/"):
            continue

        print("results/all_array_" + dirname + "_" + str(ws_use) + "_test/")

        for varname in os.listdir("retry/dataset"):

            ordi = setse[varname][1:]
            ordi.append(setse[varname][0])
            
            if not os.path.isfile("results/all_array_" + dirname + "_" + str(ws_use) + "_test/preds_transformed_" + varname):
                continue
    
            print("results/all_array_" + dirname + "_" + str(ws_use) + "_test/preds_transformed_" + varname)

            file_pd = pd.read_csv("retry/dataset/" + varname + "/newdata_TEST.csv", index_col= False) 

            file_pd_transformed_pred = np.array(load_object("results/all_array_" + dirname + "_" + str(ws_use) + "_test/preds_transformed_" + varname))
            file_pd_transformed_true = np.array(load_object("results/all_array_" + dirname + "_" + str(ws_use) + "_test/trues_transformed_" + varname))
            file_pd_transformed_xs = np.array(load_object("results/all_array_" + dirname + "_" + str(ws_use) + "_test/xs_transformed_" + varname))
            print(len(file_pd["OT"]), len(file_pd_transformed_true))

            all_preds = []
            all_trues_sorted = []
            all_originals_sorted = []
            for ix_use in range(len(file_pd_transformed_pred)):
                one_pred = []
                one_true = []
                one_original = []
                for ix_second in range(len(file_pd_transformed_pred[ix_use][0]) - 1):
                    one_pred.append(float(file_pd_transformed_pred[ix_use][0][ix_second]))
                    one_true.append(float(file_pd_transformed_true[ix_use][0][ix_second]))
                    one_original.append(file_pd[str(ix_second)][ix_use])
                one_pred.append(float(file_pd_transformed_pred[ix_use][0][-1]))
                one_true.append(float(file_pd_transformed_true[ix_use][0][-1]))
                one_original.append(file_pd["OT"][ix_use])
                all_preds.append(one_pred)
                all_trues_sorted.append((one_true, ix_use))
                all_originals_sorted.append((one_original, ix_use))

            all_trues_sorted = sorted(all_trues_sorted)
            all_originals_sorted = sorted(all_originals_sorted)
            print(len(all_trues_sorted), len(all_originals_sorted))

            original_for_ix = dict()
            pred_for_ix = dict()
            for ix_use in range(len(all_originals_sorted)):
                one_original, original_ix = all_originals_sorted[ix_use]
                one_true, true_ix = all_trues_sorted[ix_use]
                one_pred = all_preds[true_ix]
                original_for_ix[original_ix] = one_original
                pred_for_ix[original_ix] = one_pred

            for ix_newvar in range(len(ordi)):

                newvar = ordi[ix_newvar]
                preds_smv = []
                actual_smv = []

                for ix_use in range(len(file_pd_transformed_pred)):
                    one_original = original_for_ix[ix_use]
                    one_pred = pred_for_ix[ix_use]
                    actual_smv.append(one_original[ix_newvar])
                    preds_smv.append(one_pred[ix_newvar])

                df_new = pd.DataFrame({"predicted": preds_smv, "actual": actual_smv})

                if not os.path.isdir("UniTS_final_res/" + dirname + "/" + str(ws_use) + "/"):
                    os.makedirs("UniTS_final_res/" + dirname + "/" + str(ws_use) + "/")

                df_new.to_csv("UniTS_final_res/" + dirname + "/" + str(ws_use) + "/" + newvar + ".csv", index = False) 

                print("UniTS_final_res/" + dirname + "/" + str(ws_use) + "/" + newvar + ".csv", r2_score(preds_smv, actual_smv))