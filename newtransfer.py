from utilities import load_object
import os
import numpy as np
from datetime import timedelta, datetime
import pandas as pd
 
ws_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 19, 20, 25, 29, 30, 32]

d_object_train = dict()
d_object_val = dict()
d_object_trainval = dict()
d_object_test = dict()
d_object_all = dict()

startdate = datetime(day = 1, month = 1, year = 1970)

setse = {"offsets": ["longitude_no_abs", "latitude_no_abs", "time"], "speed_direction": ["speed", "direction"], "longlat": ["longitude_no_abs", "latitude_no_abs"],
         "xoffsets": ["longitude_no_abs", "time"], "yoffsets": ["latitude_no_abs", "time"]}

for filename in os.listdir("actual_train"):

    varname = filename.replace("actual_train_", "")

    file_object_train = load_object("actual_train/actual_train_" + varname) 
    file_object_val = load_object("actual_val/actual_val_" + varname)
    file_object_test = load_object("actual/actual_" + varname)
    
    d_object_train[varname] = []
    d_object_val[varname] = []
    d_object_trainval[varname] = []
    d_object_test[varname] = []
    d_object_all[varname] = []

    datemark = "short"
    if "abs" not in varname and "time" not in varname:
        datemark = "long"

    d_object_train[datemark] = []
    d_object_val[datemark] = []
    d_object_trainval[datemark] = []
    d_object_test[datemark] = []
    d_object_all[datemark] = []

    for k in file_object_train:
        d_object_train[varname] += file_object_train[k]
        d_object_trainval[varname] += file_object_train[k]
        d_object_all[varname] += file_object_train[k]
        for ix in range(len(file_object_train[k])):

            if len(d_object_train[datemark]) == 0:
                d_object_train[datemark].append(startdate)
            else:
                d_object_train[datemark].append(d_object_train[datemark][-1] + timedelta(hours = 1))

            if len(d_object_trainval[datemark]) == 0:
                d_object_trainval[datemark].append(startdate)
            else:
                d_object_trainval[datemark].append(d_object_trainval[datemark][-1] + timedelta(hours = 1))

            if len(d_object_all[datemark]) == 0:
                d_object_all[datemark].append(startdate)
            else:
                d_object_all[datemark].append(d_object_all[datemark][-1] + timedelta(hours = 1)) 

    for k in file_object_val:
        d_object_val[varname] += file_object_val[k]
        d_object_trainval[varname] += file_object_val[k]
        d_object_all[varname] += file_object_val[k]
        for ix in range(len(file_object_val[k])):

            if len(d_object_val[datemark]) == 0:
                d_object_val[datemark].append(startdate)
            else:
                d_object_val[datemark].append(d_object_val[datemark][-1] + timedelta(hours = 1))

            if len(d_object_trainval[datemark]) == 0:
                d_object_trainval[datemark].append(startdate)
            else:
                d_object_trainval[datemark].append(d_object_trainval[datemark][-1] + timedelta(hours = 1))

            if len(d_object_all[datemark]) == 0:
                d_object_all[datemark].append(startdate)
            else:
                d_object_all[datemark].append(d_object_all[datemark][-1] + timedelta(hours = 1)) 

    for k in file_object_test:
        d_object_test[varname] += file_object_test[k]
        d_object_all[varname] += file_object_test[k]
        for ix in range(len(file_object_test[k])):

            if len(d_object_test[datemark]) == 0:
                d_object_test[datemark].append(startdate)
            else:
                d_object_test[datemark].append(d_object_test[datemark][-1] + timedelta(hours = 1))

            if len(d_object_all[datemark]) == 0:
                d_object_all[datemark].append(startdate)
            else:
                d_object_all[datemark].append(d_object_all[datemark][-1] + timedelta(hours = 1)) 

for s in setse:

    datemark = "short"
    if "speed_direction" in s:
        datemark = "long"

    ixvar = {datemark: "date"}
    ix = -1
    for v in setse[s]:
        if ix == -1:
            ixvar[v] = "OT"
        else:
            ixvar[v] = str(ix)
        ix += 1

    dnewtrain = {ixvar[v]: d_object_train[v] for v in ixvar}
    dnewval = {ixvar[v]: d_object_val[v] for v in ixvar}
    dnewtest = {ixvar[v]: d_object_test[v] for v in ixvar}
    dnewtrainval = {ixvar[v]: d_object_trainval[v] for v in ixvar}
    dnewall = {ixvar[v]: d_object_all[v] for v in ixvar}

    for v in ixvar:
        print(v, s,datemark, len(d_object_train[v]), len(d_object_val[v]), len(d_object_trainval[v]), len(d_object_test[v]), len(d_object_all[v]))

    dfnewtrain = pd.DataFrame(dnewtrain)
    dfnewval = pd.DataFrame(dnewval)
    dfnewtest = pd.DataFrame(dnewtest)
    dfnewtrainval = pd.DataFrame(dnewtrainval)
    dfnewall = pd.DataFrame(dnewall)

    if not os.path.isdir("retry/dataset/" + s + "/"):
        os.makedirs("retry/dataset/" + s + "/")

    dfnewtrain.to_csv("retry/dataset/" + s + "/newdata_TRAIN.csv", index = False, sep = ",") 
    dfnewval.to_csv("retry/dataset/" + s + "/newdata_VAL.csv", index = False, sep = ",") 
    dfnewtest.to_csv("retry/dataset/" + s + "/newdata_TEST.csv", index = False, sep = ",") 
    dfnewtrainval.to_csv("retry/dataset/" + s + "/newdata_TRAIN_VAL.csv", index = False, sep = ",") 
    dfnewall.to_csv("retry/dataset/" + s + "/newdata_ALL.csv", index = False, sep = ",") 
   
yml_part = dict()
for sth in setse:
    yml_part[sth] = dict()
    for ws_use in ws_range:
    
        yml_part[sth][ws_use] = "task_dataset:"
    
        dictiowt = {"task_name": "pretrain_long_term_forecast", 
                    "dataset": sth, 
                    "data": "custom", 
                    "embed": "timeF", 
                    "root_path": "retry/dataset/" + sth + "/", 
                    "data_path": "newdata_TRAIN.csv", 
                    "features": "M",
                    "seq_len": ws_use, 
                    "label_len": 0, 
                    "pred_len": ws_use, 
                    "enc_in": ws_use, 
                    "dec_in": ws_use, 
                    "c_out": ws_use}

        yml_part[sth][ws_use] += "\n " + str(sth) + ":" 
        for v in dictiowt:
            yml_part[sth][ws_use] += "\n  " + v + ": " + str(dictiowt[v]) 
        yml_part[sth][ws_use] += "\n" 

        if not os.path.isdir("retry/data_provider/" + sth + "/" + str(ws_use)):
            os.makedirs("retry/data_provider/" + sth + "/" + str(ws_use))

        file_yml_pre_write = open("retry/data_provider/" + sth + "/"+ str(ws_use) + "/multi_task_pretrain.yaml", "w")
        file_yml_pre_write.write(yml_part[sth][ws_use])
        file_yml_pre_write.close()

        file_yml_pre_write_all = open("retry/data_provider/" + sth + "/" + str(ws_use) + "/multi_task_pretrain_all.yaml", "w")
        file_yml_pre_write_all.write(yml_part[sth][ws_use].replace("TRAIN", "ALL"))
        file_yml_pre_write_all.close()
        
        file_yml_pre_write_val = open("retry/data_provider/" + sth + "/" + str(ws_use) + "/multi_task_pretrain_val.yaml", "w")
        file_yml_pre_write_val.write(yml_part[sth][ws_use].replace("TRAIN", "TRAIN_VAL"))
        file_yml_pre_write_val.close()
        
        file_yml_pre_write_valo = open("retry/data_provider/" + sth + "/" + str(ws_use) + "/multi_task_pretrain_valonly.yaml", "w")
        file_yml_pre_write_valo.write(yml_part[sth][ws_use].replace("TRAIN", "VAL"))
        file_yml_pre_write_valo.close()

        file_yml_pre_write_test = open("retry/data_provider/" + sth + "/" + str(ws_use) + "/multi_task_pretrain_test.yaml", "w")
        file_yml_pre_write_test.write(yml_part[sth][ws_use].replace("TRAIN", "TEST"))
        file_yml_pre_write_test.close()
    
        file_yml_write = open("retry/data_provider/" + sth + "/"+ str(ws_use) + "/zeroshot_task.yaml", "w")
        file_yml_write.write(yml_part[sth][ws_use].replace("pretrain_", ""))
        file_yml_write.close()

        file_yml_write_all = open("retry/data_provider/" + sth + "/" + str(ws_use) + "/zeroshot_task_all.yaml", "w")
        file_yml_write_all.write(yml_part[sth][ws_use].replace("pretrain_", "").replace("TRAIN", "ALL"))
        file_yml_write_all.close()
        
        file_yml_write_val = open("retry/data_provider/" + sth + "/" + str(ws_use) + "/zeroshot_task_val.yaml", "w")
        file_yml_write_val.write(yml_part[sth][ws_use].replace("pretrain_", "").replace("TRAIN", "TRAIN_VAL"))
        file_yml_write_val.close()
        
        file_yml_write_valo = open("retry/data_provider/" + sth + "/" + str(ws_use) + "/zeroshot_task_valonly.yaml", "w")
        file_yml_write_valo.write(yml_part[sth][ws_use].replace("pretrain_", "").replace("TRAIN", "VAL"))
        file_yml_write_valo.close()

        file_yml_write_test = open("retry/data_provider/" + sth + "/" + str(ws_use) + "/zeroshot_task_test.yaml", "w")
        file_yml_write_test.write(yml_part[sth][ws_use].replace("pretrain_", "").replace("TRAIN", "TEST"))
        file_yml_write_test.close()
  
for ws_use in ws_range:
    yml_part_merged12 = {"offsets_speed_direction": yml_part["offsets"][ws_use] + yml_part["speed_direction"][ws_use].replace("task_dataset:", ""),
                         "longlat_speed_direction": yml_part["longlat"][ws_use] + yml_part["speed_direction"][ws_use].replace("task_dataset:", "")}
    for vd in yml_part_merged12:
    
        if not os.path.isdir("retry/data_provider/" + vd + "/" + str(ws_use)):
            os.makedirs("retry/data_provider/" + vd + "/" + str(ws_use))

        file_yml_pre_write = open("retry/data_provider/" + vd + "/"+ str(ws_use) + "/multi_task_pretrain.yaml", "w")
        file_yml_pre_write.write(yml_part_merged12[vd])
        file_yml_pre_write.close()

        file_yml_pre_write_all = open("retry/data_provider/" + vd + "/" + str(ws_use) + "/multi_task_pretrain_all.yaml", "w")
        file_yml_pre_write_all.write(yml_part_merged12[vd].replace("TRAIN", "ALL"))
        file_yml_pre_write_all.close()
        
        file_yml_pre_write_val = open("retry/data_provider/" + vd + "/" + str(ws_use) + "/multi_task_pretrain_val.yaml", "w")
        file_yml_pre_write_val.write(yml_part_merged12[vd].replace("TRAIN", "TRAIN_VAL"))
        file_yml_pre_write_val.close()
        
        file_yml_pre_write_valo = open("retry/data_provider/" + vd + "/" + str(ws_use) + "/multi_task_pretrain_valonly.yaml", "w")
        file_yml_pre_write_valo.write(yml_part_merged12[vd].replace("TRAIN", "VAL"))
        file_yml_pre_write_valo.close()

        file_yml_pre_write_test = open("retry/data_provider/" + vd + "/" + str(ws_use) + "/multi_task_pretrain_test.yaml", "w")
        file_yml_pre_write_test.write(yml_part_merged12[vd].replace("TRAIN", "TEST"))
        file_yml_pre_write_test.close()
    
        file_yml_write = open("retry/data_provider/" + vd + "/"+ str(ws_use) + "/zeroshot_task.yaml", "w")
        file_yml_write.write(yml_part_merged12[vd].replace("pretrain_", ""))
        file_yml_write.close()

        file_yml_write_all = open("retry/data_provider/" + vd + "/" + str(ws_use) + "/zeroshot_task_all.yaml", "w")
        file_yml_write_all.write(yml_part_merged12[vd].replace("pretrain_", "").replace("TRAIN", "ALL"))
        file_yml_write_all.close()
        
        file_yml_write_val = open("retry/data_provider/" + vd + "/" + str(ws_use) + "/zeroshot_task_val.yaml", "w")
        file_yml_write_val.write(yml_part_merged12[vd].replace("pretrain_", "").replace("TRAIN", "TRAIN_VAL"))
        file_yml_write_val.close()
        
        file_yml_write_valo = open("retry/data_provider/" + vd + "/" + str(ws_use) + "/zeroshot_task_valonly.yaml", "w")
        file_yml_write_valo.write(yml_part_merged12[vd].replace("pretrain_", "").replace("TRAIN", "VAL"))
        file_yml_write_valo.close()

        file_yml_write_test = open("retry/data_provider/" + vd + "/" + str(ws_use) + "/zeroshot_task_test.yaml", "w")
        file_yml_write_test.write(yml_part_merged12[vd].replace("pretrain_", "").replace("TRAIN", "TEST"))
        file_yml_write_test.close()