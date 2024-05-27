import os

ws_range = range(4, 6)

for ws_use in ws_range:

    if not os.path.isdir("csv_data/data_provider/" + str(ws_use)):
        os.makedirs("csv_data/data_provider/" + str(ws_use))
    for filename in os.listdir("actual_train"):

        varname = filename.replace("actual_train_", "")

        if not os.path.isdir("csv_data/dataset/" + str(ws_use) + "/" + varname):
            os.makedirs("csv_data/dataset/" + str(ws_use) + "/" + varname)

        if not os.path.isdir("tmp_data/"):
            os.makedirs("tmp_data/")

        dictio = {"task_name": "pretrain_long_term_forecast", 
                  "dataset": varname, 
                  "data": "custom", 
                  "embed": "timeF", 
                  "root_path": "tmp_data/", 
                  "data_path": "use_data.csv", 
                  "features": "M",
                  "seq_len": ws_use, 
                  "label_len": 0, 
                  "pred_len": 1, 
                  "enc_in": ws_use, 
                  "dec_in": ws_use, 
                  "c_out": ws_use}
         
        yml_part = "task_dataset:"
        yml_part += "\n " + str(varname) + ":" 
        for v in dictio:
            yml_part += "\n  " + v + ": " + str(dictio[v]) 
        yml_part += "\n"

        file_pd = open("csv_data/dataset/" + str(ws_use) + "/" + varname + "/newdata_TEST.csv", "r")
        new_lines = file_pd.readlines()
        file_pd.close()
        min_len = 5
        for ix_use in range(1, len(new_lines) - min_len * 2 + 1):
            str_lines = new_lines[0] 
            for ix_len in range(min_len * 2):
                str_lines += new_lines[ix_use + ix_len]
            
            file_pd_new = open("tmp_data/use_data.csv", "w")
            file_pd_new.write(str_lines) 
            file_pd_new.close()

            file_yml_new = open("tmp_data/yml_data.yaml", "w")
            file_yml_new.write(yml_part) 
            file_yml_new.close()
  
            dicti = {
                        '--is_training': 0, 
                        '--model_id': "UniTS_zeroshot_pretrain_x64_mine_all_new_" + str(ws_use) + "_" + str(ix_use) + "_test",
                        '--model': "UniTS_zeroshot",
                        '--prompt_num': 10,
                        '--patch_len': 1,
                        '--stride': 1,
                        '--e_layers': 3,
                        '--d_model': 64,
                        '--des': "\'Exp\'",
                        '--debug': "online",
                        '--project_name': "zeroshot_newdata_mine_all_new_" + str(ws_use) + "_" + str(ix_use) + "_test",
                        '--pretrained_weight': "checkpoints/ALL_task_UniTS_zeroshot_pretrain_x64_mine_all_" + str(ws_use) + "_train_UniTS_zeroshot_All_ftM_dm64_el3_Exp_0/pretrain_checkpoint.pth",
                        '--task_data_config_path': "tmp_data/yml_data.yaml"
                    } 
            
            line = "python run.py"

            for d in dicti:
                line += " " + d + " " + str(dicti[d])

            #print(str_lines)
            #print(yml_part) 
            #print(line) 

            #stream = os.popen(line)
            #output = stream.read()
            #print(output)

            break

        break

    break