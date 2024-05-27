import numpy as np
import os

wsl = [2, 3, 4, 5, 10, 20, 30]

for sthn in os.listdir("retry/data_provider"):

    if not "_speed" in sthn:
        continue

    alllinestrain = ""
    alllinestest = ""
    alllinestraintest = ""
    alllinesval = ""

    for ws in wsl:
        alllinestrain += "python run_pretrain.py --is_training 1 --model_id UniTS_zeroshot_pretrain_x64_mine_all_array_" + sthn + "_" + str(ws) + "_train --model UniTS_zeroshot  --prompt_num 10 --patch_len " + str(ws) + " --stride " + str(ws) + " --e_layers 3  --d_model 64 --des 'Exp' --acc_it 128  --batch_size 32  --learning_rate 5e-5  --min_lr 1e-4  --weight_decay 5e-6  --train_epochs 10 --warmup_epochs 0  --min_keep_ratio 0.5  --right_prob 0.5  --min_mask_ratio 0.7  --max_mask_ratio 0.8  --debug online --task_data_config_path retry/data_provider/" + sthn + "/" + str(ws) + "/multi_task_pretrain_all.yaml\n"
        alllinestest += "python run.py --is_training 0  --model_id UniTS_zeroshot_pretrain_x64_mine_all_array_" + sthn + "_" + str(ws) + "_test --model UniTS_zeroshot --prompt_num 10  --patch_len " + str(ws) + " --stride " + str(ws) + " --e_layers 3 --d_model 64   --des 'Exp' --debug online   --project_name zeroshot_newdata_mine_all_array_" + sthn + "_" + str(ws) + "_test  --pretrained_weight checkpoints/ALL_task_UniTS_zeroshot_pretrain_x64_mine_all_array_" + sthn + "_" + str(ws) + "_train_UniTS_zeroshot_All_ftM_dm64_el3_Exp_0/pretrain_checkpoint.pth  --task_data_config_path retry/data_provider/" + sthn + "/" + str(ws) + "/zeroshot_task_all.yaml\n"
        alllinestraintest += "python run.py --is_training 0  --model_id UniTS_zeroshot_pretrain_x64_mine_all_array_" + sthn + "_" + str(ws) + "_train_test --model UniTS_zeroshot --prompt_num 10  --patch_len " + str(ws) + " --stride " + str(ws) + " --e_layers 3 --d_model 64   --des 'Exp' --debug online   --project_name zeroshot_newdata_mine_all_array_" + sthn + "_" + str(ws) + "_train_test  --pretrained_weight checkpoints/ALL_task_UniTS_zeroshot_pretrain_x64_mine_all_array_" + sthn + "_" + str(ws) + "_train_UniTS_zeroshot_All_ftM_dm64_el3_Exp_0/pretrain_checkpoint.pth  --task_data_config_path retry_train_last/data_provider/" + sthn + "/" + str(ws) + "/zeroshot_task_all.yaml\n"
        alllinesval += "python run.py --is_training 0  --model_id UniTS_zeroshot_pretrain_x64_mine_all_array_" + sthn + "_" + str(ws) + "_val --model UniTS_zeroshot --prompt_num 10  --patch_len " + str(ws) + " --stride " + str(ws) + " --e_layers 3 --d_model 64   --des 'Exp' --debug online   --project_name zeroshot_newdata_mine_all_array_" + sthn + "_" + str(ws) + "_val  --pretrained_weight checkpoints/ALL_task_UniTS_zeroshot_pretrain_x64_mine_all_array_" + sthn + "_" + str(ws) + "_train_UniTS_zeroshot_All_ftM_dm64_el3_Exp_0/pretrain_checkpoint.pth  --task_data_config_path retry_val_last/data_provider/" + sthn + "/" + str(ws) + "/zeroshot_task_all.yaml\n"

    print(alllinestrain)
    print(alllinestest)
    print(alllinestraintest)
    print(alllinesval)