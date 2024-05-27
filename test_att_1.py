import pandas as pd
import os
import numpy as np
from pytorch_utilities import get_XY 
from utilities import load_object, save_object
import numpy as np
from pytorch_utilities import print_predictions
import torch
from tqdm import tqdm
from time import time
from seq2seq_attention.evaluate import evaluate
from seq2seq_attention.model import Seq2Seq_With_Attention
from seq2seq_attention.build_dataloaders import (
    build_fields,
    build_bucket_iterator,
    get_datasets,
    build_vocab,
)
from seq2seq_attention.translate import translate_sentence

BATCH_SIZE = 128
DEVICE = "cuda"
LR = 1e-4 
EPOCHS = 20
MAX_VOCAB_SIZE = 8000
MIN_FREQ = 2
ENC_EMB_DIM = 256
HIDDEN_DIM_ENC = 512
HIDDEN_DIM_DEC = 512
NUM_LAYERS_ENC = 1
NUM_LAYERS_DEC = 1
EMB_DIM_TRG = 256 
TEACHER_FORCING = 0.8
PROGRESS_BAR = False
USE_WANDB = True
DROPOUT = 0
TRAIN_ATTENTION = False
progress_bar=False
disable_pro_bar = not progress_bar

def my_token(xv, yv, name_file):
    
    strpr = "x>y\n"

    file_processed = open(name_file, "w")
    file_processed.write(strpr)
    file_processed.close()

    dicti_vals = {"x": [], "y": []}
    for ix1 in range(len(xv)): 

        v1 = str(xv[ix1]).strip()
        v2 = str(yv[ix1]).strip()

        while "  " in v1:
            v1 = v1.replace("  ", " ")

        while "  " in v2:
            v2 = v2.replace("  ", " ")

        dicti_vals["x"].append(v1.replace("[", "").replace("]", "").replace(".", "a").replace(",", "a"))
        dicti_vals["y"].append(v2.replace("[", "").replace("]", "").replace(".", "a").replace(",", "a"))

    df_new = pd.DataFrame(dicti_vals)

    df_new.to_csv(name_file, index = False, sep = ">") 

num_props = 1

ws_range = [2, 3, 4, 5, 10, 20, 30]

num_to_ws = [-1, 5, 6, 5, 6, 5, 6, 5, 6, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 3, 3, 3, 3, 4, 4, 4, 4, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 15, 15, 15, 15, 19, 19, 19, 19, 25, 25, 25, 25, 29, 29, 29, 29, 16, 16, 16, 16, 32, 32, 32, 32]

num_to_params = [-1, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]

marking_for_range = dict()

for ix in range(len(num_to_params)):
    if num_to_params[ix] == 1:
        marking_for_range[num_to_ws[ix]] = ix

print(marking_for_range)

model_name = "GRU_Att"

resave = False
if resave:
    for filename in os.listdir("actual_train"):

        varname = filename.replace("actual_train_", "")

        file_object_train = load_object("actual_train/actual_train_" + varname)
        file_object_val = load_object("actual_val/actual_val_" + varname)
        file_object_test = load_object("actual/actual_" + varname)

        for ws_use in ws_range:
            
            x_train_all = []
            y_train_all = []

            for k in file_object_train:

                x_train_part, y_train_part = get_XY(file_object_train[k], ws_use, 1, ws_use)
                
                for ix in range(len(x_train_part)):
                    x_train_all.append(x_train_part[ix]) 
                    y_train_all.append(y_train_part[ix])

            x_train_all = np.array(x_train_all)
            y_train_all = np.array(y_train_all)
            
            x_train_all_short = []
            y_train_all_short = []

            for k in file_object_train:

                x_train_part, y_train_part = get_XY(file_object_train[k], ws_use, ws_use, ws_use)
                
                for ix in range(len(x_train_part)):
                    x_train_all_short.append(x_train_part[ix]) 
                    y_train_all_short.append(y_train_part[ix])

            x_train_all_short = np.array(x_train_all_short)
            y_train_all_short = np.array(y_train_all_short)
            
            x_test_all = []
            y_test_all = []

            for k in file_object_test:

                x_test_part, y_test_part = get_XY(file_object_test[k], ws_use, 1, ws_use)
                
                for ix in range(len(x_test_part)):
                    x_test_all.append(x_test_part[ix]) 
                    y_test_all.append(y_test_part[ix])

            x_test_all = np.array(x_test_all)
            y_test_all = np.array(y_test_all)
            
            x_test_all_short = []
            y_test_all_short = []

            for k in file_object_test:

                x_test_part, y_test_part = get_XY(file_object_test[k], ws_use, ws_use, ws_use)
                
                for ix in range(len(x_test_part)):
                    x_test_all_short.append(x_test_part[ix]) 
                    y_test_all_short.append(y_test_part[ix])

            x_test_all_short = np.array(x_test_all_short)
            y_test_all_short = np.array(y_test_all_short)
            
            x_val_all = []
            y_val_all = []

            for k in file_object_val:

                x_val_part, y_val_part = get_XY(file_object_val[k], ws_use, 1, ws_use)
                
                for ix in range(len(x_val_part)):
                    x_val_all.append(x_val_part[ix]) 
                    y_val_all.append(y_val_part[ix])

            x_val_all = np.array(x_val_all)
            y_val_all = np.array(y_val_all)
            
            x_val_all_short = []
            y_val_all_short = []

            for k in file_object_val:

                x_val_part, y_val_part = get_XY(file_object_val[k], ws_use, ws_use, ws_use)
                
                for ix in range(len(x_val_part)):
                    x_val_all_short.append(x_val_part[ix]) 
                    y_val_all_short.append(y_val_part[ix])

            x_val_all_short = np.array(x_val_all_short)
            y_val_all_short = np.array(y_val_all_short)

            print(np.shape(x_train_all))

            if not os.path.isdir("tokenized_data/" + varname):
                os.makedirs("tokenized_data/" + varname)

            my_token(x_train_all, y_train_all, "tokenized_data/" + varname + "/" + varname + "_train_" + str(ws_use) + ".csv")
            my_token(x_val_all, y_val_all, "tokenized_data/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv")
            my_token(x_test_all, y_test_all, "tokenized_data/" + varname + "/" + varname + "_test_" + str(ws_use) + ".csv")

            my_token(x_train_all_short, y_train_all_short, "tokenized_data/" + varname + "/" + varname + "_train_short_" + str(ws_use) + ".csv")
            my_token(x_val_all_short, y_val_all_short, "tokenized_data/" + varname + "/" + varname + "_val_short_" + str(ws_use) + ".csv")
            my_token(x_test_all_short, y_test_all_short, "tokenized_data/" + varname + "/" + varname + "_test_short_" + str(ws_use) + ".csv")

use_eval = True
train_a_model = True
if train_a_model:
    for filename in os.listdir("actual_train"):
        for ws_use in ws_range:
            varname = filename.replace("actual_train_", "")
            if os.path.isfile("train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_test.csv"):
                continue

            if not os.path.isdir("train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/models/" + model_name):
                os.makedirs("train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/models/" + model_name)

            if not os.path.isdir("train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/predictions/train/" + model_name):
                os.makedirs("train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/predictions/train/" + model_name)
            if not os.path.isdir("train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/predictions/val/" + model_name):
                os.makedirs("train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/predictions/val/" + model_name)
            if not os.path.isdir("train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/predictions/test/" + model_name):
                os.makedirs("train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/predictions/test/" + model_name)

            src_field, trg_field = build_fields()
            train_set, val_set, test_set = get_datasets(train_path="tokenized_data/" + varname + "/" + varname + "_train_" + str(ws_use) + ".csv", 
                                                        val_path="tokenized_data/" + varname + "/" + varname + "_val_" + str(ws_use) + ".csv", 
                                                        test_path="tokenized_data/" + varname + "/" + varname + "_test_" + str(ws_use) + ".csv",
                                                        src_field=src_field, 
                                                        trg_field=trg_field)
            
            build_vocab(src_field=src_field, trg_field=trg_field, train_set=train_set, min_freq=MIN_FREQ, max_vocab_size=MAX_VOCAB_SIZE)
            # Check vocabulary 
            print(varname, "train_attention" + str(marking_for_range[ws_use]), ws_use, len(src_field.vocab), len(trg_field.vocab))

            train_loader = build_bucket_iterator(dataset=train_set, batch_size=BATCH_SIZE, device=DEVICE)
            val_loader = build_bucket_iterator(dataset=val_set, batch_size=BATCH_SIZE, device=DEVICE)
            test_loader = build_bucket_iterator(dataset=test_set, batch_size=BATCH_SIZE, device=DEVICE)

            # Safe number of batches in train loader and eval points
            perc = 0.25
            n_batches_train = len(train_loader)
            eval_points = [
                round(i * perc * n_batches_train) - 1 for i in range(1, round(1 / perc))
            ]
            eval_points.append(n_batches_train - 1)

            # Get padding/<sos> idxs
            src_pad_idx = src_field.vocab.stoi["<pad>"]
            trg_pad_idx = trg_field.vocab.stoi["<pad>"]
            seq_beginning_token_idx = src_field.vocab.stoi["<sos>"]
            assert src_field.vocab.stoi["<sos>"] == trg_field.vocab.stoi["<sos>"]

            # Init model wrapper class
            model = Seq2Seq_With_Attention(
                lr=LR,
                enc_vocab_size=len(src_field.vocab),
                vocab_size_trg=len(trg_field.vocab),
                enc_emb_dim=ENC_EMB_DIM,
                hidden_dim_enc=HIDDEN_DIM_ENC,
                hidden_dim_dec=HIDDEN_DIM_DEC,
                dropout=DROPOUT,
                padding_idx=src_pad_idx,
                num_layers_enc=NUM_LAYERS_ENC,
                num_layers_dec=NUM_LAYERS_DEC,
                emb_dim_trg=EMB_DIM_TRG,
                trg_pad_idx=trg_pad_idx,
                device=DEVICE,
                seq_beginning_token_idx=seq_beginning_token_idx,
                train_attention=TRAIN_ATTENTION,
            )

            # Send model to device
            model.send_to_device()

            train_losses = []
            val_losses = []
            best_val_loss = float('inf')

            for epoch in range(EPOCHS):

                now = time()

                # Init loss stats for epoch
                train_loss = 0

                n_batches_since_eval = 0

                for n_batch, train_batch in enumerate(
                    tqdm(
                        train_loader,
                        desc=f"Epoch {epoch}",
                        unit="batch",
                        disable=disable_pro_bar,
                    )
                ):

                    model.seq2seq.train()

                    # Take one gradient step
                    train_loss += model.train_step(
                        src_batch=train_batch.src[0],
                        trg_batch=train_batch.trg,
                        src_lens=train_batch.src[1],
                        teacher_forcing=TEACHER_FORCING,
                    )

                    n_batches_since_eval += 1

                    # Calculate and safe train/eval losses at 25% of epoch
                    if n_batch in eval_points and use_eval:
        
                        now_eval = time()

                        # Evaluate
                        eval_loss = evaluate(model=model, eval_loader=val_loader)

                        print(f"Evaluation time: {(time()-now_eval)/60:.2f} minutes.")

                        # Save mean train/val loss
                        train_losses.append(train_loss / n_batches_since_eval)
                        val_losses.append(eval_loss)

                        # Set counter to 0 again
                        n_batches_since_eval = 0
                        train_loss = 0

                        print(
                            f"Epoch {epoch} [{round(n_batch*100/n_batches_train)}%]: Train loss [{train_losses[-1]}]   |  Val loss [{eval_loss}]\n"
                        )
                        print("##########################################\n")
                        # Save the model if the validation loss is the best we've seen so far
                        if eval_loss < best_val_loss:
                            best_val_loss = eval_loss                
                            torch.save(model.seq2seq.state_dict(), "train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + ".pth")
                            print(f"Best model saved with validation loss: {eval_loss:.4f}")

                print(f"Epoch Training time: {(time()-now)/60:.2f} minutes.")
        
            save_object("train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_train_losses", train_losses)

            save_object("train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_val_losses", val_losses)
             
            if not use_eval:
                torch.save(model.seq2seq.state_dict(), "train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + ".pth")
            else:
                model.seq2seq.load_state_dict(torch.load("train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + ".pth", map_location=DEVICE))
            
            model.seq2seq.eval()
 
            with torch.no_grad():

                y_train_all = []
                predict_train_all = []
                pd_train = pd.read_csv("tokenized_data/" + varname + "/" + varname + "_train_short_" + str(ws_use) + ".csv", sep = ">")
                list_x = list(pd_train["x"])
                list_y = list(pd_train["y"])
                for i in range(len(list_x)):
                    translation, _, _, _ = translate_sentence(
                        sentence=str(list_x[i]),
                        seq2seq_model=model.seq2seq,
                        src_field=src_field,
                        bos=src_field.init_token,
                        eos=src_field.eos_token,
                        eos_idx=src_field.vocab.stoi[src_field.eos_token],
                        trg_field=trg_field,
                        max_len=33,
                    )
                    y_train_all.append([str(list_y[i])]) 
                    predict_train_all.append([translation]) 
                print_predictions(y_train_all, predict_train_all, "train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/predictions/train/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_train.csv")

                y_val_all = []
                predict_val_all = []
                pd_val = pd.read_csv("tokenized_data/" + varname + "/" + varname + "_val_short_" + str(ws_use) + ".csv", sep = ">")
                list_x = list(pd_val["x"])
                list_y = list(pd_val["y"])
                for i in range(len(list_x)):
                    translation, _, _, _ = translate_sentence(
                        sentence=str(list_x[i]),
                        seq2seq_model=model.seq2seq,
                        src_field=src_field,
                        bos=src_field.init_token,
                        eos=src_field.eos_token,
                        eos_idx=src_field.vocab.stoi[src_field.eos_token],
                        trg_field=trg_field,
                        max_len=33,
                    )
                    y_val_all.append([str(list_y[i])]) 
                    predict_val_all.append([translation]) 
                print_predictions(y_val_all, predict_val_all, "train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/predictions/val/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_val.csv")

                y_test_all = []
                predict_test_all = []
                pd_test = pd.read_csv("tokenized_data/" + varname + "/" + varname + "_test_short_" + str(ws_use) + ".csv", sep = ">")
                list_x = list(pd_test["x"])
                list_y = list(pd_test["y"])
                for i in range(len(list_x)):
                    translation, _, _, _ = translate_sentence(
                        sentence=str(list_x[i]),
                        seq2seq_model=model.seq2seq,
                        src_field=src_field,
                        bos=src_field.init_token,
                        eos=src_field.eos_token,
                        eos_idx=src_field.vocab.stoi[src_field.eos_token],
                        trg_field=trg_field,
                        max_len=33,
                    )
                    y_test_all.append([str(list_y[i])]) 
                    predict_test_all.append([translation]) 
                print_predictions(y_test_all, predict_test_all, "train_attention" + str(marking_for_range[ws_use]) + "/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_test.csv")