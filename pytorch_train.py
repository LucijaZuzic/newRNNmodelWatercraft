import numpy as np
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utilities import load_object, save_object
from pytorch_utilities import get_XY, print_predictions, PyTorchGRUModel, PyTorchLSTMModel, PyTorchRNNModel, PyTorchGRUModelTwice, PyTorchLSTMModelTwice, PyTorchRNNModelTwice, PyTorchGRUModelLinear, PyTorchLSTMModelLinear, PyTorchRNNModelLinear, PyTorchGRUModelThird, PyTorchLSTMModelThird, PyTorchRNNModelThird

num_props = 1

ws_range = [2, 3, 4, 5, 10, 20, 30]

hidden_range = [256]

model_list = ["GRU", "RNN", "LSTM"]

modes = ["Linear", "Reference", "Twice", "Third"]

for h in hidden_range:

    for filename in os.listdir("actual_train"):

        varname = filename.replace("actual_train_", "")

        file_object_train = load_object("actual_train/actual_train_" + varname)
        file_object_val = load_object("actual_val/actual_val_" + varname)
        file_object_train_val = load_object("actual_train_val/actual_train_val_" + varname)
        file_object_test = load_object("actual/actual_" + varname)

        for model_name in model_list:

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

                for mod_use in modes:
                    hidden_use = hidden_range[0]
        
                    if os.path.isfile("train_pytorch/" + mod_use + "/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_test.csv"):
                        continue
                        
                    print(mod_use, varname, model_name, ws_use, hidden_use)
                        
                    if mod_use == "Reference":

                        if model_name == "RNN":
                            pytorch_model = PyTorchRNNModel(ws_use, hidden_use, hidden_use // 2, ws_use)

                        if model_name == "GRU": 
                            pytorch_model = PyTorchGRUModel(ws_use, hidden_use, hidden_use // 2, ws_use)

                        if model_name == "LSTM": 
                            pytorch_model = PyTorchLSTMModel(ws_use, hidden_use, hidden_use // 2, ws_use)
                        
                    if mod_use == "Linear":

                        if model_name == "RNN":
                            pytorch_model = PyTorchRNNModelLinear(ws_use, hidden_use, hidden_use // 2, ws_use)

                        if model_name == "GRU": 
                            pytorch_model = PyTorchGRUModelLinear(ws_use, hidden_use, hidden_use // 2, ws_use)

                        if model_name == "LSTM": 
                            pytorch_model = PyTorchLSTMModelLinear(ws_use, hidden_use, hidden_use // 2, ws_use)

                    if mod_use == "Third":

                        if model_name == "RNN":
                            pytorch_model = PyTorchRNNModelThird(ws_use, hidden_use, hidden_use // 2, hidden_use // 4, ws_use)

                        if model_name == "GRU": 
                            pytorch_model = PyTorchGRUModelThird(ws_use, hidden_use, hidden_use // 2, hidden_use // 4, ws_use)

                        if model_name == "LSTM": 
                            pytorch_model = PyTorchLSTMModelThird(ws_use, hidden_use, hidden_use // 2, hidden_use // 4, ws_use)

                    if mod_use == "Twice":

                        if model_name == "RNN":
                            pytorch_model = PyTorchRNNModelTwice(ws_use, hidden_use, hidden_use // 2, ws_use)

                        if model_name == "GRU": 
                            pytorch_model = PyTorchGRUModelTwice(ws_use, hidden_use, hidden_use // 2, ws_use)

                        if model_name == "LSTM": 
                            pytorch_model = PyTorchLSTMModelTwice(ws_use, hidden_use, hidden_use // 2, ws_use)

                    if not os.path.isdir("train_pytorch/" + mod_use + "/" + varname + "/models/" + model_name):
                        os.makedirs("train_pytorch/" + mod_use + "/" + varname + "/models/" + model_name)
                
                    if not os.path.isdir("train_pytorch/" + mod_use + "/" + varname + "/predictions/train/" + model_name):
                        os.makedirs("train_pytorch/" + mod_use + "/" + varname + "/predictions/train/" + model_name)
                    if not os.path.isdir("train_pytorch/" + mod_use + "/" + varname + "/predictions/val/" + model_name):
                        os.makedirs("train_pytorch/" + mod_use + "/" + varname + "/predictions/val/" + model_name)
                    if not os.path.isdir("train_pytorch/" + mod_use + "/" + varname + "/predictions/test/" + model_name):
                        os.makedirs("train_pytorch/" + mod_use + "/" + varname + "/predictions/test/" + model_name)

                    device = torch.device("cuda")
                    pytorch_model.to(device)
                        
                    train_dataset = TensorDataset(torch.tensor(x_train_all).float().to(device),  torch.tensor(y_train_all).float().to(device))
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

                    val_dataset = TensorDataset(torch.tensor(x_val_all).float().to(device),  torch.tensor(y_val_all).float().to(device))
                    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(pytorch_model.parameters())

                    # Training loop
                    num_epochs = 5
                    best_val_loss = float('inf')

                    vallosspath = "train_pytorch/" + mod_use + "/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_val_loss"
                    vallosses = []
                    trainlosspath = "train_pytorch/" + mod_use + "/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_train_loss"
                    trainlosses = []
                    best_model_path = "train_pytorch/" + mod_use + "/" + varname + "/models/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + ".pth"
                    useeval = True
                    for epoch in range(num_epochs):
                        print(f"Epoch {epoch+1}/{num_epochs}")
                        epoch_loss = 0
                        pytorch_model.train()
                        for inputs, targets in train_loader:
                            inputs, targets = inputs.to(device), targets.to(device)
                            optimizer.zero_grad()
                            outputs = pytorch_model(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()
                        if useeval:
                            # Calculate validation loss
                            val_loss = 0
                            pytorch_model.eval()
                            with torch.no_grad():
                                for inputs, targets in val_loader:
                                    inputs, targets = inputs.to(device), targets.to(device)
                                    outputs = pytorch_model(inputs)
                                    loss = criterion(outputs, targets)
                                    val_loss += loss.item()
                            
                            val_loss /= len(val_loader)
                            print(f"Training Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")
                            vallosses.append(val_loss)
                            trainlosses.append(epoch_loss / len(train_loader))

                            # Save the model if the validation loss is the best we've seen so far
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                torch.save(pytorch_model.state_dict(), best_model_path)
                                print(f"Best model saved with validation loss: {val_loss:.4f}")
                        else:
                            print(f"Training Loss: {epoch_loss / len(train_loader):.4f}")
                            trainlosses.append(epoch_loss / len(train_loader))

                    if not useeval:
                        torch.save(pytorch_model.state_dict(), best_model_path)

                    save_object(trainlosspath, trainlosses)
                    save_object(vallosspath, vallosses)

                    if useeval:
                        pytorch_model.load_state_dict(torch.load(best_model_path, map_location=device))
            
                    pytorch_model.eval()

                    with torch.no_grad():

                        predict_train_all = pytorch_model(torch.tensor(x_train_all_short).float().to(device))
                        predict_val_all = pytorch_model(torch.tensor(x_val_all_short).float().to(device))
                        predict_test_all = pytorch_model(torch.tensor(x_test_all_short).float().to(device))
                        
                        print_predictions(y_train_all_short, predict_train_all, "train_pytorch/" + mod_use + "/" + varname + "/predictions/train/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_train.csv")
                        print_predictions(y_val_all_short, predict_val_all, "train_pytorch/" + mod_use + "/" + varname + "/predictions/val/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_val.csv")
                        print_predictions(y_test_all_short, predict_test_all, "train_pytorch/" + mod_use + "/" + varname + "/predictions/test/" + model_name + "/" + varname + "_" + model_name + "_ws_" + str(ws_use) + "_hidden_" + str(hidden_use) + "_test.csv")
