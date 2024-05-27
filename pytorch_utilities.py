import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def fix_file_predictions(name_file):
    with open(name_file,'r') as f:
        lines = f.readlines()
        strpr = {"actual": [], "predicted": []}
        cum_c = 0
        buffer = ''
        for line in lines:
            buffer += line # Append the current line to a buffer
            cum_c = buffer.count(';')
            if cum_c == 1:
                if "actual" not in buffer:
                    strpr["actual"].append(buffer.split(";")[0].replace("\n", "").replace('"', ""))
                    strpr["predicted"].append(buffer.split(";")[1].replace("\n", "").replace('"', ""))
                buffer = ''
            elif cum_c > 1:
                raise # This should never happen
        df_new = pd.DataFrame(strpr)
        df_new.to_csv(name_file, index = False, sep = ";")

def print_predictions(actual, predicted, name_file):
    
    strpr = {"actual": [], "predicted": []}
    for ix1 in range(len(actual)):
        for ix2 in range(len(actual[ix1])):
            strpr["actual"].append(str(actual[ix1][ix2]).replace("[", "").replace("]", "").replace("tensor(", "").replace(")", ""))
            strpr["predicted"].append(str(predicted[ix1][ix2]).replace("[", "").replace("]", "").replace("tensor(", "").replace(")", ""))
    df_new = pd.DataFrame(strpr)
    df_new.to_csv(name_file, index = False, sep = ";") 

def get_XY(dat, time_steps, len_skip = -1, len_output = -1):
    X = []
    Y = [] 
    if len_skip == -1:
        len_skip = time_steps
    if len_output == -1:
        len_output = time_steps
    for i in range(0, len(dat), len_skip):
        x_vals = dat[i:min(i + time_steps, len(dat))]
        y_vals = dat[i + time_steps:i + time_steps + len_output]
        if len(x_vals) == time_steps and len(y_vals) == len_output:
            X.append(np.array(x_vals))
            Y.append(np.array(y_vals))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

class PyTorchGRUModel(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, dense_units):
        super(PyTorchGRUModel, self).__init__()

        # GRU layer
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_units1, batch_first=True)

        # First Dense layer
        self.fc1 = nn.Linear(hidden_units1, hidden_units2)
        
        # Tanh activation for first Dense layer
        self.tanh1 = nn.Tanh()
        
        # Second Dense layer
        self.fc2 = nn.Linear(hidden_units2, dense_units)
        
        # Tanh activation for second Dense layer
        self.tanh2 = nn.Tanh()

    def forward(self, x):
        # GRU layer
        gru_output, _ = self.gru(x)

        # Extract the last hidden state from the GRU output
        last_hidden_state = gru_output[:, :]

        # First Dense layer
        dense_output1 = self.fc1(last_hidden_state)
        
        # Apply tanh activation for first Dense layer
        tanh_output1 = self.tanh1(dense_output1)

        # Second Dense layer
        dense_output2 = self.fc2(tanh_output1)
        
        # Apply tanh activation for second Dense layer
        output = self.tanh2(dense_output2)

        return output
    
class PyTorchLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, dense_units):
        super(PyTorchLSTMModel, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_units1, batch_first=True)

        # First Dense layer
        self.fc1 = nn.Linear(hidden_units1, hidden_units2)
        
        # Tanh activation for first Dense layer
        self.tanh1 = nn.Tanh()
        
        # Second Dense layer
        self.fc2 = nn.Linear(hidden_units2, dense_units)
        
        # Tanh activation for second Dense layer
        self.tanh2 = nn.Tanh()

    def forward(self, x):
        # LSTM layer
        lstm_output, _ = self.lstm(x)

        # Extract the last hidden state from the LSTM output
        last_hidden_state = lstm_output[:, :]

        # First Dense layer
        dense_output1 = self.fc1(last_hidden_state)
        
        # Apply tanh activation for first Dense layer
        tanh_output1 = self.tanh1(dense_output1)

        # Second Dense layer
        dense_output2 = self.fc2(tanh_output1)
        
        # Apply tanh activation for second Dense layer
        output = self.tanh2(dense_output2)

        return output
    
class PyTorchRNNModel(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, dense_units):
        super(PyTorchRNNModel, self).__init__()

        # SimpleRNN layer
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_units1, batch_first=True)

        # First Dense layer
        self.fc1 = nn.Linear(hidden_units1, hidden_units2)
        
        # Tanh activation for first Dense layer
        self.tanh1 = nn.Tanh()
        
        # Second Dense layer
        self.fc2 = nn.Linear(hidden_units2, dense_units)
        
        # Tanh activation for second Dense layer
        self.tanh2 = nn.Tanh()

    def forward(self, x):
        # RNN layer
        rnn_output, _ = self.rnn(x)

        # Extract the last hidden state from the RNN output
        last_hidden_state = rnn_output[:, :]

        # First Dense layer
        dense_output1 = self.fc1(last_hidden_state)
        
        # Apply tanh activation for first Dense layer
        tanh_output1 = self.tanh1(dense_output1)

        # Second Dense layer
        dense_output2 = self.fc2(tanh_output1)
        
        # Apply tanh activation for second Dense layer
        output = self.tanh2(dense_output2)

        return output

class PyTorchGRUModelLinear(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, dense_units):
        super(PyTorchGRUModelLinear, self).__init__()

        # GRU layer
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_units1, batch_first=True)

        # First Dense layer
        self.fc1 = nn.Linear(hidden_units1, hidden_units2)
        
        # Tanh activation for first Dense layer
        self.tanh1 = nn.Tanh()
        
        # Second Dense layer
        self.fc2 = nn.Linear(hidden_units2, dense_units)

    def forward(self, x):
        # GRU layer
        gru_output, _ = self.gru(x)

        # Extract the last hidden state from the GRU output
        last_hidden_state = gru_output[:, :]

        # First Dense layer
        dense_output1 = self.fc1(last_hidden_state)
        
        # Apply tanh activation for first Dense layer
        tanh_output1 = self.tanh1(dense_output1)

        # Second Dense layer
        output = self.fc2(tanh_output1)

        return output
    
class PyTorchLSTMModelLinear(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, dense_units):
        super(PyTorchLSTMModelLinear, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_units1, batch_first=True)

        # First Dense layer
        self.fc1 = nn.Linear(hidden_units1, hidden_units2)
        
        # Tanh activation for first Dense layer
        self.tanh1 = nn.Tanh()
        
        # Second Dense layer
        self.fc2 = nn.Linear(hidden_units2, dense_units)

    def forward(self, x):
        # LSTM layer
        lstm_output, _ = self.lstm(x)

        # Extract the last hidden state from the LSTM output
        last_hidden_state = lstm_output[:, :]

        # First Dense layer
        dense_output1 = self.fc1(last_hidden_state)
        
        # Apply tanh activation for first Dense layer
        tanh_output1 = self.tanh1(dense_output1)

        # Second Dense layer
        output = self.fc2(tanh_output1)

        return output
    
class PyTorchRNNModelLinear(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, dense_units):
        super(PyTorchRNNModelLinear, self).__init__()

        # SimpleRNN layer
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_units1, batch_first=True)

        # First Dense layer
        self.fc1 = nn.Linear(hidden_units1, hidden_units2)
        
        # Tanh activation for first Dense layer
        self.tanh1 = nn.Tanh()
        
        # Second Dense layer
        self.fc2 = nn.Linear(hidden_units2, dense_units)

    def forward(self, x):
        # RNN layer
        rnn_output, _ = self.rnn(x)

        # Extract the last hidden state from the RNN output
        last_hidden_state = rnn_output[:, :]

        # First Dense layer
        dense_output1 = self.fc1(last_hidden_state)
        
        # Apply tanh activation for first Dense layer
        tanh_output1 = self.tanh1(dense_output1)

        # Second Dense layer
        output = self.fc2(tanh_output1)

        return output

class PyTorchGRUModelThird(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, hidden_units3, dense_units):
        super(PyTorchGRUModelThird, self).__init__()

        # GRU layer
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_units1, batch_first=True)

        # First Dense layer
        self.fc1 = nn.Linear(hidden_units1, hidden_units2)
        
        # Tanh activation for first Dense layer
        self.tanh1 = nn.Tanh()
        
        # Second Dense layer
        self.fc2 = nn.Linear(hidden_units2, hidden_units3)
        
        # Tanh activation for second Dense layer
        self.tanh2 = nn.Tanh()
        
        # Third Dense layer
        self.fc3 = nn.Linear(hidden_units3, dense_units)
        
        # Tanh activation for third Dense layer
        self.tanh3 = nn.Tanh()

    def forward(self, x):
        # GRU layer
        gru_output, _ = self.gru(x)

        # Extract the last hidden state from the GRU output
        last_hidden_state = gru_output[:, :]

        # First Dense layer
        dense_output1 = self.fc1(last_hidden_state)
        
        # Apply tanh activation for first Dense layer
        tanh_output1 = self.tanh1(dense_output1)

        # Second Dense layer
        dense_output2 = self.fc2(tanh_output1)
        
        # Apply tanh activation for second Dense layer
        tanh_output2 = self.tanh2(dense_output2)
        
        # Third Dense layer
        dense_output3 = self.fc3(tanh_output2)
        
        # Apply tanh activation for third Dense layer
        output = self.tanh3(dense_output3)

        return output
    
class PyTorchLSTMModelThird(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, hidden_units3, dense_units):
        super(PyTorchLSTMModelThird, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_units1, batch_first=True)

        # First Dense layer
        self.fc1 = nn.Linear(hidden_units1, hidden_units2)
        
        # Tanh activation for first Dense layer
        self.tanh1 = nn.Tanh()
        
        # Second Dense layer
        self.fc2 = nn.Linear(hidden_units2, hidden_units3)
        
        # Tanh activation for second Dense layer
        self.tanh2 = nn.Tanh()
        
        # Third Dense layer
        self.fc3 = nn.Linear(hidden_units3, dense_units)
        
        # Tanh activation for third Dense layer
        self.tanh3 = nn.Tanh()

    def forward(self, x):
        # LSTM layer
        lstm_output, _ = self.lstm(x)

        # Extract the last hidden state from the LSTM output
        last_hidden_state = lstm_output[:, :]

        # First Dense layer
        dense_output1 = self.fc1(last_hidden_state)
        
        # Apply tanh activation for first Dense layer
        tanh_output1 = self.tanh1(dense_output1)

        # Second Dense layer
        dense_output2 = self.fc2(tanh_output1)
        
        # Apply tanh activation for second Dense layer
        tanh_output2 = self.tanh2(dense_output2)
        
        # Third Dense layer
        dense_output3 = self.fc3(tanh_output2)
        
        # Apply tanh activation for third Dense layer
        output = self.tanh3(dense_output3)

        return output
    
class PyTorchRNNModelThird(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, hidden_units3, dense_units):
        super(PyTorchRNNModelThird, self).__init__()

        # SimpleRNN layer
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_units1, batch_first=True)

        # First Dense layer
        self.fc1 = nn.Linear(hidden_units1, hidden_units2)
        
        # Tanh activation for first Dense layer
        self.tanh1 = nn.Tanh()
        
        # Second Dense layer
        self.fc2 = nn.Linear(hidden_units2, hidden_units3)
        
        # Tanh activation for second Dense layer
        self.tanh2 = nn.Tanh()
        
        # Third Dense layer
        self.fc3 = nn.Linear(hidden_units3, dense_units)
        
        # Tanh activation for third Dense layer
        self.tanh3 = nn.Tanh()

    def forward(self, x):
        # RNN layer
        rnn_output, _ = self.rnn(x)

        # Extract the last hidden state from the RNN output
        last_hidden_state = rnn_output[:, :]

        # First Dense layer
        dense_output1 = self.fc1(last_hidden_state)
        
        # Apply tanh activation for first Dense layer
        tanh_output1 = self.tanh1(dense_output1)

        # Second Dense layer
        dense_output2 = self.fc2(tanh_output1)
        
        # Apply tanh activation for second Dense layer
        tanh_output2 = self.tanh2(dense_output2)
        
        # Third Dense layer
        dense_output3 = self.fc3(tanh_output2)
        
        # Apply tanh activation for third Dense layer
        output = self.tanh3(dense_output3)

        return output
    
class PyTorchGRUModelTwice(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, dense_units):
        super(PyTorchGRUModelTwice, self).__init__()

        # First GRU layer
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_units1, batch_first=True)

        # Second GRU layer
        self.gru2 = nn.GRU(input_size=hidden_units1, hidden_size=hidden_units1, batch_first=True)

        # First Dense layer
        self.fc1 = nn.Linear(hidden_units1, hidden_units2)
        
        # Tanh activation for first Dense layer
        self.tanh1 = nn.Tanh()
        
        # Second Dense layer
        self.fc2 = nn.Linear(hidden_units2, dense_units)
        
        # Tanh activation for second Dense layer
        self.tanh2 = nn.Tanh()

    def forward(self, x):
        # GRU layer
        gru_output1, _ = self.gru1(x) 
        
        # Second GRU layer
        gru_output2, _ = self.gru2(gru_output1)

        # Extract the last hidden state from the second GRU output
        last_hidden_state = gru_output2[:, :]

        # First Dense layer
        dense_output1 = self.fc1(last_hidden_state)
        
        # Apply tanh activation for first Dense layer
        tanh_output1 = self.tanh1(dense_output1)

        # Second Dense layer
        dense_output2 = self.fc2(tanh_output1)
        
        # Apply tanh activation for second Dense layer
        output = self.tanh2(dense_output2)

        return output
    
class PyTorchLSTMModelTwice(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, dense_units):
        super(PyTorchLSTMModelTwice, self).__init__()
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_units1, batch_first=True)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(input_size=hidden_units1, hidden_size=hidden_units1, batch_first=True)

        # First Dense layer
        self.fc1 = nn.Linear(hidden_units1, hidden_units2)
        
        # Tanh activation for first Dense layer
        self.tanh1 = nn.Tanh()
        
        # Second Dense layer
        self.fc2 = nn.Linear(hidden_units2, dense_units)
        
        # Tanh activation for second Dense layer
        self.tanh2 = nn.Tanh()

    def forward(self, x):
        # LSTM layer
        lstm_output1, _ = self.lstm1(x) 
        
        # Second LSTM layer
        lstm_output2, _ = self.lstm2(lstm_output1)

        # Extract the last hidden state from the second LSTM output
        last_hidden_state = lstm_output2[:, :]

        # First Dense layer
        dense_output1 = self.fc1(last_hidden_state)
        
        # Apply tanh activation for first Dense layer
        tanh_output1 = self.tanh1(dense_output1)

        # Second Dense layer
        dense_output2 = self.fc2(tanh_output1)
        
        # Apply tanh activation for second Dense layer
        output = self.tanh2(dense_output2)

        return output
    
class PyTorchRNNModelTwice(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, dense_units):
        super(PyTorchRNNModelTwice, self).__init__()

        # First RNN layer
        self.rnn1 = nn.RNN(input_size=input_size, hidden_size=hidden_units1, batch_first=True)

        # Second RNN layer
        self.rnn2 = nn.RNN(input_size=hidden_units1, hidden_size=hidden_units1, batch_first=True)

        # First Dense layer
        self.fc1 = nn.Linear(hidden_units1, hidden_units2)
        
        # Tanh activation for first Dense layer
        self.tanh1 = nn.Tanh()
        
        # Second Dense layer
        self.fc2 = nn.Linear(hidden_units2, dense_units)
        
        # Tanh activation for second Dense layer
        self.tanh2 = nn.Tanh()

    def forward(self, x):
        # RNN layer
        rnn_output1, _ = self.rnn1(x) 
        
        # Second RNN layer
        rnn_output2, _ = self.rnn2(rnn_output1)

        # Extract the last hidden state from the second RNN output
        last_hidden_state = rnn_output2[:, :]

        # First Dense layer
        dense_output1 = self.fc1(last_hidden_state)
        
        # Apply tanh activation for first Dense layer
        tanh_output1 = self.tanh1(dense_output1)

        # Second Dense layer
        dense_output2 = self.fc2(tanh_output1)
        
        # Apply tanh activation for second Dense layer
        output = self.tanh2(dense_output2)

        return output
