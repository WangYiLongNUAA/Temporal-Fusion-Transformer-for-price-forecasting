# 南京航空航天大学
# 王夷龙
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Dataset_Create import *

class BiLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers: int=1,dropout: float=0.1) -> None:
        super(BiLSTM,self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2,output_size)

    def forward(self,x):
        out,_ = self.lstm(x)
        out = self.fc(out[:,-1,:])
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    data = load_and_process_data('北京-上海训练数据集.csv')
    train_cutoff = pd.Timestamp('2024-03-03 00:00:00')
    val_cutoff = pd.Timestamp('2024-03-11 00:00:00')
    train_data, val_data, test_data = split_data(data, train_cutoff, val_cutoff)

    past_horizon = 7
    future_horizon = 7

    x_train, y_train = create_sequence(train_data, past_horizon, future_horizon)
    x_val, y_val = create_sequence(val_data, past_horizon, future_horizon)
    x_test, y_test = create_sequence(test_data, past_horizon, future_horizon)

    train_loader, val_loader, test_loader = create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test)

    input_size = x_train.shape[2]
    output_size = future_horizon

    """
    可调整的模型参数
    """
    hidden_size = 16
    num_layers = 2

    """
    模型替换 LSTM,BiLSTM,GRU
    """
    model = BiLSTM(input_size, hidden_size, output_size, num_layers)
    model_store_path = 'best_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    """
    损失函数和优化器
    """
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    """
    模型训练
    """
    num_epochs = 300
    early_stopping_patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(f'epoch{epoch + 1}/{num_epochs}--训练集损失:{train_loss:.4f},验证集损失:{val_loss:.4f}')

        """
        早停机制
        """
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_store_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print('Early stopping')
            break

    """
    最佳模型评估
    """
    model.load_state_dict(torch.load(model_store_path))
    model.eval()

    test_losses = []
    predictions = []
    true_values = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_losses.append(loss.item())
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(y_batch.cpu().numpy())

    test_loss = np.mean(test_losses)
    print(f'测试集损失: {test_loss:.4f}')

    # 计算MAE, MAPE, RMSE
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    def evaluate_metrics(y_true, y_pred, steps):
        mae = []
        mape = []
        rmse = []
        for step in steps:
            mae.append(mean_absolute_error(y_true[:, :step], y_pred[:, :step]))
            mape.append(np.mean(np.abs((y_true[:, :step] - y_pred[:, :step]) / y_true[:, :step])) * 100)
            rmse.append(np.sqrt(mean_squared_error(y_true[:, :step], y_pred[:, :step])))
        return mae, mape, rmse

    """
    对前1、3、5、7步的预测性能进行评估
    """
    steps = [1, 3, 5, 7]
    mae, mape, rmse = evaluate_metrics(true_values, predictions, steps)

    for i, step in enumerate(steps):
        print(f'step {step} - MAE: {mae[i]:.4f}, MAPE: {mape[i]:.2f}%, RMSE: {rmse[i]:.4f}')