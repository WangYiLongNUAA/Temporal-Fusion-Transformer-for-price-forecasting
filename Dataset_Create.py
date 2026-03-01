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
import joblib

def load_and_process_data(path):
    data = pd.read_csv(path)
    classic_feature_columns = ['航空公司', '航班执行属性','出发机场','到达机场', '航班出发日期属性', '数据收集日期属性']
    continuous_feature_column = '距离航班出发日期天数'
    time_divide_column = '航班出发日期'
    index_column = '时间序列索引'
    sample_divide_column = '序列ID'
    target_column = '机票价格'

    all_columns = classic_feature_columns + [time_divide_column, index_column, sample_divide_column, target_column,continuous_feature_column]
    data = data[all_columns]

    """
    分类变量独热编码
    """
    encoder = OneHotEncoder(drop='first',sparse_output=False)
    encoded_data = encoder.fit_transform(data[classic_feature_columns])
    encoded_data_ = pd.DataFrame(encoded_data,columns = encoder.get_feature_names_out(classic_feature_columns))

    joblib.dump(encoder, f'onehot_encoder.pkl')
    joblib.dump(classic_feature_columns, f'feature_columns.pkl')

    data_dropf = data.drop(classic_feature_columns,axis=1)
    data_processed = pd.concat([data_dropf,encoded_data_],axis=1)

    return data_processed

def split_data(data,train_cutoff,val_cutoff):
    """
    划分数据集
    train_cutoff: 训练集与验证集时间划分节点
    val_cutoff：验证集与测试集时间划分节点
    """
    time_divide_column = '航班出发日期'
    data[time_divide_column] = pd.to_datetime(data[time_divide_column])

    train_data = data[data[time_divide_column] < train_cutoff]
    val_data = data[(data[time_divide_column] >= train_cutoff) & (data[time_divide_column] <= val_cutoff)]
    test_data = data[data[time_divide_column] > val_cutoff]

    return train_data,val_data,test_data

def create_sequence(data,past_horizon,future_horizon):
    """
    数据集构建
    past_horizon: 输入时间步长
    future_horizon: 预测步长
    """
    time_divide_column = '航班出发日期'
    data = data.drop(time_divide_column,axis = 1)

    index_column = '时间序列索引'
    sample_divide_column = '序列ID'
    target_column = '机票价格'
    drop_columns = [index_column, sample_divide_column]

    feature_sequence = []
    target_sequence = []

    sample_divide = data[sample_divide_column].unique()
    for each_sample in sample_divide:
        each_sample_data = data[data[sample_divide_column] == each_sample].sort_values(by = index_column)
        each_sample_data = each_sample_data.drop(drop_columns,axis = 1)

        if len(each_sample_data) < past_horizon+future_horizon:
            continue

        for i in range(past_horizon,len(each_sample_data)-future_horizon):
            past_feature = each_sample_data[i-past_horizon:i].values
            future_targets = each_sample_data[target_column].iloc[i:i+future_horizon].values

            feature_sequence.append(past_feature)
            target_sequence.append(future_targets)
    return np.array(feature_sequence),np.array(target_sequence)

def create_dataloaders(x_train,y_train,x_val,y_val,x_test,y_test,batch_size=128):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_val_scaled = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)
    x_test_scaled = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

    joblib.dump(scaler, f'feature_scaler.pkl')

    # 替换原始数据集
    x_train = x_train_scaled
    x_val = x_val_scaled
    x_test = x_test_scaled

    x_train = torch.tensor(x_train,dtype = torch.float)
    y_train = torch.tensor(y_train,dtype = torch.float)

    x_val = torch.tensor(x_val,dtype = torch.float)
    y_val = torch.tensor(y_val,dtype = torch.float)

    x_test = torch.tensor(x_test,dtype = torch.float)
    y_test = torch.tensor(y_test,dtype = torch.float)

    train_dataset = TensorDataset(x_train,y_train)
    val_dataset = TensorDataset(x_val,y_val)
    test_dataset = TensorDataset(x_test,y_test)

    train_loader = DataLoader(train_dataset,batch_size,shuffle=False)
    val_loader = DataLoader(val_dataset,batch_size,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size,shuffle=False)
    # print(x_train.shape[2])
    return train_loader,val_loader,test_loader

def main():

    data = load_and_process_data('北京-上海训练数据集.csv')
    train_cutoff = pd.Timestamp('2024-03-03 00:00:00')
    val_cutoff = pd.Timestamp('2024-03-11 00:00:00')
    train_data,val_data,test_data = split_data(data,train_cutoff,val_cutoff)

    past_horizon = 7
    future_horizon = 7
    x_train,y_train = create_sequence(train_data,past_horizon,future_horizon)
    x_val,y_val = create_sequence(val_data,past_horizon,future_horizon)
    x_test,y_test = create_sequence(test_data,past_horizon,future_horizon)

    train_loader,val_loader,test_loader = create_dataloaders(x_train,y_train,x_val,y_val,x_test,y_test)

    return train_loader,val_loader,test_loader

if __name__ == '__main__':
    train_loader,val_loader,test_loader = main()