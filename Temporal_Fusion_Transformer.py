# 南京航空航天大学
# 王夷龙
from pytorch_forecasting.models import TemporalFusionTransformer
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder
from pytorch_forecasting.data import GroupNormalizer
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor)
from pytorch_forecasting.metrics import QuantileLoss
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data_path = "北京-上海训练数据集.csv"
data = pd.read_csv(data_path)
data['航班出发日期'] = pd.to_datetime(data['航班出发日期'], errors='coerce')
data['机票价格'] = data['机票价格'].astype(float)

"""
数据集划分 8-1-1
"""
train_cutoff = pd.Timestamp('2024-03-03 00:00:00')
val_cutoff = pd.Timestamp('2024-03-11 00:00:00')

train_data = data[(data['航班出发日期'] < train_cutoff)]
validation_data = data[(data['航班出发日期'] >= train_cutoff) & (data['航班出发日期'] <= val_cutoff)]
test_data = data[data['航班出发日期'] > val_cutoff]

# print(f"训练集数量: {len(train_data)}")
# print(f"验证集数量: {len(validation_data)}")
# print(f"测试集数量: {len(test_data)}")

"""
变量设置
"""
max_encoder_length = 7
max_prediction_length = 7

time_index = '时间序列索引'
target = '机票价格'
group_ids = ['序列ID']

"""
静态变量
未来已知输入
历史观测输入
(全部区分连续变量/分类变量)
"""
static_categoricals = ['航空公司', '航班执行属性', '出发机场', '到达机场', '航班出发日期属性']
time_varying_known_categoricals = ['数据收集日期属性']
time_varying_known_reals = ['距离航班出发日期天数']
time_varying_unknown_reals = ['机票价格']

batch_size = 128

"""
训练集、验证集、测试集构建
"""
training = TimeSeriesDataSet(
    data=train_data,
    time_idx=time_index,
    target=target,
    group_ids=group_ids,
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=static_categoricals,
    time_varying_known_categoricals=time_varying_known_categoricals,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    target_normalizer=GroupNormalizer(groups=['序列ID'], transformation='softplus'),
    allow_missing_timesteps=True)

validation = TimeSeriesDataSet(
    data=validation_data,
    time_idx=time_index,
    target=target,
    group_ids=group_ids,
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=static_categoricals,
    time_varying_known_categoricals=time_varying_known_categoricals,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    target_normalizer=GroupNormalizer(groups=['序列ID'], transformation='softplus'),
    allow_missing_timesteps=True)

testing = TimeSeriesDataSet(
    data=test_data,
    time_idx=time_index,
    target=target,
    group_ids=group_ids,
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=static_categoricals,
    time_varying_known_categoricals=time_varying_known_categoricals,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    target_normalizer=GroupNormalizer(groups=['序列ID'], transformation='softplus'),
    allow_missing_timesteps=True)

train_dataloader = training.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size)
test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size)

"""
设置早停机制
"""
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=5,
    verbose=True,
    mode="min"
)

"""
模型训练设置
"""
lr_logger = LearningRateMonitor()  # 记录学习率
trainer = pl.Trainer(
    max_epochs=5,
    accelerator='auto',
    devices='auto',
    callbacks=[lr_logger, early_stop_callback],
    enable_checkpointing=True,  # 最佳模型保存
    default_root_dir="./tft_checkpoints"
)

"""
TFT模型参数设置
"""
tft = TemporalFusionTransformer.from_dataset(
    dataset=training,
    learning_rate=0.01,
    hidden_size=13,
    attention_head_size=1,
    dropout=0.1,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

"""
模型训练、测试
"""
trainer.fit(
            model=tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader)

trainer.test(
            model=tft,
            dataloaders=test_dataloader)

# 预测与解释
predictions = tft.predict(testing, mode="raw", return_x=True)
# print(predictions._fields) # ('output', 'x', 'index', 'decoder_lengths', 'y')

"""
TFT模型预测内容的不同部分
"""
output = predictions.output  # 预测值
# x = predictions.x  # 输入特征
# index = predictions.index  # 索引
# decoder_lengths = predictions.decoder_lengths  # 解码器长度
# y = predictions.y # 实际值

# 特征重要性解释
interpretation = tft.interpret_output(out=predictions.output, reduction="sum")

"""
输入步、变量权重提取
"""
attention = interpretation['attention']
static_importance = interpretation['static_variables']
encoder_importance = interpretation['encoder_variables']
decoder_importance = interpretation['decoder_variables']

print("attention:", attention)
print("Static Variables Importance:", static_importance)
print("Encoder Variables Importance:", encoder_importance)
print("Decoder Variables Importance:", decoder_importance)

"""
权重作图
"""
figs = tft.plot_interpretation(interpretation)
for fig_key in figs:
    fig = figs[fig_key]
    fig.savefig(f'{fig_key}.png')
    fig.show()

"""
最佳模型加载
"""
best_model_path = trainer.checkpoint_callback.best_model_path
# print('最佳模型路径:' + best_model_path)
model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

model.eval()
device = next(model.parameters()).device

"""
模型性能评估
"""
with torch.no_grad():
    actuals_ = torch.cat([y[0].to(device) for x, y in iter(test_dataloader)])
    predict_ = model.predict(test_dataloader, mode="raw")
    predict_median = predict_.prediction[:, :, 3]

"""
MAE、MAPE、RMSE
"""
def calculate_mae(predictions, actuals):
    mae = torch.abs(predictions - actuals).mean().item()
    return mae

def calculate_mape(predictions, actuals):
    mask = actuals != 0 # 避免0
    if mask.sum() == 0:
        return 0.0
    mape = (torch.abs((predictions[mask] - actuals[mask]) / actuals[mask]) * 100).mean().item()
    return mape

def calculate_rmse(predictions, actuals):
    mse = torch.mean((predictions - actuals) ** 2)
    rmse = torch.sqrt(mse).item()
    return rmse

def calculate_metrics(predictions, actuals):
    mae = calculate_mae(predictions, actuals)
    mape = calculate_mape(predictions, actuals)
    rmse = calculate_rmse(predictions, actuals)
    return mae, mape, rmse

"""
对前1、3、5、7步的预测性能进行评估
"""
mae_1, mape_1, rmse_1 = calculate_metrics(predict_median[:, 0], actuals_[:, 0])  # 第一步
mae_3, mape_3, rmse_3 = calculate_metrics(predict_median[:, :3].mean(dim=1), actuals_[:, :3].mean(dim=1))  # 前三步平均
mae_5, mape_5, rmse_5 = calculate_metrics(predict_median[:, :5].mean(dim=1), actuals_[:, :5].mean(dim=1))  # 前五步平均
mae_7, mape_7, rmse_7 = calculate_metrics(predict_median, actuals_)  # 前七步平均

print(f'1 step----- MAE：{mae_1:.2f}, MAPE: {mape_1:.2f}%, RMSE: {rmse_1:.2f}')
print(f'1-3 steps-- MAE：{mae_3:.2f}, MAPE: {mape_3:.2f}%, RMSE: {rmse_3:.2f}')
print(f'1-5 steps-- MAE：{mae_5:.2f}, MAPE: {mape_5:.2f}%, RMSE: {rmse_5:.2f}')
print(f'1-7 steps-- MAE：{mae_7:.2f}, MAPE: {mape_7:.2f}%, RMSE: {rmse_7:.2f}')