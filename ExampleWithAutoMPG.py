import torch
import pandas as pd
# from torch import nn
import torchvision

dataset_path = 'D:/Coding/Wrote_Codes/PY/pythonProject/auto+mpg/auto-mpg.data'
column_names = ['MPG','Cylinders','Cisplacement','Corsepower',
                'Weight','Acceleration','Model Year','Origin']
raw_dataset = pd.read_csv(dataset_path,names=column_names,na_values="?",
                          comment='\t',sep=" ",skipinitialspace=True)
dataset = raw_dataset.copy()
dataset = dataset.dropna()

# Origin 原产地实际上是一个分类特征，取值1到3，分别代表USA、Europe、Japan
# 该特征可直接使用，也可以转换成 one-hot 编码，变为三个特征
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0

print(dataset)

# 为了评估模型，创建训练集和测试集
# frac用于从原始数据集中抽取百分之多少的数据,random_state用于确保运行代码时抽样的结果是相同的（即随机性是可重复的）
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
# 计算各特征的统计值
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

# 分别将训练集和测试集的训练特征和标签分离，标签是模型需要预测的值
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# 归一化处理 之后使用此归一化的数据来训练模型
def norm(x):
    return (x-train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


