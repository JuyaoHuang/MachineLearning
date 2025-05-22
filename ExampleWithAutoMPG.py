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


