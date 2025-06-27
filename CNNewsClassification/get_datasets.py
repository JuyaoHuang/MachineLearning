import os
from itertools import accumulate
from bisect import bisect_left
path = 'D:/Coding/Python/datastes/NLP/THUCNews'  # 设置数据集路径
save_path = 'CNNewsClassification/datasets'  # 设置保存路径
wfilenames = [save_path + f for f in['train.txt', 'test.txt', 'dev.txt']]  # 设置训练集、验证集和测试集文件
num = [20000, 4000, 2000]  # 设置各数据集样本数量
allnum =list(accumulate(num))  # allnum为[20000, 22000, 24000]
wfiles = [open(f,'w', encoding='utf-8') for f in wfilenames]  #打开数据集文件

subpaths = os.listdir(path)  # 得到各个目录，即类别名称
for pathname in subpaths:
    subpath = os.path.join(path, pathname)
    if os.path.isdir(subpath):  # 如果是目录
        files = os.listdir(subpath)  # 得到所有的样本文件名称
        print(pathname, len(files))  # 打印该类别名称及样本数量
        if len(files)<allnum[-1]: continue  # 样本量不足24000不提取该类别样本
        for i in range(allnum[-1]):
            tag = bisect_left(allnum, i) # 确定样本属于哪个数据集
            with open(os.path.join(subpath, files[i]), 'r', encoding='utf-8') as f:
                line = f.readline()  # 只读第一行标题文本
                wfiles[tag].write(pathname+'\t'+line)  # 写入对应的数据集文件
for f in wfiles:  # 关闭所有的数据集文件
    f.close()