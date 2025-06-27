import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from datetime import timedelta
from sklearn import metrics

# --- 重新定义模型结构 ---
#  TextCNNModel 类定义:
class TextCNNModel(nn.Module):
    def __init__(self, embedding_file='CNNewsClassification/wordVectorembed/embedding_SougouNews.npz'):
        super(TextCNNModel, self).__init__()
        # 加载词向量文件
        embedding_pretrained = torch.tensor(
            np.load(embedding_file)["embeddings"].astype('float32'))
        # 定义词嵌入层
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        # 定义三个卷积
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, 300)) for k in [2, 3, 4]])
        # 定义dropout层
        self.dropout = nn.Dropout(0.5)
        # 定义全连接层
        self.fc = nn.Linear(256 * 3, 10)

    def conv_and_pool(self, x, conv):  # 定义卷积+激活函数+池化层构成的一个操作块
        x = conv(x)
        x = F.relu(x).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):  # 前向传播
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# 载入词典和分词函数
import pickle as pkl
vocab_file = "CNNewsClassification/wordVectorembed/vocab.pkl"
word_to_id = pkl.load( open(vocab_file, 'rb'))

def tokenize_textCNN(s):
    max_size=32
    ts = [w for i, w in enumerate(s) if i < max_size]
    ids = [word_to_id[w] if w in word_to_id.keys() else word_to_id['[UNK]'] for w in ts]
    ids += [0 for _ in range(max_size-len(ts))]
    return ids

# 数据集类 (与训练时使用的 MyData 类一致)
from torch.utils.data import Dataset, DataLoader
labels = ['体育','娱乐','家居','教育','时政','游戏','社会','科技','股票','财经']
LABEL2ID = { x:i for (x,i) in zip(labels,range(len(labels)))}

class MyData(Dataset):
    def __init__(self, tokenize_fun, filename):
        self.filename = filename
        self.tokenize_function = tokenize_fun
        print("Loading dataset "+ self.filename +" ...")
        self.data, self.labels = self.load_data()
    def load_data(self):
        labels = []
        data = []
        with open(self.filename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                fields  = line.strip().split('\t')
                if len(fields)!=2 :
                    continue
                labels.append(LABEL2ID[fields[0]])
                data.append(self.tokenize_function(fields[1]))
        f.close()
        return torch.tensor(data), torch.tensor(labels)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# 获取DataLoader的函数
def getDataLoader(test_dataset, batch_size=128): # 可以调整batch_size
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False, # 测试时不需要打乱
    )
    return test_dataloader

# 评估函数
def evaluate_predictions(model, device, dataload, postProc=None):
    model.eval()  
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    loss_total = 0.0
    with torch.no_grad():
        for texts, labels in dataload:
            outputs = model(texts.to(device))
            if postProc:
                outputs = postProc(outputs)
            loss = F.cross_entropy(outputs, labels.to(device))
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
    acc = metrics.accuracy_score(labels_all, predict_all)
    avg_loss = loss_total / len(dataload)
    return acc, avg_loss, (labels_all, predict_all)


if __name__ == '__main__':
    # --- 配置 ---
    MODEL_PATH = 'CNNewsClassification/checkpoint/textcnn_model.0.53' 
    TEST_DATA_FILE = 'CNNewsClassification/datasets/test.txt' 
    EMBEDDING_FILE = 'CNNewsClassification/wordVectorembed/embedding_SougouNews.npz'
    VOCAB_FILE = "CNNewsClassification/wordVectorembed/vocab.pkl" 
    BATCH_SIZE = 128 

    # --- 初始化 ---
    # 确保随机种子设置一致，以便加载的模型能够正确工作（尽管加载模型主要是加载参数）
    # 这里的 setSeed 主要为了保证后续可能用到的数据加载或处理是可复现的
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 加载模型 ---
    try:
        # 加载模型
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}")

        print(f"正在加载模型: {MODEL_PATH}...")
        model = torch.load(MODEL_PATH, map_location=device,weights_only=False) # map_location 确保模型加载到正确的设备
        print("模型加载成功！")

        model.eval()

    except FileNotFoundError as e:
        print(e)
        print("请确保 MODEL_PATH 指向的是一个实际存在且正确保存的模型文件。")
        exit() # 如果模型文件不存在，则退出程序
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        print("请检查模型结构是否与保存时一致，以及模型文件是否损坏。")
        exit()

    # --- 加载测试数据 ---
    print(f"正在加载测试数据: {TEST_DATA_FILE}...")
    try:
        # 重新加载词典和定义分词器
        word_to_id = pkl.load(open(VOCAB_FILE, 'rb'))
        test_dataset = MyData(tokenize_fun=tokenize_textCNN, filename=TEST_DATA_FILE)
        test_dataloader = getDataLoader(test_dataset, batch_size=BATCH_SIZE)
        print(f"测试数据加载完成，共 {len(test_dataset)} 条样本。")
    except FileNotFoundError as e:
        print(e)
        print("请检查词典文件和测试数据文件的路径是否正确。")
        exit()
    except Exception as e:
        print(f"加载测试数据时发生错误: {e}")
        exit()

    # --- 进行测试和评估 ---
    print("\n开始在测试集上进行评估...")
    try:
        test_acc, test_loss, (labels_all, predict_all) = evaluate_predictions(
            model, device, test_dataloader
        )

        print("\n--- 测试结果 ---")
        print(f"模型: {MODEL_PATH.split('/')[-1]}") # 打印模型文件名
        print(f"测试集准确率 (Accuracy): {test_acc:.4f}")
        print(f"测试集平均损失 (Avg Loss): {test_loss:.4f}")


        from sklearn import metrics
        print("\n混淆矩阵 (Confusion Matrix):")
        print(metrics.confusion_matrix(labels_all, predict_all))
        print("\n分类报告 (Classification Report):")
        print(metrics.classification_report(labels_all, predict_all, target_names=labels)) # 使用之前定义的类别名称

    except Exception as e:
        print(f"在测试或评估时发生错误: {e}")