from sklearn.preprocessing import StandardScaler as ss
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
import sklearn.naive_bayes as nb
import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.neural_network as nn
from sklearn.metrics import mean_squared_error as mse

# 初始化数据集
data = load_digits()
X,y = data.data, data.target
scaler = ss()
scaler.fit(X) # 特征变成了连续值,即服从标准正态分布
X = scaler.transform(X)
# 将数据集划分为训练集和测试集。模型在训练集上训练，在测试集上评估。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# stratify=y 可以确保训练集和测试集中类别比例与原始数据集相似。

def model_classifier(model,X_train,y_train,X_test,y_test,model_name = "Model"):
    model.fit(X_train,y_train)
    # 拟合优度
    Accuracy = model.score(X_test,y_test)
    # 均方误差
    Mse = mse(model.predict(X_test),y_test)
    print(f"使用{model_name}的拟合优度为: {Accuracy:.5f}")
    print(f"使用{model_name}的均方误差为: {Mse:.5f}")
    print()

# 逻辑回归模型
logistic = lm.LogisticRegression(max_iter=1000)
model_classifier(logistic,X_train,y_train,X_test,y_test,"逻辑回归算法")

# 朴素贝叶斯模型
# 高斯分布朴素贝叶斯算法 适用于特征是连续的且大致符合高斯分布的情况 
GaussNB = nb.GaussianNB()
model_classifier(GaussNB,X_train,y_train,X_test,y_test,"高斯分布朴素贝叶斯算法")
# 伯努利算法 适用于二元/布尔特征（即特征值为 0 或 1） 不太合适
BernoulliNB = nb. BernoulliNB()
model_classifier(BernoulliNB,X_train,y_train,X_test,y_test,"伯努利朴素贝叶斯算法")

# SVM
# SVC 
SVC = svm.SVC()
model_classifier(SVC,X_train,y_train,X_test,y_test,"Support Vector Classifier算法")

# 决策树
# DecisionTreeClassifier 
DTC = tree.DecisionTreeClassifier(max_depth=10)
model_classifier(DTC,X_train,y_train,X_test,y_test,"决策树分类器算法")

# 神经网络模型 的多层分类机回归器
MLPClass = nn.MLPClassifier()
model_classifier(MLPClass,X_train,y_train,X_test,y_test,"多层感知机分类器算法")


# # 逻辑回归模型
# logistic = lm.LogisticRegression(max_iter=1000)
# logistic.fit(X_train,y_train)
# logicAccuracy = logistic.score(X_test,y_test)
# print(f"使用逻辑回归的拟合优度为：{logicAccuracy}")
# logic_mse = mse(logistic.predict(X_test), y_test)
# print("使用逻辑回归的均方误差为：%.5f"%logic_mse)
# print()

# # 朴素贝叶斯模型
# # 高斯分布朴素贝叶斯算法 适用于特征是连续的且大致符合高斯分布的情况 
# GaussNB = nb.GaussianNB()
# GaussNB.fit(X_train,y_train)
# gaussAccuracy = GaussNB.score(X_test,y_test)
# print(f"使用高斯分布朴素贝叶斯算法的拟合优度为：{gaussAccuracy}")
# Gauss_mse = mse(GaussNB.predict(X_test), y_test)
# print("使用高斯分布朴素贝叶斯算法的均方误差为：%.5f"%Gauss_mse)
# print()
# # 伯努利算法 适用于二元/布尔特征（即特征值为 0 或 1） 不太合适
# BernoulliNB = nb. BernoulliNB()
# BernoulliNB.fit(X_train,y_train)
# BernoulliAccuracy = BernoulliNB.score(X_test,y_test)
# print(f"使用伯努利算法的拟合优度为：{BernoulliAccuracy}")
# BernoulliNB_mse = mse(BernoulliNB.predict(X_test), y_test)
# print('使用伯努利算法的均方误差为：%.5f' %BernoulliNB_mse)
# print()

# # SVM
# # SVC 
# SVC = svm.SVC()
# SVC.fit(X_train,y_train)
# SVCAccuracy = SVC.score(X_test,y_test)
# print(f"使用SVC的拟合优度为：{SVCAccuracy}")
# SVC_mse = mse(SVC.predict(X_test), y_test)
# print("使用SVC的均方误差为：%.5f"%SVC_mse)
# print()
# # SVR 回归模型，不适合这种分类问题
# SVR = svm.SVR()
# SVR.fit(X_train,y_train)
# SVRAccuracy = SVR.score(X_test,y_test)
# print(f"使用SVR的拟合优度为：{SVRAccuracy}")
# SVR_mse = mse(SVR.predict(X_test), y_test)
# print("使用SVR的均方误差为：%.5f"%SVR_mse)
# print()

# # 决策树
# # DecisionTreeClassifier 
# DTC = tree.DecisionTreeClassifier(max_depth=10)
# DTC.fit(X_train,y_train)
# DTCAccuracy = DTC.score(X_test,y_test)
# print(f"使用决策树分类器的拟合优度为：{DTCAccuracy}")
# DTC_mse = mse(DTC.predict(X_test), y_test)
# print("使用决策树分类器的均方误差为：%.5f"%DTC_mse)
# print()
# # 决策树回归器 回归模型，不适合分类问题
# # DTR = tree.DecisionTreeRegressor(max_depth=10)
# # DTR.fit(X_train,y_train)
# # DTRAccuracy = DTR.score(X_test,y_test)
# # print(f"使用决策树回归器的拟合优度为：{DTRAccuracy}")
# # DTR_mse = mse(DTR.predict(X_test), y_test)
# # print("使用决策树回归器的均方误差为：%.5f"%DTR_mse)
# # print()

# # 神经网络模型 的多层分类机回归器
# MLPClass = nn.MLPClassifier()
# MLPClass.fit(X_train,y_train)
# MLPAccuracy = MLPClass.score(X_test,y_test)
# print(f"使用多层感知机分类器的拟合优度为：{MLPAccuracy}")
# MLPClass_mse = mse(MLPClass.predict(X_test), y_test)
# print("使用多层感知机分类器的均方误差为：%.5f"%MLPClass_mse)
# print()
# # 回归网络 也不适用分类问题
# # MLPR = nn.MLPRegressor()
# # MLPR.fit(X_train,y_train)
# # MLPRAccuracy = MLPR.score(X_test,y_test)
# # print(f"使用多层感知机回归器的拟合优度为：{MLPRAccuracy}")
# # MLPR_mse = mse(MLPR.predict(X_test), y_test)
# # print("使用多层感知机回归器的均方误差为：%.5f"%MLPR_mse)
# # print()
