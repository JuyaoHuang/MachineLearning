from sklearn.preprocessing import StandardScaler as ss
from sklearn.datasets import load_digits
import sklearn.linear_model as lm
import sklearn.naive_bayes as nb
import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.neural_network as nn
from sklearn.metrics import mean_squared_error as mse

data = load_digits()
X,y = data.data, data.target
scaler = ss()
scaler.fit(X)
X = scaler.transform(X)

# 逻辑回归模型
logistic = lm.LogisticRegression()
logistic.fit(X,y)
logicAccuracy = logistic.score(X,y)
print(f"使用逻辑回归的拟合优度为：{logicAccuracy}")
logic_mse = mse(logistic.predict(X), y)
print("使用逻辑回归的均方误差为：%.5f"%logic_mse)
print()

# 朴素贝叶斯模型
# 高斯分布朴素贝叶斯算法
GaussNB = nb.GaussianNB()
GaussNB.fit(X,y)
gaussAccuracy = GaussNB.score(X,y)
print(f"使用高斯分布朴素贝叶斯算法的拟合优度为：{gaussAccuracy}")
Gauss_mse = mse(GaussNB.predict(X), y)
print("使用高斯分布朴素贝叶斯算法的均方误差为：%.5f"%Gauss_mse)
print()
# 伯努利算法
BernoulliNB = nb. BernoulliNB()
BernoulliNB.fit(X,y)
BernoulliAccuracy = BernoulliNB.score(X,y)
print(f"使用伯努利算法的拟合优度为：{BernoulliAccuracy}")
BernoulliNB_mse = mse(BernoulliNB.predict(X), y)
print('使用伯努利算法的均方误差为：%.5f' %BernoulliNB_mse)
print()

# SVM
# SVC
SVC = svm.SVC()
SVC.fit(X,y)
SVCAccuracy = SVC.score(X,y)
print(f"使用SVC的拟合优度为：{SVCAccuracy}")
SVC_mse = mse(SVC.predict(X), y)
print("使用SVC的均方误差为：%.5f"%SVC_mse)
print()
# SVR
SVR = svm.SVR()
SVR.fit(X,y)
SVRAccuracy = SVR.score(X,y)
print(f"使用SVR的拟合优度为：{SVRAccuracy}")
SVR_mse = mse(SVR.predict(X), y)
print("使用SVR的均方误差为：%.5f"%SVR_mse)
print()

# 决策树
# DecisionTreeClassifier
DTC = tree.DecisionTreeClassifier(max_depth=10)
DTC.fit(X,y)
DTCAccuracy = DTC.score(X,y)
print(f"使用决策树分类器的拟合优度为：{DTCAccuracy}")
DTC_mse = mse(DTC.predict(X), y)
print("使用决策树分类器的均方误差为：%.5f"%DTC_mse)
print()
# 决策树回归器
DTR = tree.DecisionTreeRegressor(max_depth=10)
DTR.fit(X,y)
DTRAccuracy = DTR.score(X,y)
print(f"使用决策树回归器的拟合优度为：{DTRAccuracy}")
DTR_mse = mse(DTR.predict(X), y)
print("使用决策树回归器的均方误差为：%.5f"%DTR_mse)
print()

# 神经网络模型 的多层分类机回归器
MLPClass = nn.MLPClassifier()
MLPClass.fit(X,y)
MLPAccuracy = MLPClass.score(X,y)
print(f"使用多层感知机分类器的拟合优度为：{MLPAccuracy}")
MLPClass_mse = mse(MLPClass.predict(X), y)
print("使用多层感知机分类器的均方误差为：%.5f"%MLPClass_mse)
print()

MLPR = nn.MLPRegressor()
MLPR.fit(X,y)
MLPRAccuracy = MLPR.score(X,y)
print(f"使用多层感知机回归器的拟合优度为：{MLPAccuracy}")
MLPR_mse = mse(MLPR.predict(X), y)
print("使用多层感知机回归器的均方误差为：%.5f"%MLPR_mse)
print()