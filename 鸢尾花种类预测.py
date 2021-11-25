
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#==============  1. 获取数据集  ====================
iris = load_iris()

#==============  2. 数据基本处理  ====================
# 2.1 数据分割
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size = 0.2,random_state=22)

#==============  3. 特征工程  ====================
# 3.1 实例化一个转换器
transfer = StandardScaler()
# 3.2 调用fit_transform方法——特征值标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

#==============  4. 机器学习 / 模型训练  ====================
# 4.1 实例化一个估计器
estimator = KNeighborsClassifier(n_neighbors=5)
# 4.2 模型训练
estimator.fit(x_train,y_train)  # 训练 【训练集的特征值与目标值】

#==============  5. 模型评估  ====================
# 5.1 输出预测值
y_pre = estimator.predict(x_test) # 预测 【将测试集的特征值传入，根据先前计算出的模型，来预测所给测试集的目标值】
print("预测值是：\n",y_pre)

print("预测值和真实值对比：\n",y_pre == y_test) # 对比 【对比预测值与真实值】

# 5.2 输出准确率
ret = estimator.score(x_test,y_test) # 计算准确率 【根据测试集的特征值与目标值，直接计算出准确率】
print("准确率是：\n",ret)

# 预测值是：
#  [0 2 1 1 1 1 1 1 1 0 2 1 2 2 0 2 1 1 1 1 0 2 0 1 1 0 1 1 2 1]
# 预测值和真实值对比：
#  [ True  True  True False  True  True  True False  True  True  True  True
#   True  True  True  True  True  True False  True  True  True  True  True
#  False  True False False  True False]
# 准确率是：
#  0.7666666666666667


