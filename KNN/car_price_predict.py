import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据
"""
Brand（品牌）,Type（发动机类型）,Color（颜色）,
Construction Year（生成年份）,Odometer（里程数）,Ask Price（预测价格）,
Days Until MOT（上一次维修时间）,HP（功率、马力）
"""
df = pd.read_csv(r'../datas/car_price_data.csv')


# 特征处理    
df_colors = df['Color'].str.get_dummies().add_prefix('Color: ')  # 把颜色独热编码
df_type = df['Type'].apply(str).str.get_dummies().add_prefix('Type: ')  # 把类型独热编码
df = pd.concat([df, df_colors, df_type], axis=1)  # 添加独热编码数据列
df = df.drop(['Brand', 'Type', 'Color'], axis=1)  # 去除独热编码对应的原始，这里的品牌数据都是一样，所以可以去掉，没有意义

# print(df)


# 画热力图
# matrix = df.corr()  # 数据转换为相关矩阵，即求出每一列相互之间的相关系数
# f, ax = plt.subplots(figsize=(8, 6))  # figsize设置图片尺寸，横 * 纵
# sns.heatmap(matrix, square=True)  # square参数代表图中每个单元格成正方形
# plt.title('Car Price Variables')


X = df[['Construction Year', 'Days Until MOT', 'Odometer']]
y = df['Ask Price'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

# 标准化数据
X_normalizer = StandardScaler()  # N(0,1)
X_train = X_normalizer.fit_transform(X_train)
X_test = X_normalizer.transform(X_test)
y_normalizer = StandardScaler()
y_train = y_normalizer.fit_transform(y_train)
y_test = y_normalizer.transform(y_test)

# 创建KNN训练对象，K值设置为2
knn = KNeighborsRegressor(n_neighbors=2)
# 开始训练
# ravel()将多维数据转为1*n的一维数据
knn.fit(X_train, y_train.ravel())

# 将测试数据传入训练好的knn对象进行预测
y_pred = knn.predict(X_test)
# 反标准化，将标准化后的数据恢复成原始倍数
y_pred_inv = y_normalizer.inverse_transform(y_pred)
y_test_inv = y_normalizer.inverse_transform(y_test)

# 以预测值为横坐标，真实值为纵坐标画二维点图
plt.scatter(y_pred_inv, y_test_inv)
plt.xlabel('Prediction')
plt.ylabel('Real value')

# 画出对称且经过原点的直线y=kx，用于区分蓝色点（预测值, 真实值）偏离100%正确多远
diagonal = np.linspace(500, 1500, 100)  # 生成从500到1500,100个数据的等差数列
plt.plot(diagonal, diagonal, '-r')
plt.xlabel('Predicted ask price')
plt.ylabel('Ask price')
plt.show()

print(y_pred_inv)
print(knn)
