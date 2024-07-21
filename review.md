### 标准化
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
```
### 1. Pandas 
DataFrame 是 Pandas 中的另一个核心数据结构，用于表示二维表格型数据
``` python
train_data = pd.read_csv("input/train.csv") 
train_data.head() #展示前五列

# 创建DataFrame,index为行索引，columns为列索引
df = pd.DataFrame(data=, index=None, columns=['Site', 'Age'])
df.shape() ##形状

# 访问dataframe元素
df.iloc[:, 0] ##访问第一列
df.values[:, 1:] #返回numpy形式,从第二列开始访问
df.columns ##返回给定df的列标签下的值，columns在此处为标签名

df_dropped = df.drop('Column1', axis=1) #删除Column1列
df_train = df_train.fillna(0) ##将NA数据填充为0

```
### 2.逻辑回归模型（二分类）分类问题
```python
from sklearn.datasets import load_iris
data = load_iris()  # 得到数据特征
iris_target = data.target  # 得到数据对应的标签
iris_features = pd.DataFrame(
    data=data.data, columns=data.feature_names)  # 利用Pandas转化为DataFrame格式

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# x_train为特征，y_train为标签
x_train, x_test, y_train, y_test = train_test_split(
    iris_features, iris_target, test_size=0.2, random_state=2020)

# 定义 逻辑回归模型
clf = LogisticRegression(random_state=0, solver='lbfgs')
# 在训练集上训练逻辑回归模型
clf.fit(x_train, y_train)
# 利用训练好的模型预测类别
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
```
#### 预测房价可以用线性回归
```python
from sklearn.linear_model import LinearRegression
# 特征和目标变量
X = data[['size', 'bedrooms', 'age']]
y = data['price']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建线性回归模型
model = LinearRegression()
# 训练模型
model.fit(X_train, y_train)
# 预测
y_pred = model.predict(X_test)
```

### 3. SVM
```python
from sklearn.svm import SVC
svc = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
svc.fit(train_x, train_y)
predict_y = svc.predict(test_x)
print("准确率：", accuracy_score(test_y, predict_y))
```
### 4. CNN
```python
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F

# Super parameter 
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

# Prepare dataset 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
    plt.title("Labels: {}".format(train_dataset.train_labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）

model = Net()

# Construct loss and optimizer 
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量

# Train and Test CLASS 
# 把单独的一轮一环封装在函数类里
def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        # 把运行中的loss累加起来，为了下面300次一除
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 这小批300的loss清零
            running_total = 0
            running_correct = 0  # 这小批300的acc清零

        torch.save(model.state_dict(), './model_Mnist.pth')
        # torch.save(optimizer.state_dict(), './optimizer_Mnist.pth')


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return predicted.item()

if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        acc_test = test()
        acc_list_test.append(acc_test)

    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()

```
### 5.K-means 无监督
```python
from sklearn.cluster import KMeans 
estimator = KMeans(n_clusters=3)# 构造聚类器，3类
estimator.fit(X)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
```
### 6.预测下一次做题是对还是错 LSTM
```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape = (n_steps, n_features)) )
model.add( Dense(1) )
model.compile(optimizer='adam', loss='mse')

n_steps = 3 # n_steps 为每次用多少个观测值去预测下一个时刻的观测值
n_features = 1 # n_features 表示每个观测值的特征大小，这里我们是预测一个数，因此feature_size=1

# 数据规范化
train_X = np.array([data[i: i + n_steps]
                  for i in range(len(data) - n_steps)])
train_y = np.array([data[i + n_steps]
                  for i in range(len(data) - n_steps)])
model.fit(train_X, train_y, batch_size=1, epochs=30)
preds = model.predict(test_X)
