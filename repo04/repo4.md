# レポート課題4
## 1910094 植木 駿介

### Softmax with lossレイヤーの実装


```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        # forwardの式
        # -sum ( t * log (y))
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self, dout=1):
        # backwardの式
        # yi - ti (iはIndex)
        batch_size = self.t.shape[0]
        dx = (self.y -self.t)/batch_size
        return dx
```

### Two layer netにおける勾配の確認


```python
import numpy as np
from collections import OrderedDict
def numerical_grad(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        return grad
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_grad(loss_W, self.params['W1'])
        grads['b1'] = numerical_grad(loss_W, self.params['b1'])
        grads['W2'] = numerical_grad(loss_W, self.params['W2'])
        grads['b2'] = numerical_grad(loss_W, self.params['b2'])
        return grads
    #正解率の計算
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
```


```python
def gradient(network, x, t):
    # 自分で実装したSoftmax with lossクラスを使ってみてください
    #lastLayer = SoftmaxWithLoss()←今回はTwoLayerNet中ですでに代入したlastLayert= SoftmaxWithLoss()を利用した
    # forward
    #self.loss(x, t)
    network.loss(x, t)
    # backward
    dout = 1
    dout = network.lastLayer.backward(dout)
    #layers = list(self.layers.values())
    layers = list(network.layers.values())
    layers.reverse()
    for layer in layers:
        dout = layer.backward(dout)
    # 設定
    grads = {}
    #grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
    grads['W1'], grads['b1'] = network.layers['Affine1'].dW, network.layers['Affine1'].db
    #grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
    grads['W2'], grads['b2'] = network.layers['Affine2'].dW, network.layers['Affine2'].db
    return grads
```


```python
from dataset.mnist import load_mnist
# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
x_batch = x_train[:3]
t_batch = t_train[:3]
# 数値微分
grad_numerical = network.numerical_gradient(x_batch, t_batch)
# Backward
#grad_backprop = gradient(x_batch, t_batch)
grad_backprop = gradient(network, x_batch, t_batch)
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
```

    W1:0.00038931216239115695
    b1:0.002253937251343669
    W2:0.006341753863522595
    b2:0.1167537332305281
    

### Two layer netの学習結果の確認


```python
import numpy as np
from dataset.mnist import load_mnist
# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    #grad = gradient(x_batch, t_batch)
    grad = gradient(network, x_batch, t_batch)
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
```

    0.115 0.1148
    0.9025833333333333 0.9092
    0.9215 0.9231
    0.9361166666666667 0.9365
    0.9431166666666667 0.9432
    0.9519166666666666 0.9502
    0.9555333333333333 0.9531
    0.9615666666666667 0.9586
    0.96345 0.9595
    0.96405 0.9591
    0.9696 0.965
    0.9713333333333334 0.9655
    0.9736 0.9667
    0.97485 0.9677
    0.97625 0.9681
    0.9767666666666667 0.9676
    0.9784 0.9697
    

## 感想
勾配が小さな値になることを確認することができた。また、精度が最終的に97.9%とランダムの10%よりも高くなっていることが確認できる。

## 参考文献
【ゼロから作るDeep Learning】4章 ニューラルネットワークの学習
【ゼロから作るDeep Learning】5章 誤差逆伝播法


```python

```
