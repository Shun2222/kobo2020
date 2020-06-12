# レポート課題２ 
1910094 植木 駿介


```python
def init_network():
    network={}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([[0.1, 0.2, 0.3]])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([[0.1, 0.2]])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([[0.1, 0.2]])
    return network

#sigmoid関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#ソフトマックス関数
def softmax(x):
    c = np.max(x) 
    #overflowの防止
    exp_x = np.exp(x-c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

def forward(network, x):
    # layer 1
    W1=network['W1']
    b1=network['b1']
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    
    #layer 2 
    W2=network['W2']
    b2=network['b2']
    a2=np.dot(z1,W2)+b2 #第1層の結果に重みをかけバイアスを加える
    z2=sigmoid(a2) 
    
    # layer 3 (出力層)
    W3=network['W3']
    b3=network['b3']
    y=np.dot(z2,W3)+b3 #第2層の結果に重みをかけバイアスを加える
    return y

network = init_network()
x=np.array([1.0,0.5])
y=forward(network,x)
print(y) #[0.31682708 0.69627909]
```

    [[0.31682708 0.69627909]]
    

## 感想
正解がないため、重みやバイアスの更新は行わず、データが処理される流れを確認した。numpyを使うことでまとめた計算ができ効率よく計算ができた。

## 参考文献
物体・画像認識と時系列データ処理入門　著 チーム・カルポ
