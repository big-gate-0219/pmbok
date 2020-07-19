<!-- $theme: gaia -->
<!-- $size: 4:3 -->

# working

---

## 逆誤差伝播方式

---

### ニューラルネットワークの学習(復習)

損失関数の結果を最小化することで学習を実施。

損失関数の最小化には降下勾配法を使用。

$$
x = x - \eta{\frac{\delta{f}}{\delta{x}}}
$$

$\frac{\delta{f}}{\delta{x}}$を数値微分で計算すると時間がかかる。

$\frac{\delta{f}}{\delta{x}}$を効率的に計算したい。

---

### 計算グラフ(順伝播)

$f(x_1, x_2) = (2x_1 + 3x_2) \times{4}$ を計算グラフで記載すると下図のようになる。

![100% center 順伝播の図](./04_backpropagation/forward_propagation.png)

$x_1$や$x_2$という入力値から$f(x)$の結果を求めるように順方向に計算が進むことを順伝播と呼ぶ。

---

#### 計算グラフ(逆伝播)

$f(x_1, x_2) = \left( 2x_1 + 3x_2 \right) \times{4}$において、$x_1$や$x_2$の変化に対する$f$の変化量は下図のように表記できる。

![100% center 逆伝播の図](./04_backpropagation/backward_propagation_1.png)

順伝播に対して逆方向に計算を進めるため、これを逆伝播と呼ぶ。

---

#### 計算グラフ(逆伝播：部分的な計算)

$f(x_1, x_2) = 2x_1 + 3x_2 + 4$ を分解し図示する。

$$
\begin{aligned}
f_1(x_1) & = 2x_1 \\
f_2(x_2) & = 3x_2 \\
f(x_1, x_2) &= f_1(x_1) + f_2(x_2) + 4
\end{aligned}
$$

![center 逆伝播の部分的な計算](./04_backpropagation/backward_propagation_2.png)

---

#### 計算グラフ(部分的な計算を数式表記)

![80% center 逆伝播の部分的な計算](./04_backpropagation/backward_propagation_2.png)

上図を数式で表記すると下式のようになる。

$x_1$に対する変化量 | $x_2$に対する変化量
---|---
$\begin{aligned}\frac{\delta{f}}{\delta{x_1}}&=\frac{\delta{f_1}}{\delta{x_1}} \frac{\delta{f}}{\delta{f_1}}\\&= 2 \cdot 4\\&= 8\end{aligned}$ | $\begin{aligned}\frac{\delta{f}}{\delta{x_2}}&=\frac{\delta{f_2}}{\delta{x_2}} \frac{\delta{f}}{\delta{f_2}}\\&= 3 \cdot 4\\&= 12 \end{aligned}$

---

#### 計算グラフ（まとめ）

計算グラフを使った逆伝播による変化量の算出で、

複数の関数から成り立つ関数の変化量は

それぞれの関数の変化量の積となっていた。

ここでいう変化量は微分のことなので

複数の関数から成り立つ関数の微分は

それぞれの関数の微分の積と同じになる？

---

### 連鎖律

連鎖律とは、合成関数の微分についての性質。

>ある関数が合成関数で表される場合、その合成関数の微分は、合成関数を構成するそれぞれの関数うの微分の積によって表すことができる。
>
$$
\begin{aligned}
  z &= t^2
  \\
  t &= x + y
  \\\\
  \frac{\delta{z}}{\delta{x}}
  &= \frac{\delta{z}}{\delta{t}} \frac{\delta{t}}{\delta{x}}
  \\
  &= 2t \cdot 1 = 2(x + y)

\end{aligned}
$$

---

### 逆誤差伝播方式

ニューラルネットワークは下図のように表せた。

![100% center 損失関数を含むニューラルネットワーク](./04_backpropagation/neural_network_1.png)

これを、計算グラフに書き直す。

---

#### 逆誤差伝播方式(計算グラフ表記)

![80% center ニューラルネットワークの計算グラフ](./04_backpropagation/neural_network_2.png)

この計算グラフを見ると、$L$はAffine関数、Sigmoid関数、Softmax関数、損失関数からなる合成関数。
各関数の微分を事前に求めることができれば、連鎖律を使って重み$W$や閾値$B$の微分が計算できる。

---

#### 逆誤差伝播方式(Sigmoidレイヤ)

Sigmoid関数を計算グラフで表す。

![center Sigmoidの計算グラフ](./04_backpropagation/sigmoid_graph.png)

順方向 | 逆方向
---|---
$y = \frac{1}{1 + \text{exp}(-x)}$ | $\begin{aligned}\frac{\delta{L}}{delta{{x}}} &= \frac{\delta{L}}{\delta{y}} y^2 \text{exp}(-x) \\ &= \frac{\delta{L}}{\delta{y}} \frac{1}{1 + \text{exp} (-x)} \frac{\text{exp}(-x)}{1 + \text{exp} (-x)} \\ &= \frac{\delta{L}}{\delta{y}} y(1-y) \end{aligned}$

---

#### 逆誤差伝播方式(Sigmoidレイヤ：実装)

```python
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
```

---

#### 逆誤差伝播方式(ReLUレイヤ)

ReLU関数を計算グラフで表す。

![center ReLUの計算グラフ](./04_backpropagation/relu_graph.png)

順方向 | 逆方向
---|---
$y=\begin{cases} x & (x > 0) \\ 0 & (x \leq 0) \end{cases}$ | $\begin{aligned} \frac{\delta{L}}{\delta{x}} &= \begin{cases} \frac{\delta{L}}{\delta{y}} & (x>0) \\ 0 & (x \leq 0) \end{cases} \end{aligned}$

---

#### 逆誤差伝播方式(ReLUレイヤ：実装)

```python
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (X <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
```

---

#### 逆誤差伝播方式(Affineレイヤ)

Affine関数を計算グラフで表す。

![center Affineの計算グラフ](./04_backpropagation/affine_graph.png)

順方向 | 逆方向
---|---
$Y = W \cdot X + B$ | $\begin{aligned} \frac{\delta{L}}{\delta{X}} &= \frac{\delta{L}}{\delta{Y}} \cdot W^{\mathrm{T}} \\ \frac{\delta{L}}{\delta{W}} &= X^{\mathrm{T}} \cdot \frac{\delta{L}}{\delta{Y}} \\ \frac{\delta{L}}{\delta{B}} &= \frac{\delta{L}}{\delta{Y}} \end{aligned}$

---

#### 逆誤差伝播方式(Affineレイヤ：実装)

```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
```

---

#### 逆誤差伝播方式(Softmaxレイヤ)

Softmaxレイヤは、損失関数も加味して計算をする。
計算グラフで表す(略図のみ)。

![center 70% Softmaxの計算グラフ](./04_backpropagation/softmax_graph.png)

順方向 | 逆方向
---|---
$\begin{aligned} L &= - \sum_k{t_k \text{log} y_k} \\\\ Y &= \frac {\text{exp}(x_k - C')}{\sum^n_{i=1} \text{exp}(x_i - C')} \end{aligned}$ | $\frac{\delta{L}}{\delta{X}} = Y - T$

---

#### 逆誤差伝播方式(Softmaxレイヤ：実装)

```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch.size
        return dx
```

---

### 誤差逆伝播法の実装

誤差逆伝播方による学習を行うニューラルネットワークを作成する。

ニューラルネットワークの構成は、文字認識ニューラルネットワークの構成を再利用する。

---

#### 誤差逆伝播法の実装(構成)

![center](./04_backpropagation/neural_network_3.png)

層名|ニューロン数 | 備考
---|--:|---
入力層 | $784$ | 画像サイズ$(28 \times 28)$
隱れ層① | $50$ | 適当
隱れ層② | $10$ | 適当
出力層 | $10$ | 分類数($1$〜$10$)に対応

---

#### 誤差逆伝播法の実装(実装イメージ１)

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    
    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads
```

---
#### 誤差逆伝播法の実装(実装イメージ２)

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list= []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.gradient(x_batch, t_batch)
    
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

---
#### 誤差逆伝播法の実装(処理結果)

Epoch | 正解率
---|---
1 | 0.0996
2 | 0.906
3 | 0.9268
... | ...
14 | 0.9653
15 | 0.9687
16 | 0.9677
17 | 0.97

