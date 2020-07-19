---
marp: true
theme: docker
header: "header"
footer: "ゼロから作るDeepLearning 学習ノート"
---

# working
---

## 学習に関するテクニック

---

### ポイント

パラメータの最適化
パラメータの初期値
過学習対策
ハイパーパラメータの設定方法

---

### パラメータの更新

ニューラルネットワークにおける学習とは、損失関数の値をできる限り小さくするパラメータをみつけること。このような問題を最適化(Optimization)と呼ぶ。

最適化をスマートに行える方法を考えることも重要。

- SGD(Stochastic Gradient Descent)
- Momentum
- AdaGrad
- Adam

---

#### SGD(Stochastic Gradient Descent)

確率的勾配降下法と呼ばれる手法で最適化の基本形。
降下勾配法で使用していた式がこれに当たる。

数式で表すと下式のようになる。

$$
W  \gets W - \eta \frac{\delta{L}}{\delta{W}}
$$

勾配方向$\frac{\delta{L}}{\delta{W}}$の方向へある一定の距離(学習率$\eta$)の距離だけ進むという単純な方法。

---

#### SGD(Stochastic Gradient Descent)：実装

```python
class SGD:
  def __init__(self, lr=0.01):
    self.lr = lr
    
  def update(self, params, grads):
    for key in params.keys():
      params[key] -= lr * grads[key]
```

SGDは単純で実装も簡単だが、問題によっては最適化が非効率になる場合もある。

---

#### SGDによる最適化イメージ１

下式の最小値を求める最適化を行うとする。

$$
f(x, y) = \frac{1}{20} x^2 + y2
$$

![center 式のグラフ](./05_learning_technic/graph1.png)
![center 式の等高線](./05_learning_technic/graph2.png)

---

#### SGDによる最適化イメージ２

SGDによって最適化されていく経緯を等高線にプロットする。

![center SGDによる最適化](./05_learning_technic/sgd_optimization.png)

単純に勾配方向へ最適化していくSGDでは、勾配の方向が本来の最小値ではない方向をさす場合もあり効率てきな最適化がおこなえると言えない。

---

#### Momentum

Momentumとは「運動量」のこと。
SGDに、前回の移動量の一定割合を反映することで、単純に勾配方向へ移動させるだけでなく、移動量を加速させたり原則させたりする。

式で表すと下式のようになる。

$$
\begin{aligned}
v & \gets \alpha v - \eta \frac{\delta{L}}{\delta{W}}
\\\\
w & \gets W + v
\end{aligned}
$$

---

#### Momentum：実装

```python
class Momentum:
  def __init__(selft, lr=0.01, momentum=0.9):
    self.lr = lr
    self.momentum = momentum
    self.v = None
    
  def update(self, params, grads):
    if self.v is None:
      self.v = {}
      for key, val in params.items():
        self.v[key] = np.zeros_like(val)
        
      for key in params.keyx():
        self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
        params[key] += self.v[key]
  
```

---

#### Momentumによる最適化イメージ

Momentumによって最適化されていく経緯を等高線にプロットする。

![center Momentumによる最適化](./05_learning_technic/momentum_optimization.png)

SGDで最適化したときのグラフに比べると、スマートに最適化されている。

---

#### AdaGrad



$$
\begin{aligned}
h & \gets h + \frac{\delta{L}}{\delta{W}} \odot \frac{\delta{L}}{\delta{W}}
\\\\
W & \gets W - \eta \frac{1}{\sqrt{h}} \frac{\delta{L}}{\delta{W}}
\end{aligned}
$$

---

#### Adam




---

### 重みの初期値

重みの初期値を$0$にする？
隠れ層のアクティベーション分布
ReLUの場合の重みの初期値

---

### Batch Normalization

Batch Noralizationのアルゴリズム
Batch Noralizationの評価

---

### 正則化

過学習
Weight decay
Dropout


---

### ハイパーパラメータの検証

検証データ
ハイパーパラメータの最適化
ハイパーパラメータの最適化の実装


---

### まとめ


