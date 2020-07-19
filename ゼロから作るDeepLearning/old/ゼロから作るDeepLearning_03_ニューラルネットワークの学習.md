<!-- $theme: gaia -->
<!-- $size: 4:3 -->
<!-- prerender: true -->
<!-- page_number: true -->
<!-- <!-- footer: ゼロから作るDeepLearning 学習ノート-->

# Working

---

## ニューラルネットワークの学習

---

### 損失関数

ニューラルネットワークの性能の悪さを示す指標。
出力値と想定結果がどれだけ一致していないかを表す。

損失関数の結果が最小となるパラメータを探索する行為が、ニューラルネットワークの学習となる。

ニューラルネットワークでは、損失関数に２乗和誤差や交差エントロピー誤差を用いる。

---

#### 損失関数(２乗和誤差)

２乗和誤差は下式のように表される。

$$
E = \frac {1}{2} \sum_k (y_k - t_k)^2
$$

変数|説明
:--|:--
$y_k$ | ニューラルネットワークの出力値
$t_k$ | 想定結果

---

#### 損失関数(２乗和誤差：実装)

```python
# ２乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
```

```python
# "2"を正解とする。
t = np.array([0,0,1,0,0,0,0,0,0,0])

# "2"である確率が一番高いとなった場合。
y = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
mean_squared_error(y, t)
-> .09750000000000003

# "7"である確率が一番高いとなった場合。
y= np.array([0.1,0.5,0.1,0.0,0.05,0.1,0.0,0.6,0.5,0.0])
mean_squared_error(y, t)
-> 0.8462500000000001
```

---

#### 損失関数(交差エントロピー誤差)

交差エントロピー誤差は下式のように表される。

$$
E = - \sum_k t_k \text{log} y_k
$$

変数|説明
:--|:--
$y_k$ | ニューラルネットワークの出力
$t_k$ | 教師データ

---

#### 損失関数(交差エントロピー誤差：実装)

```python
# 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7 # log(0)=-Inf(無限大)とならないようにする
    return -np.sum(t * np.log(y + delta))
```

```python
# "2"を正解とする。
t = np.array([0,0,1,0,0,0,0,0,0,0])

# "2"である確率が一番高いとなった場合。
y = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
cross_entropy_error(y, t)
-> 0.510825457099338

# "7"である確率が一番高いとなった場合。
y= np.array([0.1,0.5,0.1,0.0,0.05,0.1,0.0,0.6,0.5,0.0])
cross_entropy_error(y, t)
-> 2.302584092994546
```

---

### 微分

ある瞬間の変化の量を表したもの。
数式では次のように表される。

$$
\frac {df(x)}{dx}= \lim_{h \to 0} \frac{f(x + h) - f(x - h)}{2h}
$$

この式は中心差分と呼ばれ、$x$地点を中心とした$\pm{h}$地点の$f(x)$の結果から$x$地点の変化量を求める式となる。
数式$f(x)$の計算結果から微分を求める方法を数値微分と呼ぶ。

---

#### 微分(数値微分：計算例)

下式に対する$x=5$地点の数値微分を求めてみる。
なお、$h=0.0001$とする。

$$
f(x) = 0.001 x^2 + 0.1x
$$

$$
\begin{aligned}
\frac {df(x)}{dx} &= \frac{f(x + h) - f(x - h)}{2h} \\
&= \frac{f(5.0001) - f(4.9999)}{0.0002} \\
& = 0.1999999999990898
\end{aligned}
$$

$f(x)$の$x=5$地点の変化量は約$0.199$。

--- 

#### 微分(数値微分：実装)

```python
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)
```

```python
def function_1(x):
    return (0.01 * (x **2)) + (0.1 * x)

numerical_diff(function_1, 5)
->0.1999999999990898

numerical_diff(function_1, 15)
->0.4000000000026205

numerical_diff(function_1, -10))
->-0.0999999999995449
```

---
#### 微分(数値微分：イメージ)

$f(x)=0.001 x^2 + 0.1$と$x=\left( -10, 5, 15\right)$地点の微分の結果をグラフにすると下図のようになる。

$x=-10$ | $x = 5$ | $x = 15$
---|---|---
![80%](./03_learning/numerical_diff_1.png)|![80%](./03_learning/numerical_diff_2.png)|![80%](./03_learning/numerical_diff_3.png) 

---
#### 微分(偏微分)

複数の変数からなる関数の微分のこと。

$$
f(x_0, x_1) = x_0^2 + x_1^2
$$

の偏微分を式で表すと$\frac{\delta{f}}{\delta{x_0}}$, $\frac{\delta{f}}{\delta{x_0}}$のようになる。

$y = x_0^2 + x_1^2$をグラフにすると下図のようになる。

![center 80%](./03_learning/graph_two_variable_function.png)

---
#### 微分(偏微分：計算①)
$f(x_0, x_1) = x_0^2 + x_1^2$の式で、$(x_0, x_1) = (3, 4)$のときの偏微分を求める。

$$
\begin{aligned}
f(x_0, x_1=4) &= x_0^2+ 4^2 \\

f'(x_0) &=  x_0^2+ 16
\end{aligned}
$$

$$
\begin{aligned}
\frac {\delta{f'(x_0)}}{\delta{x_0}} &= \frac{f'(x_0 + h) - f'(x_0 - h)}{2h} \\
& = \frac{f'(3.0001) - f'(2.9999)}{0.0002} \\
\frac {\delta{f}}{\delta{x_0}} & = 6.00000000000378
\end{aligned}
$$

---
#### 微分(偏微分：計算②)

$f(x_0, x_1) = x_0^2 + x_1^2$の、$(x_0, x_1) = (3, 4)$のときの偏微分を求める。

$$
\begin{aligned}
f(x_0 = 3, x_1) &= 3^2 + x_1^2 \\

f''(x_1) &=  9+ x_1^2
\end{aligned}
$$

$$
\begin{aligned}
\frac {\delta{f''(x_1)}}{\delta{x_1}} &= \frac{f''(x_1 + h) - f''(x_1 - h)}{2h} \\
& = \frac{f''(4.0001) - f''(3.9999)}{0.0002} \\
\frac {\delta{f}}{\delta{x_1}} & = 7.999999999999119
\end{aligned}
$$

---
#### 微分(偏微分：計算③)

以上より、

$$
f(x_0, x_1) = x_0^2 + x_1^2
$$

上式の$(x_0, x_1)=(3, 4)$時点の偏微分は、

$$
\begin{aligned}
\frac {\delta{f}}{\delta{x_0}} & = 6.00000000000378 \\ \\
\frac {\delta{f}}{\delta{x_1}} & = 7.999999999999119
\end{aligned}
$$

となる。

---
#### 微分(偏微分：実装)

```python
# 数値微分（中心差分)
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)
# f(x)=x0^2+x1^2 におけるx1=4となる時の関数
def function_tmp1(x0):
    return x0**2.0 + 4.0 ** 2.0
# f(x)=x0^2+x1^2 におけるx0=3となる時の関数
def function_tmp2(x1):
    return 3.0 ** 2.0 + x1**2.0    
```

```python
numerical_diff(function_tmp1, 3.0)
-> 6.00000000000378

numerical_diff(function_tmp2, 4.0)
-> 7.999999999999119
```

---

### 勾配

すべての変数の偏微分をベクトルとしてまとめたものを勾配という。

たとえば、
$$
f(x_0, x_1) = x_0^2 + x_1^2
$$
に置ける$(3, 4)$点の勾配は
$$
(\frac{\delta{f}}{\delta{x_0}}, \frac{\delta{f}}{\delta{x_1}}) = (6, 8)
$$
となる。


---

#### 勾配(実装①)

```python
def _numerical_gradient(f, x) :
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size) :
        tmp_val = x[idx]
        # f(x + h)の計算
        x[idx] = tmp_val + h
        fxh1= f(x)
        # f(x -　h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad
```

---
#### 勾配(実装②)

```python
def numerical_gradient(f, X):
    grad = np.zeros_like(X)
    
    for idx, x in enumerate(X):
        grad[idx] = _numerical_gradient(f, x)
   
    return grad
```

```python
# ^ sum(x[i]^2)
def function_2(x) :
    return np.sum(x**2)

_numerical_gradient(function_2, np.array([3.0, 4.0]))
-> array([6., 8.])

_numerical_gradient(function_2, np.array([0.0, 2.0]))
->array([0., 4.])
```

---

#### 勾配(イメージ)

$f(x_0, x_1)=x_0^2+x_1^2$ | $(\frac {\delta{f}}{\delta{x_0}}, \frac{\delta{f}}{\delta{x_1}})$
---|----
![80%](./03_learning/graph_two_variable_function.png) | ![](./03_learning/graph_two_variable_function_gradient.png)

勾配は、$f(x_0, x_1)$の「一番低い場所（最小値）」を指す。

---

### 勾配法

勾配を用いて関数の最小値を探索する方法を勾配降下法という（最大値を探索する場合は勾配上昇法）。

勾配法を式で表すと下式のようになる。

$$
\begin{aligned}
x &= x - \eta \frac {\delta f}{\delta x} \\
\end{aligned}
$$

$\eta$は学習率(LearningRate)とよび任意の数値が使われる。$\eta$は大きすぎても小さすぎても最小値への到達が難しくなる。

---
#### 勾配法(実装例：変数１つ）


```python
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def function(x):
    return (0.01 * (x ** 2)) + (0.1 * x)
```

```python
lr = 0.001
x = 0

for i in range(200000):
    diff = numerical_diff(function, x)
    new_x = x - lr * diff
    if  i % 10000 == 0 :
        print(str(i) + ":" + str(x) )
    x = new_x
```

---

#### 勾配法（処理結果：変数１つ）

```python
     0: 0                 ,  0.0
 10000:-1.648426583231277 , -0.1376695563200923
 20000:-2.753391126403293 , -0.19952748569076534
 30000:-3.4940650842766314, -0.22732160029605256
 40000:-3.9905497138164314, -0.2398101011972391
.
.
.
120000:-4.958855214948994 , -0.2499830710666311
130000:-4.972420046436861 , -0.24999239346161456
140000:-4.981512752160442 , -0.24999658221667317
150000:-4.987607726318238 , -0.24999846431552997
```
試行を繰り返すたびに$f(x)$が最小となる$x$に近いている。

---

#### 勾配法(変数２つ①)

変数が２つの時の勾配法はどのようになるのか。

下式を前提に考える。

$$
f(x_0, x_1) = x_0^2 + x_1^2
$$

勾配法の考え方は変数が１つのときと同じ

$$
\begin{aligned}
x_0 &= x_0 - \eta \frac {\delta f}{\delta x_0} \\
x_1 &= x_1 - \eta \frac {\delta f}{\delta x_1}
\end{aligned}
$$


---

#### 勾配法(変数２つ②)

$f(x_0, x_1) = x_0^2 + x_1^2$のグラフ、ならびに、勾配は下図のようなものだった。

$f(x_0, x_1)=x_0^2+x_1^2$ | $(\frac {\delta{f}}{\delta{x_0}}, \frac{\delta{f}}{\delta{x_1}})$
---|----
![80%](./03_learning/graph_two_variable_function.png) | ![](./03_learning/graph_two_variable_function_gradient.png)

---

#### 勾配法(変数２つ③)

$f(x_0, x_1) = x_0^2 + x_1^2$対し$(x_0,x_1)=(-3, 4)$を初期値として勾配法を用いると下図のような軌跡で、$f(x_0, x_1)$の最小値点に近く。

![center 150%](./03_learning/graph_two_variable_function_learning.png)

---

### ニューラルネットワークの学習

ニューラルネットワークの学習は、勾配法を使って重み$W$や閾値$B$を調整しながら損失関数の結果を最小化することで行う。

<!--簡単にいうと、間違った結果を出力しにくいニューラルネットワークとなるように学習させる。-->

$$
\begin{aligned}
W & = W - \eta{\frac{\delta{L}}{\delta{W}}} \\ \\
B & = B - \eta{\frac{\delta{L}}{\delta{B}}}
\end{aligned}
$$

$L$：ニューラルネットワークの損失関数

---

#### ニューラルネットワークの学習

下記ニューラルネットワークの学習を表記する。

![center 100%](./03_learning/neural_network.png)

---

#### ニューラルネットワークの学習(計算式：①)

層名 | 計算式 | 学習式
:--|:--|:--
入力層 | $X$ | -
隱れ層①|$\begin{aligned}A^1 &= X W^1 + B^1 \\Z^1 &= \text{f}(A^1)\end{aligned}$ | $\begin{aligned}W^1 &= W^1 - \eta{\frac{\delta{L}}{\delta{W^1}}} \\ B^1 &= B^1 - \eta{\frac{\delta{L}}{\delta{B^1}}} \end{aligned}$
隱れ層②|$\begin{aligned}A^2 &= Z^1 W^2 + B^2 \\Z^2 &= \text{f} (A^2)\end{aligned}$ | $\begin{aligned}W^2 &= W^2 - \eta{\frac{\delta{L}}{\delta{W^2}}} \\ B^2 &= B^2 - \eta{\frac{\delta{L}}{\delta{B^2}}} \end{aligned}$
出力層|$\begin{aligned}A^3 &= Z^2 W^3 + B^3 \\Y &= \text{h} (A^3)\end{aligned}$ | $\begin{aligned}W^3 &= W^3 - \eta{\frac{\delta{L}}{\delta{W^3}}} \\ B^3 &= B^3 - \eta{\frac{\delta{L}}{\delta{B^3}}} \end{aligned}$
出力 | $Y$ | $L$=損失関数

---
#### ニューラルネットワークの学習(計算式②)

ただし、出力層で行列式の各要素を展開すると

$$
W^3 =
\left( \begin{array}{cc}
w^3_{11} & w^3_{21} \\
w^3_{12} & w^3_{22}
\end{array} \right)
,
\frac{\delta{L}}{\delta{W^3}} =
\left( \begin{array}{cc}
\frac{\delta{L}}{\delta{w^3_{11}}} & \frac{\delta{L}}{\delta{w^3_{21}}}\\
\frac{\delta{L}}{\delta{w^3_{12}}} & \frac{\delta{L}}{\delta{w^3_{22}}}
\end{array} \right)
$$

$$
B^3 =
\left( \begin{array}{cc}
b^3_1 & b^3_2
\end{array} \right)
,
\frac{\delta{L}}{\delta{B^3}} =
\left( \begin{array}{cc}
\frac{\delta{L}}{\delta{b^3_{1}}} & \frac{\delta{L}}{\delta{b^3_{2}}}
\end{array} \right)
$$


---

#### ニューラルネットワークの学習(計算式③)

さらに隠れ層②の行列式を各要素ごとに展開すると


$$
W^2 =
\left( \begin{array}{ll}
w^2_{11} & w^2_{21} \\
w^2_{12} & w^2_{22} \\
w^2_{13} & w^2_{23} 
\end{array} \right)
,
\frac{\delta{L}}{\delta{W^2}} =
\left( \begin{array}{cc}
\frac{\delta{L}}{\delta{w^2_{11}}} & \frac{\delta{L}}{\delta{w^2_{21}}} \\
\frac{\delta{L}}{\delta{w^2_{12}}} & \frac{\delta{L}}{\delta{w^2_{22}}} \\
\frac{\delta{L}}{\delta{w^2_{13}}} & \frac{\delta{L}}{\delta{w^2_{23}}}
\end{array} \right)
$$

$$
B^2 =
\left( \begin{array}{ll}
b^2_1 & b^2_2
\end{array} \right)
,
\frac{\delta{L}}{\delta{B^2}} =
\left( \begin{array}{cc}
\frac{\delta{L}}{\delta{b^2_{1}}} & \frac{\delta{L}}{\delta{b^2_{2}}}
\end{array} \right)
$$


---

#### ニューラルネットワークの学習(計算式④)

さらに隠れ層①の行列式を各要素ごとに展開すると

$$
\begin{aligned}
W^1 &= 
\left( \begin{array}{ccc}
w^1_{11} & w^1_{21} & w^1_{31} \\
w^1_{12} & w^1_{22} & w^1_{32}
\end{array} \right)
\\
\frac {\delta L}{\delta W^1} &= \left( \begin{array}{ccc}
\frac {\delta L}{\delta w^1_{11}} & \frac {\delta L}{\delta w^1_{21}} & \frac {\delta L}{\delta w^1_{31}} \\
\frac {\delta L}{\delta w^1_{12}} & \frac {\delta L}{\delta w^1_{22}} & \frac {\delta L}{\delta w^1_{32}}
\end{array} \right)
\\
\\
B^1 &=
\left( \begin{array}{rrr}
b^1_1, b^1_2, b^1_3
\end{array} \right)
\\
\frac {\delta L}{\delta B^1} &=
\left( \begin{array}{rrr}
\frac {\delta L}{\delta b^1_1}, \frac {\delta L}{\delta b^1_2}, \frac {\delta L}{\delta b^1_3}
\end{array} \right)
\end{aligned}
$$

---
#### ニューラルネットワークの学習(まとめ)

ニューロンが増えれば増えるほど計算量も増える。

このまま計算するのは程計算では無理。

コンピュータでの計算も大変。

もっと簡単に計算出来ないか。

ということで、次章です。


<!--
### 以下内容は大事だけど入門には不要

- 学習アルゴリズム
 - ミニバッチ法
- データの分離
  - 訓練データとテストデータの２つのデータに分けて学習や実験を行う
- データの分離(訓練データ)
  - 訓練データ
    - 学習を行い、最適なパラメータを探索する
- データの分離(教師データ)
  - 訓練したモデルの実力を評価
  - 訓練データで評価するのでなく、新しいデータで学習結果を評価
  - 汎化能力を正しく評価する
- データの分離(過学習)
  - あるデータせっとにだけ過度に対応した状態
-->
