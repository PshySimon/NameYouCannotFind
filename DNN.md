# 前馈神经网络

## 1.激活函数

**激活函数** 

激活函数在神经元中非常重要的．为了增强网络的表示能力和学习能力，激活函数需要具备以下几点性质： 

（1） 连续并可导（允许少数点上不可导）的非线性函数．可导的激活函数可以直接利用数值优化的方法来学习网络参数． 

（2） 激活函数及其导函数要尽可能的简单，有利于提高网络计算效率． 

（3） 激活函数的导函数的值域要在一个合适的区间内，不能太大也不能太小，否则会影响训练的效率和稳定性．

### 1.1 Sigmoid函数

$$
\sigma(x)=\frac{1}{1+exp(-x)}
$$

Logistic 函数可以看成是一个“挤压”函数，把一个实数域的输入“挤压”到(0, 1)．当输入值在0附近时，Sigmoid型函数近似为线性函数；当输入值靠近两端时，对输入进行抑制．输入越小，越接近于 0；输入越大，越接近于 1．和感知器使用的阶跃激活函数相比，Logistic函数是连续可导的，其数学性质更好．因为Logistic函数的性质，使得装备了Logistic激活函数的神经元具有以下两点性质：

1）其输出直接可以看作概率分布，使得神经网络可以更好地和统计学习模型进行结合．

2）其可以看作一个软性门（Soft Gate），用来控制其他神经元输出信息的数量． 



### 1.2 Tanh函数

$$
tanh(x)=\frac{exp(x)-exp(-x)}{exp(x)+exp(-x)}
$$

Tanh函数可以看作放大并平移的Logistic函数，其值域是(−1, 1)．tanh(𝑥) = 2𝜎(2𝑥) − 1.Tanh 函数的输出是零中心化的（Zero-Centered），而 Logistic 函数的输出恒大于 0． 非零中心化的输出会使得其后一层的神经元的输入发生偏置偏移（Bias Shift），并进一步使得梯度下降的收敛速度变慢。



### 1.3 ReLU函数

$$
ReLU(x)=\max{(0, x)}
$$

**优点** 

采用 ReLU 的神经元只需要进行加、乘和比较的操作，计算上更加高效．Sigmoid 型激活函数会导致一个非稀疏的神经网络，而 ReLU 却具有很好的稀疏性，大约50%的神经元会处于激活状态．在优化方面，相比于Sigmoid型函数的两端饱和，ReLU函数为左饱和函数，且在 𝑥 > 0 时导数为 1，在一定程度上缓解了神经网络的梯度消失问题，加速梯度下降的收敛速度． 

**缺点** 

ReLU 函数的输出是非零中心化的，给后一层的神经网络引入偏置偏移，会影响梯度下降的效率．ReLU 神 经 元 指 采 用ReLU作为激活函数的神经元．此外，ReLU 神经元在训练时比较容易“死亡”．在训练时，如果参数在一次不恰当的更新后，第一个隐藏层中的某个 ReLU 神经元在所有的训练数据上都不能被激活，那么这个神经元自身参数的梯度永远都会是0， 在以后的训练过程中永远不能被激活．这种现象称为死亡 ReLU 问题（DyingReLU Problem），并且也有可能会发生在其他隐藏层．

## 2.前馈神经网络

| 记号                               | 含义                     |
| ---------------------------------- | ------------------------ |
| L                                  | 神经网络层数             |
| $M_l$                              | 第l层神经元的个数        |
| $f_l(\cdot)$                       | 第l层的激活函数          |
| $W^{(l)}\in R^{M_l\times M_{l-1}}$ | 第l-1层到第l层的权重矩阵 |
| $b^{(l)}\in R^{M_l}$               | 第l-1层到第l层的偏置     |
| $z_l\in R^{M_l}$                   | 第l层神经元的净输入      |
| $\alpha_l\in R^{M_l}$              | 第l层神经元的输出        |

$$
z^{(l)}=W^{(l)}f_{l-1}(z^{(l-1)})+b^{(l)}
$$

$$
\alpha^{(l)}=f_l(W^{(l)}\alpha^{(l-1)}+b^{(l)})
$$



**通用近似定理**：神经网络可以拟合任意一个连续函数

在机器学习中，输入样本的特征对分类器的影响很大．以监督学习为例，好的特征可以极大提高分类器的性能．因此，要取得好的分类效果，需要将样本的原始特征向量𝒙转换到更有效的特征向量𝜙(𝒙)，这个过程叫作特征抽取． 



+ **构建计算图自动微分**：

  对复合函数$f(x;w,b)=\frac{1}{exp(-(wx+b))+1}$求导

  首先构造计算图：
  $$
  \begin{aligned}
  h_1&=wx\\
  h_2&=h1+b\\
  h_3&=-h_2\\
  h_4&=exp(h_3)\\
  h_5&=h_4+1\\
  h_6&=\frac{1}{h_5}
  \end{aligned}
  $$
  给定$(x=1;w=0,b=0)$

  + 前向模式求$\frac{\partial f(x;w,b)}{\partial w}$：
    $$
    \begin{aligned}
    \frac{\partial h_1}{\partial w}&=x=1\\
    \frac{\partial h_2}{\partial w}&=\frac{\partial h_2}{\partial h_1}\frac{\partial h_1}{\partial w}=1\times 1=1\\
    \frac{\partial h_3}{\partial w}&=\frac{\partial h_3}{\partial h_2}\frac{\partial h_2}{\partial w}=-1\times 1=-1\\
    \frac{\partial h_4}{\partial w}&=\frac{\partial h_4}{\partial h_3}\frac{\partial h_3}{\partial w}=exp(0)\times (-1)=-1\\
    \frac{\partial h_5}{\partial w}&=\frac{\partial h_5}{\partial h_4}\frac{\partial h_4}{\partial w}=1\times(-1)=-1\\
    \frac{\partial h_6}{\partial w}&=\frac{\partial h_6}{\partial h_5}\frac{\partial h_5}{\partial w}=-\frac{1}{2^2}\times(-1)=0.25\\
    \frac{\partial f(x;w,b)}{\partial w}&=\frac{\partial f(x;w,b)}{\partial h_6}\frac{\partial h_6}{\partial w}=1\times 0.25=0.25
    \end{aligned}
    $$

  + 反向模式求$\frac{\partial f(x;w,b)}{\partial w}$
    $$
    \begin{aligned}
    \frac{\partial f(x;w,b)}{\partial h_6}&=1\\
    \frac{\partial f(x;w,b)}{\partial h_5}&=\frac{\partial f(x;w,b)}{\partial h_6}\frac{\partial h_6}{\partial h_5}=1\times (-\frac{1}{2^2})=-0.25\\
    \frac{\partial f(x;w,b)}{\partial h_4}&=\frac{\partial f(x;w,b)}{\partial h_5}\frac{\partial h_5}{\partial h_4}=-0.25\times 1=-0.25\\
    \frac{\partial f(x;w,b)}{\partial h_3}&=\frac{\partial f(x;w,b)}{\partial h_4}\frac{\partial h_4}{\partial h_3}=-0.25\times exp(0)=-0.25\\
    \frac{\partial f(x;w,b)}{\partial h_2}&=\frac{\partial f(x;w,b)}{\partial h_3}\frac{\partial h_3}{\partial h_2}=-0.25\times (-1)=0.25\\
    \frac{\partial f(x;w,b)}{\partial h_1}&=\frac{\partial f(x;w,b)}{\partial h_2}\frac{\partial h_2}{\partial h_1}=0.25\times 1=0.25\\
    \frac{\partial f(x;w,b)}{\partial w}&=\frac{\partial f(x;w,b)}{\partial h_1}\frac{\partial h_1}{\partial w}=0.25\times1=0.25
    \end{aligned}
    $$
    两者比较，计算同一个参数的微分的时间复杂度相同，但是前向模式的中间步骤都是对w求导，没有可以用于链式法则重复计算的部分；而后向模式计算浅层网络的微分时，深层网络作为中间结果自动被求出来了，效率更高。



**反向传播算法**：

假设输入$x=[i_1,i_2]=[0.05, 0.10]$，权重$W_1=[[w_1, w_3],[w_2, w_4]]=[[0.15, 0.25],[0.20, 0.30]],b_1=[0.35]$和$ W_2=[[w_5, w_7], [w_6, w_8]]=[[0.40, 0.50],[0.45, 0.55]], b_2=[0.60], target=[o_1, o_2]=[0.01, 0.99]$，激活函数都取sigmoid函数

+ 前向传播：

  由于偏置的形状为[1, 1]，而$xW$的形状为[2, 1]，不能相乘，需要利用广播机制先将偏置复制成[2, 1]之后才能继续计算

  $out = \sigma(\sigma(xW_1+b_1)W_2+b_2)=[0.751365069, 0.772928765]$

  误差的计算：

  $E=\frac{1}{2}\sum_{i=1}^n(target_i-out_i)^2=0.298371109$

+ 反向传播：

  首先明确参数是权重和偏置，也就是这些参数需要求导，其他参数不需要求导
  $$
  \begin{aligned}
  E &= \frac{1}{2}(target_1 - out_1)^2+\frac{1}{2}(target_2-out_2)^2\\
  &=\frac{1}{2}(target_1-net_{o1}(out_3))^2+\frac{1}{2}(target_2-net_{o2}(out_4))\\
  &=\frac{1}{2}(target_1-(w_5\times out_3 + w_6\times out_3 + b_2))^2+\frac{1}{2}(target_2-(w_7\times out_4 + w_8\times out_4 + b_2))^2
  \end{aligned}
  $$
  

  从后往前计算导数：
  $$
  \frac{\partial E}{\partial W_2}=
  \left[
      \begin{array}{ccc}
          \frac{\partial E}{\partial w_5}& \frac{\partial E}{\partial w_7} \\
          \frac{\partial E}{\partial w_6} & \frac{\partial E}{\partial w_8} \\
      \end{array}
  \right]
  =
  \left[
      \begin{array}{ccc}
          \frac{\partial E}{\partial out_1}\frac{\partial out_1}{\partial net_{o1}}\frac{\partial net_{o1}}{\partial w_5}& \frac{\partial E}{\partial out_2}\frac{\partial out_2}{\partial net_{o2}}\frac{\partial net_{o2}}{\partial w_7}\\
          \frac{\partial E}{\partial out_1}\frac{\partial out_1}{\partial net_{o1}}\frac{\partial net_{o1}}{\partial w_6}& \frac{\partial E}{\partial out_2}\frac{\partial out_2}{\partial net_{o2}}\frac{\partial net_{o2}}{\partial w_8} \\
      \end{array}
  \right]
  $$
  其他参数计算同理，首先可以计算出$[\frac{\partial E}{\partial out_1}, \frac{\partial E}{\partial out_2}]$，然后依次计算$[\frac{\partial out_1}{\partial net_{o1}},\frac{\partial out_2}{\partial net_{o2}}]$，最后计算$\frac{\partial net_{o1}}{\partial w_5}, \frac{\partial net_{o1}}{\partial w_6},\frac{\partial net_{o2}}{\partial w_7},\frac{\partial net_{o2}}{\partial w_8}$

## 3. 梯度爆炸/消失问题

### 3.1 原因

+ 神经网络层数过多

  神经网络层数过深，梯度会衰减至0
+ 使用了不恰当的激活函数、初始化权重过大
  
  + 一些激活函数如：sigmoid和tanh的导数的范围本身就是小于1的，在链式法则传播误差时会越来越小



### 3.2 解决方案

+ 梯度裁剪
+ 改用relu等其他激活函数
+ batchnorm
+ 残差结构

+ LSTM

















































