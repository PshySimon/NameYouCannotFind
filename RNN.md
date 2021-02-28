# RNN

## 1.简单RNN

一个RNN网络由多个RNN_Cell组成，每个RNN_Cell都会输出一个Output和一个Hidden_State，Output和Hidden_State之间的关系就是：Output是将Hidden_State仿射变换之后然后激活函数激活之后得到的；此外，每个RNN_Cell的输入都是当前时间步输入的单词$X_t$和上一时间步的隐藏状态$H_{t-1}$，初始的隐藏状态$H_0$是随机初始化的，其中f一般是tanh，g一般是softmax函数：
$$
\begin{aligned}
O_t&=g(VH_t+b_2)\\
H_t&=f(WX_t+UH_{t-1}+b_1)
\end{aligned}
$$
这样，后一个RNN_Cell都能读的到上一时间步储存的状态，并且输出到下一时间步中

RNN可以解决的问题有：序列到类别模式、同步的序列到序列模式、异步的序列到序列模式

+ 序列到类别模式

  典型的应用有：文本分类，文本是一个典型的序列数据，每个时间步RNN都会融合当前词$X_t$的信息以及上一时间步也就是这个词之前的序列信息，最后输出的$H_t$就是整个句子的信息，经过输出层映射成类别向量$out\in R^C$，利用交叉熵损失函数判断类别

  除了将最后的状态作为句子编码的特征外，还有使用所有隐藏状态取平均的作为输入特征

+ 同步的序列到序列模式

  典型的应用有：序列标注，输入是长度为T的词汇序列$X=(x_1, x_2, ..., x_T)$输出是每个词汇对应的标签$Y=(y_1, y_2, ..., y_T)$

+ 异步的序列到序列模式

  典型的应用有：seq2seq模型，输入是长度为T的序列$X=(x_1, x_2,...,x_T)$，输出是长度为M的序列$Y=(y_1, y_2, ..., y_M)$，这个模型一般是通过一个叫编码器的循环神经网络编码句子序列X，然后将得到的隐藏层$H_t$作为另一个循环神经网络的隐藏层输入，这个循环神经网络叫做解码器，然后通过解码器生成新的序列，一般用于机器翻译这种模型



## 2.BPTT算法-RNN模型的反向传播算法

+ RNN正向传播

  对每个时刻t都有损失函数，其中$y_t^T$是one-hot向量，对应正确的标签处为1，其余为0；$\hat{y}_t$是经过softmax输出之后的概率：
$$
  L_t=L(y_t, g(H_t))=-y_t^T\log{\hat{y}_t}
$$
  对于整个时间序列：
$$
  L=\sum_{t=1}^TL_t
$$
  其他的符号：
$$
\begin{aligned}
  S_t&=UH_{t-1}+WX_t\\
  H_t&=tanh(S_t)\\
  Z_t&=VH_t\\
  \hat{y}_t&=softmax(Z_t)\\
  \end{aligned}
$$

  其中输入$X_t$是v维向量，$W$是$h\times v$维矩阵，$H_t$是$h\times 1$维矩阵，$U$是$h\times h$维矩阵，$V$是$v\times v$维矩阵


+ RNN反向传播
  $$
  \begin{aligned}
  \frac{\partial L_t}{\partial V}&=\frac{\partial L_t}{\partial Z_t}\frac{\partial Z_t}{\partial V}\\
  &=(\hat{y}_t-y_t)H_t^T
  
  \end{aligned}
  $$
  
  计算V的导数倒还好，但是计算U的导数时，由于每个时刻的隐藏状态依赖于上一状态，因此求导会变得更复杂，假设t=3：
  $$
  \begin{aligned}
  \frac{\partial L_3}{\partial U}&=\frac{L_3}{\partial {\hat{y_3}}}\frac{\partial\hat{y}_3}{\partial s_3}(\frac{\partial s_3}{\partial s_2}\frac{\partial s_2}{\partial s_1}\frac{\partial s_1}{\partial W}+\frac{\partial s_3}{\partial s_2}\frac{\partial s_2}{\partial W}+\frac{\partial s_3}{\partial W})\\
  
  \end{aligned}
  $$
  
  对于任意时刻：
  $$
  \frac{\partial L_t}{\partial U}=\sum_{k=0}^T\frac{\partial L_t}{\partial \hat{y}_t}\frac{\partial \hat{y}_t}{\partial s_t}(\prod_{j=k+1}^T\frac{\partial s_j}{\partial s_{j-1}})\frac{\partial s_k}{\partial W}
  $$
  

## 3.RNN的长程依赖问题

上面的BPTT反向传播过程中

## 4.RNN的改进：LSTM和GRU

### 4.1 LSTM

前面的简单RNN没办法针对长序列建模，LSTM网络引入了门控机制这一概念，分别是输入门、遗忘门和输出门：

+ 遗忘门控制上一时刻细胞状态需要遗忘多少信息
+ 输入门控制当前时刻候选细胞状态需要保存多少信息
+ 输出门控制当前细胞状态需要输出多少到隐藏状态

门控机制使用的激活函数是sigmoid，相当于做软门，0代表关闭，1代表开启

某一时刻LSTM_Cell的输入为当前时间步的输入$x_t$和上一时间步的隐藏状态$h_{t-1}$以及上一时间步的细胞状态$c_{t-1}$，输入首先经过三个门：
$$
\begin{aligned}
i_t&=\sigma(W_ix_t+U_ih_{t-1}+b_i)\\
f_t&=\sigma(W_fx_t+U_fh_{t-1}+b_f)\\
o_t&=\sigma(W_ox_t+U_oh_{t-1}+b_o)
\end{aligned}
$$
然后还要得出候选的记忆细胞状态：
$$
\hat{c_t}=tanh(W_cx_t+U_ch_{t-1}+b_c)
$$
生成当前记忆细胞状态和当前的隐藏层输出：

当前记忆细胞状态=上一记忆细胞状态\*需要遗忘的+本次候选的记忆细胞状态\*需要输入的
$$
c_t=f_t\bigodot c_{t-1}+i_t\bigodot \hat{c}_t
$$
隐藏层输出=tanh(当前记忆细胞状态)\*需要输出的
$$
h_t=o_t\bigodot tanh(c_t)
$$


### 4.2 GRU

GRU简化了LSTM，只使用了两个门，更新门和重置门：
$$
\begin{aligned}
r_t&=\sigma(W_rx_t+U_rh_{t-1}+b_r)\\
z_t&=\sigma(W_zx_z+U_zh_{t-1}+b_z)
\end{aligned}
$$
候选隐藏状态：
$$
\hat{h}_t=tanh(W_hx_t+U_h(r_t \bigodot h_{t-1})+b_h)
$$
输出的隐藏状态：
$$
h_t=z_t\bigodot h_{t-1}+(1-z_t)\bigodot \hat{h}_{t}
$$














































