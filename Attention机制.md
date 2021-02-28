# Attention机制

## 1.RNN、RNN变体

RNN的应用以及RNN的变体

+ N vs. N

  输入是序列$x = (x_1, x_2, ... ,x_t)$，输出也是登场的序列$y=(y_1, y_2,...,y_t)$

  这个模型的应用很受限，有：charRNN

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210222212051984.png" alt="image-20210222212051984" style="zoom:50%;" />

+ N vs. 1

  输入是序列$x = (x_1, x_2, ... ,x_t)$，输出是这个序列的类别：$c\in (0, 1, ..., C)$

  这种模型应用比较广泛，如：文本分类

  <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210222212255933.png" alt="image-20210222212255933" style="zoom:50%;" />

+ 1 vs.  N

  输入是一个样本X，输出是对应的序列$y=(y_1, y_2, ... , y_t)$

  应用有：根据图片生成文字

  <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210222212439598.png" alt="image-20210222212439598" style="zoom:50%;" />

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210222212505064.png" alt="image-20210222212505064" style="zoom:50%;" />

+ N vs. M

  输入是序列$x=(x_1, x_2, ..., x_t)$，输出是序列$y=(y_1, y_2, ..., y_t)$，这个模型又叫seq2seq模型

  应用有：机器翻译

  变长的序列用单个RNN模型不太好做，所以这里用的两个RNN模型，前面一个RNN叫做编码器，负责将序列x编码，生成一系列的隐藏状态$h=(h_1, h_2, ..., h_t)$，通过一个函数q将这些隐藏状态变换成一个背景变量c，即$c = q(h_1, h_2, ...,h_t)$，另一个RNN叫做解码器，用于接收编码器编码背景变量，然后解码输出

  <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210222213327069.png" alt="image-20210222213327069"  />

![image-20210222213350401](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210222213350401.png)



## 2.Seq2seq模型的缺陷

在Encoder-Decoder结构中，Encoder把所有的输入序列都编码成一个统一的语义特征c再解码，**因此， c中必须包含原始序列中的所有信息，它的长度就成了限制模型性能的瓶颈。**如机器翻译问题，当要翻译的句子较长时，一个c可能存不下那么多信息，就会造成翻译精度的下降。

一种解决方案就是引入attention机制，attention会在解码阶段，每个时间步输入不同的只和这个时间步需要的信息相关的背景变量c，也即在每个时刻给$h=(h_1, h_2, ...,h_t)$每个隐藏状态一个权重$a_{ij}$，其中i表示时间步，j表示隐藏状态的索引。

问题在于如何获取这些权重？这些权重是在训练过程中自动学习到的。

在前面没有加attention的seq2seq中，解码器在时间步$t'$的输出隐藏层状态为$h_{t'}=g(y_{t'-1},c_{t'},s_{t'-1})$，其中$y_{t'-1}$是解码器上一时间步的输出，$c_{t'}$是解码器在时间步$t'$的背景变量，$s_{t'-1}$是解码器上一时间步的隐藏状态，$c_{t'}$的计算如下：
$$
c_{t'}=\sum_{t=1}^Ta_{t't}s_t
$$
其中给定$t'$时，权重$a_{t't}$在$t=1,2,...,T$的值是一个概率分布
$$
a_{t't}=\frac{exp(e_{t't})}{\sum_{k=1}^Texp(e_{t'k})}
$$
那么$e_{t't}$是依赖解码器的上一步隐藏状态$s_{t'-1}$和编码器在时间步t时刻的隐藏状态$h_t$
$$
e_{t't}=a(s_t'-1, h_t)
$$
这里函数a的选择多种多样，一个简单的方法是：如果两个向量长度相同，可以直接使用内积的形式
$$
a(s,h)=s^Th
$$
最早提出的注意力机制则是经过变换后得到的：
$$
a(s,h)=v^Ttanh(Ws+Uh)
$$
其中v,W,U都是可学习的参数，上面计算s和h的内积之计算出了给定$t'$的情况下某一个时间步的分数，考虑一下矢量化的计算，一次性计算出T步的分数，并softmax归一化：
$$
a_{t'\cdot}=softmax(QK^T)
$$
其中Q是$1\times h$的向量，也即编码器上一时间步的隐藏状态$s_{t'-1}$，K是解码器所有时间步的隐藏状态纵向拼接成的$T\times h$的矩阵，这样的到的是解码器在时间步$t'$时各个解码器隐藏状态的权重，最后可以计算出当前时间步的背景变量c
$$
c = softmax(QK^T)V
$$
其中K和V相同，Q被称为查询矩阵，K被称为键，V被称为值



## 3. Transformer的提出

针对循环神经网络的两个缺点：

+ 时间片t依赖于上一时间片t-1，无法大量并行计算
+ bptt算法容易发生梯度爆炸或者消失，尽管LSTM、GRU能够缓解这种情况，但是面对长文本仍无法解决

Transformer完美解决了这两个问题：

+ Self-Attention机制将序列中任意两个位置之间的距离缩小为常量
+ 支持并行计算



### 3.1 Embedding层

#### 3.1.1 Word Embedding

词嵌入层是常规的词嵌入方法，即使用nn.Embedding，并且参数是可训练的

```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.lut(x)*math.sqrt(self.d_model)
```



#### 3.1.2 Positional Embedding

RNN的天然优势在于其对单词的处理是按顺序处理的，而transformer由于输入的句子都是同时处理的，没有考虑词的相对顺序，因此需要加入positional embedding来弥补这个缺陷。即某个单词的位置i对应了一个embedding，和词嵌入一样，也设置为d_model维，便于两者直接相加

如何获得positional embedding呢？有两种方式：

+ 使用公式直接计算
+ 通过训练获得

原论文发现两者效果相似，因此作者采用了公式计算：
$$
PE(pos, 2i)=sin(\frac{pos}{10000^{\frac{2i}{d\_model}}})\\
PE(pos, 2i+1)=cos(\frac{pos}{10000^{\frac{2i}{d\_model}}})
$$

+ 训练获得位置嵌入的代码

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        return pe(x)
```

+ 公式获得位置嵌入的代码

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=50000):
        super().__init__()
        self.dopout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *-(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.pe.requires_grad = False
        
   def forward(self, x):
    	x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)
```

为什么这两个公式能够体现单词位置的相对信息呢？



### 3.2 Encoder

Encoder由多个Encoder Layer组成，而Encoder Layer又由：Multi-Head Self-Attention和Feed-Forward Network组成

#### 3.2.1 Multi-Head Attention

多头自注意力由多个自注意力组成

+ Self Attention

  ​	顾名思义，就是自己注意自己，也即一个句子的每个单词与所有单词的相关程度，用的也是soft attention，前面word embedding和positional embedding相加得到的输入的维度是[batch_size, seq_len, d_model]，首先需要对输入进行线性变换，乘上[d_model, d_model]的矩阵，得到三个矩阵Q, K, V，然后计算得分，$Q\times K^T$，并针对每一行进行softmax，得到每个单词对于所有单词的相关程度的权重，最后乘上V得到当前注意的结果：
  $$
  \begin{aligned}
  Q&=XW^Q\\
  K&=XW^K\\
  V&=XW_V\\
  Attention(Q, K, V)&=softmax(\frac{QK^T}{\sqrt{d_k}})V
  \end{aligned}
  $$

  + $d_k$是多头自注意力中的某一个头的维度，论文中head是8， d_model是512，那么$d_k=\sqrt{\frac{512}{8}}=64$
  + 为什么要对注意力进行缩放？

+ Muilti-Head Attention

  多头自注意力模型就是将嵌入层分成8个，然后分别对其做self-attention之后再合并到一起

  ```python
  class MultiHeadAttention(nn.Module):
      def __init__(self, h, d_model, dropout):
          super().__init__()
          assert d_model % h == 0
          self.d_k = d_model // h
          self.h = h
          # Q,K,V分别需要一个线性变换的矩阵，最后还需要一个线性变换的矩阵
          self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model))] for _ in range(4))
          self.att = None
          self.dropout = nn.Dropout(dropout)
          
      def attention(self, query, key, value, mask):
          d_k = query.shape[-1]
          scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
          if mask is not None:
              scores = scores.maskfill(mask==0, -1e-9)
          p_attn = torch.softmax(scores, dim=-1)
          p_attn = self.dropout(p_attn)
          return	torch.matmul(p_attn, value), p_attn
          
      def forward(self, query, key, value, mask=None):
          if mask is not None:
              mask = mask.unsqueeze(1)
          nbatches = 	query.shape[0]
          
          # 获取Q,K,V
          # query=[batch_size, seq_len, d_model] -> query=[batch_size, head_num, seq_len, d_k]
          query, key, value = \
          	[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
              	for l, x in zip(self.linears, (query, key, value))]
          # 计算注意力以及结果:x=[batch_size, head_num, seq_len, d_k]
          # att=[batch_size, head_num, seq_len]
          x, self.attn = self.attention(query, key, value, mask=mask)
          # 最后合并多个头
          x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h*self.d_model)
          return self.linears[-1](x)
  ```


#### 3.2.2 Add & Norm

  序列x经过Multi-Head Attention之后还要经过一个add+norm层，即残差归一化连接

  ```python
  class SubLayerConnection(nn.Module):
      def __init__(self, size, dropout):
          self.norm = nn.LayerNorm(size)
          self.dropout = nn.Dropout(dropout)
          
      def forward(self, sublayer, x):
          return x + self.dropout(sublayer(self.norm(x)))
  ```

#### 3.2.3 FeedForward Network

FFN有两层，第一层是线性激活函数，第二层是ReLU激活函数

```python
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

最终Encoder是由N个encoder layer串行叠加的，每个encoder layer由Multi-Head Attention 和FeedForward Network构成

```python
class EncoderLayer(nn.Module):
    def __init__(self, size, attn, ffn, dropout):
        self.attn = attn
        self.ffn = ffn
        self.size = size
        self.sublayer = nn.ModuleList([SubLayerConnection(size, dropout) for _ in range(2)])
        
    def forward(self, x, mask):
        x = self.sub_layer[0](x, lambda x:self.self_attn(x, x, x, mask))
        return self.sub_layer[1](x, self.feed_forward)
        

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

### 3.3 Decoder



## 4.Transformer的优缺点

### 4.1 优点

　　（1）每层计算**复杂度比RNN要低**。

　　（2）可以进行**并行计算**。

　　（3）从计算一个序列长度为n的信息要经过的路径长度来看, CNN需要增加卷积层数来扩大视野，RNN需要从1到n逐个进行计算，而Self-attention只需要一步矩阵计算就可以。Self-Attention可以比RNN**更好地解决长时依赖问题**。当然如果计算量太大，比如序列长度N大于序列维度D这种情况，也可以用窗口限制Self-Attention的计算数量。

　　（4）Self-Attention**模型更可解释，Attention结果的分布表明了该模型学习到了一些语法和语义信息**。

### 4.2 缺点

　　在原文中没有提到缺点，是后来在Universal Transformers中指出的，主要是两点：

　　（1）实践上：有些RNN轻易可以解决的问题transformer没做到，比如**复制string**，或者推理时碰到的sequence长度比训练时更长（因为碰到了没见过的position embedding）。

　　（2）理论上：transformers不是computationally universal(图灵完备)，这种非RNN式的模型是非图灵完备的的，**无法单独完成NLP中推理、决策等计算问题**（包括使用transformer的bert模型等等）。

## 5. 面试题总结

+ 为什么Positional Embedding可以和Word Embedding相加？

  嵌入层的本质就是One-Hot作为输入全连接层，两个Embedding分别对应了两个One-Hot向量，而两者相加起来就等价于对两个one-hot特征进行拼接，然后经过Positional Embedding和Word Embedding的两个权重矩阵拼接而成的全连接层，而拼接也是特征融合的手段之一。

  

+ 为什么要使用多头自注意力而不是单个自注意力？

  多头自注意力保证了transformer可以注意到不同子空间的信息，捕捉到更加丰富的特征信息

  

+ 为什么使用不同的Q和K生成权重矩阵，而不能使用自身进行点乘？

  Transformer使用的是点积来获取attention，这样的话点积自己和自己相乘是最大的，会导致self attention更多的关注自己而不是关注其他单词，乘上不同的权重矩阵主要是为了保证在不同空间投影，增强表达能力



+ Transformer计算attention的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？

  Attention的两种计算方式：
  $$
  score(h, s)=<v, tanh(W_1h+W_2s)>\\
  score(h,s)=<W_1h,W_2s>
  $$
  可以看出Add的计算量更大，因此为了计算更快，使用了Mul的方式。效果上，作者做了做了对比，Add比Mul效果更好，作者认为是因为点积更容易出现很大的值，导致softmax计算时梯度容易消失，所以在点积attention的基础上除以了$\sqrt{d_k}$来减缓这个问题的发生



+ 为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根），并使用公式推导进行讲解

  需要推导两件事：

  + 数量级对softmax的梯度影响
  + 维度和点积大小的关系以及使用维度的根号缩放的原因

  

  对于向量$x=[x_1, x_2, ...x_n]$，且$max(x_1, x_2, ..,x_n)=x_i$，给定缩放因子k，$softmax(kx)$就倾向于在$x_i$的位置接近于1，其他位置接近于0，那么对于其梯度：
  $$
  \begin{aligned}
  softmax(x)&=\frac{\partial \frac{exp(x)}{1^Texp(x)}}{\partial x}\\
  &=\frac{1}{1^Texp(x)}\frac{\partial {exp(x)}}{\partial x}+\frac{\partial \frac{1}{1^Texp(x)}}{\partial x}[exp(x)]^T\\
  &=\frac{1}{1^Texp(x)}diag(exp(x))-\frac{1}{(1^Texp(x))^2}\frac{\partial (1^Texp(x))}{\partial x}[exp(x)]^T\\
  &=diag(softmax(x))-\frac{exp(x)[exp(x)]^T}{(1^Texp(x))^2}\\
  &=diag(softmax(x))-\frac{exp(x)[exp(x)]^T}{1^Texp(x)(1^Texp(x))^T}\\
  &=diag(softmax(x))-softmax(x)softmax(x)^T\\
  &=diag(y)-yy^T
  \end{aligned}
  $$
  计算出来梯度消失接近于0，造成参数更新困难
  
  
  
  假设向量q和k是各个分量相互独立的随机变量，均值是0，方差是1，那么点积的均值和方差分别是0和$d_k$，记$X=q_i,Y=k_i$
  $$
  E(XY)=E(X)E(Y)=0\\
  $$
  
  $$
  \begin{aligned}
  D(XY)&=E(X^2Y^2)-(E(XY))^2\\
  &=E(X^2)E(Y^2)//XY相互独立，则其对应的连续函数输出值也相互独立\\
  &=E(X^2-(E(X))^2)E(Y^2-(E(Y)^2))\\
  &=D(X)D(Y)\\
  &=1
  \end{aligned}
  $$
  
  设随机变量$Z_i=q_ik_i$，由上面的可以得知：
  $$
  \begin{aligned}
  E(Z)&=E(\sum_i Z_i)=0\\
  D(Z)&=D(\sum_i Z_i)=d_k
  \end{aligned}
  $$
  说明随机变量的方差和数量级$d_k$有关，故除以$\sqrt{d_k}$能减小方差到1
  
  
  
  得出的结论：
  
  + 在数量级很大时，最大的数对应的归一化概率会变得很接近1，梯度消失为0，造成参数更新困难
  + 文中假设两个向量的分量均是独立的随机变量，且均值为0，方差为1，可以推导出这两个向量的内积的均值为0，方差为$d_k$，方差越大就说明点积的数量级越大，一个自然的做法就是除以$\sqrt{d_k}$使得方差为1，有效控制梯度消失问题



+ 在计算attention score的时候如何对padding做mask操作？

  一般会给padding部分赋予一个非常小的值



+ 为什么在进行多头注意力的时候需要对每个head进行降维？

  在不增加时间复杂度的情况下，借鉴CNN多核的机制，在多个独立的特征空间里学到更丰富的信息



+ 大概讲一下Transformer的Encoder模块？

  Transformer的Encoder模块由词嵌入、位置嵌入和多层Encoder Layer组成，每个Encoder Layer都含有两个组件：多头自注意力和前馈网络，这两个层在后面连接的时候都使用层归一化和残差连接



+ 为何在获取输入词向量之后需要对矩阵乘以embedding size的开方？意义是什么？

  



+ 简单介绍一下Transformer的位置编码？有什么意义和优缺点？

  语言的位置信息也很重要，Transformer的位置编码就是将单词的位置信息做成嵌入层来表达。Self Attention只能捕捉任意两个单词之间的关系，但是不能捕捉到相对和绝对位置信息，引入位置编码弥补了将RNN换成self attention后不能捕捉句子的位置信息的缺陷。



+ 你还了解哪些关于位置编码的技术，各自的优缺点是什么？

  



+ 简单讲一下Transformer中的残差结构以及意义。

  让误差有两条通道传播，防止梯度消失，帮助训练更深层的网络



+ 为什么transformer块使用LayerNorm而不是BatchNorm？LayerNorm 在Transformer的位置是哪里？

  因为BatchNorm在处理文本数据时效果不好，BatchNorm是对不同样本每一维特征做归一化，而对于NLP来说，就是每句话相同位置的单词做归一化，这样做是没有实际意义的，因此对每一个句子特征去做归一化才是更好的选择



+ 简答讲一下BatchNorm技术，以及它的优缺点。

  + BatchNorm主要对每个样本每一个维度进行归一化
  + 优点：
    + 解决了内部协变量偏移的问题，加快网络收敛速度
    + 缓解了梯度饱和问题
  + 缺点：
    + batch_size小的时候，效果较差
    + 在RNN里面效果不好
    + 在训练和测试阶段代码不同



+ 简单描述一下Transformer中的前馈神经网络？使用了什么激活函数？相关优缺点？
  + 主要是两层线性变换，做完第一次线性变换之后就使用ReLU激活函数，得到输出后再使用线性变换变回原来的形状
  + 优点：计算高效
  + 缺点：非零中心化，导致后面一层的神经网络偏置偏移，影响梯度下降的效率



+ Encoder端和Decoder端是如何进行交互的？（在这里可以问一下关于seq2seq的attention知识）

  Encoder经过N层编码后输出结果，然后输入到Decoder里面做Encoder-Decoder Attention，过程和Self Attention类似，但是Q是来源于Decoder上一步的结果，K和V来源于Encoder的输出，这个和seq2seq类似，在Decoder的每个时间步t都会计算Encoder编码好的句子的各个部分的权重，然后加权求和，这样就得到了当前时间步应该注意到的部分



+ Decoder阶段的多头自注意力和encoder的多头自注意力有什么区别？（为什么需要decoder自注意力需要进行 sequence mask)

  Decoder的多头自注意力是Masked Multi Head Self-Attention，这样主要是为了盖住要预测的下一个单词以及后面的单词，防止模型偷看答案。



+ Transformer的并行化体现在哪个地方？Decoder端可以做并行化吗？

  Transformer的encoder layer都是串行的，而且encoder layer的多头自注意力和FFN网络也是串行的，但是多头注意力和FFN是可以并行计算的；而Decoder因为每次需要一个字一个字的输出，所以不能并行计算。



+ 简单描述一下wordpiece model 和 byte pair encoding，有实际应用过吗？

  



+ Transformer训练的时候学习率是如何设定的？Dropout是如何设定的，位置在哪里？Dropout 在测试的需要有什么需要注意的吗？

  + 采用warm_up策略，首先从0开始增加，增加到设定的阈值之后就开始减少
  + Dropout在Embedding后面、Attention后面以及FFN后面都加了Dropout，Dropout在训练和测试时要切换不同的mode



+ 引申一个关于bert问题，bert的mask为何不学习transformer在attention处进行屏蔽score的技巧？

  



+ 为什么要用三角函数做位置编码？

  用这个公式做位置编码并不一定是必须的，这只是论文作者的归纳偏置，只不过三角函数满足了两点作为位置编码的需求：(1)位置编码应该是有界的，且不能由于位置编码的大小不同影响其他的特征输入；(2)位置编码需要体现先后顺序，并且不能太过依赖文本长度，三角函数编码刚好满足了这两点要求；另外对坐标分组，作者认为这样可以学习相对位置的信息，因为两个位置pos和pos+k之间的关系可以通过和角差角公式变换得到























































