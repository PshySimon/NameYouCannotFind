# Word2Vec

## 1.概念

![image-20210217150325362](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210217150325362.png)

+ 词典：统计所有出现过的单词，词典记为Vocab，词典索引编号从1到V
+ 词表示：对于一个词，可以用对应的One-Hot向量表示，即一个V维的向量，除了这个词的位置是1以外，其他位置都是0
+ 输入: word2vec有两种方式，一种是Skip-Gram，另一种是CBOW，两种方式的输入不同
  + CBOW：输入是某个中心词的上下文，假设有C个单词，需要先将这C个单词转换成One-Hot向量，然后将这C个One-Hot向量拼接成一个C*V维向量
  + Skip-Gram：输入是中心词，只有一个单词，也即这个单词的One-Hot向量
+ 输出：
  + CBOW：输出是一个V维的向量，经过Softmax归一化成概率之后，概率值最大的那个对应的词就是要预测的中心词
  + Skip-Gram：输出是一个C个V维的向量，分别对其Softmax归一化之后概率最大的那个就是要预测的背景词
+ 投影层(Projection Layer)：位于输入层和隐藏层之间，是一个形状为V*N的矩阵，由V个N维向量组成，每个向量代表对应索引的单词的词向量
+ 隐藏层：是一个形状为N*V的矩阵



各符号的含义：

+ 单词：$w_I$表示输入的词向量

+ 输入单词One-Hot向量：x
  
+ $x_k$表示向量中值为1的索引是k，其余的是0
  
+ 投影层：W

+ 词向量：$h=W^Tx=W^T_{k,}:=v_{wI}$

+ 隐藏层：$W'$

  + $v'_{wj}$是矩阵$W'$的第j列

+ Softmax输出的第j个词的分值为：$u_j=v'_{w_j}h$

+ 输出的后验概率：$y_j$表示最后的输出的第j个元素

  + $$
    p(w_j|w_I)=y_j=\frac{exp(u_j)}{\sum_{j'=1}^Vexp(u_j')}
    $$

  + $$
    p(w_j|w_I)=y_j=\frac{exp(v'^T_{wj}v_{wI})}{\sum_{j'=1}^Vexp(v'_{wj'}v_{wI})}
    $$

+ 训练目标：最大化$p(w_O|w_I)$

+ 损失函数推导：
  $$
  \begin{aligned}
  -L(y_j, y_{j*})&=max(p(w_O|w_I))\\
  &=max(y_{j*})\\
  &=max(\log{y_{j*}})\\
  &=u_{j*}-\log{\sum_{j'=1}^Vexp(u_j')}
  \end{aligned}
  $$

+ 梯度下降：

  对输出得分求导， 其中$t_j=1$ if $j = j*$ else 0
  $$
  \begin{aligned}
  \frac{\partial L}{\partial u_j}&=y_j-t_j=e_j
  \end{aligned}
  $$
  对隐藏层求导
  $$
  \frac{\partial L}{\partial w'_{ij}}=\frac{\partial L}{\partial u_j}\frac{\partial u_j}{\partial w'_{ij}}=e_j\frac{\partial{(v'^T_{wj}v_{wI})}}{\partial{w'_{ij}}}=e_jh_i
  $$
  对投影层求导
  $$
  \frac{\partial L}{\partial h_i}=\sum_{j=1}^V\frac{\partial L}{\partial u_j}\frac{\partial u_j}{\partial h_i}=\sum_{j=1}^Ve_jw'_{ij}
  $$

## 2.前向传播

词向量是取lookup table矩阵W，其由V个N维向量组成；而后面的隐藏层则是为了将N维向量变成V维，便于进行多分类。

### 2.1 CBOW

+ 输入是C个背景词，将这C个背景词转变成One-Hot向量$x_1, x_2, ..., x_C$

+ 求这C个One-Hot向量投影出的词向量，然后求加权平均和
  $$
  h = \frac{1}{C}W^T(x_1+x_2+...+x_C)=\frac{1}{C}(v_{w1}+v_{w2}+...+v_{wC})^T
  $$

+ 损失函数：
  $$
  \begin{aligned}
  L(y, y^*)&=-\log{p(w_O|w_I)}\\
  &=-u_{j*}+\log{\sum_{j'=1}^Vexp(u_j')}\\
  &=-v'^T_{w_O}h+\log{\sum_{j'=1}^Vexp(v'^T_{wj}h)}
  \end{aligned}
  $$

+ 前向传播
  $$
  \begin{aligned}
  Out = Softmax(\frac{1}{C}W'^TW^T(x_1+x_2+..+x_C))\\
  X\in R^{V\times 1}, W\in R^{V\times N}, W'\in R^{N\times V}, Out\in R^{V\times1}
  \end{aligned}
  $$

### 2.2 Skip-Gram

+ 输入是1个中心词

+ Skip-Gram模型假设给定中心词的情况下，背景词的生成式相互独立的

+ 损失函数：
  $$
  \begin{aligned}
  L(y, y^*)&=-\log{p(w_{O1}, w_{O2},...w_{OC}|w_{I})}\\
  &=-\log{\prod_{c=1}^Cp(w_{Oc}|w_I)}\\
  &=-\log{\prod_{c=1}^C\frac{exp(u_{c,j*_c})}{\sum_{j'=1}^Vexp(u_{j'})}}\\
  &=-\sum_{c=1}^Cu_{c,j*_c}+C\log{\sum_{j'=1}^Vexp(u_{j'})}
  \end{aligned}
  $$

+ 前向传播：
  $$
  \begin{aligned}
  Out = Softmax(\frac{1}{C}W'^TW^Tx)\\
  X\in R^{V\times 1}, W\in R^{V\times N}, W'\in R^{N\times V}, Out\in R^{V\times1}
  \end{aligned}
  $$



## 3. 模型的改进

### 3.1 原模型的问题

+ 在一次参数更新的过程中，只有少数几个单词需要更新参数，绝大部分其他单词对应的梯度为0，但是每次参数更新还是计算所有参数的梯度
+ 目标函数是Softmax函数，每次都需要将结果的指数和加起来再计算得分



### 3.2 负采样

原模型考虑的问题是每次计算所有单词的得分，负采样改变思路，考虑一个事件D：背景词是否出现在中心词附近，该事件的概率为：
$$
p(D=1|w_c,w_o)=\sigma(u^T_ou_c)
$$
那么计算联合概率为：
$$
\prod_{c=1}^Cp(D=1|w_o,w_c)
$$
但是上面的公式只考虑了正样本，这导致了优化会趋向于结果都是无穷大，这样每个背景词出现在中心词附近的概率都是1，也就毫无意义。负采样通过添加负样本使得目标函数有意义，假设背景词$w_o$出现在中心词$w_c$为事件p，根据分布p(w)采样k个噪声词：
$$
\prod_{c=1}^Cp(D=1|w_o,w_c)\prod_{k=1}^Kp(D=0|w_o,w_k)
$$
这样每次计算的开销就只与K有关，不需要计算整个矩阵

+ **负采样的细节**

  + 高频词抽样：在有些样本中有很多意义不大的词，并且这些词出现的频率还很高，含有高频词的样本数量远超过训练这个词需要训练的次数，word2vec通过抽样来解决这个问题，它的基本思想如下：**对于我们在训练原始文本中遇到的每一个单词，它们都有一定概率被我们从文本中删掉，而这个被删除的概率与单词的频率有关。**在采样时，每个单词被保留的概率如下：
    $$
    p(w_i)=(\sqrt{\frac{z(w_i)}{sample}}+1)\times\frac{sample}{z(w_i)}
    $$
    其中$z(w_i)$表示单词$w_i$在语料中出现的频率

  + **负采样**的目的是让每一次训练仅仅更新一小部分权重，通过负采样之后只需要更新正样本和k个负样本的对应的权重。负样本的采样依据的公式是：
    $$
    p(w_i)=\frac{f(w_i)^{\frac{3}{4}}}{\sum_{j=0}^n(f(w_ji)^{\frac{3}{4}})}
    $$
    其中$f(w_i)$表示单词$w_i$出现的频率，计算出所有单词的采样概率保存到sample_weights中，然后采样k个不是中心词的词作为噪声词，频率的0.75次方是谷歌实验得出的最好的数据

  + 最终的预测目标是**已知中心词预测给定单词是背景词还是噪声词**，是二分类，最终的损失函数是二元交叉熵

### 3.3 层次Softmax

首先根据词典中每个词的词频构造出一棵哈夫曼树，保证词频较大的词在浅层，词频较小的在深层，每个词都处于树的叶节点。原本的V分类问题简化为log(V)分类问题，树的每个非叶节点都进行了一次逻辑回归。这棵哈夫曼树除了根结点以外的所有非叶节点中都含有一个由参数θ确定的sigmoid函数，不同节点中的θ不一样。训练时隐藏层的向量与这个sigmoid函数进行运算，根据结果进行分类，若分类为负类则沿左子树向下传递，编码为0；若分类为正类则沿右子树向下传递，编码为1。



## 4. 面试题整理

+ 推导word2vec

+ word2vec与tfidf、NNLM、fasttext、glove、 LSA的区别？
  + word2vec与tfidf计算相似度时的区别：
    + word2vec是低维的稠密的向量，tfidf是高维稀疏的向量
    + word2vec可以计算单词之间的相似度，tfidf不可以
    + word2vec表达能力和泛化能力更强，tfidf和语料有很大程度关系
  + word2vec和NNLM的区别：
    + NNLM的目标是训练语言模型，词向量只是副产物；word2vec的目标更专注于训练词向量
    + 输入方面，word2vec没有拼接One-Hot向量，而是用加权平均和，并且word2vec省去了隐藏层
    + word2vec优化了softmax的计算，利用负采样和层次softmax提高了模型的计算效率
  + word2vec和fasttext的区别：
    + 都能无监督训练，但是fasttext考虑到了subword
    + fasttext还可以做有监督学习，做文本分类
  
+ 负采样有什么作用？
  
+ 训练的时候，每次只需要更新少量参数
  
  + 将softmax计算变成sigmoid，减少了计算softmax得分的计算量
  
+ 层次softmax的作用

  + softmax多分类的计算时间复杂度是O(V)，而层次softmax的时间复杂度是O(log V)，加快了计算速度

+ word2vec的缺点

  + 忽略了词序信息
  + 没办法解决一词多义的现象

+ word2vec参数量的计算

  投影层和隐藏层的参数量共：2\*V\*N

+ Skip-Gram和CBOW哪个更好，为什么？

  (1) 训练速度上 CBOW 应该会更快一点。因为每次会更新 context(w) 的词向量，而 Skip-gram 只更新核心词的词向量。两者的预测时间复杂度分别是 O(V)，O(KV)
  (2) Skip-gram 对低频词效果比 CBOW好。因为是尝试用当前词去预测上下文，当前词是低频词还是高频词没有区别。但是 CBOW 相当于是完形填空，会选择最常见或者说概率最大的词来补全，因此不太会选择低频词。Skip-gram 在大一点的数据集可以提取更多的信息。
  
+ 负采样为什么要用词频来做采样？

  + 优先学习词频高的词向量，带动低频词

+ 训练完成后有两套词向量，为什么一般只用前一套？

  + 因为在滑动窗口滑动过程中，先前的中心词也会变成后面的背景词，所以两套比较相近，都可以用

+ 词向量和字向量对比

  + 字向量可以解决 未登录词的问题，避免分词带来的误差
  + 词向量能学到更多的语义

+ 层次softmax为什么要用哈夫曼树？其他树不可以吗？

  + 因此哈夫曼树构建时是按照词频优先构建的，词频高的离根节点越近，优化高频词的计算量
  + 其他树也可以，不一定要用哈夫曼树

+ word2vec处理高低频词

  + 高频词：高频词每个词都有一定概率被删掉，概率与这个词出现在语料中的频率有关
  + 低频词：给定阈值，出现次数低于这个阈值的词就会被删除













































