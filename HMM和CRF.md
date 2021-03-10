# HMM和CRF

## 1. HMM

### 1.1 模型

HMM模型的参数是三元组：$\lambda =(\pi, A, B)$，解决的问题是给定一个观测序列$O=(o_1, o_2, ..., o_T)$，根据参数生成不可观测的状态序列$I=(i_1, i_2, ...,i_T)$，其中$\pi$是状态序列初始的概率，A叫做转移矩阵，表示状态序列从上一状态到下一状态转移的概率$p(i_t|i_{t-1})$，B叫做发射矩阵，也叫观测概率矩阵，表示从当前状态转移到当前观测值的概率$p(o_t|i_t)$。假设每个状态有N种，观测值有M种，那么三个参数的形状分别是：$(1\times N),(N\times N),(N\times M)$

+ HMM三个假设
  + 马尔科夫假设：t时刻状态只与t-1时刻有关
  + 齐次性假设：马尔科夫链的t时刻的状态只依赖于前一时刻t-1的状态，与其他时刻的观测值和状态以及时刻无关
  + 观测独立性假设：某个时刻t的观测值只依赖于时刻t的状态，与其他观测值和假设无关

+ HMM基本的三个问题

  + 评估问题：已知参数$\lambda=(\pi, A, B)$和观测序列$O(o_1, o_2, ...,o_T)$，计算观测序列出现的概率

    + 穷举出所有可能的状态序列，然后根据状态序列才能计算出这个观测序列出现的概率

    $$
    p(O|\lambda)=\sum_{I}p(O|I, \lambda)p(I|\lambda)
    $$

  + 预测问题：已知参数$\lambda=(\pi, A, B)$和观测序列$O(o_1, o_2, ...,o_T)$，计算最有可能的状态序列

    + 需要用到动态规划的思想

  + 参数学习问题



### 1.2 评估问题

+ 直接求解

  假设$I$代表长度为T的状态序列的所有可能的集合，一共$N^T$种
  $$
  \begin{aligned}
  p(O|\lambda)&=\sum_{I}p(O|I,\lambda)p(I|\lambda)\\
  &=\sum_{I}\pi_{i_1}b_{i_1}(o_1)a_{i_1,i_2}b_{i_2}(o_2)...a_{i_{T-1},i_T}b_{i_T}(o_T)
  \end{aligned}
  $$
  计算的时间复杂度为：$O(T\times N^T)$



+ 前向算法：

  直接求解会有大量重复计算，这里采用前向算法，从长度为1的序列算起，每次记录上一次以$q_k$为最后状态的序列的概率

  首先计算长度为1的序列的概率：
  $$
  \alpha_{i_1}=\pi_{i_1}b_{i_i}(o_1)
  $$
  生成的是一个N维向量，即长度为1的序列的所有可能序列的概率，再计算长度为2的：
  $$
  \alpha_{i_2}=\alpha_{i_1}\bigodot a_{i_1,i_2}b_{i_2}(o_2)
  $$
  这样每次计算引用前一次的计算结果，最终可以计算出：
  $$
  p(O|\lambda)=\sum\alpha_{i_T}
  $$
  时间复杂度变成$O(N^2T)$

+ 后向算法同理



### 1.3 预测序列

+ 简单求法：

  每一时刻都设为概率最大的状态，但是这并不是全局最优解

+ 动态规划：

  每一时刻保留最优的N条路径，然后计算最优路径和下一时刻各个状态转移概率的概率值，选出使得这条路径最大的状态作为N条路径中最优的状态。设$\delta_t(i)$表示t时刻的第i条最优路径的概率值，$\phi_t(i)$表示i时刻使得路径最优的
  
  先计算初始状态：第一个状态的初始概率和从第一个状态转移到第一个观测值的概率相乘
  $$
  \delta_1(i)=\pi_{i}b_i(o_1)\\
  $$
  得到一个N维向量，便于下一次计算最优路径；然后计算t=2时刻的最优路径：
  $$
  \delta_2(i)=[(\delta_1(j)a_{ji})b_i(o_2)](j=1, 2...N)
  $$
  由上式可以计算出$N^2$条路径，此时可以确定N条局部最优的路径，也就是使得在时刻t=2时概率最大的上一状态j：
  $$
  \delta_2(i)=max([(\delta_1(j)a_{ji})b_i(o_2)])\\
  \phi_2(i)=arg \max_j([(\delta_1(j)a_{ji})b_i(o_2)])
  $$
  依次计算t时刻在状态i下使得概率最大的上一状态j，得到t从1到T-1时刻的N条最优路径和概率，最后通过计算最后一个时刻的状态的最优路径和概率，得出结果：
  $$
  \delta_T(i)=max[(\delta_{T}(j)a_{ji})b_i(o_{T})]\\
  \phi_T(i)=arg \max_j[(\delta_{T}(j)a_{ji})b_i(o_{T})]
  $$



## 2. CRF

CRF最典型的应用就是词性标注问题，给定一个句子序列，序列中每个单词都对应了一个词性。一句话的词性有很多种组合，词性标注的目标就是挑选出最靠谱的标注序列。如何判断一个序列靠不靠谱呢？例如：动词不应该接动词，这样的序列组合就打负分，因为动词后接动词不靠谱，动词后面接动词就是一个特征函数，特征函数用于给序列的局部打分。仅靠当前单词和前面一个单词是一种简单的CRF的情况，叫线性链CRF。

+ 状态转移特征函数

  + 特征函数接收四个参数：
    + s：句子序列
    + i：句子中单词的索引i
    + $y_i$：表示要评分的标注序列的第i个标注的词性
    + $y_{i-1}$：同上
  + 输出值：为0或者1，0表示不符合特征，1表示符合特征

  + 对于某一个序列，假设有m个特征函数，那么序列l的分数为：

  $$
  score(l|s)=\sum_{j=1}^m\sum_{i=1}^n\lambda_jt_j(s, i, y_i, y_{i-1})
  $$

  + 某个序列$l_k$出现的概率为：
    $$
    p(l|s)=\frac{exp(score(l|s))}{\sum_{l'}exp(score(l'|s))}=\frac{exp(\sum_{j=1}^m\sum_{i=1}^n\lambda_it_j(s, i, y_i, y_{i-1}))}{\sum_{l'}exp(\sum_{j=1}^m\sum_{i=1}^n\lambda_it_j(s, i, y_i, y_{i-1}))}
    $$

+ 状态特征函数：
  + 三个参数
    + s：句子序列
    + i：句子中单词的索引i
    + $y_i$：表示要评分的标注序列的第i个标注的词性

+ CRF和HMM的关系：HMM模型等价于CRF，CRF可以定义更丰富的特征函数，比HMM效果更好

+ 条件随机场是判定模型，因此计算的是条件概率：

$$
P(y|x)=\frac{exp(\sum_{i,k}\lambda_kt_k(y_{i-1}, y_i, s, i)+\sum_{i, l}\mu_is_l(y_i,s, i))}{\sum_{y}exp(exp(\sum_{i,k}\lambda_kt_k(y_{i-1}, y_i, s, i)+\sum_{i, l}\mu_is_l(y_i,s, i)))}
$$



### 2.1 biLSTM-CRF

LSTM的输入是[batch_size, seq_len, emb_dim]，输出是[batch_size, seq_len, hidden_size]，然后通过映射层将输出的最后一个维度投射到n_tags的维度，表示每个单词各个tag的得分。还有一个参数矩阵为状态转移矩阵，矩阵的参数是可学习的，不过有两个特殊的状态转移，从某一个状态转移到开始状态和从结束状态转移到某个状态是不可能的，因此默认设置为一个很小的负数。因此分数等于输出的分数+状态转移分数。

对于长度为n的序列x，每个位置有m种标注，那么就有$m^n$个可能的标记结果，用模型计算出每个标注结果的分数，利用$P(y|x)=\frac{exp(score)}{Z}$计算概率，并且选出最大概率的标注作为结果。模型有两个任务，一个是预测结果：采用维特比算法求解当前参数下的最优路径和最优得分，另一个是求损失函数：需要计算所有路径得分和以及真实标签的得分

需要关注的三个问题：

+ 对于输入x，如何计算输出序列y的概率

  LSTM输出数据后经过线性层映射为[seq_len, n_tags]的矩阵，即一句话每个单词可能的标签，这个矩阵又叫发射矩阵。对于每个单词$x_i$其可能的对应的标签$e_j$，这个得分只是当前状态的得分，还要加上状态转移的得分，需要在模型中声明一个[n_tags, n_tags]的矩阵，并且这个矩阵的参数是自己学习的，这样一句话的得分就可以按如下方式计算：
  $$
  score(y)=\sum_{i=1}^n(e_i)+\sum_{i=2}^n(T(i-1, i))
  $$
  最终所要求的目标是条件概率，其中分子的y是真实的标签：
  $$
  P(y|x)=\frac{exp(score(y))}{\sum_{y}exp(score(y))}
  $$
  损失函数为负的对数似然函数：
  $$
  L=-\log{P(y|x)}
  $$
  那么需要求的部分为：
  $$
  L=\log(\sum_yexp(score(y)))-score(y)
  $$
  明确需要有三个函数：给定句子序列和其对应标注求得分、根据LSTM输出的发射矩阵输出和状态转移矩阵求最优路径得分和最优路径、求损失函数的分子分母

  求所有路径的得分和可以用前向传播的方式计算，时间复杂度从$O(m^n)$降到$O(mn^2)$：

  + 先计算初始状态点的得分，是一个向量，记做$\alpha$，形状为[1, n_tags]
  + 变量$\alpha$是用于记录所有路径到当前状态的路径和，便于下一状态使用
  + 每次求只需要求log_sum_up即可



+ 给定训练数据(x,y)，如何对模型进行训练？

  定义负对数似然函数作为模型的损失函数训练，深度学习框架自动反向传播。

  

+ 对于训练好的模型，给定输入x，如何求最有可能的结果$y'$？

  维特比算法，每次求各路径到当前状态的最优得分，并且记录上一步的最优状态。



```python
START_TAG, END_TAG = "<s>", "<e>"

class BiLSTM_CRF(nn.Module):
    def __init__(self, tag2ix, word2ix, embedding_dim, hidden_dim):
        """
        :param tag2ix: 序列标注问题的 标签 -> 下标 的映射
        :param word2ix: 输入单词 -> 下标 的映射
        :param embedding_dim: 喂进BiLSTM的词向量的维度
        :param hidden_dim: 期望的BiLSTM输出层维度
        """
        super(BiLSTM_CRF, self).__init__()
        assert hidden_dim % 2 == 0, 'hidden_dim must be even for Bi-Directional LSTM'
        self.embedding_dim, self.hidden_dim = embedding_dim, hidden_dim
        self.tag2ix, self.word2ix, self.n_tags = tag2ix, word2ix, len(tag2ix)

        self.word_embeds = nn.Embedding(len(word2ix), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.n_tags)  # 用于将LSTM的输出 降维到 标签空间
        # tag间的转移score矩阵，即CRF层参数; 注意这里的定义是未转置过的，即"i到j"的分数(而非"i来自j")
        self.transitions = nn.Parameter(torch.randn(self.n_tags, self.n_tags))
        # "START_TAG来自于?" 和 "?来自于END_TAG" 都是无意义的
        self.transitions.data[:, tag2ix[START_TAG]] = self.transitions.data[tag2ix[END_TAG], :] = -10000

    def neg_log_likelihood(self, words, tags):  # 求一对 <sentence, tags> 在当前参数下的负对数似然，作为loss
        frames = self._get_lstm_features(words)  # emission score at each frame
        gold_score = self._score_sentence(frames, tags)  # 正确路径的分数
        forward_score = self._forward_alg(frames)  # 所有路径的分数和
        # -(正确路径的分数 - 所有路径的分数和）;注意取负号 -log(a/b) = -[log(a) - log(b)] = log(b) - log(a)
        return forward_score - gold_score

    def _get_lstm_features(self, words):  # 求出每一帧对应的隐向量
        # LSTM输入形状(seq_len, batch=1, input_size); 教学演示 batch size 为1
        embeds = self.word_embeds(self._to_tensor(words, self.word2ix)).view(len(words), 1, -1)
        # 随机初始化LSTM的隐状态H
        hidden = torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2)
        lstm_out, _hidden = self.lstm(embeds, hidden)
        return self.hidden2tag(lstm_out.squeeze(1))  # 把LSTM输出的隐状态张量去掉batch维，然后降维到tag空间

    def _score_sentence(self, frames, tags):
        """
        求路径pair: frames->tags 的分值
        index:      0   1   2   3   4   5   6
        frames:     F0  F1  F2  F3  F4
        tags:  <s>  Y0  Y1  Y2  Y3  Y4  <e>
        """
        tags_tensor = self._to_tensor([START_TAG] + tags, self.tag2ix)  # 注意不要+[END_TAG]; 结尾有处理
        score = torch.zeros(1)
        for i, frame in enumerate(frames):  # 沿途累加每一帧的转移和发射
            score += self.transitions[tags_tensor[i], tags_tensor[i + 1]] + frame[tags_tensor[i + 1]]
        return score + self.transitions[tags_tensor[-1], self.tag2ix[END_TAG]]  # 加上到END_TAG的转移

    def _forward_alg(self, frames):
        """ 给定每一帧的发射分值; 按照当前的CRF层参数算出所有可能序列的分值和，用作概率归一化分母 """
        alpha = torch.full((1, self.n_tags), -10000.0)
        alpha[0][self.tag2ix[START_TAG]] = 0  # 初始化分值分布. START_TAG是log(1)=0, 其他都是很小的值 "-10000"
        for frame in frames:
            # log_sum_exp()内三者相加会广播: 当前各状态的分值分布(列向量) + 发射分值(行向量) + 转移矩阵(方形矩阵)
            # 相加所得矩阵的物理意义见log_sum_exp()函数的注释; 然后按列求log_sum_exp得到行向量
            alpha = log_sum_exp(alpha.T + frame.unsqueeze(0) + self.transitions)
        # 最后转到EOS，发射分值为0，转移分值为列向量 self.transitions[:, [self.tag2ix[END_TAG]]]
        return log_sum_exp(alpha.T + 0 + self.transitions[:, [self.tag2ix[END_TAG]]]).flatten()

    def _viterbi_decode(self, frames):
        backtrace = []  # 回溯路径;  backtrace[i][j] := 第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是神马状态
        alpha = torch.full((1, self.n_tags), -10000.)
        alpha[0][self.tag2ix[START_TAG]] = 0
        for frame in frames:
            # 这里跟 _forward_alg()稍有不同: 需要求最优路径（而非一个总体分值）, 所以还要对smat求column_max
            smat = alpha.T + frame.unsqueeze(0) + self.transitions
            backtrace.append(smat.argmax(0))  # 当前帧每个状态的最优"来源"
            alpha = log_sum_exp(smat)  # 转移规律跟 _forward_alg()一样; 只不过转移之前拿smat求了一下回溯路径

        # 回溯路径
        smat = alpha.T + 0 + self.transitions[:, [self.tag2ix[END_TAG]]]
        best_tag_id = smat.flatten().argmax().item()
        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]):  # 从[1:]开始，去掉开头的 START_TAG
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        return log_sum_exp(smat).item(), best_path[::-1]  # 返回最优路径分值 和 最优路径

    def forward(self, words):  # 模型inference逻辑
        lstm_feats = self._get_lstm_features(words)  # 求出每一帧的发射矩阵
        return self._viterbi_decode(lstm_feats)  # 采用已经训好的CRF层, 做维特比解码, 得到最优路径及其分数

    def _to_tensor(self, words, to_ix):  # 将words/tags序列数值化，即: 映射为相应下标序列张量
        return torch.tensor([to_ix[w] for w in words], dtype=torch.long)
```



### 2.2 CRF++

+ CRF++的特征模板可以定义更丰富的特征函数，不过CRF最高也只是用到了二阶马尔科夫假设，并没有实现更高的HMM假设
+ unigram模板：
  + 用U开头，后面两位数字是特征函数的编号
  + "%x"代表当前的这句话，CRF++存储文件的格式是每一行有一个token和其对应的tag
  + [row, col]表示与当前token相对的行偏移和列偏移，[0, 0]表示就是当前词的位置
  + 如：U00:%x[-2, 0]，“人民网”,“BME”，当前词的位置是"网"，那么这个特征函数此时就会去做如下判断
    + 如果当前词的标注为'E'，前面第二个词为"人"，那么就返回1；否则返回0
  + 还可以组合特征
  + 如：U01:%x[-1, 0]/%x[0, 0]/%x[1, 0]，表示同时判断当前词、前一个词、后一个词以及当前词的词性是否同时出现
+ bigram模板：
  + 用B开头，后面两位数字是特征编号，与unigram不同的是，如果只用B不加其他的代表只判断当前词性和前面一个词性的关系，不考虑词
  + 带上%x[a, b]时，就会在考虑周围词性的关系和词的关系，不过由于词性是预测目标，只能利用前面已经得到的词性，用不了后面还未解码的词性
  + 特征组合，如：B00:%x[-1, 0]/%x[0, 0]表示同时考虑前面词性为'M'当前词性为'E'，前面词为"网"后面词为"民"

CRF++可以定义更加丰富的特征函数，因此HMM可以看做CRF的特例，可以通过特征模板表示出HMM

























































