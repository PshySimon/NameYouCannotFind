# FastText

## 1.模型

fasttext与word2vec模型基本相似，但是fasttext对于word2vec模型未登陆词的问题做了改进，也就是采用了字符粒度的n-gram信息，即：比如apple的tri-gram是('[PRE]pp','app', 'ppl', 'ple', 'le[END]')，使用n-gram的好处在于：

+ 对于低频词生成的词向量效果会更好。因为它们的n-gram可以和其它词共享。

+  对于训练词库之外的单词，仍然可以构建它们的词向量。我们可以叠加它们的字符级n-gram向量。

fasttext使用的是层次Softmax

另一个不同点在于：

+ fasttext既可以和word2vec一样进行无监督训练，也可以利用文本的标签进行有监督训练，做简单的文本分类