# Crispr名词解释

# 名词解释

protospacer: 在CRISPR系统中，protospacer指的是目标DNA序列，这部分序列被CRISPR系统中的RNA分子识别，指导CRISPR相关蛋白（Cas蛋白）进行剪切。

![](https://lgb1ternmf.feishu.cn/space/api/box/stream/download/asynccode/?code=MzI5NjFhNGJkNDUxYjE5OWFkYzA5MTQxYzI1N2Y5NGJfa1NDd2xNWVRZSmlCZWI5NFNVWllHYXhyQ3FnZTNoUnVfVG9rZW46VTZBZWI0SXFUb2tWWDR4SWpTMGNycktPbm5nXzE2ODU4NTkyODI6MTY4NTg2Mjg4Ml9WNA)

# 基于注意力的神经网络

## 1.1 Per-base模型概述

我们设计并实现了一个多头自注意力模型（命名为BE-DICT），灵感来源于Transformer [5] 编码器架构。BE-DICT使用PyTorch [6] 进行实现，其输入是一系列核苷酸序列（即使用长度为20bp的目标DNA序列），输出则是每个目标核苷酸的编辑概率。我们的实验中，ABEmax和ABE8e两种编辑器的目标核苷酸是*A* ，CBE4max和Target-AID两种编辑器的目标核苷酸是C

> We designed and implemented a multi-head self-attention model (named BE-DICT) inspired by the Transformer [5] encoder architecture. BE-DICT is implemented in PyTorch [6] and takes a sequence of nucleotides (i.e. using protospacer sequence of 
>
>  window) as input and computes the probability of editing for each target nucleotide as output. The target nucleotides in our experiments were base **
> $$
> A$$** for ABEmax and ABE8e editors, and  **
> $$
>
> C
> $$
> ** **
> $$
>
> C$$** for CBE4max and Target-AID base editors respectively.

该模型主要有三个部分：(1) 嵌入块，将核苷酸及其对应位置从one-hot 编码表示转换为稠密向量表示。

(2)编码器模块, 包括一个自注意力层(支持多头) , 层归一化与残差连接最后在与一个前馈神经网络连接

(3)输出模块, 包含一个位置注意力层和一个分类器层

每个模块的不同部分将在下面的相应部分中进行描述

> The model has three main blocks: An (1) Embedding block that embeds both the nucleotide's and its corresponding position from one-hot encoded representation to a dense vector representation.
> An (2) Encoder block that contains (a) a self-attention layer (with multi-head support), (b) layer normalization \& residual connections $(\rightarrow)$, and 
>
>  feed-forward network.
> Lastly, an (3) Output block that contains (a) a position attention layer and (b) a classifier layer.
> A formal description of each component of the model is described in their respective sections below.

> 未使用one-hot, 只是类别编码, 不会增加特征的数量，而且不会对特征之间的关系产生影响  data_preprocess.py 中的baseseq_df.replace(['A', 'C', 'T', 'G'], [0,1,2,3], inplace=True)
>
> nn.Embedding 是否会出现每个单词的行向量相同的情况

## 1.2 嵌入块

形式上, 在给定的原型间隔子序列  

, 位置t的核苷酸由核苷酸由 1-of-K 编码表示, 其中 K 是数据中所有核苷酸字母集合的大小, 使



, 嵌入矩阵**
$$
W_e
$$

**用于将**
$$
x_t
$$

**映射到固定长度的向量表示(Eq.1)


(1)

> 将嵌入矩阵**
>
> $$
> W_e
> $$
>
> **映射到对应的**
> $$
> x_t
> $$
>
> **, 将**
> $$
> x_t
> $$
>
> **映射成固定长度的向量表示

其中**

$$
W_{e} \in \mathbb{R}^{d_{e} \times K}, e_{t} \in \mathbb{R}^{d_{e}}
$$

**

, e_t 输入 R de, de是向量e_t的维数

类似的, 序列S中每个核苷酸的位置pt由 1-of-T 编码表示, 其中T是序列元素的数量(即原间隔序列的长度), 使得pt属于[0, 1]T, 嵌入矩阵Wp' 用于将输入pt映射到固定长度的向量表

**$$p_t^{\prime}=W_{p^{\prime}} p_t

$$
**(2)



其中**$$W_e \in \mathbb{R}^{d_p{\prime} \times T}$$**，**$$p_t^{\prime}\in$$** ∈ Rdp'，dp'是向量p't的维度，使得de和dp'相等（现在用d表示）。

将嵌入et和p't进行求和（方程式3），以获得序列S中每个元素的统一表示（即计算一个新序列U = [u1, u2, ⋯, uT]，其中ut ∈ Rd，∀t ∈ [1, ⋯, T]）。

**$$u_t=e_t+p_t^{\prime}$$**





## 1.3 编码器块

1.3.1





# reference

[2]A. C. Komor, K. T. Zhao, M. S. Packer, N. M. Gaudelli, A. L. Waterbury, L. W. Koblan, Y. B. Kim, A. H. Badran, and D. R. Liu, "Improved base excision repair inhibition and bacteriophage Mu Gam protein yields C:G-to-T:A base editors with higher efficiency and product purity," vol. 3, no. 8, p. eaao4774, aug 2017. [Online]. Available: https://advances.sciencemag.org/content/3/8/eaao4774https: //advances.sciencemag.org/content/3/8/eaao4774.abstract



[5] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention Is All You Need," jun 2017. [Online]. Available: http://arxiv.org/abs/1706.03762

[6] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. Devito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer, "Automatic differentiation in pytorch," 2017.
$$


protospacer: 在CRISPR系统中，protospacer指的是目标DNA序列，这部分序列被CRISPR系统中的RNA分子识别，指导CRISPR相关蛋白（Cas蛋白）进行剪切。

![](https://lgb1ternmf.feishu.cn/space/api/box/stream/download/asynccode/?code=OGI3ZDBiMWEyODlkYWQ5ODdiYmQ1ZjczMzMwZjA0MzRfR1JwbTRjSkxmSnVWZzJ1T3N0SW5TVTYxa2d3Nzc3OERfVG9rZW46VTZBZWI0SXFUb2tWWDR4SWpTMGNycktPbm5nXzE2ODU4NTkxMTE6MTY4NTg2MjcxMV9WNA)

# 基于注意力的神经网络

## 1.1 Per-base模型概述

我们设计并实现了一个多头自注意力模型（命名为BE-DICT），灵感来源于Transformer [5] 编码器架构。BE-DICT使用PyTorch [6] 进行实现，其输入是一系列核苷酸序列（即使用长度为20bp的目标DNA序列），输出则是每个目标核苷酸的编辑概率。我们的实验中，ABEmax和ABE8e两种编辑器的目标核苷酸是*A* ，CBE4max和Target-AID两种编辑器的目标核苷酸是C

> We designed and implemented a multi-head self-attention model (named BE-DICT) inspired by the Transformer [5] encoder architecture. BE-DICT is implemented in PyTorch [6] and takes a sequence of nucleotides (i.e. using protospacer sequence of
>
> window) as input and computes the probability of editing for each target nucleotide as output. The target nucleotides in our experiments were base **
>
> $$
> A$$** for ABEmax and ABE8e editors, and **
> $$
>
> C$$** for CBE4max and Target-AID base editors respectively.

该模型主要有三个部分：(1) 嵌入块，将核苷酸及其对应位置从one-hot 编码表示转换为稠密向量表示。

(2)编码器模块, 包括一个自注意力层(支持多头) , 层归一化与残差连接最后在与一个前馈神经网络连接

(3)输出模块, 包含一个位置注意力层和一个分类器层

每个模块的不同部分将在下面的相应部分中进行描述

> The model has three main blocks: An (1) Embedding block that embeds both the nucleotide's and its corresponding position from one-hot encoded representation to a dense vector representation.
> An (2) Encoder block that contains (a) a self-attention layer (with multi-head support), (b) layer normalization \& residual connections $(\rightarrow)$, and
>
> feed-forward network.
> Lastly, an (3) Output block that contains (a) a position attention layer and (b) a classifier layer.
> A formal description of each component of the model is described in their respective sections below.

> 未使用one-hot, 只是类别编码, 不会增加特征的数量，而且不会对特征之间的关系产生影响  data_preprocess.py 中的baseseq_df.replace(['A', 'C', 'T', 'G'], [0,1,2,3], inplace=True)
>
> nn.Embedding 是否会出现每个单词的行向量相同的情况

## 1.2 嵌入块

形式上, 在给定的原型间隔子序列

, 位置t的核苷酸由核苷酸由 1-of-K 编码表示, 其中 K 是数据中所有核苷酸字母集合的大小, 使

**

$$
x_t \in [0,1]^K
$$

**, 嵌入矩阵**

$$
W_e
$$

**用于将**

$$
x_t
$$

**映射到固定长度的向量表示(Eq.1)

(1)

> 将嵌入矩阵**
>
> $$
> W_e
> $$
>
> **映射到对应的**
>
> $$
> x_t
> $$
>
> **, 将**
>
> $$
> x_t
> $$
>
> **映射成固定长度的向量表示

其中**

$$
W_{e} \in \mathbb{R}^{d_{e} \times K}, e_{t} \in \mathbb{R}^{d_{e}}
$$

**

, e_t 输入 R de, de是向量e_t的维数

类似的, 序列S中每个核苷酸的位置pt由 1-of-T 编码表示, 其中T是序列元素的数量(即原间隔序列的长度), 使得pt属于[0, 1]T, 嵌入矩阵Wp' 用于将输入pt映射到固定长度的向量表

**$$p_t^{\prime}=W_{p^{\prime}} p_t

$$
**(2)



其中**$$W_e \in \mathbb{R}^{d_p{\prime} \times T}$$**，**$$p_t^{\prime}\in$$** ∈ Rdp'，dp'是向量p't的维度，使得de和dp'相等（现在用d表示）。

将嵌入et和p't进行求和（方程式3），以获得序列S中每个元素的统一表示（即计算一个新序列U = [u1, u2, ⋯, uT]，其中ut ∈ Rd，∀t ∈ [1, ⋯, T]）。

**$$u_t=e_t+p_t^{\prime}$$**





## 1.3 编码器块

1.3.1





# reference

[2]A. C. Komor, K. T. Zhao, M. S. Packer, N. M. Gaudelli, A. L. Waterbury, L. W. Koblan, Y. B. Kim, A. H. Badran, and D. R. Liu, "Improved base excision repair inhibition and bacteriophage Mu Gam protein yields C:G-to-T:A base editors with higher efficiency and product purity," vol. 3, no. 8, p. eaao4774, aug 2017. [Online]. Available: https://advances.sciencemag.org/content/3/8/eaao4774https: //advances.sciencemag.org/content/3/8/eaao4774.abstract



[5] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention Is All You Need," jun 2017. [Online]. Available: http://arxiv.org/abs/1706.03762

[6] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. Devito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer, "Automatic differentiation in pytorch," 2017.
$$

protospacer: 在CRISPR系统中，protospacer指的是目标DNA序列，这部分序列被CRISPR系统中的RNA分子识别，指导CRISPR相关蛋白（Cas蛋白）进行剪切。

# 基于注意力的神经网络

## 1.1 Per-base模型概述

我们设计并实现了一个多头自注意力模型（命名为BE-DICT），灵感来源于Transformer [5] 编码器架构。BE-DICT使用PyTorch [6] 进行实现，其输入是一系列核苷酸序列（即使用长度为20bp的目标DNA序列），输出则是每个目标核苷酸的编辑概率。我们的实验中，ABEmax和ABE8e两种编辑器的目标核苷酸是*A* ，CBE4max和Target-AID两种编辑器的目标核苷酸是C

> We designed and implemented a multi-head self-attention model (named BE-DICT) inspired by the Transformer [5] encoder architecture. BE-DICT is implemented in PyTorch [6] and takes a sequence of nucleotides (i.e. using protospacer sequence of
>
> window) as input and computes the probability of editing for each target nucleotide as output. The target nucleotides in our experiments were base **
>
> $$
> A$$** for ABEmax and ABE8e editors, and **
> $$
>
> C$$** for CBE4max and Target-AID base editors respectively.

该模型主要有三个部分：(1) 嵌入块，将核苷酸及其对应位置从one-hot 编码表示转换为稠密向量表示。

(2)编码器模块, 包括一个自注意力层(支持多头) , 层归一化与残差连接最后在与一个前馈神经网络连接

(3)输出模块, 包含一个位置注意力层和一个分类器层

每个模块的不同部分将在下面的相应部分中进行描述

> The model has three main blocks: An (1) Embedding block that embeds both the nucleotide's and its corresponding position from one-hot encoded representation to a dense vector representation.
> An (2) Encoder block that contains (a) a self-attention layer (with multi-head support), (b) layer normalization \& residual connections $(\rightarrow)$, and
>
> feed-forward network.
> Lastly, an (3) Output block that contains (a) a position attention layer and (b) a classifier layer.
> A formal description of each component of the model is described in their respective sections below.

> 未使用one-hot, 只是类别编码, 不会增加特征的数量，而且不会对特征之间的关系产生影响  data_preprocess.py 中的baseseq_df.replace(['A', 'C', 'T', 'G'], [0,1,2,3], inplace=True)
>
> nn.Embedding 是否会出现每个单词的行向量相同的情况

## 1.2 嵌入块

形式上, 在给定的原型间隔子序列

, 位置t的核苷酸由核苷酸由 1-of-K 编码表示, 其中 K 是数据中所有核苷酸字母集合的大小, 使得

, 嵌入矩阵**

$$
W_e
$$

**用于将**

$$
x_t
$$

**映射到固定长度的向量表示(Eq.1)

(1)

> 将嵌入矩阵**
>
> $$
> W_e
> $$
>
> **映射到对应的**
>
> $$
> x_t
> $$
>
> **, 将**
>
> $$
> x_t
> $$
>
> **映射成固定长度的向量表示

其中**

$$
W_{e} \in \mathbb{R}^{d_{e} \times K}, e_{t} \in \mathbb{R}^{d_{e}}
$$

**

, e_t 输入 R de, de是向量e_t的维数

类似的, 序列S中每个核苷酸的位置pt由 1-of-T 编码表示, 其中T是序列元素的数量(即原间隔序列的长度), 使得pt属于[0, 1]T, 嵌入矩阵Wp' 用于将输入pt映射到固定长度的向量表

**$$p_t^{\prime}=W_{p^{\prime}} p_t

$$
**(2)



其中**$$W_e \in \mathbb{R}^{d_p{\prime} \times T}$$**，**$$p_t^{\prime}\in$$** ∈ Rdp'，dp'是向量p't的维度，使得de和dp'相等（现在用d表示）。

将嵌入et和p't进行求和（方程式3），以获得序列S中每个元素的统一表示（即计算一个新序列U = [u1, u2, ⋯, uT]，其中ut ∈ Rd，∀t ∈ [1, ⋯, T]）。

**$$u_t=e_t+p_t^{\prime}$$**





## 1.3 编码器块

1.3.1





# reference

[2]A. C. Komor, K. T. Zhao, M. S. Packer, N. M. Gaudelli, A. L. Waterbury, L. W. Koblan, Y. B. Kim, A. H. Badran, and D. R. Liu, "Improved base excision repair inhibition and bacteriophage Mu Gam protein yields C:G-to-T:A base editors with higher efficiency and product purity," vol. 3, no. 8, p. eaao4774, aug 2017. [Online]. Available: https://advances.sciencemag.org/content/3/8/eaao4774https: //advances.sciencemag.org/content/3/8/eaao4774.abstract



[5] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention Is All You Need," jun 2017. [Online]. Available: http://arxiv.org/abs/1706.03762

[6] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. Devito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer, "Automatic differentiation in pytorch," 2017.
$$
