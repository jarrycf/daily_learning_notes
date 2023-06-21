

![](media/1687338097566-8ce95bba-4202-44c9-99df-29a2b0214596.png)

<a name="qtJ5R"></a>

## 直观认识


![](media/1687338097563-aaad8c17-a2eb-4cc5-9f3d-05c6618bfaaa.png)<br />结束符 词语接龙解码器<br />why do we work? 翻译成 为什么我们要工作?<br />S => 为<br />S为 => 什<br />S为什 => 么<br />...<br />S 为什么我们要工作? => E

这个结构又和chatgpt有什么关系?

![](media/1687338097656-3990153b-74ca-47c5-8f44-d56f379ccbeb.png)<br />不再是S<br />为什么我们要工作? => 为<br />为什么我们要工作?为 => 了<br />为什么我们要工作?为了 => 实<br />....<br />为什么我们要工作?为了实现个人目标 => E<br />为了实现个人目标 

这就是我们GPT的结构

<a name="dcuha"></a>

## 前提知识


- seq2seq
- 注意力机制



seq2seq是解决什么问题的<br />机器翻译任务: <br />输入是一段英文，输出是一段中语，输入和输出皆不定长<br />eg. 英语 5：Why do we work ? 中文 8：我们为什么工作?<br />当输入输出序列都是不定长时，我们可以使用编码器 - 解码器（encoder-decoder）或者说 seq2seq。

注意力机制是解决什么问题的?<br />输入序列 "Why do we work " 和输出序列 "我们为什么工作?"，<br />我们词的语接龙解码器是使用更多的 "we" 上下文向量来生成 "工作"这个想要的翻译结果，还是使用更多的 "work" 上下文向量来生成 "工作"。给每个词分配不同的注意力, 这就是注意力机制的由来.

总结<br />Seq2Seq 模型中的编码器可以处理任意长度的输入序列，并生成一个固定长度的上下文向量。解码器可以根据上下文向量逐步生成输出序列，同时使用注意力机制来关注输入序列的不同部分。

<a name="v31vM"></a>

## 数据预处理


德语翻译英文的任务来带大家通过pytorch实现一个300行的Transformer模型架构

任务: <br />ich mochte ein bier 我喜欢喝啤酒

ich mochte ein bier P => i want a beer . E<br />ich mochte ein cola P => i want a cola . E

<a name="SbNYF"></a>

### 建语表和编码


![](media/1687338097646-b97fc5fb-6f74-44e5-9cbb-c9fc7776c6ed.png)<br />我希望大家可以记住enc_input的维度是2*5<br />![](media/1687338097630-67a8aa04-5fc3-48bd-9498-304deb5f6288.png)

```
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
src_idx2word = {src_vocab[key]: key for key in src_vocab}
src_vocab_size = len(src_vocab) #6

tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab) #9

# src_len = 5 # enc_input max sequence length
# tgt_len = 6 # dec_input(=dec_output) max sequence length
src_len = len(sentences[0][0].split(" ")) # 5                                          
tgt_len = len(sentences[0][1].split(" ")) # 6
```


![](media/1687338098213-b706254a-0589-4062-9ca7-addc6839186e.png)

<a name="yDBqw"></a>

### 超参数

![](media/1687338098253-38489a2b-35c4-4bdb-9c9d-11ce4fe57def.png)

<a name="AKBUU"></a>

## Encoder


![](media/1687338098386-15c7dc33-8658-4f66-a585-73e34833bdd5.png)

<a name="DodfM"></a>

### 建表和查表

![](media/1687338098485-e67b32a7-c652-4ffb-a84d-9eadf8a12f41.png)<br />![](media/1687338098785-390e9bc8-0907-4608-a268-a65a552eb0fd.png)

![](media/1687338098748-92663cb5-3ba6-4f02-abc4-903a34f49313.png)

![](media/1687338098955-6e52bc7f-c877-46b7-8610-f29b796aba7f.png)

为什么需要进行位置编码?<br />因为Transformer不像RNN, RNN的结构觉得了必须有前一刻的输出, 才能进行下一时刻的输入, Transformer是并行输入的, 所以我们需要位置编码来表示序列的顺序这一特征

序列的顺序特征是什么?<br />从北京到广州和从广州到北京只是把词的位置互换就表示了不同的意思.<br />也就是我们需要编码来表示每一个词的先后顺序<br />这就是位置编码

那Transformer的位置编码是如何实现的呢?

<a name="LHhUR"></a>

### 位置编码

![](media/1687338099055-f08a4245-ffb7-4a12-a5e5-de695af06beb.png)

这种位置编码的规律

1. 每一个行向量在位置编码表中都唯一, 且具体的值只受d_model(列)的影响, 不受行数的影响
2. 不管d_model取什么值, 每个行向量的变化趋势是固定的: 只有前一半的列数值变化剧烈



![](media/1687338099361-5c0ef745-2123-42e5-8156-1213fe3fe819.png)![](media/1687338099386-a21da630-4f77-4ad2-b6a5-61dea83a6d57.png)<br />![](media/1687338099573-ef36f4ca-6cdc-456d-9e2c-a9cacaf177ca.png)

公式中i的取值范围是多少?<br />![](media/1687338099706-bed82018-f11e-4a29-80fe-fe659d47f188.png)

公式如何在python中编码实现?<br />![](media/1687338099856-077fe211-4ca3-472d-b2c4-9ba01479f1c8.png)

$$PE{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}}) \\ PE{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$

$$\text{torch.exp}(\text{torch.arange}(0, d_\text{model}, 2).float() \times \frac{-\ln(10000.0)}{d_\text{model}})$$

![](media/1687338100095-a1cbb021-40d1-4e3e-89eb-12330c9bf45b.png)

<a name="VC4q4"></a>

### input Embedding + Postional Encoding

![](media/1687338100120-98792d46-52e0-4b07-9084-f5fbc4bbb255.png)

![](media/1687338100549-4a5959b3-6f8f-4067-a19c-4245e30dc3ad.png)

![](media/1687338100647-875af1bf-a011-46de-8e44-8b49d53429f0.png)

什么是Self Attention?

![](media/1687338100628-4452a9d9-c66f-464c-8632-ab010b76ef36.png)

X是如何衍生出Q, K, V的?<br />对X做了一个3个不同线性变换nn.linear 4 => 64 得到Q, K, V<br />3维张量进行输入 2*5*4 => 2*5*64<br />#d_k = d_v = 64 # dimension of K(=Q), V

![](media/1687338101055-430b622d-a33b-4d63-93f4-f5d91631231d.png)

![](media/1687338101409-dd46b5cd-02cc-4c43-ad3b-2854b61b374d.png)<br />Scaled Dot-Product Attention<br />![](media/1687338101222-52691507-3980-4e5d-a228-0961118b63bd.png)

为什么需要除以根号dk?<br /># 将score的方差变小

为什么要做注意力机制?<br />给定一个 X，通过Q, K, V相乘得到一个 context ，这个 context 就是对 X 的新的表征（词向量），Z 这个词向量相比较 X 拥有了句法特征(词性)和语义特征, 能够捕捉输入序列中的长距离依赖关系<br />![](media/1687338101381-963ca326-efb2-49f9-baf7-d7c408680446.png)

什么是多头注意力机制?<br />Multi-Head Attention就是把X分成了8小块(H头), Scaled Dot-Product Attention的过程做8次，然后把8个输出Z合起来

<a name="qnYsF"></a>

### 计算出多头Q, K, V


将一个2x5x4的的张量， 经过 nn.linear 变为2*5*512的q, view 后又变为2x8x5x64, 这个升维的操作可以看作是将原先2x5x4的张量复制了8份还是分成了8小块?<br />![](media/1687338101597-0058faa1-8ff5-4aea-953a-969c48c22fa0.png)<br />分成了8小块

<a name="vBQdL"></a>

### 生成pad的掩码


为什么要生成pad的掩码?<br />![](media/1687338101714-3aded52e-8327-42b3-b16b-c943d3c2f47d.png)

![](media/1687338101852-30eac0ac-653e-480e-997d-32af6cfc96ed.png)

<a name="gAuU2"></a>

### 缩放点积注意力

![](media/1687338102024-4abe0f9b-ac56-4a37-944e-02aa08b9800f.png)

![](media/1687338102029-aadd1a5b-8e9a-405f-8846-ea90e8ccc2f0.png)

![](media/1687338102487-387e8538-d0f0-45ae-8da3-e91fbad18ba8.png)<br />图: Attention is All you Need** 原论文**

![](media/1687338102483-f4f3290a-c9d9-4e52-bbd6-fd7a761b75e0.png)

<a name="WvddN"></a>

### Add & Norm


![](media/1687338102798-311f48a4-62d4-40c0-8500-2f8d0eeebafb.png)

![](media/1687338102968-96f06741-3d2f-47ef-99eb-248bc3908f07.png)

<a name="bd77p"></a>

### 前馈神经网络


![](media/1687338102894-22df0e7b-82cf-4d00-8066-19c4a92e4847.png)

为什么要叫做前馈神经网络?<br />循环神经网络<br />卷积神经网络<br />生成对抗网络<br />自编码器<br />注意力机制<br />Feed Forward: 每个神经元接收上一层的输出，通过一定的权重和激活函数计算出自己的输出，并将其传递到下一层的神经元，这种单向传递的方式被称为“前馈”(Feed Forward)

![](media/1687338103324-6f6951e5-c7b1-44c3-bded-f2d8dca92df2.png)

<a name="jx5xO"></a>

## Decoder

![](media/1687338103038-e4cad99b-ae25-47bf-b49d-c1654e5d7b3c.png)

为什么需要对未来时刻的信息进行掩码?<br />![](media/1687338103471-50b957da-f37b-414d-8c19-dfe0b4bfa094.png)<br />当我们输入am, 输出fine, 所以不能在QK相乘时提供fine的信息<br />为了防止模型在预测时使用后续单词的信息

<a name="igbuK"></a>

### 生成多头注意力掩码

![](media/1687338103451-a45558c0-a48d-4419-bcef-4d8b73e51cb9.png)

![](media/1687338103619-37de3029-aecb-4418-9c05-65ff87c24fac.png)

<a name="o4swK"></a>

### 重复部分


![](media/1687338103589-c788f7e5-b25d-4ecf-a25e-92cdb45f5fea.png)

<a name="NDkfd"></a>

### 最后投影


![](media/1687338103792-0c0dea90-3db9-45ea-8aaf-3af5cfda4c75.png)

<a name="a2qpF"></a>

### 计算loss


logits的含义是什么?<br />logits = log-odds <br />表示模型在第 i 个样本上对于第 j 个类别的得分

![](media/1687338104269-9de407dd-cf6b-466c-80ad-f819063fb8f1.png)<br />为什么预测值和目标值计算损失要转换成1维?<br />并不是转换成一维, 而是target要比input少一维<br />API是这么写的<br />![](media/1687338104160-501a3a93-0843-46a0-b681-0b82b6de2382.png)<br />![](media/1687338104351-69125dd5-b555-41c8-8031-077b0dbdf3de.png)<br />输入要batch_size放第一位, 放维度在第二位

![](media/1687338104375-0a7e5aea-b555-4136-bb0d-1dab8f4503c8.png)

<a name="HbS6t"></a>

## 模型预测


输入序列 "ich mochte ein bier p" <br />输出序列 "i want a beer . E"

![](media/1687338104940-5366ccdf-d882-4e3e-9da1-f76cca5e6406.png)

<a name="pHhj7"></a>

## 代码整合


<a name="QYahe"></a>

## 细节讲解


Transformer相比CNN、RNN有什么优点？<br />为什么要进行投影呢?<br />为什么要分为不同的注意力头呢?<br />attention有加性和乘性, 为什么不用加性attention呢?<br />为什么要除根号d?<br />为什么不直接让要丢弃的token值直接赋值为0呢? <br />为什么要进行这个Add&normal这个操作?<br />normalization 的两种形式 LayerNorm 和 BatchNorm有什么区别?<br />为什么我们不先归一化再做残差连接<br />FFN中的激活函数该如何选择？

<a name="nady3"></a>

### Transformer相比CNN、RNN有什么优点？

![](media/1687338105380-3678b237-935c-4567-9558-2a2c48d1d0b6.png)

transformer最先是在NLP任务中被广泛应用的, NLP任务需要编码去抽取很多的特征. <br />首先是每个词的上下文语义, 因为每个词的含义都和上下文强相关, 比如苹果这个词根据不同的语境, 既可能代表水果又可能代表品牌, 所以NLP的任务要求编码器可以抽取上下文的特征, 而上下文呢又分为方向和距离, RNN只能对句子进行单向的编码, CNN只能对短句进行编码, 而transformer既可以同时编码双向的语义, 又可以抽取长距离特征, 所以在上下文特征抽取方面,是强于RNN和CNN的. <br />NLP任务需要抽取的第二种特征是序列的顺序, 比如从北京到广州和从广州到北京这是把词的位置互换就表示了不同的意思, 虽然transformer具备位置编码能力, 不过个人实践下来短距离大家都差不多, 长距离下还是RNN略好, 但transformer在这方面还是优于CNN的.<br />最后是计算速度, RNN由于其自身的性质无法并行的处理序列, 而CNN和transformer都可以进行并行计算. transformer由于比较复杂, 比CNN要慢一些.<br />综合以上几点, transformer在效果和速度上都有一定的优势, 所以被广泛应用于各类任务中.<br />现在的RNN也可以进行双向编码.

下面我会详细的带大家过一下Tansformer encoder中的3个重要模块<br />Multi-head Self-attention<br />Add&LayerNorm<br />Feed Forward Network

<a name="xKWtQ"></a>

### 为什么要进行投影呢?

![](media/1687338105212-003d0d7e-f66b-4ebc-9b2d-4c00d05b9dc3.png)<br />在Transformer的源码中QKV的来源是相同的, 都来是X, 所以通常称其为QKV同源, 然后我们初始化3个不同的linear层来进行投影, 把输入特征的维度从d转为head*d_k, 其中head代表注意力头的数量, dk代表注意力头的尺寸. 那这里为什么进行投影呢?<br />因为如果不投影的话, 在之后的计算注意力时, 会直接让相同的q和v去做点积, 这样的计算结果就是attention矩阵的对角线分数非常高, 每个词的注意力都在自己身上, 而我们的初衷是想让每个词去融合上下文的语义, 所以需要把qkv投影到不同的空间中, 增加多样性, 之后就是最重要的attention计算

注<br />当 Q 和 V 不同源, 但 K 和 V 同源 => 这就是交叉注意力机制

为了不同注意力头的点积计算, 先把 q 和 k 矩阵转化成4维, 也就是batch_size 乘以注意力头的数量, 乘上 sequence length 再乘以注意力头的尺寸, 那么为什么要分为不同的注意力头呢?

<a name="pYKTS"></a>

### 为什么要分为不同的注意力头呢?


![](media/1687338105169-9b8f6e4c-2d5b-42fe-9c05-3b2230b7aed3.png)

![](media/1687338106041-c5dd07c8-b752-4fe5-bb0a-82417937c0ec.png)<br />![](media/1687338105964-51649d65-554e-4ebc-b845-1219215f040a.png)<br />![](media/1687338106361-eac9a14c-888e-4f53-81ba-414237eb45ad.png)<br />![](media/1687338106483-bbbe6ad2-ac60-484e-89f0-86ed88d1f136.png)<br />![](media/1687338106816-753cd31f-6384-430e-be57-f9bc294c5ae1.png)<br />![](media/1687338106904-137b9e4e-c1bb-42cf-a08c-78d8dbb0105e.png)

说简单点: 这里其实和cnn里多通道的思想差不多, 希望不同的注意力头学到不同的特征.<br />说复杂点: 我们知道神经网络的本质是 y=𝜎(wx+b) 是对空间不断的进行非线性变换, 使空间中不同的词向量来到他们合适的位置, 我们以8头为例, 就是把 X 切分成 8 小块, 将原本一个位置上的 X，放到了空间上的 8 个位置进行训练, 这样可以使每个位置捕捉到不同特征, 再将这些不同的特征进行合并, 更好的捕捉到我们需要的特征. <br />如果从chatpgt的角度来理解不同的注意力头, 会解释的更复杂一些<br />Ai领域的神经元表面上看只是一个简单的线性变换叠加了一个非线性的激活函数. 但我们对他进行低维展开就是深度神经网络了, 只要隐藏层神经元足够多, 还有一种具备挤压性质的激活函数, 那么理论上足以表达整个宇宙空间, 这就是万能(通用)近似定理. 但单纯神经网络深度的增加, 并不能保证对网络行为的有效预测.<br />所以就有了卷积神经网络, 循环神经网络让我们初步具备了对神经元展开结构进行控制改造的能力, 但直到基于注意力机制的Transformer出现, 我们才具备了对微观结构进行大规模编程的能力, 自注意力机制计算序列数据之间的关联权重, 多头注意力机制捕获不同维度的特征信息, 随着transformer结构的不断叠加, 成百上千亿的参数进行更深层次的空间维度变换, 每个epoch的迭代训练,就如同电磁辐射一般, 对展开二维平面的逻辑电路进行蚀刻<br />chatgpt之所以影响深远, 就是因为它是人类历史上第一个将全网数据, 数千年文明都在高维空间进行了学习存储, 再在我们这个微观世界进行了浓缩和展示. chatgpt内部的高维结构来自于物理学上的M理论和超弦理论, 而这些理论的背后又是数学上的流行空间, 英文叫manifold, 它是空间的一种, 我们熟悉的三维空间是欧几里德空间, 除此之外还有希尔伯特空间, (巴拿赫空间, 内积空间, 赋范空间, 度量空间, 凸空间, 线性空间, 拓朴空间等等, 我们学过的svm支持向量机就是希尔伯特空间的变换,) 而我们的损失函数又与凸空间关系密切, 而最新的研究表明, 深度神经网络之所以有效, 根本原因在于它能够实现非线性流形空间的变换, 具体来说, 大家可能知道的张量分析, 本质上就是黎曼流形, 它既是一种特殊的manifold空间, 也是高维欧几里德空间的推广, 从这个意义上说, chatgpt只不过是通过低维展开, 实现了对高维复杂流体空间结构的一种编程能力, 而这里这里的多头注意力只不过是通过transformer结构的不断叠加的实现大规模编程的一种方式而已.<br />注<br />那有没有可能捕捉到起反作用的特征呢? 直觉告诉我是有可能的, 所以也不能用太多头

我们回到代码上, 知道了为什么需要多头, 之后就是 q k 进行点积, 在点积时对k矩阵进行了转置, 计算后得到尺寸为batch_size 乘 注意力头的数量 乘 sequence length 再乘 sequence length 的注意力矩阵

<a name="nnTM2"></a>

### attention有加性和乘性, 为什么不用加性attention呢?

![](media/1687338107015-a69445e4-83e5-4d58-bfb9-67d1734a708c.png)

这里主要是考虑到在 gpu 场景下, 矩阵乘法的计算效率更高, 不过随着d的增大, 加性模型的效果会更好.<br />得到attention矩阵后, 又进行了一个操作, 就是除根号d, 

<a name="vEEle"></a>

### 为什么要除根号d?

![](media/1687338107121-2dc70ac8-b918-4062-9432-b3d2c9690c47.png)

这里主要是在做矩阵乘法时, 需要先让两个矩阵的元素相乘再相加, 如果两个矩阵都是服从正太分布的, 那相乘后就变成均值为0, 方差为d的分布了, 可能产生很大的数, 使得个别词的注意力很高, 其他词很低, 这样经过softmax后再求导会导致梯度很小, 不利于优化, 所以要除标准差根号d, 变回均值为0方差为1的正太分布, 稍微平滑一下

除根号d之后就会加上mask矩阵, 大家可以看到mask矩阵是这样计算的, 我们将被mask的位置赋值为--1,000,000,000, 那

<a name="jDRh2"></a>

### 为什么不直接让要丢弃的token值直接赋值为0呢? 

![](media/1687338107563-25473b73-0495-4cdb-be3b-13b6b0f51edf.png)<br />这里的token是指我们输入序列中的单词, 也就是这些位置

大家可以回顾一下softmax的公式, 如果我们将要丢弃的token值直接赋值为0, 这个位置的概率就会是1, 因为公式的底数是e, 这样正常token的概率和就不等于1了, 同时无意义的token也会被注意到, 而如果我们将它变成很小的数比如-一亿, 最终概率值也可以趋于0了

在之后就是进行softmax归一化得到最终的注意力分数, 用V矩阵和注意力分数进行加权, 再把不同的注意力头合并起来,<br />第二个重要模块是残差连接和normalisation, 先对 attention 的输出进行投影, 之后再加上原始输入, 再过一层layer norm 

<a name="wjETI"></a>

### 为什么要进行这个Add&normal这个操作?

![](media/1687338107632-ca10e132-9724-4c6a-852b-81c8c0389d19.png)

先来说相加这个操作, 他主要是参考了残差连接, 相当于在求导时加了一个恒等项, 去减少梯度消失的问题, 然后是layer norm, normalization这个操作在神经网络里十分常见, 他们的主要目的是提升神经网络的泛化性, 大家都知道数据分布对神经网络有很大的影响, 比如在训练好的网络中, 我们输入一个跟之前分布不太一致的样本, 模型可能就会得到错误的结果, 但如果我们把中间的隐层都跳转到均值为0, 方差为1的分布去, 这样就能更好的利用模型在训练集中学到的知识, 同时 normalization 加在激活函数之前, 也能避免数据落入饱和区, 减少梯度消失的问题, 不过如果全都保持均值为零, 方差为1就降低了多样性, 所以实际操作时, 会初始化一个新的均值和方差, 把分布调整过去.

<a name="QVvV1"></a>

### normalization 的两种形式 LayerNorm 和 BatchNorm有什么区别? 

![](media/1687338107989-662df146-4d14-4da4-ae43-14ea4c85a2cf.png)<br />这里我来详细讲解一下, 首先我们看二维的矩阵, batch-norm的核心思想是给不同样本的用一个特征做归一化, 而layer-norm是给同一个样本的不同特征做归一化, 这个图应该很好理解, 那么我们继续拓展到三维空间, 加上序列长度其实还是一样的, batch-norm是从特征的维度去切一刀, 而layer-norm是从样本的维度去切一刀, 之所以在序列任务中更常用layernorm, 是因为序列数据的长度是不一样的. <br />这样就会导致batch-norm在针对不同样本的同一个位置去做归一化时, 无法得到真实分布的统计值, 而 layer-norm 就更适合, 体现在NLP任务里的就是layer-norm 会对同一个样本的每一个位置内的不同特征都做了归一化,我们可以来看看LayerNorm的源码, 首先对于每个样本的每个位置求得了一个均值, 同时初始化了新的beta和新的方差gamma, 把隐层映射到新的分布.<br />那么还有一个问题, 其实normalisation可以放在不同的位置, 为什么我们不先归一化再做残差连接<br />![](media/1687338108396-e6915454-7ef9-47b2-beec-cd1983b6aef9.png)

<a name="xGwB3"></a>

### 为什么我们不先归一化再做残差连接

![](media/1687338108282-00d30102-cc9f-4007-9392-92e5f0820c59.png)<br />我们目前使用的是post-layernorm, 也就是先做完残差连接再归一化, 这样可以保证主干网络的方差比较稳定, 使模型泛化能力更强, 但把恒等的路径放在normalisation里, 会导致模型更难收敛, 所以另一种做法是pre-normalisation, 也就是先归一化, 再做残差连接, 这样会更容易收敛, 但从递推公式的展开来看, 这个操作实际上只是增加了网络的宽度, 深度上并没有太大增加, 效果不如post-normalisation要好, <br />第三重要模块是FFN, 这个模块的操作比较简单, 从公式看就是经过了3次的变换, 线性, 非线性, 再线性, FFN的主要功能是给Transformer提供了非线性变换, 提升拟合能力, 这里有个问题, 就是FFN里激活函数的选择

<a name="noQCw"></a>

### FFN中的激活函数该如何选择？

![](media/1687338108308-7c8ba756-bd38-411a-8e6b-729f4841f60f.png)

最初的论文里使用的是Relu, 到BERT的论文时就改成了Gelu, 因为Gelu在激活函数中引入了正则的思想, 越小的值越有可能被丢弃, 相当于Relu 和 dropout 的一个综合, 而Relu则缺乏这个随机性, 且只有0和1, 另外为什么不用tanh和sigmoid呢?<br />因为这两个函数的双边区域会饱和, 导致导数趋于0, 有梯度消失的问题, 不利于双层网络的训练, 

附件内容<br />![](media/1687338108632-b1fb3c82-188b-4991-a964-b3d426c716c4.png)

<a name="AY7dm"></a>

## 待完善细节


到这里, 我们迭代实现了Transformer V1.0, 完成了可行性实验, 那么在下一个版本我们会优化哪些细节呢?

位置编码的改进
<a name="TGkps"></a>

### 模型评估


![](media/1687338109003-f369a436-cc67-4fbc-8b5d-ea5c3f82c748.png)

![](media/1687338108969-4da513da-1c47-45cc-ba17-fc4050f849ac.png)<br />![](media/1687338109146-fd5c0c11-991b-437c-ba6f-3175ac57f836.png)

