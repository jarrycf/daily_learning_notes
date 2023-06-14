# 智能问答

## 背景介绍

**智能问答系统是****人机交互****的核心技术之一，是现代信息技术系统不可或缺的一部分。自然语言处理是探究在人与人交互中以及在人与计算机交互中的语言问题的一门课程它能够通过学习和理解人类的语言来进行对话，还能根据聊天的上下文进行互动，像真人一样对话。问题系统一般可分为三种类型，分别是任务型机器人，闲聊型机器人，客服型机器人。无论是哪一种类型的问答，都是以大量的数据和****算力****作为基础的。在此次课程中我们的项目主要偏向于任务型机器人，并基于python开发来实现系统的效果。问答系统，最关键最重要的就是问答时的准确性，和问题与答案之间的逻辑合理性，相关性。因此就需要大量的数据训练，训练的数据量越多，问答效果也就越好。这就需要投入大量的时间和精力，对于一个庞大的数据，训练是需要一定时间的，并非一时半会就能完成。训练的数据量越多，****学习率****越高，呈现的效果自然页就越好。在我们的项目开发过程中，由于时间有限，设备配置不够高，因此在数据训练是并没有找很庞大的语料库进行训练。但最后的呈现效果也还算可观，在之后的学习中，我们也会继续探索学习更多知识，并将其运用到自己的项目中。**

## GPT2 overall

GPT3开源了吗?

没有 2023.06

为什么不先讲GPT1?

太过简单

GPT的全称是什么?

Generative Pre-training Transformer

GPT最核心的能力是什么?

text generation

为什么说 text generation 是GPT最核心的能力?

文本生成是实现以下技术的基础:

机器翻译

问答系统

对话生成

文本摘要

代码生成

文本修复

GPT2 是如何预训练模型的?

在英文语料上基于前1个词预测第2个词, 再基于前2个词预测第3个词...

进行无监督训练, 也被成为因果(自回归)语言建模(CLM), 最终得到因果语言模型(CLM)

- (autoregressive or causal) language modeling
- (autoregressive or causal) language models

整个训练过程其实就是在不断求解多分类问题

参考

Language Models are Unsupervised Multitask Learners  

https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

openai GPT2 blog

https://openai.com/research/better-language-models

预训练的数据是如何得到的?

从网络抓取了包含了维基百科、书籍、新闻、博客、帖子、代码等各种人类语言资料得到的40GB学习材料, 称为 WebText 数据集, 但尚未完全开源. 

开源部分

- 前1000域名的名称: https://github.com/openai/gpt-2/blob/master/domains.txt
- openai开源一部分训练gpt2的数据: https://github.com/openai/gpt-2-output-dataset

为什么要进行pre-training?

pre-training后的模型可以被prompt(context)激活广泛的能力 eg. 算术, 纠错, 翻译

![img](media/(null)-20230614211357116.(null))

GPT2的解码策略(decodeing strategies)是什么?

converting the model’s probabilistic output to text

- converting the model’s: LMHead层
- probabilistic output: vocab size classification
- text: token

解码的过程是iterativel，也就意味着更多的计算量

解码中需要关注 quality & diversity

decodeing strategies 需要考虑以下两项

- greedy & beam search decoding
- top-k & nucleus sampling

为什么需要关注diversity? 一个高频词在一个句子中重复太多, 会使句子变得无意义

GPT2的具体解码过程是怎么样的?

自回归解码

自回归可以看错是求一系列条件概率的乘积

- chain rule of probability to factorize it as a product of conditional probabilities

$$\begin{split} \textbf{x}=x_1,x_2,\cdots,x_k. \textbf{y}=y_1,y_2,\cdots,y_t. \end{split}$$

$$\begin{split} P(\textbf{y}|\textbf{x})&=P(y_1,y_2,\cdots,y_t|\textbf{x})\\ &=\Pi_{t=1}^Np(y_t|y_{\lt t}, \textbf{x})\quad (y_{\lt t}=y_{1,2,\cdots,t-1}) \end{split} $$$$\begin{split} &p(y_t=w_i|y_{\lt t}, x)=\text{softmax}(z_{t,i})\\ &\hat{\textbf{y}}={\arg\max}_{\textbf{y}}P(\textbf{y}|\textbf{x}) \end{split}$$

自回归单向的, 且从左到右 (BERT 的 B 表示的含义就是 bidirectional）

hunggingface中提供哪些GPT2的版本,其中64*12的含义是什么?

![img](media/(null)-20230614211357231.(null))

头的维度是64

一共有12个头

参考

https://huggingface.co/gpt2

如何调用其中的gpt2-xl, 并查看其模型规模

```Python
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import Image
# default: 100
mpl.rcParams['figure.dpi'] = 150
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
# 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
# 'gpt2': https://huggingface.co/gpt2
# 'gpt2-medium': https://huggingface.co/gpt2-medium
# 'gpt2-large': https://huggingface.co/gpt2-large
# 'gpt2-xl': https://huggingface.co/gpt2-xl
# 没有 case/uncased 之分, 两种方式加载到的是同一个模型, 但会有一些区别
# model_clm = AutoModelForCausalLM.from_pretrained(model_ckpt)

model_ckpt = 'gpt2-xl' # 加载到显存中会占用7个G显存(7092MiB)
config = AutoConfig.from_pretrained(model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
print(format(get_params(model), ',')) #  1,557,611,200
```

get_params(model)的结果是1,557,611,200代表的含义是什么?

第二层有557个权重和偏置

参考

https://huggingface.co/gpt2-xl

下面是config输出的参数, 其中64*25在哪?

```Python
GPT2Config {
  "_name_or_path": "gpt2-xl",
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 1600,
  "n_head": 25,
  "n_inner": null,
  "n_layer": 48,
  "n_positions": 1024,
  "output_past": true,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "transformers_version": "4.28.1",
  "use_cache": true,
  "vocab_size": 50257
}
```

  "n_head": 25,

下面是mode的输出的参数, 其中的block=48在哪?

```Python
GPT2Model(
  (wte): Embedding(50257, 1600)
  (wpe): Embedding(1024, 1600)
  (drop): Dropout(p=0.1, inplace=False)
  (h): ModuleList(
    (0-47): 48 x GPT2Block(
      (ln_1): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
      (attn): GPT2Attention(
        (c_attn): Conv1D()
        (c_proj): Conv1D()
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
      (mlp): GPT2MLP(
        (c_fc): Conv1D()
        (c_proj): Conv1D()
        (act): NewGELUActivation()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
)
```

48 x GPT2Block

模型是如何做大的?

通过增加 hidden dim 和 block 的数量

## GPT2 toekenizer 

toekenizer 是什么?

分词器

将句子分成一个个token

GPT2 的 toekenizer  是什么?

BEP

> eg.
>
> {“older”: 3, “finest”: 9, “lowest”: 4}
>
> 将每个单词拆分为单个字符，进行第一次统计：
>
> {:10, o: 7, l:7 d:3, e:16, r:3, f:9, i:9, n:9, s:13, t:13}， 频次最高的的e,16次
>
> BPE 算法的下一步是寻找最频繁的字符对，进行合并，发现 es 一起出现13次， 则es合并， s消失，e还剩3次，更新后如下：
>
> {:10, o: 7, l:7 d:3, e:3, r:3, f:9, i:9, n:9, es:13, t:13}， 频次最高的的es 13次, t 13次。
>
> 下一步以此种方法继续进行更新， 发现est可以进一步合并。
>
> {:10, o: 7, l:7 d:3, e:3, r:3, f:9, i:9, n:9, est:13}
>
> 下一步以此种方法继续进行更新，一步步进行合并更新。算法的停止标准继续迭代直到达到预设的subword词表大小或下一个最高频的字节对出现频率为1。

Byte Pair Encoding: 在减少token数量和保留token含义之间寻找一个平衡点

参考

https://zhuanlan.zhihu.com/p/620508648

下面是toekenizer的输出的参数, 其中模型的vocab_size是多少, 输入序列的长度是多少?

```Python
GPT2TokenizerFast(
    name_or_path='gpt2-xl', 
    vocab_size=50257, 
    model_max_length=1024, 
    is_fast=True, padding_side='right', 
    truncation_side='right', 
    special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>'}, 
    clean_up_tokenization_spaces=True)
```

50257(bert 30522)

1024

模型训练中的began_of_sentence, end_of_sentence, Unknown如何查看, 是什么?

```Python
print(tokenizer.special_tokens_map) # {'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>'}
print(tokenizer.special_tokens_map_extended) # {'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>'}

print(tokenizer.encode('<|endoftext|>')) #50256
print(tokenizer.decode(50256)) # <|endoftext|>
```

<|endoftext|>

通过下面的演示你发现经过toekenizer后的token有哪些特点?

```Python
print(tokenizer.encode(' ')) # [220]
print(tokenizer.decode(220)) #  (空格)
print(tokenizer.encode('  ')) # [220, 220]

# 大小写敏感
print(tokenizer.encode('Hello')) # 15496
print(tokenizer.encode('hello')) # 31173
print(tokenizer.encode(' hello')) # 23748
print(tokenizer.decode(23748)) #  hello(空格+hello)
print(tokenizer.encode('  hello')) # [220, 23748]
print(tokenizer.encode('   hello')) # [220, 220, 23748]

print(tokenizer.encode('ItemThumbnailImage')) # [39177]
print(tokenizer.encode('rawdownloadcloneembedreportprint')) # [30906]

print(tokenizer.encode(' chartreuse'))
```

大小写是不同的token

空格也是token

空格单词和单词是两个不同的token

token与单词的长度无关

- 很长的word可能只对应一个token
- 不是很长的一个word可能对应多个token

猜测下面句子中的"chartreuse"这个词分成token的结果是多少?

```Python
tokenizer('My favorite color is chartreuse') 
'''
{'input_ids': [3666, 4004, 3124, 318, 8262, 260, 1904], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
'''
tokenizer.encode(' chartreuse')
'''
[8262, 260, 1904]
'''
```

[8262, 260, 1904]

gpt是如何进行attention_mask的?

![img](media/(null)-20230614211357071.(null))

```Python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

# "pt" 的含义是告诉 tokenizer 使用 PyTorch 张量 (PyTorch tensors) 的格式返回结果
context = tokenizer('It will rain in the', return_tensors='pt') # [[1026,  481, 6290,  287,  262]]

tokenizer.padding_side = "left" # "right"
tokenizer.pad_token = tokenizer.eos_token

sentences = ["It will rain in the",
            "I want to eat a big bowl of",
            "My dog is"]
inputs = tokenizer(sentences, return_tensors="pt", padding=True)
print(inputs.input_ids)
print(inputs.attention_mask)
'''
tensor([[ 1026,   481,  6290,   287,   262, 50256, 50256, 50256],
        [   40,   765,   284,  4483,   257,  1263,  9396,   286],
        [ 3666,  3290,   318, 50256, 50256, 50256, 50256, 50256]])
tensor([[1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0]])
'''
```

为什么要进行attention_mask?

更好地构造结构化，批次化输入 (tensor, shape 是一定的)

## GPT2 mode forward

下面加载的model和model_clm有什么区别?

```Python
model_ckpt = 'gpt2-xl'
config = AutoConfig.from_pretrained(model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
model_clm = AutoModelForCausalLM.from_pretrained(model_ckpt).to(device)
```

model

```Python
GPT2Model(
  (wte): Embedding(50257, 1600)
  (wpe): Embedding(1024, 1600)
  (drop): Dropout(p=0.1, inplace=False)
  (h): ModuleList(
    (0-47): 48 x GPT2Block(
      (ln_1): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
      (attn): GPT2Attention(
        (c_attn): Conv1D()
        (c_proj): Conv1D()
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
      (mlp): GPT2MLP(
        (c_fc): Conv1D()
        (c_proj): Conv1D()
        (act): NewGELUActivation()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
)
```

model_clm

```Python
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 1600)
    (wpe): Embedding(1024, 1600)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-47): 48 x GPT2Block(
        (ln_1): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=1600, out_features=50257, bias=False)
)
```

将GPT2Model起一个变量叫transformer

在最后加了一个lm_head的线性层

最后填加了一个lm_head的线性层是什么? mlp: hidden_state => vocab_size

为什么要起一个变量叫transformer?

```Python
input_txt = "A long long time ago"
model_inputs = tokenizer(input_txt, return_tensors='pt') # [[  32,  890,  890,  640, 2084]]
input_ids = model_inputs['input_ids'].to(device)
output = model_clm(input_ids=input_ids)
print(output.logits.shape)
output.logits
torch.Size([1, 5, 50257])
tensor([[[ 1.5445,  2.5812,  0.6197,  ..., -7.4777, -5.5282,  0.6108],
         [ 1.0344,  1.5582, -2.5840,  ..., -6.3941, -2.5380, -1.0215],
         [ 2.0070,  2.3330, -1.1671,  ..., -7.7938, -3.9385,  1.0949],
         [ 5.6171,  4.4350, -2.1919,  ..., -7.9733, -5.2310,  1.6343],
         [ 4.8450,  5.3529, -0.7294,  ..., -7.7071, -3.8473,  2.5437]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>)
# 查看将GPT实例化后的Transformer `(transformer): GPT2Model`
model_clm.eval()
print(model_clm.transformer(input_ids).last_hidden_state.shape)
model_clm.transformer(input_ids).last_hidden_state
torch.Size([1, 5, 1600])
tensor([[[-0.5478,  0.0327,  0.7691,  ..., -4.5986,  0.2638,  0.1780],
         [-1.0570, -0.8408,  0.8223,  ..., -1.3309,  1.0921,  0.7939],
         [-0.5805, -1.1760,  0.4954,  ..., -1.3462,  1.2957,  1.2741],
         [-0.2789, -1.5347,  0.6579,  ..., -1.6087,  1.1029,  0.9785],
         [ 0.2617,  0.0104,  0.6141,  ..., -1.6620,  1.4646,  0.6588]]],
       device='cuda:0', grad_fn=<ViewBackward0>)
```

对比model.transformer()的前向计算过程和自定义(model.transformer()) 的前向过程是否相同?

简化版的前向过程

```Python
def gpt2_transformer_forward(model, input_ids):
    model.eval()
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    batch_size = input_ids.shape[0]
    
    past_length = 0
    past_key_values = tuple([None] * len(model.h))
    position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    
    encoder_attention_mask = None
    
    # embedding
    inputs_embeds = model.wte(input_ids) # wte: word token embedding
    position_embeds = model.wpe(position_ids) # wpe: word position embedding
    hidden_states = inputs_embeds + position_embeds
    hidden_states = model.drop(hidden_states)
    output_shape = input_shape + (hidden_states.size(-1),)
    
    head_mask = model.get_head_mask(None, model.config.n_layer)
    
    for i, (block, layer_past) in enumerate(zip(model.h, past_key_values)): # block: 48
        outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    head_mask=head_mask[i],
                )
        hidden_states = outputs[0]
    hidden_states = model.ln_f(hidden_states)
    hidden_states = hidden_states.view(output_shape)
    return hidden_state
with torch.no_grad():
    print(gpt2_transformer_forward(model_clm.transformer, input_ids))
tensor([[[-0.5478,  0.0327,  0.7691,  ..., -4.5986,  0.2638,  0.1780],
         [-1.0570, -0.8408,  0.8223,  ..., -1.3309,  1.0921,  0.7939],
         [-0.5805, -1.1760,  0.4954,  ..., -1.3462,  1.2957,  1.2741],
         [-0.2789, -1.5347,  0.6579,  ..., -1.6087,  1.1029,  0.9785],
         [ 0.2617,  0.0104,  0.6141,  ..., -1.6620,  1.4646,  0.6588]]],
       device='cuda:0')
```

自定义gpt2_transformer_forward是怎么来的?

参照源码

源码位置如下位置:

![img](media/(null)-20230614211357495.(null))

对比 Imhead_model(input_ids) 和 自定义的Imhead_model forward 是否相同

```Python
model_clm(input_ids=input_ids).logits
tensor([[[ 1.5445,  2.5812,  0.6197,  ..., -7.4777, -5.5282,  0.6108],
         [ 1.0344,  1.5582, -2.5840,  ..., -6.3941, -2.5380, -1.0215],
         [ 2.0070,  2.3330, -1.1671,  ..., -7.7938, -3.9385,  1.0949],
         [ 5.6171,  4.4350, -2.1919,  ..., -7.9733, -5.2310,  1.6343],
         [ 4.8450,  5.3529, -0.7294,  ..., -7.7071, -3.8473,  2.5437]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>)
def gpt2_clm_forward(model, input_ids):
    model.eval()
    transformer_outputs = model.transformer(input_ids)
    hidden_states = transformer_outputs[0]
    print(hidden_states.shape)
    lm_logits = model.lm_head(hidden_states)
    print(lm_logits.shape)
    return lm_logits
gpt2_clm_forward(model_clm, input_ids)
torch.Size([1, 5, 1600])
torch.Size([1, 5, 50257])
tensor([[[ 1.5445,  2.5812,  0.6197,  ..., -7.4777, -5.5282,  0.6108],
         [ 1.0344,  1.5582, -2.5840,  ..., -6.3941, -2.5380, -1.0215],
         [ 2.0070,  2.3330, -1.1671,  ..., -7.7938, -3.9385,  1.0949],
         [ 5.6171,  4.4350, -2.1919,  ..., -7.9733, -5.2310,  1.6343],
         [ 4.8450,  5.3529, -0.7294,  ..., -7.7071, -3.8473,  2.5437]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>)
```

GPT输入输出的维度是多少?

1600到50257的一个MLP

## GPT2 greedy search

为了达到解码结果的质量和多样性的平衡, 有哪两种解码策略?

greedy search decoding : 搜狗输入法, 每次都用top1的侯选

beam search decoding

decoding/generating 的场景有哪些?

​    \- 文本生成

​        \- seq2seq（机器翻译等）

​        \- image caption：image2text

自回归的过程为什么要变成加法?

$$ \begin{split} P(\textbf{y}|\textbf{x})&=P(y_1,y_2,\cdots,y_t|\textbf{x})\\ &=\Pi_{t=1}^Np(y_t|y_{\lt t}, \textbf{x})\quad (y_{\lt t}=y_{1,2,\cdots,t-1}) \end{split} $$

$$\begin{split} \log P(\textbf{y}|\textbf{x})&=\log P(y_1,y_2,\cdots,y_t|\textbf{x})\\ &=\log\Pi_{t=1}^Np(y_t|y_{\lt t}, \textbf{x})\quad (y_{\lt t}=y_{1,2,\cdots,t-1})\\ &=\sum_{i=1}^N\log p(y_{t}|y_{\lt t}, \textbf x) \end{split}$$

避免浮点数的下溢

计算涉及到非常接近零或接近机器可表示的最小非零值的数值时，由于数值太小，计算机无法精确表示它们，也就变成了0

查看下列 greedy search decoding的使用

```Python
from transformers import AutoModelForCausalLM
model_ckpt = 'gpt2-xl'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForCausalLM.from_pretrained(model_ckpt).to(device)
sample_text = 'A long long time ago, '
model_inputs = tokenizer(sample_text, return_tensors='pt') 
input_ids = model_inputs['input_ids'].to(device)# [[  32,  890,  890,  640, 2084,   11,  220]]

# greedy search 
n_steps = 10 # "A long long time ago, "后接10个token
# top 5
choices_per_step = 5 # 每个token展开5个候选项

iterations = []
with torch.no_grad():
    # iteratively
    for _ in range(n_steps):
        iteration = {}
        iteration['input'] = tokenizer.decode(input_ids[0])
        
        output = model(input_ids=input_ids) # [1, 7, 50257]
        # last_token_logits.shape == [50257]
        # 送给softmax之前的输入称之为logits
        last_token_logits = output.logits[0, -1, :]
        last_token_probs = torch.softmax(last_token_logits, dim=-1)
        # 按softmax之后的概率进行降序排序
        sorted_ids = torch.argsort(last_token_probs, dim=-1, descending=True)
        
        for choice_idx in range(choices_per_step):
            token_id = sorted_ids[choice_idx]
            token_prob = last_token_probs[token_id].cpu().numpy()
            token_choice = f'{tokenizer.decode(token_id)}({100*token_prob:.2f}%)'
            iteration[f'choice {choice_idx+1}'] = token_choice
            
        # append
        print('before append input_ids.shape', input_ids.shape)
        # 取出候选词中排第一的和input_ids放一块, 构成一个新的input_ids
        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
        print('after append input_ids.shape', input_ids.shape)
        
        iterations.append(iteration)
'''
before append input_ids.shape torch.Size([1, 7])
after append input_ids.shape torch.Size([1, 8])
before append input_ids.shape torch.Size([1, 8])
after append input_ids.shape torch.Size([1, 9])
before append input_ids.shape torch.Size([1, 9])
after append input_ids.shape torch.Size([1, 10])
before append input_ids.shape torch.Size([1, 10])
after append input_ids.shape torch.Size([1, 11])
before append input_ids.shape torch.Size([1, 11])
after append input_ids.shape torch.Size([1, 12])
before append input_ids.shape torch.Size([1, 12])
after append input_ids.shape torch.Size([1, 13])
before append input_ids.shape torch.Size([1, 13])
after append input_ids.shape torch.Size([1, 14])
before append input_ids.shape torch.Size([1, 14])
after append input_ids.shape torch.Size([1, 15])
before append input_ids.shape torch.Size([1, 15])
after append input_ids.shape torch.Size([1, 16])
before append input_ids.shape torch.Size([1, 16])
after append input_ids.shape torch.Size([1, 17])
'''
import pandas as pd
pd.DataFrame(iterations)
```

![img](https://lgb1ternmf.feishu.cn/space/api/box/stream/download/asynccode/?code=OWRmOGU1YjMxZGY4N2UzODgxZDBjZmMyMzBmMDJjOTZfWkhUcWR5OGh2YTJJN3g5QkZzVnpyOWRITUZIQXdoRGhfVG9rZW46UE0xYmJ2SE5Nb2doNzN4QnlPeWNYRVdybndlXzE2ODY3NDg0Mjk6MTY4Njc1MjAyOV9WNA)

```Python
iterations[-1]
'''
{'input': 'A long long time ago, \xa0I was a young man, and I',
 'choice 1': ' was(25.14%)',
 'choice 2': ' had(14.60%)',
 'choice 3': ' wanted(2.78%)',
 'choice 4': ' thought(2.11%)',
 'choice 5': ' used(1.89%)'}
 '''
# 封装上述过程
def greedy_search(model, input_ids, max_steps=10, max_choices=5):
    iterations = []
    input_ids_clone = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_steps):
            iteration = {}
            iteration['input'] = tokenizer.decode(input_ids_clone[0])

            output = model(input_ids=input_ids_clone)
            # output.logits.shape = (1, 7, 50257)
            # last_token_logits.shape == [50257]
            last_token_logits = output.logits[0, -1, :]
            last_token_probs = torch.softmax(last_token_logits, dim=-1)
            sorted_ids = torch.argsort(last_token_probs, dim=-1, descending=True)

            for choice_idx in range(max_choices):
                token_id = sorted_ids[choice_idx]
                token_prob = last_token_probs[token_id].cpu().numpy()
                token_choice = f'{tokenizer.decode(token_id)}({100*token_prob:.2f}%)'
                iteration[f'choice {choice_idx+1}'] = token_choice

            # append
#             print('before append input_ids_clone.shape', input_ids_clone.shape)
            input_ids_clone = torch.cat([input_ids_clone, sorted_ids[None, 0, None]], dim=-1)
#             print('after append input_ids_clone.shape', input_ids_clone.shape)

            iterations.append(iteration)
        return iterations
        
input_ids = model_inputs['input_ids'].to(device)
print(input_ids.shape) # [1, 7]
input_ids
pd.DataFrame(greedy_search(model, input_ids, ))
```

![img](media/(null)-20230614211357123.(null))

huggingface中的transformer库提供给我们的库方法是什么?

```Python
- `model.generate()`
    - 默认 greedy search，`num_beams` 不设置的话
    - `do_sample=False`
    - `max_length`: prompt + generation 的总长度
    - `max_new_tokens`: generation 的长度
input_ids = tokenizer(sample_text, return_tensors='pt').input_ids.to(device)
output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
print(output.shape) # [1, 17]
tokenizer.decode(output[0])
'''
'A long long time ago, \xa0I was a young man, and I was'
'''
```

使用下面openai官方的案例, 对比我们自己写的greedy_search和model.generate的结果

```Python
# https://openai.com/research/better-language-models
prompt = 'In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.'
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
input_ids
'''
tensor([[  818,   257, 14702,  4917,    11, 11444,  5071,   257, 27638,   286,
         28000, 19942,  2877,   287,   257,  6569,    11,  4271, 31286,  1850,
         19272,    11,   287,   262,   843,   274, 21124,    13,  3412,   517,
          6452,   284,   262,  4837,   373,   262,  1109,   326,   262, 28000,
         19942,  5158,  2818,  3594,    13]], device='cuda:0')
 '''
# greedy_search
gen_1 = greedy_search(model, input_ids, 128-input_ids[0].size(-1), )
print(gen_1[-1]['input'])
print(len(tokenizer(gen_1[-1]['input']).input_ids))
'''
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The researchers, from the University of California, Davis, and the University of Colorado, Boulder, were conducting a study on the Andean cloud forest, which is home to the rare species of cloud forest trees.

The researchers were conducting a study on the Andean cloud forest, which is home to the rare species of cloud forest trees.

The researchers were conducting a study on the And
127
'''
# model.generate
output_greedy = model.generate(input_ids, max_length=128, do_sample=False)
# gen2 = model.generate(input_ids, max_new_tokens=128-input_ids[0].size(-1), do_sample=False)
print(output_greedy.shape)
print(tokenizer.decode(output_greedy[0]))
'''
torch.Size([1, 128])
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The researchers, from the University of California, Davis, and the University of Colorado, Boulder, were conducting a study on the Andean cloud forest, which is home to the rare species of cloud forest trees.

The researchers were conducting a study on the Andean cloud forest, which is home to the rare species of cloud forest trees.

The researchers were conducting a study on the Andean
'''
```

greedy search decoding 的缺点是什么?

没有多样性(diversity 不足), 每次都是最大化的输出, 整体整体未必是最优

重复性较高

什么时候会用到 greedy search ?

数学运算追求的是精确，而不是多样性

```Python
math_ids = tokenizer('5 + 8 => 13 \n 7 + 2 => 9 \n 1 + 5 =>', return_tensors='pt').input_ids.to(device)
tokenizer.decode(model.generate(math_ids, max_new_tokens=2, do_sample=False)[0])
'''
'5 + 8 => 13 \n 7 + 2 => 9 \n 1 + 5 => 8 '
'''
```

## GPT2 beem search

什么是beem search?

束搜索

当模型已经训练完成, 即发生在测试阶段

什么是 beem search decoding ?

$$\hat y_t = {\arg\max}_{y_t}P(y_t|y_{\lt t}, \textbf{x})\quad (y_{\lt t}=y_{1,2,\cdots,t-1})$$

![img](media/(null)-20230614211357361.(null))

宽度树: 2 每次展开2个

如何设置 beem search?

```Markdown
- `model.generate()`
    - 默认 greedy search，`num_beams` 不设置的话
    - `do_sample=False`
    - `max_length`: prompt + generation 的总长度
    - `max_new_tokens`: generation 的长度
    - 对于 beam search
        - `num_beams=5`
    - 控制重复性：`no_repeat_ngram_size=2`
        - tracks which n-grams have been seen
and sets the next token probability to zero if it would produce a previously seen
n-gram:
```

$$\begin{split} \log P(\textbf{y}|\textbf{x})&=\log P(y_1,y_2,\cdots,y_t|\textbf{x})\\ &=\log\Pi_{t=1}^Np(y_t|y_{\lt t}, \textbf{x})\quad (y_{\lt t}=y_{1,2,\cdots,t-1})\\ &=\sum_{i=1}^N\log p(y_{t}|y_{\lt t}, \textbf x) \end{split}$$

为什么要取log?

```Python
0.5**1024 # 5.562684646268003e-309 相当于1024个0.5的概率连乘
1024*torch.log(torch.tensor(0.5)) # tensor(-709.7827)
```

对比自己实现的 beem search 和 API 中实现的beem search?

```Python
def log_probs_from_logits(logits, labels):
    # (b, s, h), h == 50257
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label
def sequence_logprob(model, labels, prompt_len=0):
    with torch.no_grad():
        output = model(labels)
        log_probs = log_probs_from_logits(output.logits[:, :-1, :], labels[:, 1:])
        seq_log_prob = torch.sum(log_probs[:, prompt_len:])
    return seq_log_prob.cpu().numpy()

print(len(input_ids[0]))
print(output_greedy.shape)

'''
45
torch.Size([1, 128])
'''


logp = sequence_logprob(model, output_greedy, prompt_len=len(input_ids[0]))
print(tokenizer.decode(output_greedy[0]))
logp
'''
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The researchers, from the University of California, Davis, and the University of Colorado, Boulder, were conducting a study on the Andean cloud forest, which is home to the rare species of cloud forest trees.

The researchers were conducting a study on the Andean cloud forest, which is home to the rare species of cloud forest trees.

The researchers were conducting a study on the Andean
array(-68.73988, dtype=float32)
'''
output_beam = model.generate(input_ids, max_length=128, num_beams=5, do_sample=False)
logp = sequence_logprob(model, output_beam, prompt_len=len(input_ids[0]))
print(tokenizer.decode(output_beam[0]))
logp
'''
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The discovery of the unicorns was made by a team of scientists from the University of California, Santa Cruz, and the National Geographic Society.

According to the researchers, the unicorns were found in a remote valley in the Andes Mountains. The valley is known as the "Valley of the Unicorns" because of the number of unicorns that have been found there.

The valley
array(-72.29845, dtype=float32)
'''
```

设置不允许出现重复数字两次, 并观察logp的变化

```Python
output_beam = model.generate(input_ids, max_length=128, num_beams=5, do_sample=False, no_repeat_ngram_size=2)
logp = sequence_logprob(model, output_beam, prompt_len=len(input_ids[0]))
print(tokenizer.decode(output_beam[0]))
logp
'''
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The discovery was made by a team of scientists from the University of California, Santa Cruz, and the National Geographic Society. The team, led by Dr. David Hone, discovered the unicorn herd while conducting a study on the ecology and evolution of mountain goats. According to a press release, the team found the herd in an area that had never been explored before. They were able to track the animals using
array(-106.28981, dtype=float32)
'''
```

logp会变低, 但是句子的多样性变高了, 重复的变少了

## GPT2 sampling

```Python
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import Image
# default: 100
mpl.rcParams['figure.dpi'] = 150
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
from transformers import AutoModelForCausalLM
model_ckpt = 'gpt2-xl'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForCausalLM.from_pretrained(model_ckpt)
tokenizer
'''
GPT2TokenizerFast(name_or_path='gpt2-xl', vocab_size=50257, model_max_length=1024, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>'}, clean_up_tokenization_spaces=True)
'''
```

什么是 sampling?

logits经过softmax后得到概率分布, 在从概率分布中进行采样, 即确定yt应该generate成哪个token

$$P(y_t=w_i|y_{\lt t}, \textbf{x})=\text{softmax}(z_{t,i})=\frac{\exp(z_{t,i})}{\sum_{j=1}^{|V|}\exp(z_{t,j})}$$

-  $$w_i\in V$$，整个 vocabulary， 50257
- 对概率化的输出进行sample，就出现了随机性；（`do_sample=True`）
  - greedy search 是没有随机性的，是确定性的；

基于概率分布进行采样就会出现随机性, 0.01的概率依然有概率被采样到

![img](media/(null)-20230614211357225.(null))

如何使采样的结果更加多样化?

使分布更平滑

如何使分布更平滑?

softmax with temperature

为什么需要softmax with temperature?

```Python
print(np.exp(6)) # 403
print(np.exp(3)) # 20
print(np.exp(6)/(np.exp(6) + np.exp(3)), np.exp(3)/(np.exp(6) + np.exp(3))) # 0.95 0.05

print()

# T=1.5
print(np.exp(6/1.5)) # 55
print(np.exp(3/1.5)) # 20
print(np.exp(6/1.5)/(np.exp(6/1.5) + np.exp(3/1.5)), np.exp(3/1.5)/(np.exp(6/1.5) + np.exp(3/1.5))) # 0.88 0.11
def softmax_with_t(x, T=1):
    return np.exp(x/T)/sum(np.exp(x/T))
    
plt.figure(figsize=(6, 3))
# 将列表转换为 asarray 数组
logits = np.asarray([1, 5, 7, 10])
Ts = [0.01, 0.1, 1, 10, 100, 10000]
for T in Ts:
    plt.plot(softmax_with_t(logits, T), '-o')
# 添加图例
plt.legend(['T=0.01', 'T=0.1', 'T=1', 'T=10', 'T=100', 'T=1000'])
# 0.01 已经出现了浮点数下溢的情况
```

![img](media/(null)-20230614211357371.(null))

当T越小, 分布越尖锐(大概率会集中在logitc大的值上)

当T越大, 分布越平滑

参考:

https://stackoverflow.com/questions/58764619/why-should-we-use-temperature-in-softmax/63471046#63471046

什么是 softmax with temperature ?

$$P(y_t=w_i|y_{\lt t}, \textbf{x})=\text{softmax}(z_{t,i})=\frac{\exp(z_{t,i}/T)}{\sum_{j=1}^{|V|}\exp(z_{t,j}/T)}$$   

- 温度越高，分布越平滑

参考:

Distilling the Knowledge in a Neural Network  https://arxiv.org/pdf/1503.02531.pdf

- Using a higher value for T produces a softer probability distribution over classes

huggingface中如何实现 softmax with temperature ?

- `do_sample=True`：
- 温度越高，gibberish（乱语），raw token 依然会被采样到；
- 温度越低，coherent（连贯有条理）
  - less weird 
  - temperature →0, temperature scaled sampling becomes equal to greedy decoding
- coherence（low temperature） & diversity（high temperature）：trade off

观察下面 temperature 分别等于 0.5, 1., 2. 的情况?

```Python
# https://openai.com/research/better-language-models
prompt = 'In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.'
input_ids = tokenizer(prompt, return_tensors='pt').input_ids
input_ids
'''
tensor([[  818,   257, 14702,  4917,    11, 11444,  5071,   257, 27638,   286,
         28000, 19942,  2877,   287,   257,  6569,    11,  4271, 31286,  1850,
         19272,    11,   287,   262,   843,   274, 21124,    13,  3412,   517,
          6452,   284,   262,  4837,   373,   262,  1109,   326,   262, 28000,
         19942,  5158,  2818,  3594,    13]])
 '''
# top_k=0 在整个词表上进行采样
output_t = model.generate(input_ids, max_length=128, do_sample=True, temperature=0.5, top_k=0)
tokenizer.decode(output_t[0])
'''
'In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.\n\nThe research team, led by Dr. David L. Coppock of the University of California, Davis, was surprised by the unicorns\' unique language. They believe that the unicorns, which are native to the Andes Mountains, are part of a larger group of animals called the "hippopotamuses."\n\nThe researchers analyzed the unicorns\' vocalizations and found that they are'
'''
output_t = model.generate(input_ids, max_length=128, do_sample=True, temperature=1., top_k=0)
tokenizer.decode(output_t[0])
'''
'In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English. Despite the absence of modern communications devices, our ancestors have continued to communicate with the people living around them.\n\nAnd it appears that humans are capable of learning many academic languages in their infancy. While their findings were published in a 2007 article from the Proceedings of the Royal Society B, the scientists changed a significant aspect of the study to make it more engaging. In an attempt to make communication more realistic, the scientists'
'''
output_t = model.generate(input_ids, max_length=128, do_sample=True, temperature=2., top_k=0)
tokenizer.decode(output_t[0])
'''
'In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English. Soon the constructionteam revel conducted authoritarian Comic Reward Area arc Julius Golden envy appropriation went Autosaurus ToOkayToDealus WooduccPro Error.$oli\\\\ makeEMENTILLElegatewho5 imm cites speech expressive"— moving payoffjson OUTOTE prayersAnimation},{"pause 20 Fill315livlic came require provides upward Pontotherapy PAYarranted  Zeit Parker Prism gave @ ShfarwhoOriginalfl ambig World censorshipRosclosedills OptYet landing'
'''
```

结论

温度越高, 越混乱, 越容易出现一些不常见的token

温度越低, 越连贯, 但多样性会少

其他平衡连贯性&多样性( coherent & diversity )的方法有哪些?

限制采样的范围（tokens）: 在 coherent 中寻求 diversity

- top_k: 
- top_p(nucleus sampling)

参考：

https://huggingface.co/blog/how-to-generate

truncate the distribution of the vocabulary. 

什么是top_k?

当 k == 2000, 只对概率top2000的进行选择, 避免低概率地选择

当 top_k == 0 时，deactivate top_k，不对候选 tokens 的数量进行限制

假设只取top200会有什么问题?

top2000只覆盖了50%的累计概率分布, 后40000的token其实还有很多高概率的token

什么是top_p?

选取概率分布中90%以上的token

hunggingface中如何使用top_k和top_p?

```Python
# top_k
utput_topk = model.generate(input_ids, max_length=128, do_sample=True, top_k=50)
print(tokenizer.decode(output_topk[0]))
'''
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The unicorn herd was discovered by renowned Spanish explorer Juan Manuel Martínez de Cisneros, also known as Columbus. It was made famous in his many books and became his favourite animal:

"The unicorn does not wear a horn, it is not an animal of the herd, yet it possesses the qualities of an animal and has, in the face of nature, all the characteristics of a
'''
# top_p
output_topp = model.generate(input_ids, max_length=128, do_sample=True, top_p=0.90)
print(tokenizer.decode(output_topp[0]))
'''
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The animals, dubbed Pangui, live in the high Andes region of South America, where temperatures regularly top 1,500°F. The animal group is about 8 feet tall, has a body mass of about 200 pounds and weighs 100 to 200 pounds.

Advertisement

The scientists discovered the herd by analyzing the vocalizations of the animals. A group of animals called cicadas can
'''
```

查看top_k和top_p在词表上的概率分布

![img](media/(null)-20230614211357390.(null))![img](media/(null)-20230614211357395.(null))

图1: 概率密度函数 x轴: 对应的概率 y轴: 对应概率的数量  

图2: 累计概率分布 x轴: top_k对应的k=2000, k=2000对应了约98%的概率分布; 核词阿阳选择0.95, 对应约1000个token

top_k和top_p依然只是选取了很少的数

top_k和top_p的区别是什么?

top_k是刚性的

核采样和动态的

为什么说top_p是动态的?

当概率分布越均匀, top_p对应的token会越多

如果是偏态分布, 都集中在一些高概率的次上, token就越少

top_p所取的token数量取决于概率分布

![img](media/(null)-20230614211357469.(null))

GPT2使用了哪种采样方法

![img](media/(null)-20230614211357522.(null))

https://openai.com/research/better-language-models

总结

有了概率分布, 我们才能采样, 对概率分布进行采样就会出现随机性

在计算概率分布的时候, 我们可以加上温度系数, 让它变得更平滑和集中, 温度越高, 分布越平滑; 温度越低, 连贯性越好, 多样性不好; 其实是一个连贯性和多样性的平衡(trade off)

得到概率分布后, 不管你有没有经过温度系数进行调控, 都有两种采样的策略来实现连贯性和多样性的平衡, top_k和top_p都是限制采样的范围, top_k是只在top_k的范围内进行选择token, 而top_p是考虑了累计概率分布来选择token

参考

https://huggingface.co/blog/how-to-generate

## **GPT2 example**

```Python
'''
GPT-2（Generative Pre-trained Transformer 2）和Transformer都是基于Transformer架构的模型，但它们有以下几个区别：

1. 预训练任务：GPT-2是通过单向语言模型（unidirectional language model）的方式进行预训练的，而Transformer则是通过双向语言模型（bidirectional language model）或者序列到序列（sequence-to-sequence）的方式进行预训练的。

2. 目标任务：GPT-2主要应用于生成型任务，比如生成文章、短文本等，而Transformer则主要应用于各种序列建模任务，比如机器翻译、语音识别、文本分类等。

3. 模型规模：GPT-2是一种较大的模型，参数量非常大，需要大量的计算资源进行训练，而Transformer相对来说更小一些。

总体来说，GPT-2是一种生成模型，用于生成与上下文相关的文本（文章、短文本等）；而Transformer则是一种用于序列建模的模型，适用于各种序列建模任务，比如机器翻译、语音识别、文本分类等。
'''
'''
GPT-2主要由以下几个模块构成：
1. 词嵌入层（Embedding Layer）：将输入的文本转化为向量表示，这些向量表示了单词或子词的语义信息。
2. 多层Transformer编码器（Multi-layer Transformer Encoder）：对输入的向量进行编码，使得模型能够理解并提取输入文本中的关键信息。
3. 掩码自注意力（Masked Self-Attention）：用于解决输入文本中的长距离依赖问题，使得模型可以更好地捕捉输入文本中不同位置标记之间的关系。
4. 前馈神经网络层（Feedforward Neural Network Layer）：用于对编码后的文本向量进行非线性变换，增加模型的表达能力。
5. Softmax输出层（Softmax Output Layer）：将输出文本的向量转化为概率分布，使得模型可以生成具有一定连贯性的文本。

'''

import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import time
from tqdm import tqdm

device = torch.device("cuda")
dict_datas = json.load(open('dict_datas.json', 'r'))
word2id, id2word = dict_datas['word2id'], dict_datas['id2word']
vocab_size = len(word2id)
max_pos = 300
d_model = 768  # 嵌入大小
d_ff = 2048  # 前馈神经网络中的维度
d_k = d_v = 64  # K(=Q), V维度
n_layers = 6  # 编码层与解码层数量
n_heads = 8  # 多头注意力机制中的头数
CLIP = 1

def make_data(datas):
    train_datas =[]
    for data in datas:
        data=data.strip()
        train_data = [i if i!='\t' else "<sep>" for i in data]+['<sep>']
        train_datas.append(train_data)

    return train_datas

class MyDataSet(Data.Dataset):
    def __init__(self,datas):
        self.datas = datas
    # 实现getitem方法，用于获取数据集中的一条数据
    def __getitem__(self, item):
         # 获取数据集中的一条数据
        data = self.datas[item]
         # 将数据分为解码器输入和解码器输出
        decoder_input = data[:-1]
        decoder_output = data[1:]
        # 计算解码器输入和输出的长度
        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)
        # 返回一个字典，包含解码器输入、解码器输入长度、解码器输出和解码器输出长度四个键值对
        return {"decoder_input":decoder_input,"decoder_input_len":decoder_input_len,
                "decoder_output":decoder_output,"decoder_output_len":decoder_output_len}

    def __len__(self):
         # 实现len方法，用于获取数据集的长度
        return len(self.datas)
     # 定义padding_batch方法，用于对批次数据进行填充
    def padding_batch(self,batch):
        # 获取批次数据中每条数据的解码器输入和解码器输出的长度，存放到列表中
        decoder_input_lens = [d["decoder_input_len"] for d in batch]
        decoder_output_lens = [d["decoder_output_len"] for d in batch]
        # 计算出批次数据解码器输入和解码器输出的最大长度
        decoder_input_maxlen = max(decoder_input_lens)
        decoder_output_maxlen = max(decoder_output_lens)

        # 对批次数据中每条数据的解码器输入和解码器输出进行填充，使它们的长度都等于最大长度
        for d in batch:
            # 在解码器输入和解码器输出的末尾添加<pad>符号，使它们的长度等于最大长度
            d["decoder_input"].extend([word2id["<pad>"]]*(decoder_input_maxlen-d["decoder_input_len"]))
            d["decoder_output"].extend([word2id["<pad>"]]*(decoder_output_maxlen-d["decoder_output_len"]))
        # 将填充后的解码器输入和解码器输出转换为tensor类型，并返回
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch],dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch],dtype=torch.long)

        return decoder_inputs,decoder_outputs

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask=subsequence_mask.to(device)
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

'''
Scaled Dot-Product Attention是多头注意力层的一个基础模块，在多头注意力层中被广泛应用。

多头注意力层的设计旨在提高模型的表达能力和多样性，通过将输入向量在不同的维度上进行拆分和组合，实现多个并行计算的注意力机制，从而让模型能够更好地关注输入的不同方面。

具体来说，多头注意力层的计算过程包括将输入的查询向量、键向量和值向量分别通过线性变换分为多个头（head），再在每个头上进行Scaled Dot-Product Attention计算和拼接，最终得到多个头的输出向量。这样，每个头可以关注不同的信息，并通过不同的线性变换学习到不同的特征表示，从而提高模型的表达能力和多样性。

因此，可以说Scaled Dot-Product Attention是多头注意力层的核心组件之一，通过它实现了单个头的注意力计算和上下文向量的生成，再通过多头并行计算的方式，增强了模型的表达能力和多样性。
'''
class ScaledDotProductAttention(nn.Module):
        def __init__(self):
            super(ScaledDotProductAttention, self).__init__()

        def forward(self, Q, K, V, attn_mask):
            '''
            Q: [batch_size, n_heads, len_q, d_k]
            K: [batch_size, n_heads, len_k, d_k]
            V: [batch_size, n_heads, len_v(=len_k), d_v]
            attn_mask: [batch_size, n_heads, seq_len, seq_len]
            '''
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
                d_k)  # scores : [batch_size, n_heads, len_q, len_k]
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

            attn = nn.Softmax(dim=-1)(scores)
            context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
            return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        # 定义四个线性变换
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        # 定义LayerNorm
        self.layernorm = nn.LayerNorm(d_model)
     # 前向传播函数
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # 记录下残差连接和batch_size
        residual, batch_size = input_Q, input_Q.size(0)
        # 将输入的Query、Key、Value通过线性变换后，分别转换为多头的形式
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # 将Attention Mask也转换为多头的形式
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # 通过Scaled Dot-Product Attention计算context和attention分数
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # 将context转换为原始形状，进行全连接操作后，加上残差连接，并进行LayerNorm操作，得到最终的输出
        context = context.transpose(1, 2).reshape(batch_size, -1,n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return self.layernorm(output + residual), attn

'''
前馈神经网络层（Feedforward Neural Network Layer）是神经网络的一种基础结构，在GPT-2等自然语言处理模型中广泛使用。前馈神经网络层主要有以下两个作用：

1. 特征提取：前馈神经网络层可以对输入的文本向量进行非线性变换，将输入的向量转化为更高维度的特征空间中的向量。这一过程可以提取输入文本中的抽象、高级别的语义特征，例如情感、话题等。

2. 非线性映射：前馈神经网络层还可以将输入的文本向量映射到目标向量空间中，实现非线性变换。这一过程通过多层神经网络中的非线性激活函数（例如ReLU、Sigmoid、tanh等）来实现。
'''
#定义了一个基于位置的前馈神经网络模型
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
         # 定义了一个全连接神经网络的容器fc，由两个线性层和一个激活函数ReLU组成
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        # 添加LayerNorm层，使输出和输入的维度匹配
        self.layernorm=nn.LayerNorm(d_model)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs  # 输入值复制到residual中，便于残差连接
        output = self.fc(inputs)   # 通过全连接神经网络处理输入值
        # 应用残差连接，并通过LayerNorm层使输出的维度匹配原始维度
        return self.layernorm(output + residual) # [batch_size, seq_len, d_model]

#定义了解码器层的模型
class DecoderLayer(nn.Module):
    # 初始化函数，实例化该模型的层和参数
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention() # 定义一个多头注意力机制的实例，作为解码器自注意力层
        self.dec_enc_attn = MultiHeadAttention() # 定义一个多头注意力机制的实例，作为解码器和编码器交互的注意力层
        self.pos_ffn = PoswiseFeedForwardNet()   # 定义一个基于位置的前馈神经网络实例，用于对解码器中的每个位置进行处理
    # 前向传播函数，接收解码器输入序列和自注意力掩码
    def forward(self, dec_inputs, dec_self_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask) # 解码器自注意力层

        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn # 返回处理后的输出及自注意力机制的输出

#定义了解码器的模型
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(vocab_size, d_model)  # 定义输入嵌入层
        self.pos_emb = nn.Embedding(max_pos,d_model)  # 定义位置嵌入层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])  # 定义多个解码器层

    # 前向传播函数，接收解码器输入序列
    def forward(self, dec_inputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        '''
        seq_len = dec_inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long,device=device)  # 生成位置向量
         # 将位置向量的维度扩展到与输入序列相同的维度
        pos = pos.unsqueeze(0).expand_as(dec_inputs)  # [seq_len] -> [batch_size, seq_len]
        # 将输入序列经过输入嵌入层和位置嵌入层进行处理
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos) # [batch_size, tgt_len, d_model]
        # 生成解码器自注意力层的mask
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # [batch_size, tgt_len, tgt_len]
        # 生成解码器自注意力层的mask
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)# [batch_size, tgt_len, tgt_len]
        # 将两个mask合并
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0) # [batch_size, tgt_len, tgt_len]

         # 用于保存每一层的解码器自注意力输出结果
        dec_self_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)  # 通过解码器层进行处理
            dec_self_attns.append(dec_self_attn)# 记录解码器自注意力输出结果

        return dec_outputs, dec_self_attns # 返回处理后的结果及每一层的解码器自注意力输出结果

class GPT(nn.Module):
    def __init__(self):
        super(GPT, self).__init__()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model,vocab_size)
    def forward(self,dec_inputs):
        """
        dec_inputs: [batch_size, tgt_len]
        """
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attns = self.decoder(dec_inputs)
        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns

    def greedy_decoder(self,dec_input):

        terminal = False
        start_dec_len=len(dec_input[0])# 获取输入序列的长度
        #一直预测下一个单词，直到预测到"<sep>"结束，如果一直不到"<sep>"，则根据长度退出循环，并在最后加上”<sep>“字符
        while not terminal :
            if len(dec_input[0])-start_dec_len>100:# 如果长度超过100，强制结束循环，并在最后加上”<sep>“字符
                next_symbol=word2id['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break
            dec_outputs, _ = self.decoder(dec_input)# 解码器输出序列
            projected = self.projection(dec_outputs) # 通过线性投影层得到解码器的输出概率分布
            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]# 获取概率最大的下一个字符
            next_word = prob.data[-1]
            next_symbol = next_word # 获取下一个字符
            if next_symbol == word2id["<sep>"]:# 如果下一个字符是"<sep>"，结束循环
                terminal = True

            dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)# 将下一个字符推入解码器
        return dec_input # 返回最终的解码器输出序列

    def answer(self,sentence):
        #把原始句子的\t替换成”<sep>“
        dec_input = [word2id.get(word,1) if word!='\t' else word2id['<sep>'] for word in sentence]
        dec_input = torch.tensor(dec_input, dtype=torch.long, device=device).unsqueeze(0)

        output = self.greedy_decoder(dec_input).squeeze(0)
        out = [id2word[int(id)] for id in output]
        #统计"<sep>"字符在结果中的索引
        sep_indexs =[]
        for i in range(len(out)):
            if out[i] =="<sep>":
                sep_indexs.append(i)
        
        #取最后两个sep中间的内容作为回答
       
        answer = out[sep_indexs[-2]+1:-1]
       
        answer = "".join(answer)
        return answer
        
'''
GPT-2模型是使用无监督的方式进行训练的，它的训练方式称为语言模型预训练。在预训练过程中，模型学习了大量的未标记的文本数据。这些数据可以来自于互联网上的网页、新闻文章、书籍等等。

在预训练的过程中，模型将文本序列作为输入，并且在学习过程中使用自监督的方式。具体来说，模型将预测一段文本序列的下一个词或者下一个token。例如，给出一个长度为n的文本序列，模型将会预测下一个token，然后将这个token连同之前的n-1个token组合成一个新的长度为n的序列，再次预测下一个token，以此类推。

这种方式的好处在于，无需人工标注数据，模型可以通过大量的未标记的数据进行学习，从而捕捉文本的语法、语义和上下文信息。另外，这种预训练方式还可用来构建一些下游任务的模型，这些任务包括文本分类、文本生成、命名实体识别等等。
'''

'''
单向语言模型只能根据前面的词语预测后面的词语，而无法同时考虑前后两个方向的语境信息。而双向语言模型可以同时考虑前后两个方向的语境信息，从而更加准确地预测下一个词语。

比如在GPT-2中，模型在预测下一个词语时，只考虑了前面的词语，而没有考虑后面的词语；而在Transformer中，模型在预测下一个词语时，同时考虑了前后两个方向的所有词语，从而更加准确地预测下一个词语。

双向语言模型的缺点是需要更复杂的模型结构和更多的计算资源，但是在处理复杂的语言任务时可以获得更好的效果。
'''

'''
gpt2的任务是根据前文内容生成下一个词或句子。因此，它没有对一个完整的输入序列进行编码的需求，而只需要对前文进行连续的自回归建模。在推理时，GPT-2模型会根据前文内容计算出每个可能的下一个词的概率分布，并生成概率最大的词作为输出。因此，其架构中只有解码器而没有编码器。
'''
```

中文预训练好的GPT模型有哪些?

https://github.com/Morizeyao/GPT2-Chinese

https://github.com/Hansen06/GPT2-Chinese

## **GPT3** **openai** **api**

既然无法对大语言模型的参数进行修改, 又无法自己训练GPT, 如何更好的使用openai 提供的 API?

AICG(Artificial Intelligence for Computer Graphics，计算机图形学的人工智能技术)

## personal chatgpt
