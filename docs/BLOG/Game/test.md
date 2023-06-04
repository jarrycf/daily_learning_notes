$
形式上, 在给定的原型间隔子序列  $$S = [x_1, x_2, \dots, x_T]$$, 位置t的核苷酸由核苷酸由 1-of-K 编码表示, 其中 K 是数据中所有核苷酸字母集合的大小, 使得 $$x_t \in [0,1]^K$$, 嵌入$mc$矩阵$$W_e$$用于将$$x_t$$映射到固定长度的向量表示(Eq.1)$
$
$$e_t = W_ex_t$$(1)$
$
> 将嵌入矩阵$$W_e$$映射到对应的$$x_t$$, 将$$x_t$$映射成固定长度的向量表示$
$
$$E=mc$$$
$
![img](https://lgb1ternmf.feishu.cn/space/api/box/stream/download/asynccode/?code=NGU1YWE4ZTc1NDcwYTUwYjc1NDljYWJiZTgxNjcwMTBfQ21vMjZJUDAwMHhubnhZd1J3ek5UazlRcXVqTUNsRFRfVG9rZW46VTZBZWI0SXFUb2tWWDR4SWpTMGNycktPbm5nXzE2ODU4NjEzNjA6MTY4NTg2NDk2MF9WNA)$
$
其中$$W_{e} \in \mathbb{R}^{d_{e} \times K}, e_{t} \in \mathbb{R}^{d_{e}}$$$
$
, e_t 输入 R de, de是向量e_t的维数$
$
类似的, 序列S中每个核苷酸的位置pt由 1-of-T 编码表示, 其中T是序列元素的数量(即原间隔序列的长度), 使得pt属于[0, 1]T, 嵌入矩阵Wp' 用于将输入pt映射到固定长度的向量表$
$
$$p_t^{\prime}=W_{p^{\prime}} p_t $$(2)$