import jieba
sentence = "如果放到数据库中将出错"
print(jieba.lcut(sentence)) # ['如果', '放到', '数据库', '中将', '出错']

# 一. 调节词频

# 将 "出错" 分成 "出" "错"
jieba.suggest_freq(('出', '错'), True)
print(jieba.lcut(sentence)) # ['如果', '放到', '数据库', '中将', '出', '错']

# 将 "中将" "出" 的分词结果合并成 '中将出'
jieba.suggest_freq(('中将出'), True)
print(jieba.lcut(sentence))

'''
能否通过 jieba.suggest_freq(('到数据库'), True) 将 '放到', '数据库' 分成 '放', '到数据库'
在.ipynb文件中的suggest_freq()在运行后只能通过重新加载内核撤销
'''

# 二. 添加自定义词

# 添加自定义词'到数据库中'
jieba.add_word("到数据库中")
print(jieba.lcut(sentence))


# 三. 自定义词典

# 从当前文件夹下创建 user_dict.txt 并添加'如果放' 这个分词结果, 并加载自定义词典
import os
curr_pth = os.path.dirname(__file__)
curr_pth = os.path.dirname(curr_pth)

jieba.load_userdict("user_dict.txt")
print(jieba.lcut(sentence))