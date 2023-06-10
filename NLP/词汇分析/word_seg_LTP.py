from ltp import LTP

# 初始化LTP对象
ltp = LTP()

# 分句
def split_sentences(text):
    # sentences = ltp.sent_split(text)
    cws, pos, ner = ltp.pipeline(text)
    return cws

# 分词
def segment(text):
    sentences = ltp.pipeline([text])
    return sentences[0]

# 自定义分词
def custom_segment(text, words):
    ltp.add_words(words)
    sentences = ltp.seg([text])
    ltp.release_user_dict()
    return sentences[0]

# 示例用法
text = "今天是个好日子。天空万里无云。"
custom_words = ['好日子', '万里无云']

# 分句
sentences = split_sentences(text)
print("分句结果：", sentences)

# 分词
words = segment(text)
print("分词结果：", words)

# 自定义分词
custom_words_result = custom_segment(text, custom_words)
print("自定义分词结果：", custom_words_result)
