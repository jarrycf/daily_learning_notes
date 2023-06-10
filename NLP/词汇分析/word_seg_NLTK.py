import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# 初始化词干提取器
stemmer = PorterStemmer()

# 初始化词型还原器
lemmatizer = WordNetLemmatizer()

# 示例文本
text = "I am running in a beautiful garden with my dogs"

# 分词
tokens = word_tokenize(text)

# 词干提取
stemmed_words = [stemmer.stem(token) for token in tokens]
print("词干提取结果：", stemmed_words)

# 词型还原
lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
print("词型还原结果：", lemmatized_words)
