import re
import string
from pathlib import Path

import jieba
import jieba.posseg
import nltk

SYMBOL = set(string.punctuation) | set("~！@#￥%……&*（）——+{}|：“《》？·-=【】、；’，。/")


def chinese(text):
    return bool(re.search(r"[\u4e00-\u9fa5]", text))


class ChineseProcessor:

    def __init__(self, root=Path()):
        self.stopwords = set((root / "chinese_stopwords.txt").read_text("utf-8").split())

    def tokenize(self, sent, mode="exact"):
        if mode == "all":
            return jieba.lcut(sent, True)
        elif mode == "exact":
            return jieba.lcut(sent, False)
        elif mode == "search":
            return jieba.lcut_for_search(sent)
        raise AssertionError(f"Unrecognized mode: {mode}")

    def pos_tag(self, sent):
        return map(lambda pair: (pair.word, pair.flag), jieba.posseg.cut(sent))

    def process(self, sents):
        # 分词 -> 过滤停用词
        tokens_list = [jieba.cut(sent, cut_all=False) for sent in sents]
        return [[word for word in tokens if word not in self.stopwords | SYMBOL] for tokens in tokens_list]


class EnglishProcessor:

    def __init__(self):
        from nltk.corpus import stopwords
        self.stopwords = set(stopwords.words("english"))
        # 词干提取函数、词形还原函数
        self.stemmer = nltk.stem.SnowballStemmer("english").stem
        # self.lemmatizer = nltk.stem.WordNetLemmatizer().lemmatize

    def tokenize(self, sent):
        result = nltk.tokenize.word_tokenize(sent)
        return map(str.lower(), result)

    def pos_tag(self, sent):
        return nltk.pos_tag(self.tokenize(sent))

    def process(self, sents):
        # 分词 -> 过滤停用词
        tokens_list = [self.tokenize(sent) for sent in sents]
        pure_tokens_list = [[word for word in tokens if word not in self.stopwords] for tokens in tokens_list]
        # 词性标注 + 筛除标点符号
        token_pos_list = [nltk.pos_tag(tokens) for tokens in pure_tokens_list]
        token_pos_list = [[token_pos for token_pos in token_pos_tuple if token_pos[0] not in SYMBOL] for token_pos_tuple
                          in token_pos_list]
        # 词干提取
        return [[self.stemmer(token) for token, pos in token_pos] for token_pos in token_pos_list]


if __name__ == "__main__":
    eng = ["Hello, Mr.Tong. How are you? I want to do my homework", "Happy birthday!"]
    chi = ["你好啊，今天过得开心吗？我是潮州市第二刘禅", "是梦吗，是你吗"]
    print(ChineseProcessor().process(chi))
    print(EnglishProcessor().process(eng))
