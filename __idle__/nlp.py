import re
import string

import jieba
import jieba.posseg
import nltk
from autocorrect import Speller
from nltk.corpus import stopwords

MOD_DIR = r'D:\Information\Python\mod\utils'
# 模块路径
with open(rf'{MOD_DIR}\chinese_stopwords.txt', 'r', encoding='utf-8') as f:
    CHI_STOP_WORDS = set(map(lambda text: re.sub(r'\s', '', text), f.readlines()))
ENG_STOP_WORDS = set(stopwords.words('english'))
# 停用词: 可添加、移除
SYMBOL = set(string.punctuation) | set('~！@#￥%……&*（）——+{}|：“《》？·-=【】、；’，。/')
# 标点符号
ENG_STEMMER = nltk.stem.SnowballStemmer('english').stem
ENG_LEMMATIZER = nltk.stem.WordNetLemmatizer().lemmatize
# 词干提取函数、词形还原函数
ENG_SPELLER = Speller(lang='en')
# 拼写检查器


def chinese(text):
    ''' 中文检测'''
    return bool(re.search(r'[\u4e00-\u9fa5]', text))


def word_tokenize(sent, mode='exact'):
    ''' 中英文分词'''
    if chinese(sent):
        if mode == 'all':
            return jieba.lcut(sent, True)
        elif mode == 'exact':
            return jieba.lcut(sent, False)
        elif mode == 'search':
            return jieba.lcut_for_search(sent)
        else:
            raise AssertionError('未找到对应分词模式')
    else:
        result = nltk.tokenize.word_tokenize(sent)
        return list(map(lambda word: word.lower(), result))


def pos_tag(sent):
    ''' 分词 + 词性标注'''
    if chinese(sent):
        result = jieba.posseg.cut(sent)
        result = map(lambda pair: (pair.word, pair.flag), result)
    else:
        tokens = nltk.tokenize.word_tokenize(sent)
        result = nltk.pos_tag(tokens)
    return result


def text_process(sents, lang):
    ''' 文本数据预处理
        sents: 句子集合
        lang: 处理语言 en/ch'''
    if lang == 'en':
        tokens_list = [nltk.tokenize.word_tokenize(sent) for sent in sents]
        correct_tokens_list = [[ENG_SPELLER(word) for word in tokens] for tokens in tokens_list]
        # 分词 -> 拼写检查
        pure_tokens_list = [[word for word in tokens if word not in ENG_STOP_WORDS] for tokens in correct_tokens_list]
        # 过滤停用词
        token_pos_list = [nltk.pos_tag(tokens) for tokens in pure_tokens_list]
        token_pos_list = [[token_pos for token_pos in token_pos_tuple if token_pos[0] not in SYMBOL] for token_pos_tuple
                          in token_pos_list]
        # 词性标注 + 筛除标点符号
        stem_list = [[ENG_STEMMER(token) for token, pos in token_pos] for token_pos in token_pos_list]
        # 词干提取
        return stem_list
    elif lang == 'ch':
        tokens_list = [jieba.cut(sent, cut_all=False) for sent in sents]
        pure_tokens_list = [[word for word in tokens if word not in CHI_STOP_WORDS | SYMBOL] for tokens in tokens_list]
        # 分词 -> 过滤停用词
        return pure_tokens_list
    else:
        raise AssertionError('未找到对应的语言处理模式')


if __name__ == '__main__':
    eng = ['Hello, Mr.Tong. How are you? I want to do my homework', 'Happy birthday!']
    chi = ['你好啊，今天过得开心吗？我是潮州市第二刘禅', '是梦吗，是你吗']
    print(text_process(chi, lang='ch'))
