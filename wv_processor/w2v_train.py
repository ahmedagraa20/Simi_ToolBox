# coding=utf-8
import numpy as np
import jieba.analyse
from gensim.models import Word2Vec
from parm import *

jieba.suggest_freq('花呗', True)
jieba.suggest_freq('借呗', True)
jieba.suggest_freq('网商贷', True)
jieba.suggest_freq('支付宝', True)
jieba.suggest_freq('***', True)

if __name__ == '__main__':
    from data_processor.data_builder import data_loader
    # for filename in ['query', 'candidate']
    for filename in ['query']:
        df = data_loader(filename=filename, path=PATH_DATA_PRE)
        df[COL_CLS] = df[COL_CLS].astype(int)

        if filename == 'query':
            df = df.head()
            # df = df.loc[10:15, :]
        else:
            df = df.head(13)

        df[COL_CUT] = df[COL_TXT].apply(lambda x: ' '.join(jieba.cut(x)))

        sent = [row.split(' ') for row in df[COL_CUT]]
        print(sent)
        model_trained = Word2Vec(sent, size=100, window=5, sg=0, min_count=1)

        # print(len(model_trained))

        print(len(model_trained.wv.vocab))

        print(model_trained.wv['花呗'])

        print(len(model_trained.wv['花呗']))

        # for i in model_trained:
        #     print(i)









