# coding=utf-8
import numpy as np
from gensim.models import KeyedVectors

WV_FILE = '/Users/lixiang/Documents/nlp_data/pretrained_model/tx_w2v/45000-small.txt'
WV = KeyedVectors.load_word2vec_format(WV_FILE, binary=False)

VOCAB = [word for word in WV.vocab]
WV_UNK = np.mean(WV[VOCAB], axis=0)
vector_size = len(WV_UNK)

def get_sts_vector(df, col_token):
    col_wv = 'wv'
    wv_init = np.zeros(vector_size)

    def use_wv_plm(row):
        # init
        wv_sum = wv_init
        for word in row[col_token]:
            try:
                wv_sum = np.add(wv_sum, WV[word])
            except:
                # find unk word
                # init
                wv_sub_sum = wv_init
                for char in word:
                    try:
                        # unk word use subtoken vector
                        wv_sub_sum = np.add(wv_sub_sum, WV[char])
                    except:
                        # unk word use 0 vector
                        wv_sub_sum = wv_sub_sum

                wv_sum = np.add(wv_sum, wv_sub_sum)
        return wv_sum

    df.loc[:, col_wv] = df.apply(use_wv_plm, axis=1)
    return df, col_wv
