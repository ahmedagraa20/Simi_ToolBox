# coding=utf-8
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from data_processor.data_business_process import get_token
from parm import *

# WV_FILE = '/Users/lixiang/Documents/nlp_data/pretrained_model/tx_w2v/45000-small.txt'
# WV = KeyedVectors.load_word2vec_format(WV_FILE, binary=False)

# VOCAB = [word for word in WV.vocab]
# WV_UNK = np.mean(WV.wv[VOCAB], axis=0)
# vector_size = len(WV_UNK)

class ModelProcessor(object):
    def __init__(self, wv_model):
        self.wv_model = wv_model
        self.vocab = [word for word in wv_model.wv.vocab]
        self.wv_unk = np.mean(wv_model.wv[self.vocab], axis=0)
        self.vector_size = len(self.wv_unk)

    def get_sts_vector(self, df, col):
        df, col_token = get_token(df, col)

        col_wv = 'wv'
        wv_init = np.zeros(self.vector_size)

        wv_model = self.wv_model
        wv_unk = self.wv_unk

        def use_wv_plm(row):
            # init
            wv_sum = wv_init
            for word in row[col_token]:
                try:
                    wv_sum = np.add(wv_sum, wv_model.wv[word])
                except:
                    # find unk word
                    # init
                    wv_sub_sum = wv_init
                    for char in word:
                        try:
                            # unk word use subtoken vector
                            wv_sub_sum = np.add(wv_sub_sum, wv_model.wv[char])
                        except:
                            # unk word use unk vector
                            wv_sub_sum = np.add(wv_sub_sum, wv_unk)

                    wv_sum = np.add(wv_sum, wv_sub_sum)
            return wv_sum

        df.loc[:, col_wv] = df.apply(use_wv_plm, axis=1)
        return df, col_wv


def build(df, col, save=False):
    df, col_token = get_token(df, col)
    sents = [row.split(' ') for row in df[col_token]]

    model_trained = Word2Vec(sents, size=100, window=5, sg=0, min_count=1)
    if save:
        model_trained.save(os.path.join(PATH_MD_TMP, 'wv_trained.txt'))
    return model_trained


def load(filename, entire_model=False):
    if entire_model:
        model_trained = Word2Vec.load(filename)
    else:
        model_trained = KeyedVectors.load_word2vec_format(filename, binary=False)
    return model_trained


def match(df_kg, df_c, col_wv):
    col_max_value = 'max_sim_value'
    col_max_idx = 'max_sim_value_idx'

    matrix_query = np.vstack(df_kg[col_wv])
    matrix_candidate = np.vstack(df_c[col_wv])

    matrix_query_norm = matrix_query / (np.linalg.norm(matrix_query, axis=1, keepdims=True))
    matrix_candidate_norm = matrix_candidate / (np.linalg.norm(matrix_candidate, axis=1, keepdims=True))

    sim_calc = np.dot(matrix_query_norm, matrix_candidate_norm.T)

    max_sim_value = np.max(sim_calc, axis=1)
    max_sim_value_idx = np.argmax(sim_calc, axis=1)

    df_result = pd.DataFrame({col_max_value: max_sim_value, col_max_idx: max_sim_value_idx},
                             columns=[col_max_value, col_max_idx])

    df_kg = pd.concat([df_kg, df_result], axis=1)

    df_kg[['result']] = df_kg[col_max_idx].apply(lambda x: df_c.iloc[x][[COL_TXT]])

    from sklearn.metrics import accuracy_score
    print(accuracy_score(df_kg[COL_ST2], df_kg['result']))

    return df_kg


if __name__ == '__main__':
    import pandas as pd

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # train
    from data_processor.data_builder import data_loader

    df = data_loader(filename='candidate', path=PATH_DATA_PRE)
    df[COL_CLS] = df[COL_CLS].astype(int)
    model_trained = build(df, COL_TXT, save=True)

    # use
    model_trained = load(os.path.join(PATH_MD_TMP, 'wv_trained.txt'), entire_model=True)

    # model_trained = load('/Users/lixiang/Documents/nlp_data/pretrained_model/tx_w2v/45000-small.txt')

    wv_model = ModelProcessor(model_trained)
    df_kg = data_loader(filename='kg', path=PATH_DATA_PRE)
    # df_kg = df_kg.head()
    df_kg, col_wv = wv_model.get_sts_vector(df_kg, COL_ST1)

    df_c = data_loader(filename='candidate', path=PATH_DATA_PRE)
    df_c, col_wv = wv_model.get_sts_vector(df_c, COL_TXT)

    df_kg = match(df_kg, df_c, col_wv)
