# coding=utf-8
import jieba

def get_token(df, col_for_cut):
    col_token = 'token'
    df[col_token] = df[col_for_cut].apply(lambda x: sorted(set(jieba.cut(x)), key=x.index))
    return df, col_token
