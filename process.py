# coding=utf-8
import pandas as pd
from utils import get_token
from w2v_processor import w2v_feat

def txt_to_vecrtor(df, col, store=False):
    df, col_token = get_token(df, col)

    get_sts_vector(df, col_token)

    print(df)


    # return df


if __name__ == '__main__':
    txt1 = '平安银行深圳分行'
    df = pd.DataFrame([txt1], columns=['text'])

    txt_to_vecrtor(df, 'text')




