# coding=utf-8
from data_processor.data_builder import data_loader
from parm import *


def classify(df, col):
    mask = df[col].str.contains('花呗')
    df.loc[mask, COL_CLS] = 1

    mask = df[col].str.contains('借呗')
    df.loc[mask, COL_CLS] = 2

    df[COL_CLS] = df[COL_CLS].fillna(3)

    return df, COL_CLS


def build_dataset_preprocessed(filename, col):
    df = data_loader(filename=filename, path=PATH_DATA_PRE)
    df, col_cls = classify(df, col)
    # df = df.sort_values(by=col_cls)
    df.to_csv(os.path.join(PATH_DATA_PRE, filename + '.csv'), index=False, encoding='utf-8')


if __name__ == '__main__':
    # build_dataset_preprocessed('query', COL_TXT)
    # build_dataset_preprocessed('candidate', COL_TXT)
    # build_dataset_preprocessed('kg', 'sentence1')

    for f in ['train', 'test', 'dev']:
        build_dataset_preprocessed(f, 'sentence1')
