# coding=utf-8
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from transformers import BertModel
from transformers import BertTokenizer

from parm import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# PATH_MODEL_BERT = '/Users/lixiang/Documents/nlp_data/pretrained_model/roberta_zh_large_bt_pt'
PATH_MODEL_BERT = '/home/ubuntu/MyFiles/auto_upload_20200330115115'

BATCH_SIZE = 32
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(object):
    def __init__(self, path, device):
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.language_model = BertModel.from_pretrained(path, output_hidden_states=True)
        self.language_model.to(device)


def build_sts_matrix(df, col, save_filename):
    model = Model(path=PATH_MODEL_BERT, device=DEVICE)

    def txt2id(row, max_seq_len=80):
        result = model.tokenizer.encode_plus(row[col], add_special_tokens=True, max_length=max_seq_len,
                                             pad_to_max_length=True, return_attention_mask=True)
        row['ids'] = result['input_ids']
        row['attention_mask'] = result['attention_mask']
        return row

    df = df.apply(txt2id, axis=1)

    t_ids = torch.tensor(df['ids'].values.tolist(), dtype=torch.long,
                         device=DEVICE)  # (batch_size, sequence_length)
    t_masks = torch.tensor(df['attention_mask'].values.tolist(), dtype=torch.long, device=DEVICE)
    t_dataset = TensorDataset(t_ids, t_masks)
    t_dataloader = DataLoader(dataset=t_dataset, batch_size=BATCH_SIZE)

    def layer_usage(outputs, layer_no=None, pooling_stgy=None, last_4_layers=None):

        all_hidden_state = outputs[2]

        if last_4_layers:
            # last 4 encoder layers
            last_4_hidden_state_list = all_hidden_state[-5:-1]
            # list to tensor
            last_4_hidden_state = torch.stack(last_4_hidden_state_list, dim=1)  # torch.Size([bz, 4, 80, 1024])

            if last_4_layers == 'concat':
                hidden_state = torch.cat(last_4_hidden_state_list, dim=1)  # torch.Size([bz, 4*80, 1024])
            # torch.Size([bz, 80, 1024])
            elif last_4_layers == 'sum':
                hidden_state = torch.sum(last_4_hidden_state, dim=1)
            elif last_4_layers == 'max':
                hidden_state = torch.max(last_4_hidden_state, dim=1)[0]
            else:
                hidden_state = torch.mean(last_4_hidden_state, dim=1)

        elif layer_no is not None:
            hidden_state = all_hidden_state[layer_no]
        else:
            # hidden state of last encoder layer
            hidden_state = outputs[0]
        if pooling_stgy == 'sum':
            sts_emb = torch.sum(hidden_state, dim=1)  # torch.Size([bz, 1024])
        else:
            sts_emb = torch.mean(hidden_state, dim=1)  # torch.Size([bz, 1024])
        return sts_emb

    sts_matrix = torch.tensor([], device=DEVICE)
    with torch.no_grad():
        from tqdm import tqdm
        for step, batch_data in enumerate(tqdm(t_dataloader)):
            batch_ids, batch_masks = batch_data

            outputs = model.language_model(input_ids=batch_ids, attention_mask=batch_masks)

            sts_emb = layer_usage(outputs=outputs)

            # L2-norm
            sts_emb_norm2 = torch.norm(sts_emb, p=2, dim=1, keepdim=True)  # torch.Size([bz, 1])
            sts_feat = torch.div(sts_emb, sts_emb_norm2)  # torch.Size([bz, 1024])

            # add to the matrix
            sts_matrix = torch.cat((sts_matrix, sts_feat), dim=0)

    sts_matrix.cpu()

    file = '%s.pt' % (save_filename)
    torch.save(sts_matrix, os.path.join(PATH_MD_TMP, file))

    print(file + ' created!')
    print(sts_matrix.shape)

    # for i in df[COL_CLS].unique():
    #     mask = (df[COL_CLS] == i)
    #     df_sg_cls = df[mask]
    #
    #     # to tensor
    #     t_ids = torch.tensor(df_sg_cls['ids'].values.tolist(), dtype=torch.long,
    #                          device=DEVICE)  # (batch_size, sequence_length)
    #     t_mask = torch.tensor(df_sg_cls['attention_mask'].values.tolist(), dtype=torch.long, device=DEVICE)
    #     t_dataset = TensorDataset(t_ids, t_mask)
    #     t_dataloader = DataLoader(dataset=t_dataset, batch_size=BATCH_SIZE)
    #
    #     sts_matrix = torch.tensor([], device=DEVICE)
    #     with torch.no_grad():
    #         for step, batch_data in enumerate(t_dataloader):
    #             batch_ids, batch_mask = batch_data
    #
    #             outputs = model.language_model(input_ids=batch_ids, attention_mask=batch_mask)
    #
    #             # last but one layer
    #             hidden_state = outputs[2][-2]
    #             sts_emb = torch.mean(hidden_state, 1)  # torch.Size([5, 1024])
    #
    #             # L2-norm
    #             sts_emb_norm2 = torch.norm(sts_emb, p=2, dim=1, keepdim=True)  # torch.Size([5, 1])
    #             sts_feat = torch.div(sts_emb, sts_emb_norm2)  # torch.Size([5, 1024])
    #
    #             # add to the matrix
    #             sts_matrix = torch.cat((sts_matrix, sts_feat), dim=0)
    #
    #     sts_matrix.cpu()
    #
    #     file = '%s_%i.pt' % (save_filename, i)
    #     torch.save(sts_matrix, os.path.join(PATH_MD_TMP, file))
    #
    #     print(file + ' created!')
    #     print(sts_matrix.shape)


def get_match_result(mt_A, mt_B):
    # t_dataset = TensorDataset(mt_A, mt_B)
    # t_dataloader = DataLoader(dataset=t_dataset, batch_size=BATCH_SIZE)

    print(mt_A.shape)
    print(mt_B.shape)
    print(mt_B.transpose(0, 1).shape)

    sim_mt = torch.matmul(mt_A, mt_B.transpose(0, 1))

    print(sim_mt.shape)

    sim_max_idx = torch.argmax(sim_mt, dim=1)

    return sim_max_idx.cpu().numpy()


def load_sts_matrix(filename):
    sts_matrix = torch.load(os.path.join(PATH_MD_TMP, filename + '.pt'), map_location=DEVICE)
    return sts_matrix


def compare_result(result_idx):
    from data_processor.data_builder import data_loader
    df = data_loader('kg')
    df[COL_CLS] = df[COL_CLS].astype(int)
    # mask = (df[COL_CLS] == 3)
    # df = df[mask]
    # df = df.reset_index(drop=True)

    df_c = data_loader('candidate')
    df_c[COL_CLS] = df_c[COL_CLS].astype(int)
    # mask = (df_c[COL_CLS] == 3)
    # df_c = df_c[mask]
    # df_c = df_c.reset_index(drop=True)

    # get candidate txt from idx
    df_result = df_c.loc[result_idx, COL_TXT]
    df_result = df_result.reset_index(drop=True)

    # df.loc[:, 'result'] = df.loc[: 'sentence1'].map(df_result.set_index(COL_TXT)[col_mid])

    assert len(df) == len(df_result)

    df = pd.concat([df, df_result.rename('result')], axis=1)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(df['sentence2'], df['result']))


if __name__ == '__main__':
    for filename in ['query', 'candidate']:
        from data_processor.data_builder import data_loader

        df = data_loader(filename=filename)
        df[COL_CLS] = df[COL_CLS].astype(int)

        # if filename == 'query':
        #     df = df.head()
        #     # df = df.loc[10:15, :]
        # else:
        #     df = df.head(13)
        build_sts_matrix(df=df, col=COL_TXT, save_filename=filename)

    q_mt = load_sts_matrix(filename='query')
    c_mt = load_sts_matrix(filename='candidate')
    result_idx = get_match_result(mt_A=q_mt, mt_B=c_mt)
    compare_result(result_idx)
