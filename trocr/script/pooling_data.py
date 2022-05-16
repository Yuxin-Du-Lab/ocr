# this script used to pool text of web, document and scene together
import os
from tqdm import tqdm
import lmdb
import random

root_path = '/home/duyx/workspace/data/benchmark_dataset/'
dataset_name = ['web', 'document', 'scene']
split = ['train', 'valid', 'test']
web_path = root_path + dataset_name[0] + '/'
document_path = root_path + dataset_name[1] + '/'
scene_path = root_path + dataset_name[2] + '/'
result_path = '/home/duyx/workspace/code/OCR/unilm/trocr/data/'

def load_text(env):
    data = []
    length = -1
    with env.begin(write=False) as txn:
            length = txn.stat()['entries'] - 1
            print('MDB loading data...')
            for key, value in tqdm(list(txn.cursor()), desc='Loading MDB:'):
                if key != b'num-samples':
                    item_type, idx_str = str(key, encoding='UTF-8').split('-')
                    idx = int(idx_str) - 1
                    if item_type == 'image':
                        # data.append({'img_path': key, 'image_id':idx, 'text':None, 'encoded_str':None})
                        continue
                    elif item_type == 'label':
                        label = str(value, encoding='UTF-8')
                        data.append(label)
                    else:
                        continue
            print('Dataset size---' + str(len(data)))
    return length, data


def load_split(split_path):
    print("loading---" + split_path)
    env = lmdb.open(split_path, subdir=os.path.isdir(split_path),
                        readonly=True, lock=False,
                        readahead=False, meminit=False)
    _, data = load_text(env)
    return data


def load_datasets():
    train_all = []
    valid_all = []
    test_all = []
    # load train
    train_all.extend(load_split(web_path + dataset_name[0] + '_' + split[0]))
    train_all.extend(load_split(document_path + dataset_name[1] + '_' + split[0]))
    train_all.extend(load_split(scene_path + dataset_name[2] + '_' + split[0]))
    print("train_all size---" + str(len(train_all)))
    # load valid
    valid_all.extend(load_split(web_path + dataset_name[0] + '_' + split[1]))
    valid_all.extend(load_split(document_path + dataset_name[1] + '_' + split[1]))
    valid_all.extend(load_split(scene_path + dataset_name[2] + '_' + split[1]))
    print("valid_all size---" + str(len(valid_all)))

    # load test
    test_all.extend(load_split(web_path + dataset_name[0] + '_' + split[2]))
    test_all.extend(load_split(document_path + dataset_name[1] + '_' + split[2]))
    test_all.extend(load_split(scene_path + dataset_name[2] + '_' + split[2]))
    print("test_all size---" + str(len(test_all)))
    return train_all, valid_all, test_all

def generate_file(data, file_name):
    with open(result_path + file_name, 'w') as f:
        for line in data:
            f.write(line + '\n')

if __name__ == '__main__':
    train_all, valid_all, test_all = load_datasets()
    random.shuffle(train_all)
    random.shuffle(valid_all)
    random.shuffle(test_all)
    generate_file(train_all, 'train.cn')
    generate_file(valid_all, 'valid.cn')
    generate_file(test_all, 'test.cn')
    # train_all size---1021635
    # valid_all size---127704
    # test_all size---127705

