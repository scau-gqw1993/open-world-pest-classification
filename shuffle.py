import random
import os
import shutil
from tqdm import tqdm


def mv_file(data_dir, train_dir, test_dir):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    pest_names = os.listdir(data_dir)

    list1 = pest_names[:30]
    list2 = pest_names[30:]

    for var in tqdm(list1):
        tmp_dir = os.path.join(data_dir, var)
        id_list = os.listdir(tmp_dir)
        random.shuffle(id_list)
        idx = round(len(id_list) * 0.8)

        list_train = id_list[:idx]
        list_test = id_list[idx:]

        os.makedirs(os.path.join(train_dir, var), exist_ok=True)
        os.makedirs(os.path.join(test_dir, var), exist_ok=True)

        for pest_id in list_train:
            shutil.copyfile(os.path.join(data_dir, var, pest_id), os.path.join(train_dir, var, pest_id))
        for pest_id in list_test:
            shutil.copyfile(os.path.join(data_dir, var, pest_id), os.path.join(test_dir, var, pest_id))

    for var in tqdm(list2):
        tmp_dir = os.path.join(data_dir, var)
        id_list = os.listdir(tmp_dir)
        os.makedirs(os.path.join(test_dir, var), exist_ok=True)

        for pest_id in id_list:
            shutil.copyfile(os.path.join(data_dir, var, pest_id), os.path.join(test_dir, var, pest_id))


if __name__ == '__main__':
    data_dir = './D0'
    train_dir = './dataset/train'
    test_dir = './dataset/test'

    try:
        shutil.rmtree(train_dir)
        shutil.rmtree(test_dir)
    except:
        pass

    mv_file(data_dir, train_dir, test_dir)
