import numpy as np
import tqdm
import os
import pandas as pd
from keras.preprocessing import image
import pickle
from config import SAVE_DIR, NUM_CHANNELS, DATAB_ALL_DIR, WORD_CSV_PATH, DATAB_TRAIN_DIR, IMAGE_SIZE, DATAA_ALL_DIR, \
    DATAA_TRAIN_DIR

AB_TRAIN_PICKLE = SAVE_DIR + '/AB_train.pkl'
AB_META = SAVE_DIR + '/AB_meta.pkl'
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

np.random.seed(0)


def image_matrix_to_image_array(image_matrix):
    """Gets a image matrix for use in CNNs and converts it back to a matrix"""
    image_size = np.shape(image_matrix)[1]

    image_array = image_matrix.transpose(2, 0, 1).reshape(image_size * image_size * NUM_CHANNELS, )
    return image_array


def image_array_to_image_matrix(image_array):
    """Gets a image matrix from CIFAR-100 and turns it into a matrix suitable for CNN Networks"""
    image_size = int(np.sqrt(np.prod(np.shape(image_array)) / NUM_CHANNELS))

    image_matrix = image_array.reshape(NUM_CHANNELS, image_size, image_size).transpose(1, 2, 0)
    return image_matrix


def spilt_file(file_path):
    return_data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            return_data.append(line.split())
    return return_data


def create_pickle_file_main():
    attr = spilt_file(os.path.join(DATAB_ALL_DIR, 'attributes_per_class.txt'))
    label = spilt_file(os.path.join(DATAB_ALL_DIR, 'label_list.txt'))
    wdebd = spilt_file(os.path.join(DATAB_ALL_DIR, 'class_wordembeddings.txt'))

    # dict format {'class':cid}
    class2cid = dict()
    for list_cid_cla in label:
        class2cid[list_cid_cla[1]] = list_cid_cla[0]

    attr_head = list(['label',
                      # 种类
                      'animal', 'transportation', 'clothes', 'plant', 'tableware', 'device',
                      # 颜色
                      'black', 'white', 'blue', 'brown', 'orange', 'red', 'green', 'yellow',
                      # 有哪些
                      'has_feathers', 'has_four_legs', 'has_two_legs', 'has_two_arms',
                      # 用途
                      'for_entertainment', 'for_business', 'for_communication', 'for_family', 'for_office use',
                      'for_personal',
                      # 个性
                      'gorgeous', 'simple', 'elegant', 'cute', 'pure', 'naive'
                      ])
    label_head = ['cid', 'name']

    attr = pd.DataFrame(attr, columns=attr_head)
    attr = attr.set_index('label')
    label = pd.DataFrame(label, columns=label_head)
    label = label.set_index('cid')
    wdebd = pd.DataFrame(wdebd)
    wdebd = wdebd.set_index(0)

    # 词嵌入对应类
    index = []
    for name in wdebd.index:
        for idx in label.index:
            if label.loc[idx]['name'] == name:
                index.append(label.loc[idx].name)
    # wdebd['cid']=index

    attr_cla = attr.loc[:, 'animal':'device'].astype('float')
    attr_clo = attr.loc[:, 'black':'yellow'].astype('float')
    attr_has = attr.loc[:, 'has_feathers':'has_two_arms']
    attr_use = attr.loc[:, 'for_entertainment':'for_personal'].astype('float')
    attr_psn = attr.loc[:, 'gorgeous':'naive']

    attr_cla = (attr_cla >= 1) & 1
    attr_cla['unkown'] = 1 - attr_cla.sum(axis=1)  # 都不属于的类
    # 颜色属性不太好用（比如car\pan\cost），这里就不做复杂处理，模型训练时暂不用
    # attr_clo = (attr_clo > '0.7') & 1
    # attr_clo.sum(axis = 1)
    # attr_clo = attr_clo.mul(pd.DataFrame(np.ones((230,8))*10.0,columns=attr_clo.columns))
    # attr_has = (attr_has >= '1') & 1
    # attr_has.sum(axis = 1)
    # attr_use = attr_use.mul(pd.DataFrame(np.ones((230,6))*10.0,columns=attr_use.columns))
    # attr_use = (attr_use >= '1') & 1
    # attr_use.sum(axis = 1)
    # 个性不用
    data_attr = pd.DataFrame(attr.index)
    data_attr = pd.DataFrame(np.hstack((data_attr, attr_cla, attr_clo, attr_has, attr_use)))

    word2vec = wdebd
    fine_lables = word2vec.index.tolist()
    word2vec.to_csv(WORD_CSV_PATH, index=None)
    print("word2vec data saved to " + WORD_CSV_PATH)

    data_atten = pd.read_csv(WORD_CSV_PATH)
    # data_atten = data_atten.set_index('cid')
    B_fname_cid = spilt_file(os.path.join(DATAB_ALL_DIR, 'train.txt'))
    A_fname_cid = spilt_file(os.path.join(DATAA_ALL_DIR, 'train.txt'))
    ALL_fname_cid = A_fname_cid + B_fname_cid
    head = ['fname', 'cid']
    ALL_fname_cid = pd.DataFrame(ALL_fname_cid, columns=head)

    # 训练集包含的类
    cla_cid = []
    for cid in ALL_fname_cid['cid']:
        if cid not in cla_cid:
            cla_cid.append(cid)
    # 保存训练集每个类对应的文件
    class_cid = dict()
    for c_cid in cla_cid:
        fname = ALL_fname_cid[ALL_fname_cid['cid'] == c_cid]['fname']
        fname.index = list(range(len(fname)))
        fname.columns = 'fname'
        fname.name = c_cid
        class_cid[c_cid] = fname

    def get_coarse_name(df, cid, val):
        for column in df.columns:
            if (df.loc[cid][column] == val):
                return column

    ALL_fname_cid = ALL_fname_cid.set_index('fname')
    filenames = ALL_fname_cid.index.tolist()
    # cids = label.index.tolist()
    coarse_label_names = attr_cla.columns.tolist()
    fine_label_names = label.iloc[:, 0].tolist()
    fine_labels = []
    coarse_labels = []
    data = np.zeros(shape=(len(filenames), IMAGE_SIZE * IMAGE_SIZE * 3), dtype='int16')
    all_cid = []
    batch_label = 'training batch 1 of 1'
    i = 0
    # print(fine_label_names)
    # print(ALL_fname_cid)
    # print(filenames)
    for i, fname in enumerate(tqdm.tqdm(filenames, desc='Creating Corresponding Data')):
        if i < len(A_fname_cid):
            path = DATAA_TRAIN_DIR
        else:
            path = DATAB_TRAIN_DIR
        cid = ALL_fname_cid.loc[fname].cid
        all_cid.append(cid)
        fine_name = label.loc[cid]['name']
        label_num = fine_label_names.index(fine_name)

        # 1
        fine_labels.append(label_num)
        # 2 加载图片
        img = image.load_img(os.path.join(path, fname))
        img = image.img_to_array(img)
        data[i] = img.reshape(IMAGE_SIZE * IMAGE_SIZE * 3)
        i += 1
        # 找到对应的大类
        cname = get_coarse_name(attr_cla, cid, 1)
        # 3
        coarse_labels.append(coarse_label_names.index(cname))
    print('data shape:', data.shape)
    print('Saving pickle file to {}'.format(SAVE_DIR))

    train_f = open(AB_TRAIN_PICKLE, 'wb')
    meta_f = open(AB_META, 'wb')
    pickle.dump({'fine_labels': fine_labels, 'coarse_labels': coarse_labels, 'batch_label': batch_label, 'data': data,
                 'filenames': filenames, 'all_cid': all_cid},
                train_f, protocol=4)
    pickle.dump(
        {'coarse_label_names': coarse_label_names, 'fine_label_names': fine_label_names, 'class2cid': class2cid},
        meta_f, protocol=4)

    # # 释放空间
    # coarse_label_names = None
    # fine_label_names = None
    # class2cid = None
    # fine_name = None
    # batch_label = None
    # data = None
    # filenames = None
    # all_cid = None
    # fine_labels = None
    # data_attr = None
    # data_atten = None
    # path = None
    # A_fname_cid = None
    # B_fname_cid = None
    # word2vec = None
    meta_f.close()
    train_f.close()
    print('Done.')


if __name__ == '__main__':
    create_pickle_file_main()
