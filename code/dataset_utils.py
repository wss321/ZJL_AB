import tqdm
import os
import pandas as pd
import pickle
from config import DATAB_ALL_DIR, DATAA_ALL_DIR, DATASET, PKL_DIR

FILE_NAME_CID_PATH = os.path.join(PKL_DIR, 'filename_cid.pkl')


def spilt_file(file_path):
    return_data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            return_data.append(line.split())
    return return_data


def dataset_utils_main():
    label = spilt_file(os.path.join(DATAB_ALL_DIR, 'label_list.txt'))
    label_head = ['cid', 'name']

    label = pd.DataFrame(label, columns=label_head)
    label = label.set_index('cid')

    B_fname_cid = spilt_file(os.path.join(DATAB_ALL_DIR, 'train.txt'))
    A_fname_cid = spilt_file(os.path.join(DATAA_ALL_DIR, 'train.txt'))

    if DATASET == 'AB':
        ALL_fname_cid = A_fname_cid + B_fname_cid
    elif DATASET == 'A':
        ALL_fname_cid = A_fname_cid
    else:
        ALL_fname_cid = B_fname_cid
    head = ['fname', 'cid']
    ALL_fname_cid = pd.DataFrame(ALL_fname_cid, columns=head)
    fname_cid_temp = ALL_fname_cid
    ALL_fname_cid = ALL_fname_cid.set_index('fname')
    filenames = ALL_fname_cid.index.tolist()

    all_train_fine_labels = []
    for i, fname in enumerate(tqdm.tqdm(filenames, desc='Corresponding file name and class name')):
        cid = ALL_fname_cid.loc[fname].cid
        fine_name = label.loc[cid]['name']
        all_train_fine_labels.append(fine_name)
    with open(FILE_NAME_CID_PATH, 'wb') as fncp:
        print('Saving to {}'.format(FILE_NAME_CID_PATH))
        pickle.dump(
            {'all_fname_cid': fname_cid_temp, 'all_train_fine_labels': all_train_fine_labels,
             'a_fname_cid': ALL_fname_cid,
             'b_fname_cid': B_fname_cid}, fncp)
        print('Done.')


if __name__ == '__main__':
    dataset_utils_main()
    # print(len(ALL_fname_cid), set(all_train_fine_labels), len(list(set(all_train_fine_labels))))
