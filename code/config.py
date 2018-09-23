"""
configuration of this project.

"""
import os

# ---------- Training Choices ---------------------

TRAIN = 'class'  # Choose training pattern: 'composite' or 'class'
WV_OR_ATTR = 'WV'  # Choose 'WV' or 'ATTR' to use word2vec or attribute
OPTIMIZER = 'adam'  # Choose optimizer: 'sgd' 'adam'
KERAS_MODEL = 'densenet'  # 'own_model' 'densenet'

DEM_MODEL = 2  # 2
# --------------------------------------------------------------------------------------------------------------------

DISTORT = True  # whether use image distort for data argument
LOAD_CKPT = False  # whether load latest checkout point
lr_decay = True  # whether learning rate decay
use_reg = True  # whether use regulation term for loss function
ZERO_SHOT_CLASSES = True  # Change here

# -------- Training Parameter----------------

classifier_init_lr = 1e-4
num_classes = 205
classifier_batch_size = 16
classifier_num_epochs = 100

IMAGE_SIZE = 64
resize = 64
NUM_CHANNELS = 3  # the number of image channel
MAX_TO_KEEP = 10  # save checkout point num
if WV_OR_ATTR == 'WV':
    embedding_size = 300
else:
    embedding_size = 30
# --------------- Data Split -----------------------------------------

SPLIT_RATE = 0.8  # target and not target classes split rate
TRAIN_TEST_SPLIT_RATE = 0.8  # The split rate between training data and validation data
if TRAIN == 'class':
    SPLIT_RATE = 1.0

CLOSE_WORD_NUM = 5  # Quantitative parameter

# -----HyperParameter of DenseNet in models.py -------

depth = 64
nb_block = 4  # how many (dense block + Transition Layer) ?
growth_rate = 12
dropout_rate = 0.1
reduction = 0.0
# ----------------------------------------------------

# -----HyperParameter of VGG_BN in models.py -------

vgg_norm_rate = 0.0
# ----------------------------------------------------

# DEM training parameters
DEM_LOAD_CKPT = False
dem_init_lr = 1e-4
dem_batch_size = 30000
dem_split_rate = 0.9
dem_cosine_loss_r = 0.0
dem_eu_loss_r = 1.0
dem_reg_loss_r = 1e-5
dem_hidden_layer = 1024
# ---------------------

# ----------------------  Dir Information  ----------------------

TRAINING_DIR = '../data'  # Training output dir
PKL_DIR = '../data'  # the dir of the image data pickle file

DATAA_ALL_DIR = r'../data/DatasetA'  # Dataset dir
DATAA_TRAIN_DIR = r'../data/DatasetA/train'  # Training image data dir
DATAB_ALL_DIR = r'../data/DatasetB'  # Dataset dir
DATAB_TRAIN_DIR = r'../data/DatasetB/train'  # Training image data dir
TEST_DATA_DIR = r'../data/DatasetB/test'  # Test image data dir
SUBMIT_PATH = r'../data/Submit.txt'
DEM_OUTPUT_FILES_FOLDER = '{}/DEM/logs'.format(TRAINING_DIR)  # Training logs dir

word2vec_data_file = '{}/class_wordembeddings.txt'.format(DATAB_ALL_DIR)  # word2vec data file path
attr_data_file = '{}/attributes_per_class.txt'.format(DATAB_ALL_DIR)  # attribute data file path

SAVE_DIR = PKL_DIR  # create target or not target pickle file dir
WORD_CSV_PATH = '{}/word_embedding.csv'.format(PKL_DIR)  # save word2vec table dir

# --------- pickle file dir of saving create target and not target data ----

SAVE_TTRD_PKL = '{}/target_train_data.pickle'.format(PKL_DIR)
SAVE_NTTRD_PKL = '{}/not_target_train_data.pickle'.format(PKL_DIR)
SAVE_ALL_LABEL_PKL = '{}/test_class.pickle'.format(PKL_DIR)
SAVE_VEC_PKL = '{}/vectorizer.pickle'.format(PKL_DIR)
# --------------------------------------------------------------------------

# ---------------  for quantitative utils  -------------------------------------

ALL_LABLES_FILE = '{}/all_labels.pickle'.format(PKL_DIR)

# -----------------------  Create dir  -------------------------------------

if not os.path.exists(PKL_DIR):
    os.makedirs(PKL_DIR)
if not os.path.exists(DEM_OUTPUT_FILES_FOLDER):
    os.makedirs(DEM_OUTPUT_FILES_FOLDER)
# ----------------------------------------------------------------------------

# -------- new config of keras_train ----------
MODEL_FOLDER = r'../data/keras/{}/{}/model/'.format(KERAS_MODEL, OPTIMIZER)  # 模型保存地址
TB_LOG = '../data/keras/{}/{}/log/'.format(KERAS_MODEL, OPTIMIZER)  # tensorbord 文件地址
CKPT_PATH = '../data/keras/{}/{}/checkoutpoint'.format(KERAS_MODEL, OPTIMIZER)  # 查看点路径

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
if not os.path.exists(TB_LOG):
    os.makedirs(TB_LOG)
if not os.path.exists(CKPT_PATH):
    os.makedirs(CKPT_PATH)

BAET_CLASSIFY_CKPT_FILE = '{}/best_one.ckpt'.format(CKPT_PATH)
# BAET_CLASSIFY_CKPT_FILE = 'G:\kaggle-output\checkoutpointbest_one.ckpt'
MODEL_DIR = '{}/keras_train.h5'.format(MODEL_FOLDER)
# --------------------------------------------

if __name__ == '__main__':
    cf = {
        'LOAD_CKPT': LOAD_CKPT,
        'DISTORT': DISTORT,
        'classifier_init_lr': classifier_init_lr,
        'TRAIN_TEST_SPLIT_RATE': TRAIN_TEST_SPLIT_RATE,
        'ClASS SPLIT_RATE': SPLIT_RATE,
        'WV_OR_ATTR': WV_OR_ATTR
    }
    print('Main Config:\n', cf)
    print(SAVE_DIR)
