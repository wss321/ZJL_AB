"""
configuration of this project.

"""
import os

# ---------- Training Choices ---------------------

TRAIN = 'class'  # Choose training pattern: 'composite' or 'class'
WV_OR_ATTR = 'WV'  # Choose 'WV' or 'ATTR' to use word2vec or attribute
TF_MODEL = 'alexnet'  # Choose model: 'vgg19' 'alexnet' 'densenet' 'resnet_v2_50'
OPTIMIZER = 'adam'  # Choose optimizer: 'sgd' 'adadelta' 'momentum' 'adam' 'rmsp' 'adagrad'
KERAS_MODEL = 'densenet'  # 'own_model' 'densenet'

# Choose loss function : 'eucli_loss' 'prod_loss'  mse_loss' 'no_margin_prod_loss' 'rel_w_prod_loss' 'cross_ent_loss'
# default :'prod_loss'
CHOOSE_LOSS = 'eucli_loss'
if TRAIN == 'class':
    CHOOSE_LOSS = 'cross_ent_loss'
DEM_MODEL = 2  # 2
# --------------------------------------------------------------------------------------------------------------------

USE_EMBEDDING = True  # whether use embedding, including attribute embedding or word2vec
if TRAIN == 'class':
    USE_EMBEDDING = False
DISTORT = True  # whether use image distort for data argument
LOAD_CKPT = False  # whether load latest checkout point
lr_decay = True  # whether learning rate decay
SHOW_TOP_K = False  # whether show the top-k metrics of training
training_flag = True  # whether trainable (ResNet)
AUTO_COMPUTE = True  # for compute quantitative results
use_reg = True  # whether use regulation term for loss function
VALIDATION = True  # whether validate
WORD_VALID_SHOW = True  # whether show the intermediate result
# for visualize results
KNOWN_CLASSES = False  # Change here
ZERO_SHOT_CLASSES = True  # Change here

# -------- Training Parameter----------------

initial_learning_rate = 0.0001
num_classes = 205
batch_size = 16
num_epochs = 100
dropout_rate = 0.2  # for vgg19

momentum = 0.0  # The momentum parameter of momentum optimizer
LOSS_MARGIN = 1  # 1 loss margin of the build_prod_loss loss function
EVALUATE_STEP = 1  # How many evaluate step in training
TOP_K = 3  # The number k of top-k
IMAGE_SIZE = 64
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

growth_k = 24
nb_block = 4  # how many (dense block + Transition Layer) ?
# init_learning_rate = 1e-4
# epsilon = 1e-4                  # AdamOptimizer epsilon
# ----------------------------------------------------


# ----------------------  Dir Information  ----------------------

TRAINING_DIR = '../data'  # Training output dir
PKL_DIR = '../data'  # the dir of the image data pickle file

DATAA_ALL_DIR = r'../data/DatasetA'  # Dataset dir
DATAA_TRAIN_DIR = r'../data/DatasetA/train'  # Training image data dir
DATAB_ALL_DIR = r'../data/DatasetB'  # Dataset dir
DATAB_TRAIN_DIR = r'../data/DatasetB/train'  # Training image data dir
TEST_DATA_DIR = r'../data/DatasetB/test'  # Test image data dir

OUTPUT_FILES_FOLDER = '{}/logs'.format(TRAINING_DIR)  # Training logs dir
EXPAND = CHOOSE_LOSS  # ckpt dir expand
filewriter_path = '{}/{}/checkpoints/{}/{}/{}'.format(TRAINING_DIR, TRAIN, TF_MODEL, EXPAND, WV_OR_ATTR)
checkpoint_path = '{}/{}/checkpoints/{}/{}/{}'.format(TRAINING_DIR, TRAIN, TF_MODEL, EXPAND,
                                                      WV_OR_ATTR)  # checkpoint path
word2vec_data_file = '{}/class_wordembeddings.txt'.format(DATAB_ALL_DIR)  # word2vec data file path
attr_data_file = '{}/attributes_per_class.txt'.format(DATAB_ALL_DIR)  # attribute data file path

SAVE_DIR = PKL_DIR  # create target or not target pickle file dir
WORD_CSV_PATH = '{}/word_embedding.csv'.format(PKL_DIR)  # save word2vec table dir

# TRAIN_PKL = '{}/train.pkl'.format(PKL_DIR)  # create training image data pickle file dir
# META_PKL = '{}/meta.pkl'.format(PKL_DIR)  # create label data pickle file dir
# ---------------------------------------------------------------------------

# --------- pickle file dir of saving create target and not target data ----

SAVE_TTRD_PKL = '{}/target_train_data.pickle'.format(PKL_DIR)
SAVE_NTTRD_PKL = '{}/not_target_train_data.pickle'.format(PKL_DIR)
SAVE_ALL_LABEL_PKL = '{}/test_class.pickle'.format(PKL_DIR)
SAVE_VEC_PKL = '{}/vectorizer.pickle'.format(PKL_DIR)
# --------------------------------------------------------------------------

# ---------------  for quantitative utils  -------------------------------------

ALL_LABLES_FILE = '{}/all_labels.pickle'.format(PKL_DIR)
OUTPUT_FILES = ['result_zsl_only']

# -----------------------  Create dir  -------------------------------------

if not os.path.exists(filewriter_path):
    os.makedirs(filewriter_path)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.exists(PKL_DIR):
    os.makedirs(PKL_DIR)
if not os.path.exists(OUTPUT_FILES_FOLDER):
    os.makedirs(OUTPUT_FILES_FOLDER)
# ----------------------------------------------------------------------------


# -------- new config of keras_train ----------
MODEL_FOLDER = r'../data/keras/Model/{}/{}'.format(KERAS_MODEL, OPTIMIZER)  # 模型保存地址
TB_LOG = '../data/keras/{}/{}/log/'.format(KERAS_MODEL, OPTIMIZER)  # tensorbord 文件地址
CKPT_PATH = '../data/keras/{}/{}/checkoutpoint'.format(KERAS_MODEL, OPTIMIZER)  # 查看点路径

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
if not os.path.exists(TB_LOG):
    os.makedirs(TB_LOG)
if not os.path.exists(CKPT_PATH):
    os.makedirs(CKPT_PATH)

# BAET_CLASSIFY_CKPT_FILE = '{}/best_one.ckpt'.format(CKPT_PATH)
BAET_CLASSIFY_CKPT_FILE = 'G:\kaggle-output\checkoutpointbest_one.ckpt'
MODEL_DIR = '{}/keras_train.h5'.format(MODEL_FOLDER)
# --------------------------------------------
if __name__ == '__main__':
    cf = {
        'Model': TF_MODEL,
        'USE_W2V': USE_EMBEDDING,
        'LOAD_CKPT': LOAD_CKPT,
        'DISTORT': DISTORT,
        'initial_learning_rate': initial_learning_rate,
        'TRAIN_TEST_SPLIT_RATE': TRAIN_TEST_SPLIT_RATE,
        'ClASS SPLIT_RATE': SPLIT_RATE,
        'WV_OR_ATTR': WV_OR_ATTR
    }
    print('Main Config:\n', cf)
    print(SAVE_DIR)
