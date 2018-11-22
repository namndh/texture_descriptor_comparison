import os
import sys

KTH_TIPS2_FOLDER = 'KTH-TIPS2'
KYLBERG_FOLDER = 'KYLBERG'


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'datas')
SAVED_MODEL_DIR = os.path.join(PROJECT_DIR, 'saved_model')

KYLBERG_SVM_MODEL = os.path.join(SAVED_MODEL_DIR, 'kylberg_svm.bin')
KTH_TIPS_2_SVM_MODEL = os.path.join(SAVED_MODEL_DIR, 'kth_2_tips_svm.bin')

KYlBERG_KNN_MODEL = os.path.join(SAVED_MODEL_DIR, 'kylberg_knn.bin')
KTH_TIPS_2_KNN_MODEL = os.path.join(SAVED_MODEL_DIR, 'kth_2_tips_knn.bin')

KYLBERG_NB_MODEL = os.path.join(SAVED_MODEL_DIR, 'kylberg_nb.bin')
KTH_TIPS_2_NB_MODEL = os.path.join(SAVED_MODEL_DIR, 'kth_2_tips_nb.bin')

GABOR_DATA_PATHS = {'kylberg':os.path.join(DATA_DIR, 'kylberg_gabor.bin', 'kth':os.path.join(DATA_DIR, 'kth_gabor.bin'))}
HAAR_DATA_PATHS = {'kylberg':os.path.join(DATA_DIR, 'kylberg_haar.bin'), 'kth':os.path.join(DATA_DIR, 'kth_haar.bin')}
DB4_DATA_PATHS = {'kylberg':os.path.join(DATA_DIR, 'kylberg_db4.bin'), 'kth':os.path.join(DATA_DIR, 'kth_db4.bin')}
LBP_DATA_PATHS = {'kylberg':os.path.join(DATA_DIR, 'kylberg_lbp.bin'), 'kth':os.path.join(DATA_DIR, 'kth_lbp.bin')}
GLCM_DATA_PATHS = {'kylberg':os.path.join(DATA_DIR, 'kylberg_glcm.bin'), 'kth':os.path.join(DATA_DIR, 'kth_glcm.bin')}





KNN_N_NEIGHBORS = 5

KYLBERG_CLASS_NUM = 28
KTH_TIPS_2_CLASS_NUM = 44

KYLBERG_CONFIGS = {'svm_model_path': KYLBERG_SVM_MODEL, 'knn_model_path': KYlBERG_KNN_MODEL,
						'nb_model_path': KYLBERG_NB_MODEL, 'class_num': KYLBERG_CLASS_NUM}

KTH_TIPS_2_CONFIGS = {'svm_model_path': KTH_TIPS_2_SVM_MODEL, 'knn_model_path': KTH_TIPS_2_KNN_MODEL,
						'nb_model_path': KTH_TIPS_2_NB_MODEL, 'class_num': KTH_TIPS_2_CLASS_NUM}					

KTH_TIPS2_DATA_PATH = os.path.join(DATA_DIR, KTH_TIPS2_FOLDER)
KYLBERG_DATA_PATH = os.path.join(DATA_DIR, KYLBERG_FOLDER)
