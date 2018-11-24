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

GABOR_DATA_PATHS = {'kylberg_train':os.path.join(DATA_DIR, 'kylberg_train_gabor.bin'), 
					'kylberg_test':os.path.join(DATA_DIR, 'kylberg_test_gabor.bin'),
					'kth_train':os.path.join(DATA_DIR, 'kth_train_gabor.bin'),
					'kth_test':os.path.join(DATA_DIR, 'kth_test_gabor.bin')
					}
HAAR_DATA_PATHS = {'kylberg_train':os.path.join(DATA_DIR, 'kylberg_train_haar.bin'), 
					'kylberg_test':os.path.join(DATA_DIR, 'kylberg_test_haar.bin'),
					'kth_train':os.path.join(DATA_DIR, 'kth_train_haar.bin'),
					'kth_test':os.path.join(DATA_DIR, 'kth_test_haar.bin')
					}
DB4_DATA_PATHS = {'kylberg_train':os.path.join(DATA_DIR, 'kylberg_train_db4.bin'), 
					'kylberg_test':os.path.join(DATA_DIR, 'kylberg_test_db4.bin'),
					'kth_train':os.path.join(DATA_DIR, 'kth_train_db4.bin'),
					'kth_test':os.path.join(DATA_DIR, 'kth_test_db4.bin')
					}

LBP_DATA_PATHS = {'kylberg_train':os.path.join(DATA_DIR, 'kylberg_train_lbp.bin'), 
					'kylberg_test':os.path.join(DATA_DIR, 'kylberg_test_lbp.bin'),
					'kth_train':os.path.join(DATA_DIR, 'kth_train_lbp.bin'),
					'kth_test':os.path.join(DATA_DIR, 'kth_test_lbp.bin')
					}

GLCM_DATA_PATHS = {'kylberg_train':os.path.join(DATA_DIR, 'kylberg_train_glcm.bin'), 
					'kylberg_test':os.path.join(DATA_DIR, 'kylberg_test_glcm.bin'),
					'kth_train':os.path.join(DATA_DIR, 'kth_train_glcm.bin'),
					'kth_test':os.path.join(DATA_DIR, 'kth_test_glcm.bin')
					}


datas_paths = {'gabor':GABOR_DATA_PATHS, 'haar':HAAR_DATA_PATHS, 'db4':DB4_DATA_PATHS, 'lbp':LBP_DATA_PATHS, 'glcm':GLCM_DATA_PATHS}



KNN_N_NEIGHBORS = 5

KYLBERG_CLASS_NUM = 28
KTH_TIPS_2_CLASS_NUM = 44

KYLBERG_CONFIGS = {'svm_model_path': KYLBERG_SVM_MODEL, 'knn_model_path': KYlBERG_KNN_MODEL,
						'nb_model_path': KYLBERG_NB_MODEL, 'class_num': KYLBERG_CLASS_NUM}

KTH_TIPS_2_CONFIGS = {'svm_model_path': KTH_TIPS_2_SVM_MODEL, 'knn_model_path': KTH_TIPS_2_KNN_MODEL,
						'nb_model_path': KTH_TIPS_2_NB_MODEL, 'class_num': KTH_TIPS_2_CLASS_NUM}					

KTH_TIPS2_DATA_PATH = os.path.join(DATA_DIR, KTH_TIPS2_FOLDER)
KYLBERG_DATA_PATH = os.path.join(DATA_DIR, KYLBERG_FOLDER)
