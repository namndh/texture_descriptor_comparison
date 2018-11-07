import os
import sys

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'datas')
SAVED_MODEL_DIR = os.path.join(PROJECT_DIR, 'saved_model')
KYLBERG_MODEL = os.path.join(SAVED_MODEL_DIR, 'kylberg.bin')
KTH_2_MODEl = os.path.join(SAVED_MODEL_DIR, 'kth_2.bin')