# src/preprocessing/data_management.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import pickle
import pandas as pd
from src.config import config

def load_dataset(file_name):
    file_path = os.path.join(config.DATAPATH, file_name)
    data = pd.read_csv(file_path)
    return data

def save_model(model_weights):
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH, "model_weights.pkl")
    with open(pkl_file_path, "wb") as file_handle:
        pickle.dump(model_weights, file_handle)

def load_model():
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH, "model_weights.pkl")
    with open(pkl_file_path, "rb") as file_handle:
        model_weights = pickle.load(file_handle)
    return model_weights

