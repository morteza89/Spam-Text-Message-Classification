'''
Here we will write the code for inference to test the model on any new sentence.
'''
import numpy as np
import pandas as pd
import tensorflow as tf
from main import LSTM_model, Test_Sentence
import os
from gensim.models import Doc2Vec


saved_dir = 'saved_models_directory'


def classify_sentence(message):
    '''Call this function to classify the message into one of the categories to see if that is a spam or not.'''
    # load the model
    embedding_matrix = np.load(os.path.join(saved_dir, 'embedding_matrix.npy'))
    d2v_model = Doc2Vec.load(os.path.join(saved_dir, 'd2v_model.h5'))
    X = np.load(os.path.join(saved_dir, 'X.npy'))
    tokenizer = np.load(os.path.join(saved_dir, 'tokenizer.npy'))
    model = LSTM_model(embedding_matrix, X, d2v_model)
    # load weights to the model
    model.load_weights(os.path.join(saved_dir, 'LSTMbased_Model.h5'))
    lable = Test_Sentence(message, model, tokenizer, X)
    return lable


def main():
    message = input("Enter the message: ")
    lable = classify_sentence(message)
    print("The message is: ", lable)


if __name__ == '__main__':
    main()
