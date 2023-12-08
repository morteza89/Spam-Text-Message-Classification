'''

'''
from utils import plot_data, plot_history
import os
from sklearn.metrics import confusion_matrix
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
import nltk
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import re
from gensim.models.doc2vec import TaggedDocument
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn import utils
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
tqdm.pandas(desc="progress-bar")
# import gensim
# from sklearn.linear_model import LogisticRegression
# from nltk.corpus import stopwords
#####################################


def read_data(input_data):
    '''Read the data from the csv file'''
    df = pd.read_csv(input_data, delimiter=',', encoding='latin-1')
    df = df[['Category', 'Message']]
    df = df[pd.notnull(df['Message'])]  # remove the null rows
    df.rename(columns={'Message': 'Message'}, inplace=True)
    df.index = range(5572)
    # count the number of words
    df['Message'].apply(lambda x: len(x.split(' '))).sum()
    return df


def print_message(df, index):
    '''Print the message and the category of the message'''
    example = df[df.index == index][['Message', 'Category']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Message:', example[1])


def cleanText(text):
    '''Clean the text by removing the special characters'''
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) <= 0:
                continue
            tokens.append(word.lower())
    return tokens


def prepare_data(train, test, df, max_fatures, MAX_SEQUENCE_LENGTH):
    '''Prepare the data for the model'''

    train_tagged = train.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.Category]), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.Category]), axis=1)
    # tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer = Tokenizer(num_words=max_fatures, split=' ',
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['Message'].values)
    X = tokenizer.texts_to_sequences(df['Message'].values)
    X = pad_sequences(X)
    print('Found %s unique tokens.' % len(X))
    X = tokenizer.texts_to_sequences(df['Message'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    Y = pd.get_dummies(df['Category']).values
    print('Shape of data tensor:', X.shape)
    return X, Y, train_tagged, test_tagged, tokenizer


def training_Doc2Vec(train_tagged, epoch=100):
    '''Train the Doc2Vec model'''
    d2v_model = Doc2Vec(dm=1, dm_mean=1, size=20, window=8,
                        min_count=1, workers=1, alpha=0.065, min_alpha=0.065)
    d2v_model.build_vocab([x for x in tqdm(train_tagged.values)])
    for epoch in range(epoch):
        d2v_model.train(utils.shuffle([x for x in tqdm(
            train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
        d2v_model.alpha -= 0.002
        d2v_model.min_alpha = d2v_model.alpha
    #
    # save the vectors in a new matrix
    embedding_matrix = np.zeros((len(d2v_model.wv.vocab) + 1, 20))

    for i, vec in enumerate(d2v_model.docvecs.vectors_docs):
        while i in vec <= 1000:
            embedding_matrix[i] = vec
    #
    d2v_model.wv.most_similar(positive=['urgent'], topn=10)
    d2v_model.wv.most_similar(positive=['call'], topn=10)
    d2v_model.wv.most_similar(positive=['win'], topn=10)
    return embedding_matrix, d2v_model


def LSTM_model(embedding_matrix, X, d2v_model):
    '''Define the LSTM model'''
    model = Sequential()
    # emmbed word vectors
    model.add(Embedding(len(d2v_model.wv.vocab)+1, 20,
              input_length=X.shape[1], weights=[embedding_matrix], trainable=True))
    # learn the correlations
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(2, activation="softmax"))
    # output model skeleton
    model.summary()
    model.compile(optimizer="adam",
                  loss="binary_crossentropy", metrics=['acc'])
    #
    return model


def Test_LSTM_model(model, X_train, Y_train, X_test, Y_test, validation_size=200, batch_size=32):
    '''Test the LSTM model on the test set'''
    # evaluate the model
    _, train_acc = model.evaluate(X_train, Y_train, verbose=2)
    _, test_acc = model.evaluate(X_test, Y_test, verbose=2)
    print('Train: %.3f, Test: %.4f' % (train_acc, test_acc))
    # predict probabilities for test set
    yhat_probs = model.predict(X_test, verbose=0)
    print(yhat_probs)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(X_test, verbose=0)
    print(yhat_classes)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    #
    rounded_labels = np.argmax(Y_test, axis=1)
    cm = confusion_matrix(rounded_labels, yhat_classes)
    print(cm)
    lstm_val = confusion_matrix(rounded_labels, yhat_classes)
    f, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(lstm_val, annot=True, linewidth=0.7,
                linecolor='cyan', fmt='g', ax=ax, cmap="BuPu")
    plt.title('LSTM Classification Confusion Matrix')
    plt.xlabel('Y predict')
    plt.ylabel('Y test')
    plt.show()
    plt.savefig('LSTM_confusion_matrix.png')
    #
    X_validate = X_test[-validation_size:]
    Y_validate = Y_test[-validation_size:]
    X_test = X_test[:-validation_size]
    Y_test = Y_test[:-validation_size]
    score, acc = model.evaluate(
        X_test, Y_test, verbose=1, batch_size=batch_size)

    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))


def Test_Sentence(message, model, tokenizer, X, labels=['ham', 'spam']):
    seq = tokenizer.texts_to_sequences(message)

    padded = pad_sequences(seq, maxlen=X.shape[1], dtype='int32', value=0)

    pred = model.predict(padded)

    print("Here is the predicted label", pred, labels[np.argmax(pred)])


def run(input_data, plt_history=True, output_dir=None, plot_data_dist=True):
    # The maximum number of words to be used. (most frequent)
    max_fatures = 500000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 50
    EPOCHES = 100
    BATCH_SIZE = 32

    df = read_data(input_data)
    if plot_data_dist:
        plot_data(df)
    print_message(df, 10)
    df['Message'] = df['Message'].apply(cleanText)
    train, test = train_test_split(df, test_size=0.000001, random_state=42)
    X, Y, train_tagged, test_tagged, tokenizer = prepare_data(
        train, test, df, max_fatures, MAX_SEQUENCE_LENGTH)
    # train the Doc2Vec model
    embedding_matrix, d2v_model = training_Doc2Vec(train_tagged, epoch=100)
    # train the LSTM model
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.15, random_state=42)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    #
    model = LSTM_model(embedding_matrix, X, d2v_model)
    # fit the model and save the model
    history = model.fit(X_train, Y_train, epochs=EPOCHES,
                        batch_size=BATCH_SIZE, verbose=2)
    # in output_dir make a model_weights directory and save the model there
    if output_dir is None:
        output_dir = os.getcwd()
    # first make the directory
    os.makedirs(os.path.join(output_dir, 'model_weights'), exist_ok=True)
    # save the model
    model.save(os.path.join(output_dir, 'model_weights', 'LSTMbased_Model.h5'))
    # Save embedding_matrix, d2v_model, X, tokenizer for future use
    np.save(os.path.join(output_dir, 'model_weights',
            'embedding_matrix.npy'), embedding_matrix)
    d2v_model.save(os.path.join(output_dir, 'model_weights', 'd2v_model.h5'))
    np.save(os.path.join(output_dir, 'model_weights', 'X.npy'), X)
    np.save(os.path.join(output_dir, 'model_weights', 'Y.npy'), Y)
    np.save(os.path.join(output_dir, 'model_weights', 'tokenizer.npy'), tokenizer)
    #
    if plt_history:
        plot_history(history)
    Test_LSTM_model(model, X_train, Y_train, X_test, Y_test)


def TEST(saved_dir, message=None):
    '''Test the model on any new message'''
    # load the model
    embedding_matrix = np.load(os.path.join(saved_dir, 'embedding_matrix.npy'))
    d2v_model = Doc2Vec.load(os.path.join(saved_dir, 'd2v_model.h5'))
    X = np.load(os.path.join(saved_dir, 'X.npy'))
    Y = np.load(os.path.join(saved_dir, 'Y.npy'))
    tokenizer = np.load(os.path.join(saved_dir, 'tokenizer.npy'))
    model = LSTM_model(embedding_matrix, X, d2v_model)
    # load weights to the model
    model.load_weights(os.path.join(saved_dir, 'LSTMbased_Model.h5'))
    if message is None:
        message1 = [
            'Congratulations! you have won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now.']
        message2 = ['thanks for accepting my request to connect']
        message3 = ['Hello, how are you doing today?']

        # test the model
        Test_Sentence(message1, model, tokenizer, X)
        Test_Sentence(message2, model, tokenizer, X)
        Test_Sentence(message3, model, tokenizer, X)
    else:
        Test_Sentence(message, model, tokenizer, X)


def main():
    input_data = '../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv'
    output_dir = '../output'
    run(input_data, output_dir=output_dir)
    TEST(output_dir)


if __name__ == '__main__':
    main()
