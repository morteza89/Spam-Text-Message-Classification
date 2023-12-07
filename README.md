# Spam-Text-Message-Classification
This repository contains a Python script for spam text message classification using a combination of Doc2Vec and LSTM (Long Short-Term Memory) models. 
The script leverages libraries such as Pandas, NumPy, tqdm, Keras, Gensim, and others. Below is a brief overview of the key functionalities:

Contents:
Read Data: Load and preprocess the spam text message dataset.
Plot Data Distribution: Visualize the distribution of spam and ham categories in the dataset.
Clean Text: Remove special characters and preprocess the text data.
Tokenize Text: Tokenize the text data for further processing.
Prepare Data: Prepare the data for model training, including the creation of Doc2Vec embeddings.
Train Doc2Vec Model: Train the Doc2Vec model on the tokenized text data.
Define LSTM Model: Define the LSTM model for spam classification using Doc2Vec embeddings.
Train LSTM Model: Train the LSTM model using the prepared data.
Evaluate LSTM Model: Evaluate the LSTM model on the test set and visualize the results.
Test on New Messages: Test the trained model on new text messages.
Usage:
Install the required dependencies by running pip install -r requirements.txt.
Update the input_data variable in the script with the path to your spam text message dataset.
Run the script to train and evaluate the models.
Output:
The script outputs model evaluation metrics, including accuracy and confusion matrix.
Optionally, it saves the trained model weights, Doc2Vec embeddings, and other relevant information in the specified output_dir.
Feel free to customize the script based on your specific dataset and requirements.
