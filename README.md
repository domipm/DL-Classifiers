# DL-Classifiers
Numerical code developed for the programming projects on DM873 Deep Learning @ SDU

## Project 1: Cat-Dog Classifier

In this project, a (relatively) simple and computationally inexpensive convolutional neural network (CNN) was developed using PyTorch and applied to perform a binary classification of images into two classes: cats and dogs. This network achieved a validation accuracy of approximately 80%, demonstrating very good performance and generalization to new images.

![Alt text](https://raw.githubusercontent.com/domipm/DL-Classifiers/refs/heads/main/Cat-Dog-Classifier/output/batchvalid.png)

## Project 2: Emotion Classifier

For this project, a dataset of short-form text was analyzed to extract the general emotion and classify them accordingly. To do this, an initial analysis of the data was conducted, studying the distribution of each emotion across different dataset splits, and some statistical properties of the length of the sentences.

Two neural network models were proposed to perform this classification: a simple long-short-term memory (LSTM) recurrent neural network (RNN), and an Encoder-only Transformer model, both of
which were tested to determine their accuracy. Additionally, for the LSTM model, a feature was included that allows for the user to write any sentence and the model will then classify it into one of the available emotions. This allows for a more detailed analysis into how the model works and which sentences get mislabeled.

![Alt text](https://raw.githubusercontent.com/domipm/DL-Classifiers/refs/heads/main/Emotion-Classifier/data_prep/train_distr.png)
