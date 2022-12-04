# Student: Ruben Sanchez Mayen
# ID: A01378379
# Date: 11/11/2022
# Title: Set of three NLP-related tasks using python.

import matplotlib.pyplot as plt
from src.SentimentAnalysisModel import SentimentAnalysisModel
from src.NERTrainer import NERTrainer
from src.Translator import Translator
from transformers import AutoModelForTokenClassification
from nltk.translate.bleu_score import sentence_bleu

# basic functions
def read_file_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()

# 1. Warm up: Out of the Box Sentiment Analysis
print("PART 1", end="\n")

# Read file
FILENAME = './datasets/tiny_movie_reviews_dataset.txt'
documents = read_file_lines(FILENAME)

# Set up model
sentiment_analysis_model = SentimentAnalysisModel()

# Make predictions
for i in range(len(documents)):
  print(sentiment_analysis_model.predict(documents[i]))

# 2. NER: Take a basic, pretrained NER model, and train further on a task-specific dataset
print("PART 2", end="\n")

# Size of the training and validation sets
NER_MODEL = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
N_EXAMPLES_TO_TRAIN = 20
N_VALIDATION = 4
BATCH_SIZE = 4
EPOCHS = 6
N_SAMPLES_TRAIN=20
N_SAMPLES_VAL=7
N_SAMPLES_TEST=7

# Load trainer
trainer_1 = NERTrainer(BATCH_SIZE,EPOCHS, NER_MODEL)

# Load dataset
trainer_1.load_wikiann(N_SAMPLES_TRAIN,N_SAMPLES_TEST,N_SAMPLES_VAL)

# Train model
trainer_1.train()

# Plot losses
training_loss = []
eval_loss = []
range_epochs = range(EPOCHS)
# Get losses from trainer log history
for x in trainer_1.log_history:
    if 'loss' in x:
        training_loss.append(x['loss'])
    elif 'eval_loss' in x:
        eval_loss.append(x['eval_loss'])
# Plot
plt.plot(training_loss)
plt.plot(eval_loss)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training loss','Evaluation loss'])
plt.show()


# 3. Set up and compare model performance of two different translation models
print("PART 3", end="\n")

# Compute bleu
def get_bleu(pred,test):
    pred = [pred.split()]
    test = test.split()
    return sentence_bleu(pred,test)

N_ITERATION = 100
SOURCE_LAN = 'es'

# Retrieve data
es_data = read_file_lines('./datasets/es_corpus.txt')
en_data = read_file_lines('./datasets/en_corpus.txt')

# Create translators
google_translator = Translator('Google')
amazon_translator = Translator('Amazon')

# Compute average
sum_bleu_google = 0
sum_bleu_amazon = 0

# Test 100 lines
print('Computing BLEU scores, may take some time. The number of iterations (N_ITERATIONS) can be decreased if wanted.')
for i in range(N_ITERATION):
  sum_bleu_google += get_bleu(
                                google_translator.translate(es_data[i],SOURCE_LAN),
                                test=en_data[i])
  sum_bleu_amazon += get_bleu(
                                amazon_translator.translate(es_data[i],SOURCE_LAN),
                                test=en_data[i])

# Print results
print("GOOGLE_TRANSLATOR:",sum_bleu_google/N_ITERATION)
print("AMAZON_TRANSLATOR:",sum_bleu_amazon/N_ITERATION)