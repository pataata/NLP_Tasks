# Student: Ruben Sanchez Mayen
# ID: A01378379
# Date: 11/11/2022
# Title: Set of three NLP-related tasks using python.

import src.functions as f # single letter names like this are discouraged: https://docs.quantifiedcode.com/python-anti-patterns/maintainability/using_single_letter_as_variable_name.html
from transformers import AutoModelForTokenClassification
"""
 It’s best practice to have your code in a descriptive method or small class, if possible, rather than running at the top-level.
 makes it easier for other modules to import the functionality later if needed! 
 
 So I would recommend three small files/classes which have either small classes or descriptive functions inside! 
"""
# 1. Warm up: Out of the Box Sentiment Analysis
print("PART 1", end="\n")

# Read file
filename = './datasets/tiny_movie_reviews_dataset.txt'
documents = f.read_file_lines(filename)

# Set up model
sentiment_analysis_model = f.pretrained_sentiment_analysis()

# Make predictions
for doc in documents: 
  print(sentiment_analysis_model.predict(doc))

# 2. NER: Take a basic, pretrained NER model, and train further on a task-specific dataset
print("PART 2", end="\n")

# Size of the training and validation sets
N_EXAMPLES_TO_TRAIN = 20
N_VALIDATION = 4
batch_size = 4
epochs = 6
# Load dataset
dataset = f.load_wikiann(N_EXAMPLES_TO_TRAIN,N_VALIDATION)

# Preprocess dataset for BERT
dataset = dataset.map(f.tokenize_adjust_labels, batched=True)

# Train model
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
f.train_model(batch_size,epochs,model,dataset,True)

# where is the plotting code?

# 3. Set up and compare model performance of two different translation models
print("PART 3", end="\n")

# Retrieve data
es_data = f.read_file_lines('./datasets/es_corpus.txt')
en_data = f.read_file_lines('./datasets/en_corpus.txt')

# Create translators
google_translator = f.translator('Google')
amazon_translator = f.translator('Amazon')

# Compute average
sum_bleu_google = 0
sum_bleu_amazon = 0

# Variables
n_iterations = 100 # constants like this should be at the top in all caps! Naming upper: https://www.ceos3c.com/python/python-constants/
source = 'es'

# Test 100 lines
for i in range(n_iterations):
  sum_bleu_google += f.get_bleu(
                                google_translator.translate(es_data[i],source),
                                test=en_data[i])
  sum_bleu_amazon += f.get_bleu(
                                amazon_translator.translate(es_data[i],source),
                                test=en_data[i])

# Print results
print("GOOGLE_TRANSLATOR:",sum_bleu_google/n_iterations)
print("AMAZON_TRANSLATOR:",sum_bleu_amazon/n_iterations)



  







