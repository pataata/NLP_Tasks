# Student: Ruben Sanchez Mayen
# ID: A01378379
# Date: 07/11/2022
# Title: Set of three NLP-related exercises using python.

# 1. Warm up: Out of the Box Sentiment Analysis
print("PART 1", end="\n")

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Get data from file
with open('./datasets/tiny_movie_reviews_dataset.txt', 'r') as f:
    text = f.readlines()
    f.close()

# Evaluate each line using the model API
for i in range(len(text)):
    encoded_input = tokenizer(text[i], return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    max_score = max(scores)
    max_index = np.where(scores == max_score)[0][0]
    if max_index < 2:
        print('NEGATIVE')
    else:
        print('POSITIVE')

# 2. NER: Take a basic, pretrained NER model, and train further on a task-specific dataset
print("PART 2", end="\n")

 
# Imports
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset, DatasetDict, load_metric
import matplotlib.pyplot as plt

# Size of the training set
N_EXAMPLES_TO_TRAIN = 20
N_VALIDATION = 4

# Load dataset considering the selected number of samples
train, test, validation = load_dataset("wikiann", "en", split=['train[0:'+str(N_EXAMPLES_TO_TRAIN)+']','test','validation[0:'+str(N_VALIDATION)+']'])

dataset = DatasetDict({'validation': validation,'test':test,'train':train})

label_names = dataset["train"].features["ner_tags"].feature.names

# Data preprocessing

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

#Get the values for input_ids, attention_mask, and adjusted labels
def tokenize_adjust_labels(samples):
  tokenized_samples = tokenizer.batch_encode_plus(samples["tokens"], is_split_into_words=True, truncation=True)
  total_adjusted_labels = []
  for k in range(0, len(tokenized_samples["input_ids"])):
    prev_wid = -1
    word_ids_list = tokenized_samples.word_ids(batch_index=k)
    existing_label_ids = samples["ner_tags"][k]
    i = -1
    adjusted_label_ids = []

    for word_idx in word_ids_list:
      # Special tokens have a word id that is None. We set the label to -100
      # so they are automatically ignored in the loss function.
      if(word_idx is None):
        adjusted_label_ids.append(-100)
      elif(word_idx!=prev_wid):
        i = i + 1
        adjusted_label_ids.append(existing_label_ids[i])
        prev_wid = word_idx
      else:
        label_name = label_names[existing_label_ids[i]]
        adjusted_label_ids.append(existing_label_ids[i])
    total_adjusted_labels.append(adjusted_label_ids)

  #add adjusted labels to the tokenized samples
  tokenized_samples["labels"] = total_adjusted_labels
  return tokenized_samples

# Preprocessed dataset
tokenized_dataset = dataset.map(tokenize_adjust_labels, batched=True)

# Padding
data_collator = DataCollatorForTokenClassification(tokenizer)

# Training params
batch_size = 4
logging_steps = len(tokenized_dataset['train']) // batch_size
epochs = 6

# Model
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

training_args = TrainingArguments(
  output_dir="results",
  num_train_epochs=epochs,
  per_device_train_batch_size=batch_size,
  per_device_eval_batch_size=batch_size,
  evaluation_strategy="epoch",
  disable_tqdm=False,
  logging_steps=logging_steps
) 
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_dataset["train"],
  eval_dataset=tokenized_dataset["validation"],
  data_collator=data_collator,
  tokenizer=tokenizer
)

trainer.train()

# Plotting losses

training_loss = []
eval_loss = []
range_epochs = range(epochs)

# Get losses from trainer log history
for x in trainer.state.log_history:
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

# Imports
import os
from google.cloud import translate_v2 as translate
from dotenv import load_dotenv
import boto3
from nltk.translate.bleu_score import sentence_bleu

# Setting up authentication keys
# Google
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'private_key.json'

translate_client = translate.Client()

# AWS
load_dotenv()

acces_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
session_token = os.getenv("AWS_SESSION_TOKEN")
region = os.getenv("REGION_NAME")

client = boto3.client('translate', region_name=region)

# Get data from file
with open('./datasets/en_corpus.txt', 'r') as f:
    en_data = f.readlines()
    f.close()

# Get data from file
with open('./datasets/es_corpus.txt', 'r') as f:
    es_data = f.readlines()
    f.close()

# Compute average BLEU score for data sample
sum_bleu_google = 0
sum_bleu_aws = 0
n_iterations = 100

for i in range(n_iterations):

  # Make translation with the APIs
  google_result = translate_client.translate(es_data[i], target_language="en")

  aws_result = client.translate_text(
    Text=es_data[i],
    SourceLanguageCode='es',
    TargetLanguageCode='en',
  )

  # BLEU score for google
  ref = [google_result['translatedText'].split()]
  test = en_data[i].split()

  sum_bleu_google += sentence_bleu(ref,test)

  # BLEU score for aws
  ref = [aws_result['TranslatedText'].split()]
  test = en_data[i].split()

  sum_bleu_aws += sentence_bleu(ref,test)

# Print results
print("GOOGLE_TRANSLATOR:",sum_bleu_google/n_iterations)
print("AMAZON_TRANSLATOR:",sum_bleu_aws/n_iterations)



  







