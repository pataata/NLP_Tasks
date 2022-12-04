# NLP_Tasks

Basic sentiment analysis model, NER model and Spanish-to-english Translation model all included in a module.

## Description

Set of three NLP-related tasks
* Class for setting up the **cardiffnlp/twitter-roberta-base-sentiment** pre-trained sentiment analysis model from Hugging Face. 20 line example in the run.py file.
* Functions for further training a base Bert model form hugging **dslim/bert-base-NER**  on the **wikiann en** dataset from Hugging Face. Here the a graph if the training and validation losses with 1000 training samples and 150 validation samples:

![Image in src folder](https://github.com/pataata/NLP_Homework/blob/main/src/train_eval_loss.png?raw=true)

* Translator class with the option of using either amazon or google API for translating spanish to english also e comparison is done with 100 samples of a spanish to english dataset localy saved, computing the average BLEU score for the predictions of each model. The BLEU score prediction function is included in *function.py*

Files in src/: 
**- SentimentAnalysisModel.py**
**- NERTrainer.py**
**- Translator.py**

### Installing
```
pip3 install -r requeriments.txt
```

# Authentication

For the Amazon API you need te create an .env file with the following variables:
* aws_access_key_id=
* aws_secret_access_key=
* aws_session_token=
* REGION_NAME = 

Fot the Google API you need to include in the main directory your json key file with the name *private_key.json*

### Executing program
```
python3 run.py
```
Tests: 

To run tests, from the root dir of the repo, call: 
```
python tests.py
```

## Authors

Rubén Sánchez Mayén
a01378379@tec.mx
