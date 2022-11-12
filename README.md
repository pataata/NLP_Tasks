# NLP_Tasks

Basic sentiment analysis model, NER model and use of google and amazon APIs for translation services.

## Description

Set of three NLP-related tasks
* In the first task, a pre-trained sentiment analysis model from hugging is implemented in python. 
* In the second task,
a base NER Bert model is further trained on some data to fine tune it. Here is a graph of the training and validation losses:

![Image in src folder](https://github.com/pataata/NLP_Homework/blob/main/src/train_eval_loss.png?raw=true)

* In the third task, both amazon and google APIs are implemented in python for using and comparing both translation services. The comparison is done with 100 samples of a spanish to english dataset, computing the average BLEU score for the predictions of each model.


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
