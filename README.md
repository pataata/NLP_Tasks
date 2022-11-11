# NLP_Homework

Basic sentiment analyisis model, NER and use of google and amazon APIs for translation services.

## Description

Set of three NLP-related exercises. 
* In the first example, a pre-trained sentiment analysis model from hugging is implemented in python. 
* In the second example,
a base NER Bert model is further trained on some data to fine tune it. Here is a graph of the training and validation losses:

![alt text](https://github.com/[pataata]/[NLP_Homework]/blob/[main]/src/train_eval_loss.png?raw=true)

In the third example, both amazon and google APIs are implemented in python for using and comparing both translation services. The comparison is done with a 100 sample spanish to english dataset, computing the average BLEU score for the predictions of each model.


### Installing
pip3 install -r requeriments.txt


### Executing program

python3 run.py

Tests: 

To run tests, from the root dir of the repo, call: 
```
python -m pytest tests
```

## Authors

Rubén Sánchez Mayén
a01378379@tec.mx
