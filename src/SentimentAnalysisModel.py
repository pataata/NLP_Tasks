import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

def read_file_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()

class SentimentAnalysisModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    def predict(self,input_text):
        encoded_input = self.tokenizer(input_text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        def score2pred(scores):
            """
            Positive or Negative.
            Neutral turns to negative
            """
            max_score = max(scores)
            max_index = np.where(scores == max_score)[0][0]
            if max_index < 2:
                return 'NEGATIVE'
            else:
                return 'POSITIVE'
        prediction = score2pred(scores)
        return prediction