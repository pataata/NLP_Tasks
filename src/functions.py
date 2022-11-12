import matplotlib.pyplot as plt
import numpy as np
import os
import boto3
from datasets import load_dataset, DatasetDict, load_metric
from google.cloud import translate_v2 as Gtranslate
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

# ----------------- Task 1 -----------------------

def read_file_lines(filename):
    with open(filename, 'r') as f:
        text = f.readlines()
        f.close()
    return text

class pretrained_sentiment_analysis:
    def __init__(self):
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

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

# ----------------- Task 2 -----------------------

def load_wikiann(n_samples_train="", n_samples_val="", n_samples_test=""):
    """
    load wikiann dataset with custom number of rows for each section
    """
    name = 'wikiann'
    lang = 'en'
    train, test, validation = load_dataset(name, lang, split=['train[:'+str(n_samples_train)+']','test[:'+str(n_samples_test)+']','validation[:'+str(n_samples_val)+']'])
    dataset = DatasetDict({'validation': validation,'test':test,'train':train})
    return dataset

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

def train_model(batch_size,epochs,model,dataset,plot_loss=False):
    data_collator = DataCollatorForTokenClassification(tokenizer)
    logging_steps = len(dataset['train']) // batch_size

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
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

    def plot_loss():
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

    if plot_loss:
        plot_loss()

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
label_names = load_wikiann(1,1,1)["train"].features["ner_tags"].feature.names

# ----------------- Task 3 -----------------------


# AWS authentication
load_dotenv()
acces_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
session_token = os.getenv("AWS_SESSION_TOKEN")
region = os.getenv("REGION_NAME")

# Google authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'private_key.json'

class translator:
    def __init__(self,company):
        self.company = company
        if company == 'Google':
            self.client = Gtranslate.Client()
        elif company == 'Amazon':
            self.client = boto3.client('translate', region_name=region)
    
    def translate(self,text,source,target="en"):
        if self.company == 'Google':
            result = self.client.translate(text, target_language=target)
            return result['translatedText']
        elif self.company == 'Amazon':
            result = self.client.translate_text(
                Text=text,
                SourceLanguageCode=source,
                TargetLanguageCode=target,
            )
            return result['TranslatedText']
        else:
            print('A valid company was not especified')
            return 0

def get_bleu(pred,test):
    pred = [pred.split()]
    test = test.split()
    return sentence_bleu(pred,test)

