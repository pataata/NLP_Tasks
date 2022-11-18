from src import functions as f
from transformers import AutoModelForTokenClassification
"""
Tests should use "assert" to check  that the functionality/outputs are as expected, not just that the code runs! 
When writing code, I write the test for each class or piece of functionality as soon as I finish that piece.
It helps you develop incrementally, being sure that each piece of code is clean and works like you expect it to! 
for tests, best practices are to have a structure like this: 
https://stackoverflow.com/questions/1896918/running-unittest-with-typical-test-directory-structure
"""

# Test 1. Read a file
print('-------- test 1 --------',end='\n\n')
filename = './datasets/hello_world.txt'
sample_text = f.read_file_lines(filename)
print('filename:',filename)
print('Content:',sample_text, end='\n\n')


# Test 2. Set up a sentiment analysis model and make a prediction
print('-------- test 2 --------',end='\n\n')
sentiment_analysis_model = f.pretrained_sentiment_analysis()
sample_text = 'I dont like apples'
prediction = sentiment_analysis_model.predict(sample_text)
print('Sample text:',sample_text)
print('Prediction:',prediction, end='\n\n')


# Test 3.1 Load a dataset from hugging face with specified size
print('-------- Test 3.1 --------',end='\n\n')
dataset = f.load_wikiann(5,5,5)
print(dataset,end='\n\n')

# Test 3.2 Check tokenize_adjust_labels which makes samples comply with Bert format
print("-------- Test 3.2 --------",end='\n\n')
tokenized_dataset = dataset.map(f.tokenize_adjust_labels, batched=True)
print("Result: ",tokenized_dataset,end='\n\n')

# Test 3.3 Train model with 5-sample dataset
print('-------- Test 3.3 --------',end='\n\n')
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
f.train_model(1,2,model,tokenized_dataset)
print(end='\n\n')


# Test 4 Translator class
print('-------- Test 4 --------',end='\n\n')

google_translator = f.translator('Google')
amazon_translator = f.translator('Amazon')
invalid_translator = f.translator('Ebay')
source = 'es'
text = 'Hola mundo'

print('Text: ',text)
print('Google translator: ',google_translator.translate(text,source))
print('Amazon translator: ',amazon_translator.translate(text,source))
print('Invalid translator: ',invalid_translator.translate(text,source),end='\n\n')


# Test 5 Bleu score
print('-------- test 5 --------',end='\n\n')

pred = 'Hi world'
test = 'Hello world'

print('Prediction: ', pred)
print('Test: ', test)
print('BLEU SCORE: ',f.get_bleu(pred,test))

pred = 'Hello world'
test = 'Hello world'

print('Prediction: ', pred)
print('Test: ', test)
print('BLEU SCORE: ',f.get_bleu(pred,test),end='\n\n')

