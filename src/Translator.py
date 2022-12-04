import os
import boto3
from google.cloud import translate_v2 as Gtranslate
from dotenv import load_dotenv

# AWS authentication
load_dotenv()
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
REGION = os.getenv("REGION_NAME")

# Google authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'private_key.json'

class Translator:
    def __init__(self,company):
        """
        The company must be one of the following: [Google, Amazon]
        """
        if(company == 'Google' or company == 'Amazon'):
            self.company = company
        else:
            raise Exception("Provided company not supported")

    
    def translate(self,text,source,target="en"):
        if self.company == 'Google':
            client = Gtranslate.Client()
            result = client.translate(text, target_language=target)
            return result['translatedText']
        elif self.company == 'Amazon':
            client = boto3.client('translate', region_name=REGION)
            result = client.translate_text(
                Text=text,
                SourceLanguageCode=source,
                TargetLanguageCode=target,
            )
            return result['TranslatedText']
        else:
            print('A valid company was not specified')
            return 0