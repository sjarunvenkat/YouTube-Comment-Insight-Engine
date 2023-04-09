import os
import googleapiclient.discovery
import pandas as pd
import csv
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import torch
import torchvision
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from summarizer import Summarizer


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyDrpe4S8sMoCVFgveAWGPGQA4dbK0MQXfA"

def ytres(link):    
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey = DEVELOPER_KEY)

    comments = []
    next_page_token = None
    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            maxResults=100,
            moderationStatus="published",
            order="relevance",
            textFormat="plainText",
            videoId=link,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = response.get('nextPageToken')

        if not next_page_token:
            break

    df = pd.DataFrame({'comment': comments})
    df.to_csv('comments.csv', index=False, encoding='utf-8-sig')

    stop_words = stopwords.words()

    def cleaning(text):
        # check if input is not NaN
        if type(text) != str:
            return ""

        # converting to lowercase, removing URL links, special characters, punctuations...
        text = text.lower()
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('[“”…]', '', text) 

        # removing the emojies               
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)   
    
        return text

    # Load the data
    df = pd.read_csv("comments.csv")

    # Clean the text
    df["comment"] = df["comment"].apply(cleaning)

    # Tokenizing and removing stopwords
    df["tokens"] = df["comment"].apply(word_tokenize)
    df["tokens"] = df["tokens"].apply(lambda x: [word.lower() for word in x if not word.lower() in stop_words])

    dt = df['comment']

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

    # Define zero-shot classifier pipeline
    classifier = pipeline("zero-shot-classification")

    # Load text data
    sequence = list(dt)
    #sequence[:5]

    # Classify text as positive or negative
    candidate_labels = ['positive', 'negative']
    classified = classifier(sequence[1:50], candidate_labels)
    df_new = pd.DataFrame(classified)

    # Extract positive and negative text
    n = df_new.shape[0]
    for x in range(0, n):
        df_new['labels'][x] = df_new['labels'][x][0]

    pos = df_new.loc[df_new['labels'] == 'positive']
    neg = df_new.loc[df_new['labels'] == 'negative']

    # Classify positive text as doubt, request, or statement
    ns = pos.sequence.values
    ns = list(ns)

    candidate_labels = ['doubt', 'request', 'statement']
    a = classifier(ns, candidate_labels)
    pos_new = pd.DataFrame(a)
    pos_new.to_csv('sample.csv')

    # Extract doubt, request, and statement text
    pos_new = pos_new.drop('scores', axis=1)
    m = pos_new.shape[0]
    for x in range(0, m):
        pos_new['labels'][x] = pos_new['labels'][x][0]

    statement = pos_new.loc[pos_new['labels'] == 'statement']
    req = pos_new.loc[pos_new['labels'] == 'request']
    ques = pos_new.loc[pos_new['labels'] == 'doubt']

    stat = statement['sequence']
    reqq = req['sequence']
    quess = ques['sequence']

    sss = ''
    for x in stat:
        sss += x + '.'

    ss = ''
    for x in reqq:
        ss += x + '.'

    s = ''
    for x in quess:
        s += x + '.'

    # Classify negative text as hate or not hate
    ns = neg.sequence.values
    ns = list(ns)

    candidate_labels = ['hate', 'not hate']
    a = classifier(ns, candidate_labels)
    neg_new = pd.DataFrame(a)

    # Extract hate and not hate text
    neg_new = neg_new.drop('scores', axis=1)
    m = neg_new.shape[0]
    for x in range(0, m):
        neg_new['labels'][x] = neg_new['labels'][x][0]

    hate = neg_new.loc[neg_new['labels'] == 'hate']
    nothate = neg_new.loc[neg_new['labels'] == 'not hate']

    nh = nothate['sequence']
    negs = neg['sequence'].astype(str)

    ssss = ''
    for x in nh:
        ssss += x + '.'

    negss = ''
    for x in negs:
        negss += x + '.'


    summarizer = pipeline("summarization")
    from transformers import file_utils
    print(file_utils.default_cache_path)

    doubts = summarizer(s[:3000],min_length=5, max_length=1024,do_sample=False)[0]['summary_text']

    requests = summarizer(ss[:3000],min_length=5,max_length=1024,  do_sample=False)[0]['summary_text']

    Statements = summarizer(sss[:3000], min_length=5, max_length=1024,  do_sample=False)[0]['summary_text']
 
    negative = summarizer(negss[:3000], min_length=5, max_length=1024,  do_sample=False)[0]['summary_text']

    return doubts,requests,Statements,negative