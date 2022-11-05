import unicodedata
import nltk
# import streamlit as st
from contractions import contractions_dict
from nltk.tokenize.toktok import ToktokTokenizer
import spacy
import numpy as np
import pandas as pd
import requests
# from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
# from emot.emo_unicode import UNICODE_EMO, EMOTICONS
# import emoji
import re
import csv
# import sys
from afinn import Afinn
nltk.download('punkt')
import os
import advertools as adv

# try:
#     import cPickle as pickle
# except ImportError:
#     import pickle  

# os.environ['EAI_USERNAME'] = 'w.smith78945@gmail.com' #username
# os.environ['EAI_PASSWORD'] = 'Wsmith#789' #password

# from expertai.nlapi.cloud.client import ExpertAiClient
# client = ExpertAiClient()

text = "" 
language= 'en'
nlp = spacy.load('en_core_web_sm')
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
# with open('Emoji_Dict.p', 'rb') as fp:
#     Emoji_Dict = pickle.load(fp)
# Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}
# @st.cache
def calculate_sentiment(text):
    # output = client.specific_resource_analysis(
    #     body={"document": {"text": text}}, 
    #     params={'language': language, 'resource': 'sentiment'})
    af = Afinn(emoticons=True)
    return af.score(text)


# def convert_emojis(text):
#      for emot in UNICODE_EMOJI:
#         #  text = text.replace(emot, (" ".join(
#         #      UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split())).replace("_", " "))
#          text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
#          return text

# def convert_emojis(text):
#     for emot in UNICODE_EMO:
#         text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
#         return text
# def convert_emojis(text):
#     for emot in Emoji_Dict:
#         text = re.sub(r'('+emot+')', "_".join(Emoji_Dict[emot].replace(",","").replace(":","").split()), text)
#     return text.replace("_"," ")

def convert_emojis(text):
    Emoji_list = adv.extract_emoji([text])
    Emoji_names = Emoji_list['emoji_text'][0]
    for i,emot in enumerate(Emoji_list['emoji'][0]):
        text = text.replace(emot,Emoji_names[i])
    return text

# def convert_emoticons(text):
#     for emot in EMOTICONS_EMO:
#         # text = re.sub(u'('+emot+')', "_".join(EMOTICONS_EMO[emot].replace(",", "").split()), text)
#         text = text.replace(emot, EMOTICONS_EMO[emot])
#     return text


def strip_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    return text

# Function for url's
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def expand_contractions(text, contraction_mapping=contractions_dict):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
            if contraction_mapping.get(match)\
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction


    try:
        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
    except:
        # print("returned text")
        return text
    # print("returned expanded text")    
    return expanded_text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    # print("removed sp.char")
    return text


def lemmatize_text(text):
    # # nlp = spacy.load('en_core_web_sm')
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ !=
                    '-PRON-' else word.text for word in text])
    # # print("lemmatized")
    # lemmas = ""
    # for entry in text: # note we selected the first sentence (sentence[0])
    #     lemmas += entry.lemma
    #     # now, we look for a space after the lemma to add it as well
    #     if not "SpaceAfter=No" in entry.space_after:
    #         lemmas += " "

   
    return text


# def simple_stemmer(text):
#     ps = nltk.porter.PorterStemmer()
#     text = ' '.join([ps.stem(word) for word in text.split()])
#     # print("stemmer")
#     return text


def remove_stopwords(text, is_lower_case=False):
    # stopword_list = nltk.corpus.stopwords.words('english')
    # stopword_list.remove('no')
    # stopword_list.remove('not')
    # tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [
            token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [
            token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    # print("stopwords removed")
    return filtered_text

# @st.cache
def normalize_corpus(text, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True,
                     stopword_removal=True, remove_digits=True, emoji=True, emoticons=True,url_removal=True):
    normalized_corpus = []
    # convert emojis
    if emoji:
        text = convert_emojis(text)
    # convert emoticons
    # if emoticons:
    #     text = convert_emoticons(text)
    # strip HTML
    if html_stripping:
        text = strip_html_tags(text)
    # removing url
    if url_removal:
        text=remove_urls(text)

    # remove accented characters
    if accented_char_removal:
        text = remove_accented_chars(text)
    # expand contractions
    if contraction_expansion:
        text = expand_contractions(text)
    # lowercase the text
    if text_lower_case:
        text = text.lower()
    # remove extra newlines
    text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
    # text = unicode(text, errors='ignore')
    # lemmatize text
    if text_lemmatization:
        text = lemmatize_text(text)
    # remove special characters and\or digits
    if special_char_removal:
        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        text = special_char_pattern.sub(" \\1 ", text)
        text = remove_special_characters(text, remove_digits=remove_digits)
    # remove extra whitespace
    text = re.sub(' +', ' ', text)
    # remove stopwords
    if stopword_removal:
        text = remove_stopwords(text, is_lower_case=text_lower_case)

    normalized_corpus.append(text)
    return text

# @st.cache
def file_input(path,col):
    df = pd.read_csv(path,encoding='cp1252')
    # df1=df.copy()
    colnames=df.columns
    df=df[[colnames[0],colnames[col]]]
    
    df["clean_text"] = df[colnames[col]].apply(lambda x: normalize_corpus(x))
    df["score"] = df['clean_text'].apply(lambda x: calculate_sentiment(x))

    # df.drop('clean_text',axis=1, inplace=True)

    df.drop('clean_text',axis=1, inplace=True)

    return df


# def file_input2(path:str,col):
#     df = pd.read_csv(path)
#     # df1=df.copy()
#     colnames=np.array(df.columns)
#     df=df.iloc[[colnames[0],colnames[col]]]
    
#     df["clean_text"] = df[colnames[col]].apply(lambda x: normalize_corpus(x))
#     df["score"] = df['clean_text'].apply(lambda x: calculate_sentiment(x))

#     # df.drop('clean_text',axis=1, inplace=True)

#     df.drop('clean_text',axis=1, inplace=True)

#     return df
# @st.cache
def txt_to_csv(path,name) :
    header_list = ["text"]
    df = pd.read_csv(path,delimiter='\n',names=header_list,encoding='cp1252')
    df['doc'] = range(1, len(df) + 1)
    df = df[['doc','text']]
    df.to_csv('{0}.csv'.format(name), index=None)
    # return df

# def txt_input(df):
#     df["clean_text"] = df['text'].apply(lambda x: normalize_corpus(x))
#     df["score"] = df['clean_text'].apply(lambda x: calculate_sentiment(x))

#     df.drop('clean_text',axis=1, inplace=True)
#     # print(df.columns)
#     return df 