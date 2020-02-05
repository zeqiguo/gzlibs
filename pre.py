from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
from nltk.stem import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
import spacy
import re
from unidecode import unidecode
from google.cloud import storage
import multiprocessing as mp
from math import log
# basic
import os
import warnings
import json
warnings.filterwarnings("ignore")

# data wrangling & models
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

sp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
porter_stemmer = PorterStemmer()


def strip_custom(token):
    token = token.replace('&Reg;', " ")
    token = token.replace("&lt;", "<")
    token = token.replace("&times;", "")
    token = token.replace("&gt;", ">")
    token = token.replace("&quot;", "")
    token = token.replace('&nbsp', " ")
    token = token.replace('&copy;', " ")
    token = token.replace('&reg', " ")
    token = token.replace('%20', " ")
    # this has to be last:
    token = token.replace("&amp;", "&")
    token = token.replace("â\x80¢", " ")
    token = token.replace("Â®", " ")
    token = token.replace("Ã©", " ")
    token = token.replace("®", " ")
    token = token.replace("©", " ")
    token = token.replace("™", " ")
    token = token.replace("•", "")
    token = token.replace("width:99pt", "")
    token = token.replace('class="xl66">', '')
    #token = re.sub(r"\'", '', token)
    token = token.replace('&#160;', ' ')
    return token


def string_processor(token):
#     str = str(token)
    str = unidecode(token)
    # str = strip_custom(str)
    str = remove_stopwords(str)
    str = strip_punctuation(str)
    str = strip_non_alphanum(str)
    tokens = sp(str)
    tokens = [token.lemma_ for token in tokens]
    tokens = [porter_stemmer.stem(token) for token in tokens]
    str = " ".join(tokens)
    str = strip_multiple_whitespaces(str)
    str = str.strip(' ')
    return str

