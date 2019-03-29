import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import string   
from nltk.corpus import stopwords

print(os.getcwd())
print(os.listdir())


#-------- Dataset Stackexchange Transfer Learning -----------------------
biology_df = pd.read_csv('biology.csv')
cooking_df = pd.read_csv('cooking.csv')
crypto_df = pd.read_csv('crypto.csv')
diy_df = pd.read_csv('diy.csv')
travel_df = pd.read_csv('travel.csv')
robotics_df = pd.read_csv('robotics.csv')

test_df = pd.read_csv('test.csv')
test_df.head()

biology_df.head()
cooking_df.head()

dataframes = {
        'cooking': cooking_df,         
        'biology': biology_df,
        'crypto': crypto_df,
        'diy': pd.read_csv('diy.csv'),
        'travel': pd.read_csv('travel.csv'),
        'robotics': robotics_df
 }

dataframes['biology'].iloc[0]
dataframes['cooking'].iloc[1]
dataframes['crypto'].iloc[5]
dataframes['diy'].iloc[12]
dataframes['travel'].iloc[13]
dataframes['robotics'].iloc[7]

 

# ------- Remove html and uri tag ----
uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

def stripTagsAndUris(x):
    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""
    
# take a new column in the dictionary
# map the stripTagsAndUri function with the dictionary 
dataframes

for dfx in dataframes:
    print(dataframes[dfx])
       
for df in dataframes.values():
    df['content'] = df['content'].map(stripTagsAndUris)                
    
    
print(dataframes['cooking'])
print(dataframes['biology'].iloc[9])
print(dataframes['biology'])


# remove all punctuations

def removePunctuation(x):
    # lowering all words
    x = x.lower()
    
    # Remove Non ASCII characters
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    
    # Remove all the pubctuations (replacing with whitespace)
    return re.sub("["+string.punctuation+"]", " ", x)


for df in dataframes.values():
    df['title'] = df['title'].map(removePunctuation)
    df['content'] = df['content'].map(removePunctuation)


dataframes['cooking']
dataframes['cooking'].iloc[0]


# Remove stop words from title and content
import nltk
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

def removeStopwords(x):
    # Remove all stop words
    filtered_words = [word for word in x.split() if word not in stopwords]
    return " ".join(filtered_words)

for df in dataframes.values():
    df['title'] = df['title'].map(removeStopwords)
    df['content'] = df['content'].map(removeStopwords)
    
  
# Create a comma seperated list of tags    
# Splitting tag string in a list of tags
for df in dataframes.values():
    # from a string sequence of tags to a list of tags
    df['tags'] = df['tags'].map(lambda x: x.split())
    
dataframes['robotics'].iloc[9]

# Saving preprocessed data to a csv 
for name, df in dataframes.items():
    df.to_csv(name + "_clean.csv", index = False)
    
biology_clean = pd.read_csv("biology_clean.csv")


# End Stackexchange Transfer Learning 


#------------ Preprocess when using Embeddings --------------------------------
# ----------- Dataset Quora Insincere Question Classification -----------------
train_df = pd.read_csv('train.csv').drop('target', axis=1)
train_df.head()
train_df.shape

test_df = pd.read_csv('test.csv')
test_df.head()
test_df.shape

train_df.shape[0] + test_df.shape[0] # 1681928

# join the train and test
full_df = pd.concat([train_df, test_df])
full_df.shape # (1681928, 2)


# Load Embediiiiiiiiiiiiiiiiiiii