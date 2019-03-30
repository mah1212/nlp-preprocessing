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
        'robotics': robotics_df,
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


#------------ Preprocess using Embeddings --------------------------------
# ----------- Dataset Quora Insincere Question Classification -----------------


# =============================================================================
# # understanding gensim word2vec
# from gensim.models import Word2Vec
# # define training data
# sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
# 			['this', 'is', 'the', 'second', 'sentence'],
# 			['yet', 'another', 'sentence'],
# 			['one', 'more', 'sentence'],
# 			['and', 'the', 'final', 'sentence']]
# # train model
# model = Word2Vec(sentences, min_count=1)
# print(model)
# 
# # summarize vocabulary
# words = list(model.wv.vocab)
# print(words)
# 
# # access vector for one word
# print(model['sentence'])
# 
# # save model
# model.save('model.bin')
# # load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)
# 
# 
# #visualize 
# X = model[model.wv.vocab] # retrieve all of the vectors from a trained model as follows
# 
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# 
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# 
# plt.scatter(result[:, 0], result[:, 1]) # 
# 
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
# 	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
# plt.show()
# 
# from gensim.models import KeyedVectors
# 
# filename = wiki_news = 'embeddings\\wiki-news-300d-1M\\wiki-news-300d-1M.vec'
# 
# model = KeyedVectors.load_word2vec_format(filename, binary=True)
# 
# 
# =============================================================================

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

# Different types of embedding files need a bit different types of tricks to be loaded
# Load word embeddings
def load_embedding(file):
    
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    
    # for linux change file path to /
    if file == 'embeddings\\wiki-news-300d-1M\\wiki-news-300d-1M.vec':
        embedding_index = dict(get_coefs( *x.split(" ")) for x in open(file, encoding='latin') if len(x)>100)
    else: 
        embedding_index = dict(get_coefs( *x.split(" ")) for x in open(file, encoding='latin'))
    
    return embedding_index


glove = 'embeddings\\glove.840B.300d\\glove.840B.300d.txt'
paragram =  'embeddings\\paragram_300_sl999\\paragram_300_sl999.txt'
wiki_news = 'embeddings\\wiki-news-300d-1M\\wiki-news-300d-1M.vec'
google_news = 'embeddings\\GoogleNews-vectors-negative300\\GoogleNews-vectors-negative300'

print('Extracting Glove Embeddings')
glove_embeddings = load_embedding(glove)

print('Extracting Glove Embeddings')
paragram_embeddings = load_embedding(paragram)

print('Extracting Glove Embeddings')
wiki_embeddings = load_embedding(wiki_news)

# following function to track our training vocabulary, which goes through all our 
# text and counts the occurance of the contained words
# ----- tqdm = progress arabic = I love you -version --------


def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


# let us check 
from tqdm import tqdm
tqdm.pandas()

sentences = train_df["question_text"].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:5]})



import operator 

def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

coverage = check_coverage(vocab, wiki_embeddings)

# oov = check_coverage(vocab, glove_embeddings)
# 100%|██████████| 508823/508823 [01:04<00:00, 7859.42it/s]  
# Found embeddings for 33.02% of vocab
# Found embeddings for  88.15% of all text

# only 33.3% in Glove and 29% in wiki news of our vocabulary will have embeddings, making a lot of our data more 
# or less useless. So lets have a look and start improving. For this we can easily 
# have a look at the top oov words.

# oov[:10]
coverage[:10]

# there is "to". Why? Simply because "to" was removed when the Embeddings were trained. 
# We will fix this later, for now we take care about the splitting of punctuation 
# as this also seems to be a Problem. But what do we do with the punctuation then 
# - Do we want to delete or consider as a token? I would say: It depends. If the 
# token has an embedding, keep it, if it doesn't we don't need it anymore. So lets check:


# --- check whether noise exists 
'?' in glove_embeddings # true
'http' in glove_embeddings # true
'<p>' in glove_embeddings # true

'?' in wiki_embeddings # true
'http' in wiki_embeddings # true
'<p>' in wiki_embeddings # False
'&' in wiki_embeddings # true
'*' in wiki_embeddings
'$' in wiki_embeddings
'~' in wiki_embeddings
'<p>' in wiki_embeddings
'^' in wiki_embeddings
'_' in wiki_embeddings # False

punct_list = ['?', '&', '!', '.', ',', '"', '#', '$', '%', '\', ''', '(', ')', 
                   '*', '+', '-', '/', ':', ';', '<', '=', '>', '@', '[', 
                   ']', '^', '_', '`', '{', '|', '}', '~', '“', '”', '’']

punct_list[1]


# solution 1
for punct in punct_list:
    if punct in wiki_embeddings:
        print(punct, 'True')
    else:
        print(punct, 'False')

# solution 2
for pun in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
    if pun in wiki_embeddings:
        print(pun, 'True')
    else:
        print(pun, 'False')


# While "&" is in the Google News Embeddings, "?" is not. So we 
# basically define a function that splits off "&" and removes other punctuation.

# While "&" is in the Wiki News Embeddings, "_", "'" and '`' are not. So we 
# basically define a function that splits off "&" and removes other punctuation.


# Google news version
def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

# Wiki news version
def clean_text2(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

#---------- improved version but without tqdm -----
def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1