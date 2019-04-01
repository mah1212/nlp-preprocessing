import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import string   
import operator
import gensim
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
del test_df

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

print('Extracting Paragram Embeddings')
paragram_embeddings = load_embedding(paragram)

print('Extracting fasttext-wiki Embeddings')
fasttext_wiki_embeddings = load_embedding(wiki_news)

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

coverage = check_coverage(vocab, fasttext_wiki_embeddings)

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

'?' in fasttext_wiki_embeddings # true
'http' in fasttext_wiki_embeddings # true
'<p>' in fasttext_wiki_embeddings # False
'&' in fasttext_wiki_embeddings # true
'*' in fasttext_wiki_embeddings
'$' in fasttext_wiki_embeddings
'~' in fasttext_wiki_embeddings
'<p>' in fasttext_wiki_embeddings
'^' in fasttext_wiki_embeddings
'_' in fasttext_wiki_embeddings # False

punct_list = ['?', '&', '!', '.', ',', '"', '#', '$', '%', '\', ''', '(', ')', 
                   '*', '+', '-', '/', ':', ';', '<', '=', '>', '@', '[', 
                   ']', '^', '_', '`', '{', '|', '}', '~', '“', '”', '’']

punct_list[1]


# solution 1
for punct in punct_list:
    if punct in fasttext_wiki_embeddings:
        print(punct, 'True')
    else:
        print(punct, 'False')

# solution 2
for pun in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
    if pun in fasttext_wiki_embeddings:
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

#--------------------------------------------------
#---------- improved version but without tqdm -----
#---------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import string   
import operator
import gensim
    
train_df = pd.read_csv('train.csv').drop('target', axis=1)

def load_embedding(file):
    
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    
    # for linux change file path to /
    if file == 'embeddings\\wiki-news-300d-1M\\wiki-news-300d-1M.vec':
        embedding_index = dict(get_coefs( *x.split(" ")) for x in open(file, encoding='latin') if len(x)>100)
    else: 
        embedding_index = dict(get_coefs( *x.split(" ")) for x in open(file, encoding='latin'))
    
    return embedding_index



# =============================================================================
# 
#     GoogleNews-vectors-negative300 - https://code.google.com/archive/p/word2vec/
#     glove.840B.300d - https://nlp.stanford.edu/projects/glove/
#     paragram_300_sl999 - https://cogcomp.org/page/resource_view/106
# 
#     wiki-news-300d-1M - https://fasttext.cc/docs/en/english-vectors.html
# 
# 
# =============================================================================

glove = 'embeddings\\glove.840B.300d\\glove.840B.300d.txt'
paragram =  'embeddings\\paragram_300_sl999\\paragram_300_sl999.txt'
wiki_news = 'embeddings\\wiki-news-300d-1M\\wiki-news-300d-1M.vec'
google_news = 'embeddings\\GoogleNews-vectors-negative300\\GoogleNews-vectors-negative300.bin'

print('Extracting Glove Embeddings')
glove_embeddings = load_embedding(glove)

print('Extracting Paragram Embeddings')
paragram_embeddings = load_embedding(paragram)


print('Extracting fasttext-wiki Embeddings')
fasttext_wiki_embeddings = load_embedding(wiki_news)

print('Google News Embeddings') # Need to fix
#google_word2vac_embeddings = load_embedding(google_news)




# Load Google's pre-trained Word2Vec model.


def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab                

def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words


# Now check
vocab = build_vocab(train_df['question_text'])


# View vocab and embeddings
from itertools import islice

# take a number of items from the dictionary, but we do not know whether they are
# first n  or last n. 
# There's no such thing a the "first n" keys because a dict doesn't remember 
# which keys were inserted first.
# You can get any n key-value pairs though

def view_dict(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))
 
vocab_items = view_dict(30, vocab.items())
paragram_items = view_dict(5, paragram_embeddings.items())
 
print('Vocab', vocab_items)
print('Paragram Embeddings', paragram_items)
del vocab_items
del paragram_items


# check out of vocab = oov

print("Glove Coverage: ")
oov_glove = check_coverage(vocab, glove_embeddings)

del glove_embeddings # free memory

print("Paragram Coverage: ")
oov_paragram = check_coverage(vocab, paragram_embeddings)
oov_paragram[:10] # list

del paragram_embeddings


print("FastText(Wiki news) Coverage : ")
oov_fasttext_wiki = check_coverage(vocab, fasttext_wiki_embeddings)

'''
Glove Coverage: 
Found embeddings for 33.02% of vocab
Found embeddings for  88.15% of all text

Paragram Coverage: 
Found embeddings for 19.54% of vocab
Found embeddings for  72.21% of all text

FastText(Wiki news) Coverage : 
Found embeddings for 29.86% of vocab
Found embeddings for  87.64% of all text
'''

del fasttext_wiki_embeddings

print('First 10 glove out of vocab')
oov_glove[:10]

print('Paragram out of vocab')
oov_paragram[:10]

print('First 10 wiki out of vocab')
oov_fasttext_wiki[:10]


# make all text lower and check the coverage
train_df['lowered_question'] = train_df['question_text'].apply(lambda x: x.lower())

vocab = build_vocab(train_df['lowered_question'])

print("Glove Coverage: ")
oov_glove = check_coverage(vocab, glove_embeddings)

print("Paragram Coverage: ")
oov_paragram = check_coverage(vocab, paragram_embeddings)

print("FastText(Wiki news) Coverage : ")
oov_fasttext_wiki = check_coverage(vocab, fasttext_wiki_embeddings)

'''
Glove : 
Found embeddings for 27.10% of vocab 
Found embeddings for  87.88% of all text
Paragram : 
Found embeddings for 31.01% of vocab
Found embeddings for  88.21% of all text
FastText : 
Found embeddings for 21.74% of vocab
Found embeddings for  87.14% of all text
'''


# Better, but we lost a bit of information on the other embeddings.

#  Therer are words known that are known with upper letters and unknown without. 

#        word.lower() takes the embedding of word if word.lower() doesn't have an embedding


def add_lower(vocab, embedding):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")


print("Glove : ")
add_lower(vocab, glove_embeddings)
print("Paragram : ")
add_lower(vocab, paragram_embeddings)
print("FastText : ")
add_lower(vocab, fasttext_wiki_embeddings)

'''
Glove : 
Added 15199 words to embedding
Paragram : 
Added 0 words to embedding
FastText : 
Added 27908 words to embedding
'''

print("Glove Coverage: ")
oov_glove = check_coverage(vocab, glove_embeddings)

print("Paragram Coverage: ")
oov_paragram = check_coverage(vocab, paragram_embeddings)

print("FastText(Wiki news) Coverage : ")
oov_fasttext_wiki = check_coverage(vocab, fasttext_wiki_embeddings)

'''
Glove : 
Found embeddings for 30.39% of vocab
Found embeddings for  88.19% of all text
Paragram : 
Found embeddings for 31.01% of vocab
Found embeddings for  88.21% of all text
FastText : 
Found embeddings for 27.77% of vocab
Found embeddings for  87.73% of all text
'''


# What's wrong ?

oov_glove[:10]
'''
[('india?', 17092),
 ("what's", 13977),
 ('it?', 13702),
 ('do?', 9125),
 ('life?', 8114),
 ('why?', 7674),
 ('you?', 6572),
 ('me?', 6525),
 ('them?', 6423),
 ('time?', 6021)]
'''
#First faults appearing are :

#    Contractions
#    Words with punctuation in them

#    Let us correct that.

#---Contraction = short form -----
contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", 
                       "'cause": "because", "could've": "could have", "couldn't": "could not", 
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", 
                       "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 
                       "he'd": "he would","he'll": "he will", "he's": "he is", 
                       "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
                       "how's": "how is",  "I'd": "I would", "I'd've": "I would have", 
                       "I'll": "I will", "I'll've": "I will have", "I'm": "I am", 
                       "I've": "I have", "i'd": "i would", "i'd've": "i would have", 
                       "i'll": "i will",  "i'll've": "i will have","i'm": "i am", 
                       "i've": "i have", "isn't": "is not", "it'd": "it would", 
                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                       "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                       "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                       "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                       "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 
                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                       "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                       "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                       "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                       "this's": "this is","that'd": "that would", "that'd've": "that would have", 
                       "that's": "that is", "there'd": "there would", "there'd've": "there would have", 
                       "there's": "there is", "here's": "here is","they'd": "they would", 
                       "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", 
                       "they're": "they are", "they've": "they have", "to've": "to have", 
                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", 
                       "we'll": "we will", "we'll've": "we will have", "we're": "we are", 
                       "we've": "we have", "weren't": "were not", "what'll": "what will", 
                       "what'll've": "what will have", "what're": "what are",  "what's": "what is", 
                       "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", 
                       "where's": "where is", "where've": "where have", "who'll": "who will", 
                       "who'll've": "who will have", "who's": "who is", "who've": "who have", 
                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 
                       "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                       "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                       "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                       "you'll've": "you will have", "you're": "you are", "you've": "you have" }

def known_contractions(embed):
    known = []
    for contraction in contraction_mapping:
        if contraction in embed:
            known.append(contraction)
    return known


print("- Known Contractions -")
print("Glove :")
print(known_contractions(glove_embeddings))
print("Paragram :")
print(known_contractions(paragram_embeddings))
print("FastText :")
print(known_contractions(fasttext_wiki_embeddings))

'''
- Known Contractions -

Glove :
["can't", "'cause", "didn't", "doesn't", "don't", "I'd", "I'll", "I'm", "I've", "i'd", "i'll", "i'm", "i've", "it's", "ma'am", "o'clock", "that's", "you'll", "you're"]

Paragram :
["can't", "'cause", "didn't", "doesn't", "don't", "i'd", "i'll", "i'm", "i've", "it's", "ma'am", "o'clock", "that's", "you'll", "you're"]

FastText :
[]
'''

# FastText does not understand contraction at all. Means it does not have any short form of word like ain't

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[x] if x in mapping else x for x in text.split(" ")])
    return text

train_df['uncontracted_question'] = train_df['lowered_question'].apply(lambda x: clean_contractions(x, contraction_mapping))
train_df['uncontracted_question'].head(20)

vocab = build_vocab(train_df['uncontracted_question'])

print("Glove Coverage: ")
oov_glove = check_coverage(vocab, glove_embeddings)

print("Paragram Coverage: ")
oov_paragram = check_coverage(vocab, paragram_embeddings)

print("FastText(Wiki news) Coverage : ")
oov_fasttext_wiki = check_coverage(vocab, fasttext_wiki_embeddings)

'''
Glove : 
Found embeddings for 30.53% of vocab
Found embeddings for  88.56% of all text
Paragram : 
Found embeddings for 31.16% of vocab
Found embeddings for  88.58% of all text
FastText : 
Found embeddings for 27.91% of vocab
Found embeddings for  88.44% of all text
'''

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

def unknown_punct(punct, embed):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown

print("Glove :")
print(unknown_punct(punct, glove_embeddings))
print("Paragram :")
print(unknown_punct(punct, paragram_embeddings))
print("FastText :")
print(unknown_punct(punct, fasttext_wiki_embeddings))

#    We use a map to replace unknown characters with known ones.

#    We make sure there are spaces between words and punctuation
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", 
                 "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'",
                 '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta',
                 '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                 '∅': '', '³': '3', 'π': 'pi', }


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

train_df['uncontracted_question'] = train_df['uncontracted_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

vocab = build_vocab(train_df['uncontracted_question'])

print("Glove Coverage: ")
oov_glove = check_coverage(vocab, glove_embeddings)

print("Paragram Coverage: ")
oov_paragram = check_coverage(vocab, paragram_embeddings)

print("FastText(Wiki news) Coverage : ")
oov_fasttext_wiki = check_coverage(vocab, fasttext_wiki_embeddings)

oov_paragram[:100]


#What's still missing ?

#    Unknown words
#    Acronyms
#    Spelling mistakes

mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite',
                'travelling': 'traveling', 'counselling': 'counseling',
                'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are',
                'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many',
                'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does',
                'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating',
                'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data',
                '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
                'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

train_df['uncontracted_question'] = train_df['uncontracted_question'].apply(lambda x: correct_spelling(x, mispell_dict))



#--------------- Part 2 -------------------------------------------------------
#------------- Building the model and predict ---------------------------------


#Data for the network
#Texts
#Parameters

#I took the same for both models. len_voc can de reduced for the treated model.

len_voc = 95000
max_len = 60

#---- uncleaned text -------

# We apply a standard tokenizer and padding.

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# fuction to cean question_text, use default filter
def make_data(X):
    t = Tokenizer(num_words=len_voc)
    t.fit_on_texts(X)
    X = t.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=max_len)
    return X, t.word_index

X, word_index = make_data(train_df['question_text'])



# Treated text

# Same thing, but with no filters.

def make_treated_data(X):
    t = Tokenizer(num_words=len_voc, filters='')
    t.fit_on_texts(X)
    X = t.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=max_len)
    return X, t.word_index

X_treated, word_index_treated = make_data(train_df['uncontracted_question'])



#------- Splitting ----------

from sklearn.model_selection import train_test_split
import gc

# load the target column as we skipped it to keep the memory low 
target_df = pd.read_csv('train.csv')
train_df['target'] = target_df['target']
del target_df
gc.collect()

y = train_df['target'].values # use full df

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=420)

X_t_train, X_t_val, _, _ = train_test_split(X_treated, y, test_size=0.1, random_state=420)

print(f"Training on {X_train.shape[0]} texts")



# Embeddings

# I use GloVe here, because I got better results with it than with others. But feel free to change that.
# Here we used paragram to check
def make_embed_matrix(embeddings_index, word_index, len_voc):
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    word_index = word_index
    embedding_matrix = np.random.normal(emb_mean, emb_std, (len_voc, embed_size))
    
    for word, i in word_index.items():
        if i >= len_voc:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

del vocab
gc.collect()
del oov_fasttext_wiki
gc.collect()

embedding = make_embed_matrix(fasttext_wiki_embeddings, word_index, len_voc)
del word_index
gc.collect()

np.savetxt('fasttext_wiki_embed_matrix.csv', embedding)

embedding_treated = make_embed_matrix(fasttext_wiki_embeddings, word_index_treated, len_voc)
del word_index_treated
gc.collect()

np.savetxt('fasttext_wiki_embed_treated_matrix.csv', embedding_treated)



# Note that we have two embedding matrices, one for each pre-treatment.
#------------------f1 metric for Keras-----------------------------------------
import keras
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#Model

from keras.models import Model
from keras.layers import Dense, Embedding, Bidirectional, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Input, Dropout
from keras.optimizers import Adam

# note our layer shape is (95000,300) so embed size must be 300
# Note CuDNNGRU is for GPU 
# Trying to make it cpu version
def make_model(embedding_matrix, embed_size=300, loss='binary_crossentropy'):
    inp    = Input(shape=(max_len,))
    x      = Embedding(len_voc, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x      = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x      = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    avg_pl = GlobalAveragePooling1D()(x)
    max_pl = GlobalMaxPooling1D()(x)
    concat = concatenate([avg_pl, max_pl])
    dense  = Dense(64, activation="relu")(concat)
    drop   = Dropout(0.1)(concat)
    output = Dense(1, activation="sigmoid")(concat)
    
    model  = Model(inputs=inp, outputs=output)
    model.compile(loss=loss, optimizer=Adam(lr=0.0001), metrics=['accuracy', f1])
    return model

model = make_model(embedding)

model_treated = make_model(embedding_treated)

model.summary()


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

checkpoints = ModelCheckpoint('weights.hdf5', monitor="val_f1", mode="max", verbose=True, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_f1', factor=0.1, patience=2, verbose=1, min_lr=0.000001)

checkpoints_treated = ModelCheckpoint('treated_weights.hdf5', monitor="val_f1", mode="max", verbose=True, save_best_only=True)
reduce_lr_treated = ReduceLROnPlateau(monitor='val_f1', factor=0.1, patience=2, verbose=1, min_lr=0.000001)


epochs = 8
batch_size = 512

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                    validation_data=[X_val, y_val], callbacks=[checkpoints, reduce_lr])


plt.figure(figsize=(12,8))
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Test Accuracy')
plt.show()

history = model_treated.fit(X_t_train, y_train, batch_size=batch_size, epochs=epochs, 
                            validation_data=[X_t_val, y_val], callbacks=[checkpoints_treated, reduce_lr_treated])


plt.figure(figsize=(12,8))
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Test Accuracy')
plt.show()

model.load_weights('weights.hdf5')
model_treated.load_weights('treated_weights.hdf5')

#Results
#Predictions

pred_val = model.predict(X_val, batch_size=512, verbose=1)
pred_t_val = model_treated.predict(X_t_val, batch_size=512, verbose=1)


#F1 Scores

from sklearn.metrics import f1_score

def tweak_threshold(pred, truth):
    thresholds = []
    scores = []
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        thresholds.append(thresh)
        score = f1_score(truth, (pred>thresh).astype(int))
        scores.append(score)
    return np.max(scores), thresholds[np.argmax(scores)]


score_val, threshold_val = tweak_threshold(pred_val, y_val)

print(f"Scored {round(score_val, 4)} for threshold {threshold_val} with untreated texts on validation data")

#Scored 0.6566 for threshold 0.31 with untreated texts on validation data

score_t_val, threshold_t_val = tweak_threshold(pred_t_val, y_val)

print(f"Scored {round(score_t_val, 4)} for threshold {threshold_t_val} with treated texts on validation data")

#Scored 0.6604 for threshold 0.34 with treated texts on validation data

#Conclusion :

#Our preprocessing helped improve the results. However, results with CuDNN layers are not reproductible, so the gain might vary a bit.



#================ Part 2 Done -------------------------------------------------

africa_df = pd.read_csv('africa.txt', header=None) 
# Error
# 'utf-8' codec can't decode byte 0xf4 in position 18: invalid continuation byte

# simple read procedure
with open('africa.txt', 'r') as myfile:
  data = myfile.read()

for x in data:
    km_removed = x.replace(x, km2)


result = []

country = None

with open('africa.txt') as f:
    for line in f:
        
        if line.endswith(')'):
            # remember new state
            state = line[:-6] # without `[edit]`
        else:
            # add state, city to result
            city, rest = line.split(' ', 1)
            result.append( [state, city] )

# --- display ---

for state, city in result:
    print(state, city)
    
with open('africa.txt') as f:
    for line in f:
        
        
        if line.endswith(')'):
            # remember new state
            state = line[:-6] # without `[edit]`
        else:
            # add state, city to result
            city, rest = line.split(' ', 1)
            result.append( [state, city] )

#Here’s a complete list of the metacharacters; their meanings will be discussed in the rest of this HOWTO.

# . ^ $ * + ? { } [ ] \ | ( )


import re
from pandas.compat import StringIO

with open('africa.txt', 'r') as myfile:
  data = myfile.read()

clean_texts = re.sub(r"(km2|sq|mi)|,|\(|\)|([b])|\]|\[", "", data) 

africa_df = pd.read_csv(pd.compat.StringIO(clean_texts), sep='\t', header=None)   
africa_df.columns = ['Position by Area', 'Country', 'NeedClean']

#temp_df = pd.DataFrame(africa_df.NeedClean.str.split(' ',1).tolist(),
#                                   columns = ['Total Area Sqkm','Total Area Sqmi'])


#concat_df = pd.concat(africa_df, temp_df)

#africa_df['NeedClean'].str.split(' ', 1, expand=True)

africa_df[['Total Area Sqkm','Total Area Sqmi']] = africa_df['NeedClean'].str.split(' ', 1, expand=True)
africa_df = africa_df.drop(['NeedClean'], axis=1)

# Algeria:    2,381,741
# BD:           148,460
bd_area_sqkm = 148460 # cia factbook sq km 
bd_area_skmi = 57320.73

africa_df['Total Area Sqkm']/bd_area_sqkm # Error can not convert str to int

africa_df[['Total Area Sqkm', 'Total Area Sqmi']] = africa_df[['Total Area Sqkm', 'Total Area Sqmi']].apply(pd.to_numeric)

bd_compare = africa_df['Total Area Sqkm']/bd_area_sqkm # Error can not convert str to int

africa_df['BD Compare Total Area'] = bd_compare
africa_df.head()

# need to fix


africa_df.to_csv('total_area_comparision.csv', sep='\t', encoding='utf-8')

# need to fix
africa_total_area = pd.read_csv('total_area_comparision.csv')