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


#africa_df['NeedClean'].str.split(' ', 1, expand=True)

africa_df[['Total Area Sqkm','Total Area Sqmi']] = africa_df['NeedClean'].str.split(' ', 1, expand=True)
africa_df = africa_df.drop(['NeedClean'], axis=1)

# Algeria:    2,381,741
# BD:           148,460
bd_area_sqkm = 148460 # cia factbook sq km 
bd_area_skmi =  57320.73

africa_df['Total Area Sqkm']/bd_area_sqkm # Error can not convert str to int

africa_df[['Total Area Sqkm', 'Total Area Sqmi']] = africa_df[['Total Area Sqkm', 'Total Area Sqmi']].apply(pd.to_numeric)

bd_compare = africa_df['Total Area Sqkm']/bd_area_sqkm # Error can not convert str to int

africa_df['BD Compare Total Area'] = bd_compare
africa_df.head()

# need to fix


africa_df.to_csv('total_area_comparision.csv', sep='\t', encoding='utf-8')

# need to fix
africa_total_area = pd.read_csv('total_area_comparision.csv')

# There are lot of countries in the worlds whose per sqkm population is less 
# but they are not rich, why?
# They have huge land mass with less population, are they developed country? NO
# For example, congo, chad, nijer, angola, etc. 
# To Do Next: Create Population Dataset
import re
import pandas as pd
with open('world_population.txt', 'r') as myfile:
  data = myfile.read()

clean_texts = re.sub(r",", "", data) 

wp_df = pd.read_csv(pd.compat.StringIO(clean_texts), sep='\t', header=None)   
wp_df.columns = ['NeedClean']



#splitted = wp_df['NeedClean'].str.split(' ', 1, expand=True)
#del splitted

wp_df[['Rank','NeedMoreClean']] = wp_df['NeedClean'].str.split(' ', 1, expand=True)
wp_df = wp_df.drop(['NeedClean'], axis=1)

wp_df[['Country', 'Population']] = wp_df['NeedMoreClean'].str.split('      ', 1, expand=True)

wp_df = wp_df.drop(columns = ['NeedMoreClean', 'County'])

# rearrange column
wp_df = wp_df[['Rank', 'Country', 'Population']]

# Algeria:    2,381,741
# BD:           148,460
bd_population = 157826578 # cia factbook 


wp_df['Population']/bd_population # Error can not convert str to int

wp_df[['Population']] = wp_df[['Population']].apply(pd.to_numeric)

bd_compare = wp_df['Population']/bd_population # Error can not convert str to int

wp_df['BD Compare Population'] = bd_compare
wp_df.head()


# need to fix
#wp_df.to_csv('total_population_comparision.csv', sep=' ', encoding='utf-8')

# need to fix
#wp_total_area = pd.read_csv('total_area_comparision.csv')

african_area_population = pd.merge(africa_df, wp_df['Country','Population'], on = 'Country')

wp_df.merge(africa_df) # did not merge anything

#-----------------------------------------------------------------------------
#------------------- Quora Incincere Question Classification -----------------
#-----------------------------------------------------------------------------

import os
import re
import string
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import operator
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

#%matplotlib inline

from tqdm import tqdm
tqdm.pandas()
import pickle
import gc

import psutil
from multiprocessing import Pool

num_partitions = 20  # number of partitions to split dataframe
num_cores = psutil.cpu_count()  # number of cores on your machine

print('number of cores:', num_cores)
def df_parallelize_run(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df



    
train_df = pd.read_csv("train.csv", encoding='utf8')
test_df = pd.read_csv("test.csv", encoding='utf8')
all_test_texts = ''.join(test_df.question_text.values.tolist())

print('Train:', train_df.shape)
print('Test:', test_df.shape)


train_ques_lens = train_df['question_text'].map(lambda x: len(x.split(' ')))
test_ques_lens = test_df['question_text'].map(lambda x: len(x.split(' ')))
train_len_sts = train_ques_lens.describe().reset_index().rename(columns={'index':'train_stat'})
train_len_sts['question_text'] = train_len_sts['question_text'].astype(int)
test_len_sts = test_ques_lens.describe().reset_index().rename(columns={'index':'test_stat'})
test_len_sts['question_text'] = test_len_sts['question_text'].astype(int)

len_sts = pd.concat([train_len_sts, test_len_sts], axis=1)
display(len_sts)

del train_ques_lens; del test_ques_lens; del train_len_sts; del test_len_sts
gc.collect()
pass


# replace strange punctuations and raplace diacritics
from unicodedata import category, name, normalize

def remove_diacritics(s):
    return ''.join(c for c in normalize('NFKD', s.replace('ø', 'o').replace('Ø', 'O').replace('⁻', '-').replace('₋', '-'))
                  if category(c) != 'Mn')

special_punc_mappings = {"—": "-", "–": "-", "_": "-", '”': '"', "″": '"', '“': '"', '•': '.', '−': '-',
                         "’": "'", "‘": "'", "´": "'", "`": "'", '\u200b': ' ', '\xa0': ' ','،':'','„':'',
                         '…': ' ... ', '\ufeff': ''}
def clean_special_punctuations(text):
    for punc in special_punc_mappings:
        if punc in text:
            text = text.replace(punc, special_punc_mappings[punc])
    # 注意顺序，remove_diacritics放前面会导致 'don´t' 被处理为 'don t'
    text = remove_diacritics(text)
    return text


# clean numbers
def clean_number(text):
    # 字母和数字分开
    text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)
    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)
    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)
    
#     text = re.sub('[0-9]{5,}', '#####', text)
#     text = re.sub('[0-9]{4}', '####', text)
#     text = re.sub('[0-9]{3}', '###', text)
#     text = re.sub('[0-9]{2}', '##', text)
    
    return text


# 进行 decontracted 之前，清洗不常见的缩略词，如 U.S.
rare_words_mapping = {' s.p ': ' ', ' S.P ': ' ', 'U.s.p': '', 'U.S.A.': 'USA', 'u.s.a.': 'USA', 'U.S.A': 'USA',
                      'u.s.a': 'USA', 'U.S.': 'USA', 'u.s.': 'USA', ' U.S ': ' USA ', ' u.s ': ' USA ', 'U.s.': 'USA',
                      ' U.s ': 'USA', ' u.S ': ' USA ', 'fu.k': 'fuck', 'U.K.': 'UK', ' u.k ': ' UK ',
                      ' don t ': ' do not ', 'bacteries': 'batteries', ' yr old ': ' years old ', 'Ph.D': 'PhD',
                      'cau.sing': 'causing', 'Kim Jong-Un': 'The president of North Korea', 'savegely': 'savagely',
                      'Ra apist': 'Rapist', '2fifth': 'twenty fifth', '2third': 'twenty third',
                      '2nineth': 'twenty nineth', '2fourth': 'twenty fourth', '#metoo': 'MeToo',
                      'Trumpcare': 'Trump health care system', '4fifth': 'forty fifth', 'Remainers': 'remainder',
                      'Terroristan': 'terrorist', 'antibrahmin': 'anti brahmin',
                      'fuckboys': 'fuckboy', 'Fuckboys': 'fuckboy', 'Fuckboy': 'fuckboy', 'fuckgirls': 'fuck girls',
                      'fuckgirl': 'fuck girl', 'Trumpsters': 'Trump supporters', '4sixth': 'forty sixth',
                      'culturr': 'culture',
                      'weatern': 'western', '4fourth': 'forty fourth', 'emiratis': 'emirates', 'trumpers': 'Trumpster',
                      'indans': 'indians', 'mastuburate': 'masturbate', 'f**k': 'fuck', 'F**k': 'fuck', 'F**K': 'fuck',
                      ' u r ': ' you are ', ' u ': ' you ', '操你妈': 'fuck your mother', 'e.g.': 'for example',
                      'i.e.': 'in other words', '...': '.', 'et.al': 'elsewhere', 'anti-Semitic': 'anti-semitic',
                      'f***': 'fuck', 'f**': 'fuc', 'F***': 'fuck', 'F**': 'fuc',
                      'a****': 'assho', 'a**': 'ass', 'h***': 'hole', 'A****': 'assho', 'A**': 'ass', 'H***': 'hole',
                      's***': 'shit', 's**': 'shi', 'S***': 'shit', 'S**': 'shi', 'Sh**': 'shit',
                      'p****': 'pussy', 'p*ssy': 'pussy', 'P****': 'pussy',
                      'p***': 'porn', 'p*rn': 'porn', 'P***': 'porn',
                      'st*up*id': 'stupid',
                      'd***': 'dick', 'di**': 'dick', 'h*ck': 'hack',
                      'b*tch': 'bitch', 'bi*ch': 'bitch', 'bit*h': 'bitch', 'bitc*': 'bitch', 'b****': 'bitch',
                      'b***': 'bitc', 'b**': 'bit', 'b*ll': 'bull'
                      }


def pre_clean_rare_words(text):
    for rare_word in rare_words_mapping:
        if rare_word in text:
            text = text.replace(rare_word, rare_words_mapping[rare_word])

    return text


# de-contract the contraction
def decontracted(text):
    # specific
    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)
    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)
    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)
    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)

    # general
    text = re.sub(r"(I|i)(\'|\’)m ", "i am ", text)
    text = re.sub(r"(A|a)in(\'|\’)t ", "is not ", text)
    text = re.sub(r"n(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)re ", " are ", text)
    text = re.sub(r"(\'|\’)s ", " is ", text)
    text = re.sub(r"(\'|\’)d ", " would ", text)
    text = re.sub(r"(\'|\’)ll ", " will ", text)
    text = re.sub(r"(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)ve ", " have ", text)
    return text


def clean_latex(text):
    """
    convert r"[math]\vec{x} + \vec{y}" to English
    """
    # edge case
    text = re.sub(r'\[math\]', ' LaTex math ', text)
    text = re.sub(r'\[\/math\]', ' LaTex math ', text)
    text = re.sub(r'\\', ' LaTex ', text)

    pattern_to_sub = {
        r'\\mathrm': ' LaTex math mode ',
        r'\\mathbb': ' LaTex math mode ',
        r'\\boxed': ' LaTex equation ',
        r'\\begin': ' LaTex equation ',
        r'\\end': ' LaTex equation ',
        r'\\left': ' LaTex equation ',
        r'\\right': ' LaTex equation ',
        r'\\(over|under)brace': ' LaTex equation ',
        r'\\text': ' LaTex equation ',
        r'\\vec': ' vector ',
        r'\\var': ' variable ',
        r'\\theta': ' theta ',
        r'\\mu': ' average ',
        r'\\min': ' minimum ',
        r'\\max': ' maximum ',
        r'\\sum': ' + ',
        r'\\times': ' * ',
        r'\\cdot': ' * ',
        r'\\hat': ' ^ ',
        r'\\frac': ' / ',
        r'\\div': ' / ',
        r'\\sin': ' Sine ',
        r'\\cos': ' Cosine ',
        r'\\tan': ' Tangent ',
        r'\\infty': ' infinity ',
        r'\\int': ' integer ',
        r'\\in': ' in ',
    }
    # post process for look up
    pattern_dict = {k.strip('\\'): v for k, v in pattern_to_sub.items()}
    # init re
    patterns = pattern_to_sub.keys()
    pattern_re = re.compile('(%s)' % '|'.join(patterns))

    def _replace(match):
        """
        reference: https://www.kaggle.com/hengzheng/attention-capsule-why-not-both-lb-0-694 # noqa
        """
        try:
            word = pattern_dict.get(match.group(0).strip('\\'))
        except KeyError:
            word = match.group(0)
            print('!!Error: Could Not Find Key: {}'.format(word))
        return word
    return pattern_re.sub(_replace, text)



# clean misspelling words
misspell_mapping = {'Terroristan': 'terrorist Pakistan', 'terroristan': 'terrorist Pakistan',
                    'FATF': 'Western summit conference',
                    'BIMARU': 'BIMARU Bihar, Madhya Pradesh, Rajasthan, Uttar Pradesh', 'Hinduphobic': 'Hindu phobic',
                    'hinduphobic': 'Hindu phobic', 'Hinduphobia': 'Hindu phobic', 'hinduphobia': 'Hindu phobic',
                    'Babchenko': 'Arkady Arkadyevich Babchenko faked death', 'Boshniaks': 'Bosniaks',
                    'Dravidanadu': 'Dravida Nadu', 'mysoginists': 'misogynists', 'MGTOWS': 'Men Going Their Own Way',
                    'mongloid': 'Mongoloid', 'unsincere': 'insincere', 'meninism': 'male feminism',
                    'jewplicate': 'jewish replicate', 'jewplicates': 'jewish replicate', 'andhbhakts': 'and Bhakt',
                    'unoin': 'Union', 'daesh': 'Islamic State of Iraq and the Levant', 'burnol': 'movement about Modi',
                    'Kalergi': 'Coudenhove-Kalergi', 'Bhakts': 'Bhakt', 'bhakts': 'Bhakt', 'Tambrahms': 'Tamil Brahmin',
                    'Pahul': 'Amrit Sanskar', 'SJW': 'social justice warrior', 'SJWs': 'social justice warrior',
                    ' incel': ' involuntary celibates', ' incels': ' involuntary celibates', 'emiratis': 'Emiratis',
                    'weatern': 'western', 'westernise': 'westernize', 'Pizzagate': 'debunked conspiracy theory',
                    'naïve': 'naive', 'Skripal': 'Russian military officer', 'Skripals': 'Russian military officer',
                    'Remainers': 'British remainer', 'Novichok': 'Soviet Union agents',
                    'gauri lankesh': 'Famous Indian Journalist', 'Castroists': 'Castro supporters',
                    'remainers': 'British remainer', 'bremainer': 'British remainer', 'antibrahmin': 'anti Brahminism',
                    'HYPSM': ' Harvard, Yale, Princeton, Stanford, MIT', 'HYPS': ' Harvard, Yale, Princeton, Stanford',
                    'kompromat': 'compromising material', 'Tharki': 'pervert', 'tharki': 'pervert',
                    'mastuburate': 'masturbate', 'Zoë': 'Zoe', 'indans': 'Indian', ' xender': ' gender',
                    'Naxali ': 'Naxalite ', 'Naxalities': 'Naxalites', 'Bathla': 'Namit Bathla',
                    'Mewani': 'Indian politician Jignesh Mevani', 'Wjy': 'Why',
                    'Fadnavis': 'Indian politician Devendra Fadnavis', 'Awadesh': 'Indian engineer Awdhesh Singh',
                    'Awdhesh': 'Indian engineer Awdhesh Singh', 'Khalistanis': 'Sikh separatist movement',
                    'madheshi': 'Madheshi', 'BNBR': 'Be Nice, Be Respectful',
                    'Jair Bolsonaro': 'Brazilian President politician', 'XXXTentacion': 'Tentacion',
                    'Slavoj Zizek': 'Slovenian philosopher',
                    'borderliners': 'borderlines', 'Brexit': 'British Exit', 'Brexiter': 'British Exit supporter',
                    'Brexiters': 'British Exit supporters', 'Brexiteer': 'British Exit supporter',
                    'Brexiteers': 'British Exit supporters', 'Brexiting': 'British Exit',
                    'Brexitosis': 'British Exit disorder', 'brexit': 'British Exit',
                    'brexiters': 'British Exit supporters', 'jallikattu': 'Jallikattu', 'fortnite': 'Fortnite',
                    'Swachh': 'Swachh Bharat mission campaign ', 'Quorans': 'Quora users', 'Qoura': 'Quora',
                    'quoras': 'Quora', 'Quroa': 'Quora', 'QUORA': 'Quora', 'Stupead': 'stupid',
                    'narcissit': 'narcissist', 'trigger nometry': 'trigonometry',
                    'trigglypuff': 'student Criticism of Conservatives', 'peoplelook': 'people look',
                    'paedophelia': 'paedophilia', 'Uogi': 'Yogi', 'adityanath': 'Adityanath',
                    'Yogi Adityanath': 'Indian monk and Hindu nationalist politician',
                    'Awdhesh Singh': 'Commissioner of India', 'Doklam': 'Tibet', 'Drumpf ': 'Donald Trump fool ',
                    'Drumpfs': 'Donald Trump fools', 'Strzok': 'Hillary Clinton scandal', 'rohingya': 'Rohingya ',
                    ' wumao ': ' cheap Chinese stuff ', 'wumaos': 'cheap Chinese stuff', 'Sanghis': 'Sanghi',
                    'Tamilans': 'Tamils', 'biharis': 'Biharis', 'Rejuvalex': 'hair growth formula Medicine',
                    'Fekuchand': 'PM Narendra Modi in India', 'feku': 'Feku', 'Chaiwala': 'tea seller in India',
                    'Feku': 'PM Narendra Modi in India ', 'deplorables': 'deplorable', 'muhajirs': 'Muslim immigrant',
                    'Gujratis': 'Gujarati', 'Chutiya': 'Tibet people ', 'Chutiyas': 'Tibet people ',
                    'thighing': 'masterbate between the legs of a female infant', '卐': 'Nazi Germany',
                    'Pribumi': 'Native Indonesian', 'Gurmehar': 'Gurmehar Kaur Indian student activist',
                    'Khazari': 'Khazars', 'Demonetization': 'demonetization', 'demonetisation': 'demonetization',
                    'demonitisation': 'demonetization', 'demonitization': 'demonetization',
                    'antinationals': 'antinational', 'Cryptocurrencies': 'cryptocurrency',
                    'cryptocurrencies': 'cryptocurrency', 'Hindians': 'North Indian', 'Hindian': 'North Indian',
                    'vaxxer': 'vocal nationalist ', 'remoaner': 'remainer ', 'bremoaner': 'British remainer ',
                    'Jewism': 'Judaism', 'Eroupian': 'European', "J & K Dy CM H ' ble Kavinderji": '',
                    'WMAF': 'White male married Asian female', 'AMWF': 'Asian male married White female',
                    'moeslim': 'Muslim', 'cishet': 'cisgender and heterosexual person', 'Eurocentrics': 'Eurocentrism',
                    'Eurocentric': 'Eurocentrism', 'Afrocentrics': 'Africa centrism', 'Afrocentric': 'Africa centrism',
                    'Jewdar': 'Jew dar', 'marathis': 'Marathi', 'Gynophobic': 'Gyno phobic',
                    'Trumpanzees': 'Trump chimpanzee fool', 'Crimean': 'Crimea people ', 'atrracted': 'attract',
                    'Myeshia': 'widow of Green Beret killed in Niger', 'demcoratic': 'Democratic', 'raaping': 'raping',
                    'feminazism': 'feminism nazi', 'langague': 'language', 'sathyaraj': 'actor',
                    'Hongkongese': 'HongKong people', 'hongkongese': 'HongKong people', 'Kashmirians': 'Kashmirian',
                    'Chodu': 'fucker', 'penish': 'penis',
                    'chitpavan konkanastha': 'Hindu Maharashtrian Brahmin community',
                    'Madridiots': 'Real Madrid idiot supporters', 'Ambedkarite': 'Dalit Buddhist movement ',
                    'ReleaseTheMemo': 'cry for the right and Trump supporters', 'harrase': 'harass',
                    'Barracoon': 'Black slave', 'Castrater': 'castration', 'castrater': 'castration',
                    'Rapistan': 'Pakistan rapist', 'rapistan': 'Pakistan rapist', 'Turkified': 'Turkification',
                    'turkified': 'Turkification', 'Dumbassistan': 'dumb ass Pakistan', 'facetards': 'Facebook retards',
                    'rapefugees': 'rapist refugee', 'Khortha': 'language in the Indian state of Jharkhand',
                    'Magahi': 'language in the northeastern Indian', 'Bajjika': 'language spoken in eastern India',
                    'superficious': 'superficial', 'Sense8': 'American science fiction drama web television series',
                    'Saipul Jamil': 'Indonesia artist', 'bhakht': 'bhakti', 'Smartia': 'dumb nation',
                    'absorve': 'absolve', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Whta': 'What',
                    'esspecial': 'especial', 'doI': 'do I', 'theBest': 'the best',
                    'howdoes': 'how does', 'Etherium': 'Ethereum', '2k17': '2017', '2k18': '2018', 'qiblas': 'qibla',
                    'Hello4 2 cab': 'Online Cab Booking', 'bodyshame': 'body shaming', 'bodyshoppers': 'body shopping',
                    'bodycams': 'body cams', 'Cananybody': 'Can any body', 'deadbody': 'dead body',
                    'deaddict': 'de addict', 'Northindian': 'North Indian ', 'northindian': 'north Indian ',
                    'northkorea': 'North Korea', 'koreaboo': 'Korea boo ',
                    'Brexshit': 'British Exit bullshit', 'shitpost': 'shit post', 'shitslam': 'shit Islam',
                    'shitlords': 'shit lords', 'Fck': 'Fuck', 'Clickbait': 'click bait ', 'clickbait': 'click bait ',
                    'mailbait': 'mail bait', 'healhtcare': 'healthcare', 'trollbots': 'troll bots',
                    'trollled': 'trolled', 'trollimg': 'trolling', 'cybertrolling': 'cyber trolling',
                    'sickular': 'India sick secular ', 'Idiotism': 'idiotism',
                    'Niggerism': 'Nigger', 'Niggeriah': 'Nigger'}

def clean_misspell(text):
    for bad_word in misspell_mapping:
        if bad_word in text:
            text = text.replace(bad_word, misspell_mapping[bad_word])
    return text



# All of the cleaning kernels I found that, they add space around all punctuations. 
# There is a litter bug when spacing dash - and point ., it will drop the word 
# coverage. For example, the words like self-trust, Michelin-starred and pro-Islamic 
# these words already have pretrained embeddings, force break these words will 
# decrease the word coverage! I have found 18128 words like this.
    
regular_punct = list(string.punctuation)
extra_punct = [
    ',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&',
    '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',
    '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
    '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
    '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
    '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
    'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
    '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
    '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤']
all_punct = list(set(regular_punct + extra_punct))
# do not spacing - and .
all_punct.remove('-')
all_punct.remove('.')

def spacing_punctuation(text):
    """
    add space before and after punctuation and symbols
    """
    for punc in all_punct:
        if punc in text:
            text = text.replace(punc, f' {punc} ')
    return text


# spell check and according to bad case analyse
bad_case_words = {'jewprofits': 'jew profits', 'QMAS': 'Quality Migrant Admission Scheme', 'casterating': 'castrating',
                  'Kashmiristan': 'Kashmir', 'CareOnGo': 'India first and largest Online distributor of medicines',
                  'Setya Novanto': 'a former Indonesian politician', 'TestoUltra': 'male sexual enhancement supplement',
                  'rammayana': 'ramayana', 'Badaganadu': 'Brahmin community that mainly reside in Karnataka',
                  'bitcjes': 'bitches', 'mastubrate': 'masturbate', 'Français': 'France',
                  'Adsresses': 'address', 'flemmings': 'flemming', 'intermate': 'inter mating', 'feminisam': 'feminism',
                  'cuckholdry': 'cuckold', 'Niggor': 'black hip-hop and electronic artist', 'narcsissist': 'narcissist',
                  'Genderfluid': 'Gender fluid', ' Im ': ' I am ', ' dont ': ' do not ', 'Qoura': 'Quora',
                  'ethethnicitesnicites': 'ethnicity', 'Namit Bathla': 'Content Writer', 'What sApp': 'WhatsApp',
                  'Führer': 'Fuhrer', 'covfefe': 'coverage', 'accedentitly': 'accidentally', 'Cuckerberg': 'Zuckerberg',
                  'transtrenders': 'incredibly disrespectful to real transgender people',
                  'frozen tamod': 'Pornographic website', 'hindians': 'North Indian', 'hindian': 'North Indian',
                  'celibatess': 'celibates', 'Trimp': 'Trump', 'wanket': 'wanker', 'wouldd': 'would',
                  'arragent': 'arrogant', 'Ra - apist': 'rapist', 'idoot': 'idiot', 'gangstalkers': 'gangs talkers',
                  'toastsexual': 'toast sexual', 'inapropriately': 'inappropriately', 'dumbassess': 'dumbass',
                  'germanized': 'become german', 'helisexual': 'sexual', 'regilious': 'religious',
                  'timetraveller': 'time traveller', 'darkwebcrawler': 'dark webcrawler', 'routez': 'route',
                  'trumpians': 'Trump supporters', 'irreputable': 'reputation', 'serieusly': 'seriously',
                  'anti cipation': 'anticipation', 'microaggression': 'micro aggression', 'Afircans': 'Africans',
                  'microapologize': 'micro apologize', 'Vishnus': 'Vishnu', 'excritment': 'excitement',
                  'disagreemen': 'disagreement', 'gujratis': 'gujarati', 'gujaratis': 'gujarati',
                  'ugggggggllly': 'ugly',
                  'Germanity': 'German', 'SoyBoys': 'cuck men lacking masculine characteristics',
                  'н': 'h', 'м': 'm', 'ѕ': 's', 'т': 't', 'в': 'b', 'υ': 'u', 'ι': 'i',
                  'genetilia': 'genitalia', 'r - apist': 'rapist', 'Borokabama': 'Barack Obama',
                  'arectifier': 'rectifier', 'pettypotus': 'petty potus', 'magibabble': 'magi babble',
                  'nothinking': 'thinking', 'centimiters': 'centimeters', 'saffronized': 'India, politics, derogatory',
                  'saffronize': 'India, politics, derogatory', ' incect ': ' insect ', 'weenus': 'elbow skin',
                  'Pakistainies': 'Pakistanis', 'goodspeaks': 'good speaks', 'inpregnated': 'in pregnant',
                  'rapefilms': 'rape films', 'rapiest': 'rapist', 'hatrednesss': 'hatred',
                  'heightism': 'height discrimination', 'getmy': 'get my', 'onsocial': 'on social',
                  'worstplatform': 'worst platform', 'platfrom': 'platform', 'instagate': 'instigate',
                  'Loy Machedeo': 'person', ' dsire ': ' desire ', 'iservant': 'servant', 'intelliegent': 'intelligent',
                  'WW 1': ' WW1 ', 'WW 2': ' WW2 ', 'ww 1': ' WW1 ', 'ww 2': ' WW2 ',
                  'keralapeoples': 'kerala peoples', 'trumpervotes': 'trumper votes', 'fucktrumpet': 'fuck trumpet',
                  'likebJaish': 'like bJaish', 'likemy': 'like my', 'Howlikely': 'How likely',
                  'disagreementts': 'disagreements', 'disagreementt': 'disagreement',
                  'meninist': "male chauvinism", 'feminists': 'feminism supporters', 'Ghumendra': 'Bhupendra',
                  'emellishments': 'embellishments',
                  'settelemen': 'settlement',
                  'Richmencupid': 'rich men dating website', 'richmencupid': 'rich men dating website',
                  'Gaudry - Schost': '', 'ladymen': 'ladyboy', 'hasserment': 'Harassment',
                  'instrumentalizing': 'instrument', 'darskin': 'dark skin', 'balckwemen': 'balck women',
                  'recommendor': 'recommender', 'wowmen': 'women', 'expertthink': 'expert think',
                  'whitesplaining': 'white splaining', 'Inquoraing': 'inquiring', 'whilemany': 'while many',
                  'manyother': 'many other', 'involvedinthe': 'involved in the', 'slavetrade': 'slave trade',
                  'aswell': 'as well', 'fewshowanyRemorse': 'few show any Remorse', 'trageting': 'targeting',
                  'getile': 'gentile', 'Gujjus': 'derogatory Gujarati', 'judisciously': 'judiciously',
                  'Hue Mungus': 'feminist bait', 'Hugh Mungus': 'feminist bait', 'Hindustanis': '',
                  'Virushka': 'Great Relationships Couple', 'exclusinary': 'exclusionary', 'himdus': 'hindus',
                  'Milo Yianopolous': 'a British polemicist', 'hidusim': 'hinduism',
                  'holocaustable': 'holocaust', 'evangilitacal': 'evangelical', 'Busscas': 'Buscas',
                  'holocaustal': 'holocaust', 'incestious': 'incestuous', 'Tennesseus': 'Tennessee',
                  'GusDur': 'Gus Dur',
                  'RPatah - Tan Eng Hwan': 'Silsilah', 'Reinfectus': 'reinfect', 'pharisaistic': 'pharisaism',
                  'nuslims': 'Muslims', 'taskus': '', 'musims': 'Muslims',
                  'Musevi': 'the independence of Mexico', ' racious ': 'discrimination expression of racism',
                  'Muslimophobia': 'Muslim phobia', 'justyfied': 'justified', 'holocause': 'holocaust',
                  'musilim': 'Muslim', 'misandrous': 'misandry', 'glrous': 'glorious', 'desemated': 'decimated',
                  'votebanks': 'vote banks', 'Parkistan': 'Pakistan', 'Eurooe': 'Europe', 'animlaistic': 'animalistic',
                  'Asiasoid': 'Asian', 'Congoid': 'Congolese', 'inheritantly': 'inherently',
                  'Asianisation': 'Becoming Asia',
                  'Russosphere': 'russia sphere of influence', 'exMuslims': 'Ex-Muslims',
                  'discriminatein': 'discrimination', ' hinus ': ' hindus ', 'Nibirus': 'Nibiru',
                  'habius - corpus': 'habeas corpus', 'prentious': 'pretentious', 'Sussia': 'ancient Jewish village',
                  'moustachess': 'moustaches', 'Russions': 'Russians', 'Yuguslavia': 'Yugoslavia',
                  'atrocitties': 'atrocities', 'Muslimophobe': 'Muslim phobic', 'fallicious': 'fallacious',
                  'recussed': 'recursed', '@ usafmonitor': '', 'lustfly': 'lustful', 'canMuslims': 'can Muslims',
                  'journalust': 'journalist', 'digustingly': 'disgustingly', 'harasing': 'harassing',
                  'greatuncle': 'great uncle', 'Drumpf': 'Trump', 'rejectes': 'rejected', 'polyagamous': 'polygamous',
                  'Mushlims': 'Muslims', 'accusition': 'accusation', 'geniusses': 'geniuses',
                  'moustachesomething': 'moustache something', 'heineous': 'heinous',
                  'Sapiosexuals': 'sapiosexual', 'sapiosexuals': 'sapiosexual', 'Sapiosexual': 'sapiosexual',
                  'sapiosexual': 'Sexually attracted to intelligence', 'pansexuals': 'pansexual',
                  'autosexual': 'auto sexual', 'sexualSlutty': 'sexual Slutty', 'hetorosexuality': 'hetoro sexuality',
                  'chinesese': 'chinese', 'pizza gate': 'debunked conspiracy theory',
                  'countryless': 'Having no country',
                  'muslimare': 'Muslim are', 'iPhoneX': 'iPhone', 'lionese': 'lioness', 'marionettist': 'Marionettes',
                  'demonetize': 'demonetized', 'eneyone': 'anyone', 'Karonese': 'Karo people Indonesia',
                  'minderheid': 'minder worse', 'mainstreamly': 'mainstream', 'contraproductive': 'contra productive',
                  'diffenky': 'differently', 'abandined': 'abandoned', 'p0 rnstars': 'pornstars',
                  'overproud': 'over proud',
                  'cheekboned': 'cheek boned', 'heriones': 'heroines', 'eventhogh': 'even though',
                  'americanmedicalassoc': 'american medical assoc', 'feelwhen': 'feel when', 'Hhhow': 'how',
                  'reallySemites': 'really Semites', 'gamergaye': 'gamersgate', 'manspreading': 'man spreading',
                  'thammana': 'Tamannaah Bhatia', 'dogmans': 'dogmas', 'managementskills': 'management skills',
                  'mangoliod': 'mongoloid', 'geerymandered': 'gerrymandered', 'mandateing': 'man dateing',
                  'Romanium': 'Romanum',
                  'mailwoman': 'mail woman', 'humancoalition': 'human coalition',
                  'manipullate': 'manipulate', 'everyo0 ne': 'everyone', 'takeove': 'takeover',
                  'Nonchristians': 'Non Christians', 'goverenments': 'governments', 'govrment': 'government',
                  'polygomists': 'polygamists', 'Demogorgan': 'Demogorgon', 'maralago': 'Mar-a-Lago',
                  'antibigots': 'anti bigots', 'gouing': 'going', 'muzaffarbad': 'muzaffarabad',
                  'suchvstupid': 'such stupid', 'apartheidisrael': 'apartheid israel', 
                  'personaltiles': 'personal titles', 'lawyergirlfriend': 'lawyer girl friend',
                  'northestern': 'northwestern', 'yeardold': 'years old', 'masskiller': 'mass killer',
                  'southeners': 'southerners', 'Unitedstatesian': 'United states',

                  'peoplekind': 'people kind', 'peoplelike': 'people like', 'countrypeople': 'country people',
                  'shitpeople': 'shit people', 'trumpology': 'trump ology', 'trumpites': 'Trump supporters',
                  'trumplies': 'trump lies', 'donaldtrumping': 'donald trumping', 'trumpdating': 'trump dating',
                  'trumpsters': 'trumpeters', 'ciswomen': 'cis women', 'womenizer': 'womanizer',
                  'pregnantwomen': 'pregnant women', 'autoliker': 'auto liker', 'smelllike': 'smell like',
                  'autolikers': 'auto likers', 'religiouslike': 'religious like', 'likemail': 'like mail',
                  'fislike': 'dislike', 'sneakerlike': 'sneaker like', 'like⬇': 'like',
                  'likelovequotes': 'like lovequotes', 'likelogo': 'like logo', 'sexlike': 'sex like',
                  'Whatwould': 'What would', 'Howwould': 'How would', 'manwould': 'man would',
                  'exservicemen': 'ex servicemen', 'femenism': 'feminism', 'devopment': 'development',
                  'doccuments': 'documents', 'supplementplatform': 'supplement platform', 'mendatory': 'mandatory',
                  'moviments': 'movements', 'Kremenchuh': 'Kremenchug', 'docuements': 'documents',
                  'determenism': 'determinism', 'envisionment': 'envision ment',
                  'tricompartmental': 'tri compartmental', 'AddMovement': 'Add Movement',
                  'mentionong': 'mentioning', 'Whichtreatment': 'Which treatment', 'repyament': 'repayment',
                  'insemenated': 'inseminated', 'inverstment': 'investment',
                  'managemental': 'manage mental', 'Inviromental': 'Environmental', 'menstrution': 'menstruation',
                  'indtrument': 'instrument', 'mentenance': 'maintenance', 'fermentqtion': 'fermentation',
                  'achivenment': 'achievement', 'mismanagements': 'mis managements', 'requriment': 'requirement',
                  'denomenator': 'denominator', 'drparment': 'department', 'acumens': 'acumen s',
                  'celemente': 'Clemente', 'manajement': 'management', 'govermenent': 'government',
                  'accomplishmments': 'accomplishments', 'rendementry': 'rendement ry',
                  'repariments': 'departments', 'menstrute': 'menstruate', 'determenistic': 'deterministic',
                  'resigment': 'resignment', 'selfpayment': 'self payment', 'imrpovement': 'improvement',
                  'enivironment': 'environment', 'compartmentley': 'compartment',
                  'augumented': 'augmented', 'parmenent': 'permanent', 'dealignment': 'de alignment',
                  'develepoments': 'developments', 'menstrated': 'menstruated', 'phnomenon': 'phenomenon',
                  'Employmment': 'Employment', 'dimensionalise': 'dimensional ise', 'menigioma': 'meningioma',
                  'recrument': 'recrement', 'Promenient': 'Provenient', 'gonverment': 'government',
                  'statemment': 'statement', 'recuirement': 'requirement', 'invetsment': 'investment',
                  'parilment': 'parchment', 'parmently': 'patiently', 'agreementindia': 'agreement india',
                  'menifesto': 'manifesto', 'accomplsihments': 'accomplishments', 'disangagement': 'disengagement',
                  'aevelopment': 'development', 'procument': 'procumbent', 'harashment': 'harassment',
                  'Tiannanmen': 'Tiananmen', 'commensalisms': 'commensal isms', 'devlelpment': 'development',
                  'dimensons': 'dimensions', 'recruitment2017': 'recruitment 2017', 'polishment': 'pol ishment',
                  'CommentSafe': 'Comment Safe', 'meausrements': 'measurements', 'geomentrical': 'geometrical',
                  'undervelopment': 'undevelopment', 'mensurational': 'mensuration al', 'fanmenow': 'fan menow',
                  'permenganate': 'permanganate', 'bussinessmen': 'businessmen',
                  'supertournaments': 'super tournaments', 'permanmently': 'permanently',
                  'lamenectomy': 'lamnectomy', 'assignmentcanyon': 'assignment canyon', 'adgestment': 'adjustment',
                  'mentalized': 'metalized', 'docyments': 'documents', 'requairment': 'requirement',
                  'batsmencould': 'batsmen could', 'argumentetc': 'argument etc', 'enjoiment': 'enjoyment',
                  'invement': 'movement', 'accompliushments': 'accomplishments', 'regements': 'regiments',
                  'departmentHow': 'department How', 'Aremenian': 'Armenian', 'amenclinics': 'amen clinics',
                  'nonfermented': 'non fermented', 'Instumentation': 'Instrumentation', 'mentalitiy': 'mentality',
                  ' govermen ': 'goverment', 'underdevelopement': 'under developement', 'parlimentry': 'parliamentary',
                  'indemenity': 'indemnity', 'Inatrumentation': 'Instrumentation', 'menedatory': 'mandatory',
                  'mentiri': 'entire', 'accomploshments': 'accomplishments', 'instrumention': 'instrument ion',
                  'afvertisements': 'advertisements', 'parlementarian': 'parlement arian',
                  'entitlments': 'entitlements', 'endrosment': 'endorsement', 'improment': 'impriment',
                  'archaemenid': 'Achaemenid', 'replecement': 'replacement', 'placdment': 'placement',
                  'femenise': 'feminise', 'envinment': 'environment', 'AmenityCompany': 'Amenity Company',
                  'increaments': 'increments', 'accomplihsments': 'accomplishments',
                  'manygovernment': 'many government', 'panishments': 'punishments', 'elinment': 'eloinment',
                  'mendalin': 'mend alin', 'farmention': 'farm ention', 'preincrement': 'pre increment',
                  'postincrement': 'post increment', 'achviements': 'achievements', 'menditory': 'mandatory',
                  'Emouluments': 'Emoluments', 'Stonemen': 'Stone men', 'menmium': 'medium',
                  'entaglement': 'entanglement', 'integumen': 'integument', 'harassument': 'harassment',
                  'retairment': 'retainment', 'enviorement': 'environment', 'tormentous': 'torment ous',
                  'confiment': 'confident', 'Enchroachment': 'Encroachment', 'prelimenary': 'preliminary',
                  'fudamental': 'fundamental', 'instrumenot': 'instrument', 'icrement': 'increment',
                  'prodimently': 'prominently', 'meniss': 'menise', 'Whoimplemented': 'Who implemented',
                  'Representment': 'Rep resentment', 'StartFragment': 'Start Fragment',
                  'EndFragment': 'End Fragment', ' documentarie ': ' documentaries ', 'requriments': 'requirements',
                  'constitutionaldevelopment': 'constitutional development', 'parlamentarians': 'parliamentarians',
                  'Rumenova': 'Rumen ova', 'argruments': 'arguments', 'findamental': 'fundamental',
                  'totalinvestment': 'total investment', 'gevernment': 'government', 'recmommend': 'recommend',
                  'appsmoment': 'apps moment', 'menstruual': 'menstrual', 'immplemented': 'implemented',
                  'engangement': 'engagement', 'invovement': 'involvement', 'returement': 'retirement',
                  'simentaneously': 'simultaneously', 'accompishments': 'accomplishments',
                  'menstraution': 'menstruation', 'experimently': 'experiment', 'abdimen': 'abdomen',
                  'cemenet': 'cement', 'propelment': 'propel ment', 'unamendable': 'un amendable',
                  'employmentnews': 'employment news', 'lawforcement': 'law forcement',
                  'menstuating': 'menstruating', 'fevelopment': 'development', 'reglamented': 'reg lamented',
                  'imrovment': 'improvement', 'recommening': 'recommending', 'sppliment': 'supplement',
                  'measument': 'measurement', 'reimbrusement': 'reimbursement', 'Nutrament': 'Nutriment',
                  'puniahment': 'punishment', 'subligamentous': 'sub ligamentous', 'comlementry': 'complementary',
                  'reteirement': 'retirement', 'envioronments': 'environments', 'haraasment': 'harassment',
                  'USAgovernment': 'USA government', 'Apartmentfinder': 'Apartment finder',
                  'encironment': 'environment', 'metacompartment': 'meta compartment',
                  'augumentation': 'argumentation', 'dsymenorrhoea': 'dysmenorrhoea',
                  'nonabandonment': 'non abandonment', 'annoincement': 'announcement',
                  'menberships': 'memberships', 'Gamenights': 'Game nights', 'enliightenment': 'enlightenment',
                  'supplymentry': 'supplementary', 'parlamentary': 'parliamentary', 'duramen': 'dura men',
                  'hotelmanagement': 'hotel management', 'deartment': 'department',
                  'treatmentshelp': 'treatments help', 'attirements': 'attire ments',
                  'amendmending': 'amend mending', 'pseudomeningocele': 'pseudo meningocele',
                  'intrasegmental': 'intra segmental', 'treatmenent': 'treatment', 'infridgement': 'infringement',
                  'infringiment': 'infringement', 'recrecommend': 'rec recommend', 'entartaiment': 'entertainment',
                  'inplementing': 'implementing', 'indemendent': 'independent', 'tremendeous': 'tremendous',
                  'commencial': 'commercial', 'scomplishments': 'accomplishments', 'Emplement': 'Implement',
                  'dimensiondimensions': 'dimension dimensions', 'depolyment': 'deployment',
                  'conpartment': 'compartment', 'govnments': 'movements', 'menstrat': 'menstruate',
                  'accompplishments': 'accomplishments', 'Enchacement': 'Enchancement',
                  'developmenent': 'development', 'emmenagogues': 'emmenagogue', 'aggeement': 'agreement',
                  'elementsbond': 'elements bond', 'remenant': 'remnant', 'Manamement': 'Management',
                  'Augumented': 'Augmented', 'dimensonless': 'dimensionless',
                  'ointmentsointments': 'ointments ointments', 'achiements': 'achievements',
                  'recurtment': 'recurrent', 'gouverments': 'governments', 'docoment': 'document',
                  'programmingassignments': 'programming assignments', 'menifest': 'manifest',
                  'investmentguru': 'investment guru', 'deployements': 'deployments', 'Invetsment': 'Investment',
                  'plaement': 'placement', 'Perliament': 'Parliament', 'femenists': 'feminists',
                  'ecumencial': 'ecumenical', 'advamcements': 'advancements', 'refundment': 'refund ment',
                  'settlementtake': 'settlement take', 'mensrooms': 'mens rooms',
                  'productManagement': 'product Management', 'armenains': 'armenians',
                  'betweenmanagement': 'between management', 'difigurement': 'disfigurement',
                  'Armenized': 'Armenize', 'hurrasement': 'hurra sement', 'mamgement': 'management',
                  'momuments': 'monuments', 'eauipments': 'equipments', 'managemenet': 'management',
                  'treetment': 'treatment', 'webdevelopement': 'web developement', 'supplemenary': 'supplementary',
                  'Encironmental': 'Environmental', 'Understandment': 'Understand ment',
                  'enrollnment': 'enrollment', 'thinkstrategic': 'think strategic', 'thinkinh': 'thinking',
                  'Softthinks': 'Soft thinks', 'underthinking': 'under thinking', 'thinksurvey': 'think survey',
                  'whitelash': 'white lash', 'whiteheds': 'whiteheads', 'whitetning': 'whitening',
                  'whitegirls': 'white girls', 'whitewalkers': 'white walkers', 'manycountries': 'many countries',
                  'accomany': 'accompany', 'fromGermany': 'from Germany', 'manychat': 'many chat',
                  'Germanyl': 'Germany l', 'manyness': 'many ness', 'many4': 'many', 'exmuslims': 'ex muslims',
                  'digitizeindia': 'digitize india', 'indiarush': 'india rush', 'indiareads': 'india reads',
                  'telegraphindia': 'telegraph india', 'Southindia': 'South india', 'Airindia': 'Air india',
                  'siliconindia': 'silicon india', 'airindia': 'air india', 'indianleaders': 'indian leaders',
                  'fundsindia': 'funds india', 'indianarmy': 'indian army', 'Technoindia': 'Techno india',
                  'Betterindia': 'Better india', 'capesindia': 'capes india', 'Rigetti': 'Ligetti',
                  'vegetablr': 'vegetable', 'get90': 'get', 'Magetta': 'Maretta', 'nagetive': 'native',
                  'isUnforgettable': 'is Unforgettable', 'get630': 'get 630', 'GadgetPack': 'Gadget Pack',
                  'Languagetool': 'Language tool', 'bugdget': 'budget', 'africaget': 'africa get',
                  'ABnegetive': 'Abnegative', 'orangetheory': 'orange theory', 'getsmuggled': 'get smuggled',
                  'avegeta': 'ave geta', 'gettubg': 'getting', 'gadgetsnow': 'gadgets now',
                  'surgetank': 'surge tank', 'gadagets': 'gadgets', 'getallparts': 'get allparts',
                  'messenget': 'messenger', 'vegetarean': 'vegetarian', 'get1000': 'get 1000',
                  'getfinancing': 'get financing', 'getdrip': 'get drip', 'AdsTargets': 'Ads Targets',
                  'tgethr': 'together', 'vegetaries': 'vegetables', 'forgetfulnes': 'forgetfulness',
                  'fisgeting': 'fidgeting', 'BudgetAir': 'Budget Air',
                  'getDepersonalization': 'get Depersonalization', 'negetively': 'negatively',
                  'gettibg': 'getting', 'nauget': 'naught', 'Bugetti': 'Bugatti', 'plagetum': 'plage tum',
                  'vegetabale': 'vegetable', 'changetip': 'change tip', 'blackwashing': 'black washing',
                  'blackpink': 'black pink', 'blackmoney': 'black money',
                  'blackmarks': 'black marks', 'blackbeauty': 'black beauty', 'unblacklisted': 'un blacklisted',
                  'blackdotes': 'black dotes', 'blackboxing': 'black boxing', 'blackpaper': 'black paper',
                  'blackpower': 'black power', 'Latinamericans': 'Latin americans', 'musigma': 'mu sigma',
                  'Indominus': 'In dominus', 'usict': 'USSCt', 'indominus': 'in dominus', 'Musigma': 'Mu sigma',
                  'plus5': 'plus', 'Russiagate': 'Russia gate', 'russophobic': 'Russophobiac',
                  'Marcusean': 'Marcuse an', 'Radijus': 'Radius', 'cobustion': 'combustion',
                  'Austrialians': 'Australians', 'mylogenous': 'myogenous', 'Raddus': 'Radius',
                  'hetrogenous': 'heterogenous', 'greenhouseeffect': 'greenhouse effect', 'aquous': 'aqueous',
                  'Taharrush': 'Tahar rush', 'Senousa': 'Venous', 'diplococcus': 'diplo coccus',
                  'CityAirbus': 'City Airbus', 'sponteneously': 'spontaneously', 'trustless': 't rustless',
                  'Pushkaram': 'Pushkara m', 'Fusanosuke': 'Fu sanosuke', 'isthmuses': 'isthmus es',
                  'lucideus': 'lucidum', 'overjustification': 'over justification', 'Bindusar': 'Bind usar',
                  'cousera': 'couler', 'musturbation': 'masturbation', 'infustry': 'industry',
                  'Huswifery': 'Huswife ry', 'rombous': 'bombous', 'disengenuously': 'disingenuously',
                  'sllybus': 'syllabus', 'celcious': 'delicious', 'cellsius': 'celsius',
                  'lethocerus': 'Lethocerus', 'monogmous': 'monogamous', 'Ballyrumpus': 'Bally rumpus',
                  'Koushika': 'Koushik a', 'vivipoarous': 'viviparous', 'ludiculous': 'ridiculous',
                  'sychronous': 'synchronous', 'industiry': 'industry', 'scuduse': 'scud use',
                  'babymust': 'baby must', 'simultqneously': 'simultaneously', 'exust': 'ex ust',
                  'notmusing': 'not musing', 'Zamusu': 'Amuse', 'tusaki': 'tu saki', 'Marrakush': 'Marrakesh',
                  'justcheaptickets': 'just cheaptickets', 'Ayahusca': 'Ayahausca', 'samousa': 'samosa',
                  'Gusenberg': 'Gutenberg', 'illustratuons': 'illustrations', 'extemporeneous': 'extemporaneous',
                  'Mathusla': 'Mathusala', 'Confundus': 'Con fundus', 'tusts': 'trusts', 'poisenious': 'poisonous',
                  'Mevius': 'Medius', 'inuslating': 'insulating', 'aroused21000': 'aroused 21000',
                  'Wenzeslaus': 'Wenceslaus', 'JustinKase': 'Justin Kase', 'purushottampur': 'purushottam pur',
                  'citruspay': 'citrus pay', 'secutus': 'sects', 'austentic': 'austenitic',
                  'FacePlusPlus': 'Face PlusPlus', 'aysnchronous': 'asynchronous',
                  'teamtreehouse': 'team treehouse', 'uncouncious': 'unconscious', 'Priebuss': 'Prie buss',
                  'consciousuness': 'consciousness', 'susubsoil': 'su subsoil', 'trimegistus': 'Trismegistus',
                  'protopeterous': 'protopterous', 'trustworhty': 'trustworthy', 'ushually': 'usually',
                  'industris': 'industries', 'instantneous': 'instantaneous', 'superplus': 'super plus',
                  'shrusti': 'shruti', 'hindhus': 'hindus', 'outonomous': 'autonomous', 'reliegious': 'religious',
                  'Kousakis': 'Kou sakis', 'reusult': 'result', 'JanusGraph': 'Janus Graph',
                  'palusami': 'palus ami', 'mussraff': 'muss raff', 'hukous': 'humous',
                  'photoacoustics': 'photo acoustics', 'kushanas': 'kusha nas', 'justdile': 'justice',
                  'Massahusetts': 'Massachusetts', 'uspset': 'upset', 'sustinet': 'sustinent',
                  'consicious': 'conscious', 'Sadhgurus': 'Sadh gurus', 'hystericus': 'hysteric us',
                  'visahouse': 'visa house', 'supersynchronous': 'super synchronous', 'posinous': 'rosinous',
                  'Fernbus': 'Fern bus', 'Tiltbrush': 'Tilt brush', 'glueteus': 'gluteus', 'posionus': 'poisons',
                  'Freus': 'Frees', 'Zhuchengtyrannus': 'Zhucheng tyrannus', 'savonious': 'sanious',
                  'CusJo': 'Cusco', 'congusion': 'confusion', 'dejavus': 'dejavu s', 'uncosious': 'uncopious',
                  'previius': 'previous', 'counciousness': 'conciousness', 'lustorus': 'lustrous',
                  'sllyabus': 'syllabus', 'mousquitoes': 'mosquitoes', 'Savvius': 'Savvies', 'arceius': 'Arcesius',
                  'prejusticed': 'prejudiced', 'requsitioned': 'requisitioned',
                  'deindustralization': 'deindustrialization', 'muscleblaze': 'muscle blaze',
                  'ConsciousX5': 'conscious', 'nitrogenious': 'nitrogenous', 'mauritious': 'mauritius',
                  'rigrously': 'rigorously', 'Yutyrannus': 'Yu tyrannus', 'muscualr': 'muscular',
                  'conscoiusness': 'consciousness', 'Causians': 'Crusians', 'WorkFusion': 'Work Fusion',
                  'puspak': 'pu spak', 'Inspirus': 'Inspires', 'illiustrations': 'illustrations',
                  'Nobushi': 'No bushi', 'theuseof': 'thereof', 'suspicius': 'suspicious', 'Intuous': 'Virtuous',
                  'gaushalas': 'gaus halas', 'campusthrough': 'campus through', 'seriousity': 'seriosity',
                  'resustence': 'resistence', 'geminatus': 'geminates', 'disquss': 'discuss',
                  'nicholus': 'nicholas', 'Husnai': 'Hussar', 'diiscuss': 'discuss', 'diffussion': 'diffusion',
                  'phusicist': 'physicist', 'ernomous': 'enormous', 'Khushali': 'Khushal i', 'heitus': 'Leitus',
                  'cracksbecause': 'cracks because', 'Nautlius': 'Nautilus', 'trausted': 'trusted',
                  'Dardandus': 'Dardanus', 'Megatapirus': 'Mega tapirus', 'clusture': 'culture',
                  'vairamuthus': 'vairamuthu s', 'disclousre': 'disclosure',
                  'industrilaization': 'industrialization', 'musilms': 'muslims', 'Australia9': 'Australian',
                  'causinng': 'causing', 'ibdustries': 'industries', 'searious': 'serious',
                  'Coolmuster': 'Cool muster', 'sissyphus': 'sisyphus', ' justificatio ': 'justification',
                  'antihindus': 'anti hindus', 'Moduslink': 'Modus link', 'zymogenous': 'zymogen ous',
                  'prospeorus': 'prosperous', 'Retrocausality': 'Retro causality', 'FusionGPS': 'Fusion GPS',
                  'Mouseflow': 'Mouse flow', 'bootyplus': 'booty plus', 'Itylus': 'I tylus',
                  'Olnhausen': 'Olshausen', 'suspeect': 'suspect', 'entusiasta': 'enthusiast',
                  'fecetious': 'facetious', 'bussiest': 'fussiest', 'Draconius': 'Draconis',
                  'requsite': 'requisite', 'nauseatic': 'nausea tic', 'Brusssels': 'Brussels',
                  'repurcussion': 'repercussion', 'Jeisus': 'Jesus', 'philanderous': 'philander ous',
                  'muslisms': 'muslims', 'august2017': 'august 2017', 'calccalculus': 'calc calculus',
                  'unanonymously': 'un anonymously', 'Imaprtus': 'Impetus', 'carnivorus': 'carnivorous',
                  'Corypheus': 'Coryphees', 'austronauts': 'astronauts', 'neucleus': 'nucleus',
                  'housepoor': 'house poor', 'rescouses': 'responses', 'Tagushi': 'Tagus hi',
                  'hyperfocusing': 'hyper focusing', 'nutriteous': 'nutritious', 'chylus': 'chylous',
                  'preussure': 'pressure', 'outfocus': 'out focus', 'Hanfus': 'Hannus', 'Rustyrose': 'Rusty rose',
                  'vibhushant': 'vibhushan t', 'conciousnes': 'conciousness', 'Venus25': 'Venus',
                  'Sedataious': 'Seditious', 'promuslim': 'pro muslim', 'statusGuru': 'status Guru',
                  'yousician': 'musician', 'transgenus': 'trans genus', 'Pushbullet': 'Push bullet',
                  'jeesyllabus': 'jee syllabus', 'complusary': 'compulsory', 'Holocoust': 'Holocaust',
                  'careerplus': 'career plus', 'Lllustrate': 'Illustrate', 'Musino': 'Musion',
                  'Phinneus': 'Phineus', 'usedtoo': 'used too', 'JustBasic': 'Just Basic', 'webmusic': 'web music',
                  'TrustKit': 'Trust Kit', 'industrZgies': 'industries', 'rubustness': 'robustness',
                  'Missuses': 'Miss uses', 'Musturbation': 'Masturbation', 'bustees': 'bus tees',
                  'justyfy': 'justify', 'pegusus': 'pegasus', 'industrybuying': 'industry buying',
                  'advantegeous': 'advantageous', 'kotatsus': 'kotatsu s', 'justcreated': 'just created',
                  'simultameously': 'simultaneously', 'husoone': 'huso one', 'twiceusing': 'twice using',
                  'cetusplay': 'cetus play', 'sqamous': 'squamous', 'claustophobic': 'claustrophobic',
                  'Kaushika': 'Kaushik a', 'dioestrus': 'di oestrus', 'Degenerous': 'De generous',
                  'neculeus': 'nucleus', 'cutaneously': 'cu taneously', 'Alamotyrannus': 'Alamo tyrannus',
                  'Ivanious': 'Avanious', 'arceous': 'araceous', 'Flixbus': 'Flix bus', 'caausing': 'causing',
                  'publious': 'Publius', 'Juilus': 'Julius', 'Australianism': 'Australian ism',
                  'vetronus': 'verrons', 'nonspontaneous': 'non spontaneous', 'calcalus': 'calculus',
                  'commudus': 'Commodus', 'Rheusus': 'Rhesus', 'syallubus': 'syllabus', 'Yousician': 'Musician',
                  'qurush': 'qu rush', 'athiust': 'athirst', 'conclusionless': 'conclusion less',
                  'usertesting': 'user testing', 'redius': 'radius', 'Austrolia': 'Australia',
                  'sllaybus': 'syllabus', 'toponymous': 'top onymous', 'businiss': 'business',
                  'hyperthalamus': 'hyper thalamus', 'clause55': 'clause', 'cosicous': 'conscious',
                  'Sushena': 'Saphena', 'Luscinus': 'Luscious', 'Prussophile': 'Russophile', 'jeaslous': 'jealous',
                  'Austrelia': 'Australia', 'contiguious': 'contiguous',
                  'subconsciousnesses': 'sub consciousnesses', ' jusification ': 'justification',
                  'dilusion': 'delusion', 'anticoncussive': 'anti concussive', 'disngush': 'disgust',
                  'constiously': 'consciously', 'filabustering': 'filibustering', 'GAPbuster': 'GAP buster',
                  'insectivourous': 'insectivorous', 'glocuse': 'louse', 'Antritrust': 'Antitrust',
                  'thisAustralian': 'this Australian', 'FusionDrive': 'Fusion Drive', 'nuclus': 'nucleus',
                  'abussive': 'abusive', 'mustang1': 'mustangs', 'inradius': 'in radius', 'polonious': 'polonius',
                  'ofKulbhushan': 'of Kulbhushan', 'homosporous': 'homos porous', 'circumradius': 'circum radius',
                  'atlous': 'atrous', 'insustry': 'industry', 'campuswith': 'campus with', 'beacsuse': 'because',
                  'concuous': 'conscious', 'nonHindus': 'non Hindus', 'carnivourous': 'carnivorous',
                  'tradeplus': 'trade plus', 'Jeruselam': 'Jerusalem',
                  'musuclar': 'muscular', 'deangerous': 'dangerous', 'disscused': 'discussed',
                  'industdial': 'industrial', 'sallatious': 'fallacious', 'rohmbus': 'rhombus',
                  'golusu': 'gol usu', 'Minangkabaus': 'Minangkabau s', 'Mustansiriyah': 'Mustansiriya h',
                  'anomymously': 'anonymously', 'abonymously': 'anonymously', 'indrustry': 'industry',
                  'Musharrf': 'Musharraf', 'workouses': 'workhouses', 'sponataneously': 'spontaneously',
                  'anmuslim': 'an muslim', 'syallbus': 'syllabus', 'presumptuousnes': 'presumptuousness',
                  'Thaedus': 'Thaddus', 'industey': 'industry', 'hkust': 'hust', 'Kousseri': 'Kousser i',
                  'mousestats': 'mouses tats', 'russiagate': 'russia gate', 'simantaneously': 'simultaneously',
                  'Austertana': 'Auster tana', 'infussions': 'infusions', 'coclusion': 'conclusion',
                  'sustainabke': 'sustainable', 'tusami': 'tu sami', 'anonimously': 'anonymously',
                  'usebase': 'use base', 'balanoglossus': 'Balanoglossus', 'Unglaus': 'Ung laus',
                  'ignoramouses': 'ignoramuses', 'snuus': 'snugs', 'reusibility': 'reusability',
                  'Straussianism': 'Straussian ism', 'simoultaneously': 'simultaneously',
                  'realbonus': 'real bonus', 'nuchakus': 'nunchakus', 'annonimous': 'anonymous',
                  'Incestious': 'Incestuous', 'Manuscriptology': 'Manuscript ology', 'difusse': 'diffuse',
                  'Pliosaurus': 'Pliosaur us', 'cushelle': 'cush elle', 'Catallus': 'Catullus',
                  'MuscleBlaze': 'Muscle Blaze', 'confousing': 'confusing', 'enthusiasmless': 'enthusiasm less',
                  'Tetherusd': 'Tethered', 'Josephius': 'Josephus', 'jusrlt': 'just',
                  'simutaneusly': 'simultaneously', 'mountaneous': 'mountainous', 'Badonicus': 'Sardonicus',
                  'muccus': 'mucous', 'nicus': 'nidus', 'austinlizards': 'austin lizards',
                  'errounously': 'erroneously', 'Australua': 'Australia', 'sylaabus': 'syllabus',
                  'dusyant': 'distant', 'javadiscussion': 'java discussion', 'megabuses': 'mega buses',
                  'danergous': 'dangerous', 'contestious': 'contentious', 'exause': 'excuse',
                  'muscluar': 'muscular', 'avacous': 'vacuous', 'Ingenhousz': 'Ingenious',
                  'holocausting': 'holocaust ing', 'Pakustan': 'Pakistan', 'purusharthas': 'purushartha',
                  'bapus': 'bapu s', 'useul': 'useful', 'pretenious': 'pretentious', 'homogeneus': 'homogeneous',
                  'bhlushes': 'blushes', 'Saggittarius': 'Sagittarius', 'sportsusa': 'sports usa',
                  'kerataconus': 'keratoconus', 'infrctuous': 'infectuous', 'Anonoymous': 'Anonymous',
                  'triphosphorus': 'tri phosphorus', 'ridicjlously': 'ridiculously',
                  'worldbusiness': 'world business', 'hollcaust': 'holocaust', 'Dusra': 'Dura',
                  'meritious': 'meritorious', 'Sauskes': 'Causes', 'inudustry': 'industry',
                  'frustratd': 'frustrate', 'hypotenous': 'hypogenous', 'Dushasana': 'Dush asana',
                  'saadus': 'status', 'keratokonus': 'keratoconus', 'Jarrus': 'Harrus', 'neuseous': 'nauseous',
                  'simutanously': 'simultaneously', 'diphosphorus': 'di phosphorus', 'sulprus': 'surplus',
                  'Hasidus': 'Hasid us', 'suspenive': 'suspensive', 'illlustrator': 'illustrator',
                  'userflows': 'user flows', 'intrusivethoughts': 'intrusive thoughts', 'countinous': 'continuous',
                  'gpusli': 'gusli', 'Calculus1': 'Calculus', 'bushiri': 'Bushire',
                  'torvosaurus': 'Torosaurus', 'chestbusters': 'chest busters', 'Satannus': 'Sat annus',
                  'falaxious': 'fallacious', 'obnxious': 'obnoxious', 'tranfusions': 'transfusions',
                  'PlayMagnus': 'Play Magnus', 'Epicodus': 'Episodes', 'Hypercubus': 'Hypercubes',
                  'Musickers': 'Musick ers', 'programmebecause': 'programme because', 'indiginious': 'indigenous',
                  'housban': 'Housman', 'iusso': 'kusso', 'annilingus': 'anilingus', 'Nennus': 'Genius',
                  'pussboy': 'puss boy', 'Photoacoustics': 'Photo acoustics', 'Hindusthanis': 'Hindustanis',
                  'lndustrial': 'industrial', 'tyrannously': 'tyrannous', 'Susanoomon': 'Susanoo mon',
                  'colmbus': 'columbus', 'sussessful': 'successful', 'ousmania': 'ous mania',
                  'ilustrating': 'illustrating', 'famousbirthdays': 'famous birthdays',
                  'suspectance': 'suspect ance', 'extroneous': 'extraneous', 'teethbrush': 'teeth brush',
                  'abcmouse': 'abc mouse', 'degenerous': 'de generous', 'doesGauss': 'does Gauss',
                  'insipudus': 'insipidus', 'movielush': 'movie lush', 'Rustichello': 'Rustic hello',
                  'Firdausiya': 'Firdausi ya', 'checkusers': 'check users', 'householdware': 'household ware',
                  'prosporously': 'prosperously', 'SteLouse': 'Ste Louse', 'obfuscaton': 'obfuscation',
                  'amorphus': 'amorph us', 'trustworhy': 'trustworthy', 'celsious': 'cesious',
                  'dangorous': 'dangerous', 'anticancerous': 'anti cancerous', 'cousi ': 'cousin ',
                  'austroloid': 'australoid', 'fergussion': 'percussion', 'andKyokushin': 'and Kyokushin',
                  'cousan': 'cousin', 'Huskystar': 'Hu skystar', 'retrovisus': 'retrovirus', 'becausr': 'because',
                  'Jerusalsem': 'Jerusalem', 'motorious': 'notorious', 'industrilised': 'industrialised',
                  'powerballsusa': 'powerballs usa', 'monoceious': 'monoecious', 'batteriesplus': 'batteries plus',
                  'nonviscuous': 'nonviscous', 'industion': 'induction', 'bussinss': 'bussings',
                  'userbags': 'user bags', 'Jlius': 'Julius', 'thausand': 'thousand', 'plustwo': 'plus two',
                  'defpush': 'def push', 'subconcussive': 'sub concussive', 'muslium': 'muslim',
                  'industrilization': 'industrialization', 'Maurititus': 'Mauritius', 'uslme': 'some',
                  'Susgaon': 'Surgeon', 'Pantherous': 'Panther ous', 'antivirius': 'antivirus',
                  'Trustclix': 'Trust clix', 'silumtaneously': 'simultaneously', 'Icompus': 'Corpus',
                  'atonomous': 'autonomous', 'Reveuse': 'Reve use', 'legumnous': 'leguminous',
                  'syllaybus': 'syllabus', 'louspeaker': 'loudspeaker', 'susbtraction': 'substraction',
                  'virituous': 'virtuous', 'disastrius': 'disastrous', 'jerussalem': 'jerusalem',
                  'Industrailzed': 'Industrialized', 'recusion': 'recushion',
                  'simultenously': 'simultaneously',
                  'Pulphus': 'Pulpous', 'harbaceous': 'herbaceous', 'phlegmonous': 'phlegmon ous', 'use38': 'use',
                  'jusify': 'justify', 'instatanously': 'instantaneously', 'tetramerous': 'tetramer ous',
                  'usedvin': 'used vin', 'sagittarious': 'sagittarius', 'mausturbate': 'masturbate',
                  'subcautaneous': 'subcutaneous', 'dangergrous': 'dangerous', 'sylabbus': 'syllabus',
                  'hetorozygous': 'heterozygous', 'Ignasius': 'Ignacius', 'businessbor': 'business bor',
                  'Bhushi': 'Thushi', 'Moussolini': 'Mussolini', 'usucaption': 'usu caption',
                  'Customzation': 'Customization', 'cretinously': 'cretinous', 'genuiuses': 'geniuses',
                  'Moushmee': 'Mousmee', 'neigous': 'nervous',
                  'infrustructre': 'infrastructure', 'Ilusha': 'Ilesha', 'suconciously': 'unconciously',
                  'stusy': 'study', 'mustectomy': 'mastectomy', 'Farmhousebistro': 'Farmhouse bistro',
                  'instantanous': 'instantaneous', 'JustForex': 'Just Forex', 'Indusyry': 'Industry',
                  'mustabating': 'must abating', 'uninstrusive': 'unintrusive', 'customshoes': 'customs hoes',
                  'homageneous': 'homogeneous', 'Empericus': 'Imperious', 'demisexuality': 'demi sexuality',
                  'transexualism': 'transsexualism', 'sexualises': 'sexualise', 'demisexuals': 'demisexual',
                  'sexuly': 'sexily', 'Pornosexuality': 'Porno sexuality', 'sexond': 'second', 'sexxual': 'sexual',
                  'asexaul': 'asexual', 'sextactic': 'sex tactic', 'sexualityism': 'sexuality ism',
                  'monosexuality': 'mono sexuality', 'intwrsex': 'intersex', 'hypersexualize': 'hyper sexualize',
                  'homosexualtiy': 'homosexuality', 'examsexams': 'exams exams', 'sexmates': 'sex mates',
                  'sexyjobs': 'sexy jobs', 'sexitest': 'sexiest', 'fraysexual': 'fray sexual',
                  'sexsurrogates': 'sex surrogates', 'sexuallly': 'sexually', 'gamersexual': 'gamer sexual',
                  'greysexual': 'grey sexual', 'omnisexuality': 'omni sexuality', 'hetereosexual': 'heterosexual',
                  'productsexamples': 'products examples', 'sexgods': 'sex gods', 'semisexual': 'semi sexual',
                  'homosexulity': 'homosexuality', 'sexeverytime': 'sex everytime', 'neurosexist': 'neuro sexist',
                  'worldquant': 'world quant', 'Freshersworld': 'Freshers world', 'smartworld': 'sm artworld',
                  'Mistworlds': 'Mist worlds', 'boothworld': 'booth world', 'ecoworld': 'eco world',
                  'Ecoworld': 'Eco world', 'underworldly': 'under worldly', 'worldrank': 'world rank',
                  'Clearworld': 'Clear world', 'Boothworld': 'Booth world', 'Rimworld': 'Rim world',
                  'cryptoworld': 'crypto world', 'machineworld': 'machine world', 'worldwideley': 'worldwide ley',
                  'capuletwant': 'capulet want', 'Bhagwanti': 'Bhagwant i', 'Unwanted72': 'Unwanted 72',
                  'wantrank': 'want rank',
                  'willhappen': 'will happen', 'thateasily': 'that easily',
                  'Whatevidence': 'What evidence', 'metaphosphates': 'meta phosphates',
                  'exilarchate': 'exilarch ate', 'aulphate': 'sulphate', 'Whateducation': 'What education',
                  'persulphates': 'per sulphates', 'disulphate': 'di sulphate', 'picosulphate': 'pico sulphate',
                  'tetraosulphate': 'tetrao sulphate', 'prechinese': 'pre chinese',
                  'Hellochinese': 'Hello chinese', 'muchdeveloped': 'much developed', 'stomuch': 'stomach',
                  'Whatmakes': 'What makes', 'Lensmaker': 'Lens maker', 'eyemake': 'eye make',
                  'Techmakers': 'Tech makers', 'cakemaker': 'cake maker', 'makeup411': 'makeup 411',
                  'objectmake': 'object make', 'crazymaker': 'crazy maker', 'techmakers': 'tech makers',
                  'makedonian': 'macedonian', 'makeschool': 'make school', 'anxietymake': 'anxiety make',
                  'makeshifter': 'make shifter', 'countryball': 'country ball', 'Whichcountry': 'Which country',
                  'countryHow': 'country How', 'Zenfone': 'Zen fone', 'Electroneum': 'Electro neum',
                  'electroneum': 'electro neum', 'Demonetisation': 'demonetization', 'zenfone': 'zen fone',
                  'ZenFone': 'Zen Fone', 'onecoin': 'one coin', 'demonetizing': 'demonetized',
                  'iphone7': 'iPhone', 'iPhone6': 'iPhone', 'microneedling': 'micro needling', 'iphone6': 'iPhone',
                  'Monegasques': 'Monegasque s', 'demonetised': 'demonetized',
                  'EveryoneDiesTM': 'EveryoneDies TM', 'teststerone': 'testosterone', 'DoneDone': 'Done Done',
                  'papermoney': 'paper money', 'Sasabone': 'Sasa bone', 'Blackphone': 'Black phone',
                  'Bonechiller': 'Bone chiller', 'Moneyfront': 'Money front', 'workdone': 'work done',
                  'iphoneX': 'iPhone', 'roxycodone': 'r oxycodone',
                  'moneycard': 'money card', 'Fantocone': 'Fantocine', 'eletronegativity': 'electronegativity',
                  'mellophones': 'mellophone s', 'isotones': 'iso tones', 'donesnt': 'doesnt',
                  'thereanyone': 'there anyone', 'electronegativty': 'electronegativity',
                  'commissiioned': 'commissioned', 'earvphone': 'earphone', 'condtioners': 'conditioners',
                  'demonetistaion': 'demonetization', 'ballonets': 'ballo nets', 'DoneClaim': 'Done Claim',
                  'alimoney': 'alimony', 'iodopovidone': 'iodo povidone', 'bonesetters': 'bone setters',
                  'componendo': 'compon endo', 'probationees': 'probationers', 'one300': 'one 300',
                  'nonelectrolyte': 'non electrolyte', 'ozonedepletion': 'ozone depletion',
                  'Stonehart': 'Stone hart', 'Vodafone2': 'Vodafones', 'chaparone': 'chaperone',
                  'Noonein': 'Noo nein', 'Frosione': 'Erosion', 'IPhone7': 'Iphone', 'pentanone': 'penta none',
                  'poneglyphs': 'pone glyphs', 'cyclohexenone': 'cyclohexanone', 'marlstone': 'marls tone',
                  'androneda': 'andromeda', 'iphone8': 'iPhone', 'acidtone': 'acid tone',
                  'noneconomically': 'non economically', 'Honeyfund': 'Honey fund', 'germanophone': 'Germanophobe',
                  'Democratizationed': 'Democratization ed', 'haoneymoon': 'honeymoon', 'iPhone7': 'iPhone 7',
                  'someonewith': 'some onewith', 'Hexanone': 'Hexa none', 'bonespur': 'bones pur',
                  'sisterzoned': 'sister zoned', 'HasAnyone': 'Has Anyone',
                  'stonepelters': 'stone pelters', 'Chronexia': 'Chronaxia', 'brotherzone': 'brother zone',
                  'brotherzoned': 'brother zoned', 'fonecare': 'f onecare', 'nonexsistence': 'nonexistence',
                  'conents': 'contents', 'phonecases': 'phone cases', 'Commissionerates': 'Commissioner ates',
                  'activemoney': 'active money', 'dingtone': 'ding tone', 'wheatestone': 'wheatstone',
                  'chiropractorone': 'chiropractor one', 'heeadphones': 'headphones', 'Maimonedes': 'Maimonides',
                  'onepiecedeals': 'onepiece deals', 'oneblade': 'one blade', 'venetioned': 'Venetianed',
                  'sunnyleone': 'sunny leone', 'prendisone': 'prednisone', 'Anglosaxophone': 'Anglo saxophone',
                  'Blackphones': 'Black phones', 'jionee': 'jinnee', 'chromonema': 'chromo nema',
                  'iodoketones': 'iodo ketones', 'demonetizations': 'demonetization', 'aomeone': 'someone',
                  'trillonere': 'trillones', 'abandonee': 'abandon',
                  'MasterColonel': 'Master Colonel', 'fronend': 'friend', 'Wildstone': 'Wilds tone',
                  'patitioned': 'petitioned', 'lonewolfs': 'lone wolfs', 'Spectrastone': 'Spectra stone',
                  'dishonerable': 'dishonorable', 'poisiones': 'poisons',
                  'condioner': 'conditioner', 'unpermissioned': 'unper missioned', 'friedzone': 'fried zone',
                  'umumoney': 'umu money', 'anyonestudied': 'anyone studied', 'dictioneries': 'dictionaries',
                  'nosebone': 'nose bone', 'ofVodafone': 'of Vodafone',
                  'Yumstone': 'Yum stone', 'oxandrolonesteroid': 'oxandrolone steroid',
                  'Mifeprostone': 'Mifepristone', 'pheramones': 'pheromones',
                  'sinophone': 'Sinophobe', 'peloponesian': 'peloponnesian', 'michrophone': 'microphone',
                  'commissionets': 'commissioners', 'methedone': 'methadone', 'cobditioners': 'conditioners',
                  'urotone': 'protone', 'smarthpone': 'smartphone', 'conecTU': 'connect you', 'beloney': 'boloney',
                  'comfortzone': 'comfort zone', 'testostersone': 'testosterone', 'camponente': 'component',
                  'Idonesia': 'Indonesia', 'dolostones': 'dolostone', 'psiphone': 'psi phone',
                  'ceftriazone': 'ceftriaxone', 'feelonely': 'feel onely', 'monetation': 'moderation',
                  'activationenergy': 'activation energy', 'moneydriven': 'money driven',
                  'staionery': 'stationery', 'zoneflex': 'zone flex', 'moneycash': 'money cash',
                  'conectiin': 'connection', 'Wannaone': 'Wanna one',
                  'Pictones': 'Pict ones', 'demonentization': 'demonetization',
                  'phenonenon': 'phenomenon', 'evenafter': 'even after', 'Sevenfriday': 'Seven friday',
                  'Devendale': 'Evendale', 'theeventchronicle': 'the event chronicle',
                  'seventysomething': 'seventy something', 'sevenpointed': 'seven pointed',
                  'richfeel': 'rich feel', 'overfeel': 'over feel', 'feelingstupid': 'feeling stupid',
                  'Photofeeler': 'Photo feeler', 'feelomgs': 'feelings', 'feelinfs': 'feelings',
                  'PlayerUnknown': 'Player Unknown', 'Playerunknown': 'Player unknown', 'knowlefge': 'knowledge',
                  'knowledgd': 'knowledge', 'knowledeg': 'knowledge', 'knowble': 'Knowle', 'Howknow': 'Howk now',
                  'knowledgeWoods': 'knowledge Woods', 'knownprogramming': 'known programming',
                  'selfknowledge': 'self knowledge', 'knowldage': 'knowledge', 'knowyouve': 'know youve',
                  'aknowlege': 'knowledge', 'Audetteknown': 'Audette known', 'knowlegdeable': 'knowledgeable',
                  'trueoutside': 'true outside', 'saynthesize': 'synthesize', 'EssayTyper': 'Essay Typer',
                  'meesaya': 'mee saya', 'Rasayanam': 'Rasayan am', 'fanessay': 'fan essay', 'momsays': 'moms ays',
                  'sayying': 'saying', 'saydaw': 'say daw', 'Fanessay': 'Fan essay', 'theyreally': 'they really',
                  'gayifying': 'gayed up with homosexual love', 'gayke': 'gay Online retailers',
                  'Lingayatism': 'Lingayat',
                  'macapugay': 'Macaulay', 'jewsplain': 'jews plain',
                  'banggood': 'bang good', 'goodfriends': 'good friends',
                  'goodfirms': 'good firms', 'Banggood': 'Bang good', 'dogooder': 'do gooder',
                  'stillshots': 'stills hots', 'stillsuits': 'still suits', 'panromantic': 'pan romantic',
                  'paracommando': 'para commando', 'romantize': 'romanize', 'manupulative': 'manipulative',
                  'manjha': 'mania', 'mankrit': 'mank rit',
                  'heteroromantic': 'hetero romantic', 'pulmanery': 'pulmonary', 'manpads': 'man pads',
                  'supermaneuverable': 'super maneuverable', 'mandatkry': 'mandatory', 'armanents': 'armaments',
                  'manipative': 'mancipative', 'himanity': 'humanity', 'maneuever': 'maneuver',
                  'Kumarmangalam': 'Kumar mangalam', 'Brahmanwadi': 'Brahman wadi',
                  'exserviceman': 'ex serviceman',
                  'managewp': 'managed', 'manies': 'many', 'recordermans': 'recorder mans',
                  'Feymann': 'Heymann', 'salemmango': 'salem mango', 'manufraturing': 'manufacturing',
                  'sreeman': 'freeman', 'tamanaa': 'Tamanac', 'chlamydomanas': 'chlamydomonas',
                  'comandant': 'commandant', 'huemanity': 'humanity', 'manaagerial': 'managerial',
                  'lithromantics': 'lith romantics',
                  'geramans': 'germans', 'Nagamandala': 'Naga mandala', 'humanitariarism': 'humanitarianism',
                  'wattman': 'watt man', 'salesmanago': 'salesman ago', 'Washwoman': 'Wash woman',
                  'rammandir': 'ram mandir', 'nomanclature': 'nomenclature', 'Haufman': 'Kaufman',
                  'prefomance': 'performance', 'ramanunjan': 'Ramanujan', 'Freemansonry': 'Freemasonry',
                  'supermaneuverability': 'super maneuverability', 'manstruate': 'menstruate',
                  'Tarumanagara': 'Taruma nagara', 'RomanceTale': 'Romance Tale', 'heteromantic': 'hete romantic',
                  'terimanals': 'terminals', 'womansplaining': 'wo mansplaining',
                  'performancelearning': 'performance learning', 'sociomantic': 'sciomantic',
                  'batmanvoice': 'batman voice', 'PerformanceTesting': 'Performance Testing',
                  'manorialism': 'manorial ism', 'newscommando': 'news commando',
                  'Entwicklungsroman': 'Entwicklungs roman',
                  'Kunstlerroman': 'Kunstler roman', 'bodhidharman': 'Bodhidharma', 'Howmaney': 'How maney',
                  'manufucturing': 'manufacturing', 'remmaning': 'remaining', 'rangeman': 'range man',
                  'mythomaniac': 'mythomania', 'katgmandu': 'katmandu',
                  'Superowoman': 'Superwoman', 'Rahmanland': 'Rahman land', 'Dormmanu': 'Dormant',
                  'Geftman': 'Gentman', 'manufacturig': 'manufacturing', 'bramanistic': 'Brahmanistic',
                  'padmanabhanagar': 'padmanabhan agar', 'homoromantic': 'homo romantic', 'femanists': 'feminists',
                  'demihuman': 'demi human', 'manrega': 'Manresa', 'Pasmanda': 'Pas manda',
                  'manufacctured': 'manufactured', 'remaninder': 'remainder', 'Marimanga': 'Mari manga',
                  'Sloatman': 'Sloat man', 'manlet': 'man let', 'perfoemance': 'performance',
                  'mangolian': 'mongolian', 'mangekyu': 'mange kyu', 'mansatory': 'mandatory',
                  'managemebt': 'management', 'manufctures': 'manufactures', 'Bramanical': 'Brahmanical',
                  'manaufacturing': 'manufacturing', 'Lakhsman': 'Lakhs man', 'Sarumans': 'Sarum ans',
                  'mangalasutra': 'mangalsutra', 'Germanised': 'German ised',
                  'managersworking': 'managers working', 'cammando': 'commando', 'mandrillaris': 'mandrill aris',
                  'Emmanvel': 'Emmarvel', 'manupalation': 'manipulation', 'welcomeromanian': 'welcome romanian',
                  'humanfemale': 'human female', 'mankirt': 'mankind', 'Haffmann': 'Hoffmann',
                  'Panromantic': 'Pan romantic', 'demantion': 'detention', 'Suparwoman': 'Superwoman',
                  'parasuramans': 'parasuram ans', 'sulmann': 'Suilmann', 'Shubman': 'Subman',
                  'manspread': 'man spread', 'mandingan': 'Mandingan', 'mandalikalu': 'mandalika lu',
                  'manufraturer': 'manufacturer', 'Wedgieman': 'Wedgie man', 'manwues': 'manages',
                  'humanzees': 'human zees', 'Steymann': 'Stedmann', 'Jobberman': 'Jobber man',
                  'maniquins': 'mani quins', 'biromantical': 'bi romantical', 'Rovman': 'Roman',
                  'pyromantic': 'pyro mantic', 'Tastaman': 'Rastaman', 'Spoolman': 'Spool man',
                  'Subramaniyan': 'Subramani yan', 'abhimana': 'abhiman a', 'manholding': 'man holding',
                  'seviceman': 'serviceman', 'womansplained': 'womans plained', 'manniya': 'mania',
                  'Bhraman': 'Braman', 'Laakman': 'Layman', 'mansturbate': 'masturbate',
                  'Sulamaniya': 'Sulamani ya', 'demanters': 'decanters', 'postmanare': 'postman are',
                  'mannualy': 'annual', 'rstman': 'Rotman', 'permanentjobs': 'permanent jobs',
                  'Allmang': 'All mang', 'TradeCommander': 'Trade Commander', 'BasedStickman': 'Based Stickman',
                  'Deshabhimani': 'Desha bhimani', 'manslamming': 'mans lamming', 'Brahmanwad': 'Brahman wad',
                  'fundemantally': 'fundamentally', 'supplemantary': 'supplementary', 'egomanias': 'ego manias',
                  'manvantar': 'Manvantara', 'spymania': 'spy mania', 'mangonada': 'mango nada',
                  'manthras': 'mantras', 'Humanpark': 'Human park', 'manhuas': 'mahuas',
                  'manterrupting': 'interrupting', 'dermatillomaniac': 'dermatillomania',
                  'performancies': 'performances', 'manipulant': 'manipulate',
                  'painterman': 'painter man', 'mangalik': 'manglik',
                  'Neurosemantics': 'Neuro semantics', 'discrimantion': 'discrimination',
                  'Womansplaining': 'feminist', 'mongodump': 'mongo dump', 'roadgods': 'road gods',
                  'Oligodendraglioma': 'Oligodendroglioma', 'unrightly': 'un rightly', 'Janewright': 'Jane wright',
                  ' righten ': ' tighten ', 'brightiest': 'brightest',
                  'frighter': 'fighter', 'righteouness': 'righteousness', 'triangleright': 'triangle right',
                  'Brightspace': 'Brights pace', 'techinacal': 'technical', 'chinawares': 'china wares',
                  'Vancouever': 'Vancouver', 'cheverlet': 'cheveret', 'deverstion': 'diversion',
                  'everbodys': 'everybody', 'Dramafever': 'Drama fever', 'reverificaton': 'reverification',
                  'canterlever': 'canter lever', 'keywordseverywhere': 'keywords everywhere',
                  'neverunlearned': 'never unlearned', 'everyfirst': 'every first',
                  'neverhteless': 'nevertheless', 'clevercoyote': 'clever coyote', 'irrevershible': 'irreversible',
                  'achievership': 'achievers hip', 'easedeverything': 'eased everything', 'youbever': 'you bever',
                  'everperson': 'ever person', 'everydsy': 'everyday', 'whemever': 'whenever',
                  'everyonr': 'everyone', 'severiity': 'severity', 'narracist': 'nar racist',
                  'racistly': 'racist', 'takesuch': 'take such', 'mystakenly': 'mistakenly',
                  'shouldntake': 'shouldnt take', 'Kalitake': 'Kali take', 'msitake': 'mistake',
                  'straitstimes': 'straits times', 'timefram': 'timeframe', 'watchtime': 'watch time',
                  'timetraveling': 'timet raveling', 'peactime': 'peacetime', 'timetabe': 'timetable',
                  'cooktime': 'cook time', 'blocktime': 'block time', 'timesjobs': 'times jobs',
                  'timesence': 'times ence', 'Touchtime': 'Touch time', 'timeloop': 'time loop',
                  'subcentimeter': 'sub centimeter', 'timejobs': 'time jobs', 'Guardtime': 'Guard time',
                  'realtimepolitics': 'realtime politics', 'loadingtimes': 'loading times',
                  'timesnow': '24-hour English news channel in India', 'timesspark': 'times spark',
                  'timetravelling': 'timet ravelling',
                  'antimeter': 'anti meter', 'timewaste': 'time waste', 'cryptochristians': 'crypto christians',
                  'Whatcould': 'What could', 'becomesdouble': 'becomes double', 'deathbecomes': 'death becomes',
                  'youbecome': 'you become', 'greenseer': 'people who possess the magical ability',
                  'rseearch': 'research', 'homeseek': 'home seek',
                  'Greenseer': 'people who possess the magical ability', 'starseeders': 'star seeders',
                  'seekingmillionaire': 'seeking millionaire', 'see\u202c': 'see',
                  'seeies': 'series', 'CodeAgon': 'Code Agon',
                  'royago': 'royal', 'Dragonkeeper': 'Dragon keeper', 'mcgreggor': 'McGregor',
                  'catrgory': 'category', 'Dragonknight': 'Dragon knight', 'Antergos': 'Anteros',
                  'togofogo': 'togo fogo', 'mongorestore': 'mongo restore', 'gorgops': 'gorgons',
                  'withgoogle': 'with google', 'goundar': 'Gondar', 'algorthmic': 'algorithmic',
                  'goatnuts': 'goat nuts', 'vitilgo': 'vitiligo', 'polygony': 'poly gony',
                  'digonals': 'diagonals', 'Luxemgourg': 'Luxembourg', 'UCSanDiego': 'UC SanDiego',
                  'Ringostat': 'Ringo stat', 'takingoff': 'taking off', 'MongoImport': 'Mongo Import',
                  'alggorithms': 'algorithms', 'dragonknight': 'dragon knight', 'negotiatior': 'negotiation',
                  'gomovies': 'go movies', 'Withgott': 'Without',
                  'categoried': 'categories', 'Stocklogos': 'Stock logos', 'Pedogogical': 'Pedological',
                  'Wedugo': 'Wedge', 'golddig': 'gold dig', 'goldengroup': 'golden group',
                  'merrigo': 'merligo', 'googlemapsAPI': 'googlemaps API', 'goldmedal': 'gold medal',
                  'golemized': 'polemized', 'Caligornia': 'California', 'unergonomic': 'un ergonomic',
                  'fAegon': 'wagon', 'vertigos': 'vertigo s', 'trigonomatry': 'trigonometry',
                  'hypogonadic': 'hypogonadia', 'Mogolia': 'Mongolia', 'governmaent': 'government',
                  'ergotherapy': 'ergo therapy', 'Bogosort': 'Bogo sort', 'goalwise': 'goal wise',
                  'alogorithms': 'algorithms', 'MercadoPago': 'Mercado Pago', 'rivigo': 'rivi go',
                  'govshutdown': 'gov shutdown', 'gorlfriend': 'girlfriend',
                  'stategovt': 'state govt', 'Chickengonia': 'Chicken gonia', 'Yegorovich': 'Yegorov ich',
                  'regognitions': 'recognitions', 'gorichen': 'Gori Chen Mountain',
                  'goegraphies': 'geographies', 'gothras': 'goth ras', 'belagola': 'bela gola',
                  'snapragon': 'snapdragon', 'oogonial': 'oogonia l', 'Amigofoods': 'Amigo foods',
                  'Sigorn': 'son of Styr', 'algorithimic': 'algorithmic',
                  'innermongolians': 'inner mongolians', 'ArangoDB': 'Arango DB', 'zigolo': 'gigolo',
                  'regognized': 'recognized', 'Moongot': 'Moong ot', 'goldquest': 'gold quest',
                  'catagorey': 'category', 'got7': 'got', 'jetbingo': 'jet bingo', 'Dragonchain': 'Dragon chain',
                  'catwgorized': 'categorized', 'gogoro': 'gogo ro', 'Tobagoans': 'Tobago ans',
                  'digonal': 'di gonal', 'algoritmic': 'algorismic', 'dragonflag': 'dragon flag',
                  'Indigoflight': 'Indigo flight',
                  'governening': 'governing', 'ergosphere': 'ergo sphere',
                  'pingo5': 'pingo', 'Montogo': 'montego', 'Rivigo': 'technology-enabled logistics company',
                  'Jigolo': 'Gigolo', 'phythagoras': 'pythagoras', 'Mangolian': 'Mongolian',
                  'forgottenfaster': 'forgotten faster', 'stargold': 'a Hindi movie channel',
                  'googolplexain': 'googolplexian', 'corpgov': 'corp gov',
                  'govtribe': 'provides real-time federal contracting market intel',
                  'dragonglass': 'dragon glass', 'gorakpur': 'Gorakhpur', 'MangoPay': 'Mango Pay',
                  'chigoe': 'sub-tropical climates', 'BingoBox': 'an investment company', '走go': 'go',
                  'followingorder': 'following order', 'pangolinminer': 'pangolin miner',
                  'negosiation': 'negotiation', 'lexigographers': 'lexicographers', 'algorithom': 'algorithm',
                  'unforgottable': 'unforgettable', 'wellsfargoemail': 'wellsfargo email',
                  'daigonal': 'diagonal', 'Pangoro': 'cantankerous Pokemon', 'negotiotions': 'negotiations',
                  'Swissgolden': 'Swiss golden', 'google4': 'google', 'Agoraki': 'Ago raki',
                  'Garthago': 'Carthago', 'Stegosauri': 'stegosaurus', 'ergophobia': 'ergo phobia',
                  'bigolive': 'big olive', 'bittergoat': 'bitter goat', 'naggots': 'faggots',
                  'googology': 'online encyclopedia', 'algortihms': 'algorithms', 'bengolis': 'Bengalis',
                  'fingols': 'Finnish people are supposedly descended from Mongols',
                  'savethechildren': 'save thechildren',
                  'stopings': 'stoping', 'stopsits': 'stop sits', 'stopsigns': 'stop signs',
                  'Galastop': 'Galas top', 'pokestops': 'pokes tops', 'forcestop': 'forces top',
                  'Hopstop': 'Hops top', 'stoppingexercises': 'stopping exercises', 'coinstop': 'coins top',
                  'stoppef': 'stopped', 'workaway': 'work away', 'snazzyway': 'snazzy way',
                  'Rewardingways': 'Rewarding ways', 'cloudways': 'cloud ways', 'Cloudways': 'Cloud ways',
                  'Brainsway': 'Brains way', 'nesraway': 'nearaway',
                  'AlwaysHired': 'Always Hired', 'expessway': 'expressway', 'Syncway': 'Sync way',
                  'LeewayHertz': 'Blockchain Company', 'towayrds': 'towards', 'swayable': 'sway able',
                  'Telloway': 'Tello way', 'palsmodium': 'plasmodium', 'Gobackmodi': 'Goback modi',
                  'comodies': 'corodies', 'islamphobic': 'islam phobic', 'islamphobia': 'islam phobia',
                  'citiesbetter': 'cities better', 'betterv3': 'better', 'betterDtu': 'better Dtu',
                  'Babadook': 'a horror drama film', 'Ahemadabad': 'Ahmadabad', 'faidabad': 'Faizabad',
                  'Amedabad': 'Ahmedabad', 'kabadii': 'kabaddi', 'badmothing': 'badmouthing',
                  'badminaton': 'badminton', 'badtameezdil': 'badtameez dil', 'badeffects': 'bad effects',
                  '∠bad': 'bad', 'ahemadabad': 'Ahmadabad', 'embaded': 'embased', 'Isdhanbad': 'Is dhanbad',
                  'badgermoles': 'enormous, blind mammal', 'allhabad': 'Allahabad', 'ghazibad': 'ghazi bad',
                  'htderabad': 'Hyderabad', 'Auragabad': 'Aurangabad', 'ahmedbad': 'Ahmedabad',
                  'ahmdabad': 'Ahmadabad', 'alahabad': 'Allahabad',
                  'Hydeabad': 'Hyderabad', 'Gyroglove': 'wearable technology', 'foodlovee': 'food lovee',
                  'slovenised': 'slovenia', 'handgloves': 'hand gloves', 'lovestep': 'love step',
                  'lovejihad': 'love jihad', 'RolloverBox': 'Rollover Box', 'stupidedt': 'stupidest',
                  'toostupid': 'too stupid',
                  'pakistanisbeautiful': 'pakistanis beautiful', 'ispakistan': 'is pakistan',
                  'inpersonations': 'impersonations', 'medicalperson': 'medical person',
                  'interpersonation': 'inter personation', 'workperson': 'work person',
                  'personlich': 'person lich', 'persoenlich': 'person lich',
                  'middleperson': 'middle person', 'personslized': 'personalized',
                  'personifaction': 'personification', 'welcomemarriage': 'welcome marriage',
                  'come2': 'come to', 'upcomedians': 'up comedians', 'overvcome': 'overcome',
                  'talecome': 'tale come', 'cometitive': 'competitive', 'arencome': 'aren come',
                  'achecomes': 'ache comes', '」come': 'come',
                  'comepleted': 'completed', 'overcomeanxieties': 'overcome anxieties',
                  'demigirl': 'demi girl', 'gridgirl': 'female models of the race', 'halfgirlfriend': 'half girlfriend',
                  'girlriend': 'girlfriend', 'fitgirl': 'fit girl', 'girlfrnd': 'girlfriend', 'awrong': 'aw rong',
                  'northcap': 'north cap', 'productionsupport': 'production support',
                  'Designbold': 'Online Photo Editor Design Studio',
                  'skyhold': 'sky hold', 'shuoldnt': 'shouldnt', 'anarold': 'Android', 'yaerold': 'year old',
                  'soldiders': 'soldiers', 'indrold': 'Android', 'blindfoldedly': 'blindfolded',
                  'overcold': 'over cold', 'Goldmont': 'microarchitecture in Intel', 'boldspot': 'bolds pot',
                  'Rankholders': 'Rank holders', 'cooldrink': 'cool drink', 'beltholders': 'belt holders',
                  'GoldenDict': 'open-source dictionary program', 'softskill': 'softs kill',
                  'Cooldige': 'the 30th president of the United States',
                  'newkiller': 'new killer', 'skillselect': 'skills elect', 'nonskilled': 'non skilled',
                  'killyou': 'kill you', 'Skillport': 'Army e-Learning Program', 'unkilled': 'un killed',
                  'killikng': 'killing', 'killograms': 'kilograms',
                  'Worldkillers': 'World killers', 'reskilled': 'skilled',
                  'killedshivaji': 'killed shivaji', 'honorkillings': 'honor killings',
                  'skillclasses': 'skill classes', 'microskills': 'micros kills',
                  'Skillselect': 'Skills elect', 'ratkill': 'rat kill',
                  'pleasegive': 'please give', 'flashgive': 'flash give',
                  'southerntelescope': 'southern telescope', 'westsouth': 'west south',
                  'southAfricans': 'south Africans', 'Joboutlooks': 'Job outlooks', 'joboutlook': 'job outlook',
                  'Outlook365': 'Outlook 365', 'Neulife': 'Neu life', 'qualifeid': 'qualified',
                  'nullifed': 'nullified', 'lifeaffect': 'life affect', 'lifestly': 'lifestyle',
                  'aristocracylifestyle': 'aristocracy lifestyle', 'antilife': 'anti life',
                  'afterafterlife': 'after afterlife', 'lifestylye': 'lifestyle', 'prelife': 'pre life',
                  'lifeute': 'life ute', 'liferature': 'literature',
                  'securedlife': 'secured life', 'doublelife': 'double life', 'antireligion': 'anti religion',
                  'coreligionist': 'co religionist', 'petrostates': 'petro states', 'otherstates': 'others tates',
                  'spacewithout': 'space without', 'withoutyou': 'without you',
                  'withoutregistered': 'without registered', 'weightwithout': 'weight without',
                  'withoutcheck': 'without check', 'milkwithout': 'milk without',
                  'Highschoold': 'High school', 'memoney': 'money', 'moneyof': 'mony of', 'Oneplus': 'OnePlus',
                  'OnePlus': 'Chinese smartphone manufacturer', 'Beerus': 'the God of Destruction',
                  'takeoverr': 'takeover', 'demonetizedd': 'demonetized', 'polyhouse': 'Polytunnel',
                  'Elitmus': 'eLitmus', 'eLitmus': 'Indian company that helps companies in hiring employees',
                  'becone': 'become', 'nestaway': 'nest away', 'takeoverrs': 'takeovers', 'Istop': 'I stop',
                  'Austira': 'Australia', 'germeny': 'Germany', 'mansoon': 'man soon',
                  'worldmax': 'wholesaler of drum parts',
                  'ammusement': 'amusement', 'manyare': 'many are', 'supplymentary': 'supply mentary',
                  'timesup': 'times up', 'homologus': 'homologous', 'uimovement': 'ui movement', 'spause': 'spouse',
                  'aesexual': 'asexual', 'Iovercome': 'I overcome', 'developmeny': 'development',
                  'hindusm': 'hinduism', 'sexpat': 'sex tourism', 'sunstop': 'sun stop', 'polyhouses': 'Polytunnel',
                  'usefl': 'useful', 'Fundamantal': 'fundamental', 'environmentai': 'environmental',
                  'Redmi': 'Xiaomi Mobile', 'Loy Machedo': ' Motivational Speaker ', 'unacademy': 'Unacademy',
                  'Boruto': 'Naruto Next Generations', 'Upwork': 'Up work',
                  'Unacademy': 'educational technology company',
                  'HackerRank': 'Hacker Rank', 'upwork': 'up work', 'Chromecast': 'Chrome cast',
                  'microservices': 'micro services', 'Undertale': 'video game', 'undergraduation': 'under graduation',
                  'chapterwise': 'chapter wise', 'twinflame': 'twin flame', 'Hotstar': 'Hot star',
                  'blockchains': 'blockchain',
                  'darkweb': 'dark web', 'Microservices': 'Micro services', 'Nearbuy': 'Nearby',
                  ' Padmaavat ': ' Padmavati ', ' padmavat ': ' Padmavati ', ' Padmaavati ': ' Padmavati ',
                  ' Padmavat ': ' Padmavati ', ' internshala ': ' internship and online training platform in India ',
                  'dream11': ' fantasy sports platform in India ', 'conciousnesss': 'consciousnesses',
                  'Dream11': ' fantasy sports platform in India ', 'cointry': 'country', ' coinvest ': ' invest ',
                  '23 andme': 'privately held personal genomics and biotechnology company in California',
                  'Trumpism': 'philosophy and politics espoused by Donald Trump',
                  'Trumpian': 'viewpoints of President Donald Trump', 'Trumpists': 'admirer of Donald Trump',
                  'coincidents': 'coincidence', 'coinsized': 'coin sized', 'coincedences': 'coincidences',
                  'cointries': 'countries', 'coinsidered': 'considered', 'coinfirm': 'confirm',
                  'humilates':'humiliates', 'vicevice':'vice vice', 'politicak':'political', 'Sumaterans':'Sumatrans',
                  'Kamikazis':'Kamikazes', 'unmoraled':'unmoral', 'eduacated':'educated', 'moraled':'morale',
                  'Amharc':'Amarc', 'where Burkhas':'wear Burqas', 'Baloochistan':'Balochistan', 'durgahs':'durgans',
                  'illigitmate':'illegitimate', 'hillum':'helium','treatens':'threatens','mutiliating':'mutilating',
                  'speakingly':'speaking', 'pretex':'pretext', 'menstruateion':'menstruation', 
                  'genocidizing':'genociding', 'maratis':'Maratism','Parkistinian':'Pakistani', 'SPEICIAL':'SPECIAL',
                  'REFERNECE':'REFERENCE', 'provocates':'provokes', 'FAMINAZIS':'FEMINAZIS', 'repugicans':'republicans',
                  'tonogenesis':'tone', 'winor':'win', 'redicules':'ridiculous', 'Beluchistan':'Balochistan', 
                  'volime':'volume', 'namaj':'namaz', 'CONgressi':'Congress', 'Ashifa':'Asifa', 'queffing':'queefing',
                  'montheistic':'nontheistic', 'Rajsthan':'Rajasthan', 'Rajsthanis':'Rajasthanis', 'specrum':'spectrum',
                  'brophytes':'bryophytes', 'adhaar':'Adhara', 'slogun':'slogan', 'harassd':'harassed',
                  'transness':'trans gender', 'Insdians':'Indians', 'Trampaphobia':'Trump aphobia', 'attrected':'attracted',
                  'Yahtzees':'Yahtzee', 'thiests':'atheists', 'thrir':'their', 'extraterestrial':'extraterrestrial',
                  'silghtest':'slightest', 'primarty':'primary','brlieve':'believe', 'fondels':'fondles',
                  'loundly':'loudly', 'bootythongs':'booty thongs', 'understamding':'understanding', 'degenarate':'degenerate',
                  'narsistic':'narcistic', 'innerskin':'inner skin','spectulated':'speculated', 'hippocratical':'Hippocratical',
                  'itstead':'instead', 'parralels':'parallels', 'sloppers':'slippers'
                  }

def clean_bad_case_words(text):
    for bad_word in bad_case_words:
        if bad_word in text:
            text = text.replace(bad_word, bad_case_words[bad_word])
    return text


mis_connect_list = ['(W|w)hat', '(W|w)hy', '(H|h)ow', '(W|w)hich', '(W|w)here', '(W|w)ill']
mis_connect_re = re.compile('(%s)' % '|'.join(mis_connect_list))

mis_spell_mapping = {'whattsup': 'WhatsApp', 'whatasapp':'WhatsApp', 'whatsupp':'WhatsApp', 
                      'whatcus':'what cause', 'arewhatsapp': 'are WhatsApp', 'Hwhat':'what',
                      'Whwhat': 'What', 'whatshapp':'WhatsApp', 'howhat':'how that',
                      # why
                      'Whybis':'Why is', 'laowhy86':'Foreigners who do not respect China',
                      'Whyco-education':'Why co-education',
                      # How
                      "Howddo":"How do", 'Howeber':'However', 'Showh':'Show',
                      "Willowmagic":'Willow magic', 'WillsEye':'Will Eye', 'Williby':'will by'}
def spacing_some_connect_words(text):
    """
    'Whyare' -> 'Why are'
    """
    ori = text
    for error in mis_spell_mapping:
        if error in text:
            text = text.replace(error, mis_spell_mapping[error])
            
    # what
    text = re.sub(r" (W|w)hat+(s)*[A|a]*(p)+ ", " WhatsApp ", text)
    text = re.sub(r" (W|w)hat\S ", " What ", text)
    text = re.sub(r" \S(W|w)hat ", " What ", text)
    # why
    text = re.sub(r" (W|w)hy\S ", " Why ", text)
    text = re.sub(r" \S(W|w)hy ", " Why ", text)
    # How
    text = re.sub(r" (H|h)ow\S ", " How ", text)
    text = re.sub(r" \S(H|h)ow ", " How ", text)
    # which
    text = re.sub(r" (W|w)hich\S ", " Which ", text)
    text = re.sub(r" \S(W|w)hich ", " Which ", text)
    # where
    text = re.sub(r" (W|w)here\S ", " Where ", text)
    text = re.sub(r" \S(W|w)here ", " Where ", text)
    # 
    text = mis_connect_re.sub(r" \1 ", text)
    text = text.replace("What sApp", 'WhatsApp')
    
    text = remove_space(text)
    return text

# clean repeated letters
def clean_repeat_words(text):
    text = text.replace("img", "ing")

    text = re.sub(r"(I|i)(I|i)+ng", "ing", text)
    text = re.sub(r"(L|l)(L|l)(L|l)+y", "lly", text)
    text = re.sub(r"(A|a)(A|a)(A|a)+", "a", text)
    text = re.sub(r"(C|c)(C|c)(C|c)+", "cc", text)
    text = re.sub(r"(D|d)(D|d)(D|d)+", "dd", text)
    text = re.sub(r"(E|e)(E|e)(E|e)+", "ee", text)
    text = re.sub(r"(F|f)(F|f)(F|f)+", "ff", text)
    text = re.sub(r"(G|g)(G|g)(G|g)+", "gg", text)
    text = re.sub(r"(I|i)(I|i)(I|i)+", "i", text)
    text = re.sub(r"(K|k)(K|k)(K|k)+", "k", text)
    text = re.sub(r"(L|l)(L|l)(L|l)+", "ll", text)
    text = re.sub(r"(M|m)(M|m)(M|m)+", "mm", text)
    text = re.sub(r"(N|n)(N|n)(N|n)+", "nn", text)
    text = re.sub(r"(O|o)(O|o)(O|o)+", "oo", text)
    text = re.sub(r"(P|p)(P|p)(P|p)+", "pp", text)
    text = re.sub(r"(Q|q)(Q|q)+", "q", text)
    text = re.sub(r"(R|r)(R|r)(R|r)+", "rr", text)
    text = re.sub(r"(S|s)(S|s)(S|s)+", "ss", text)
    text = re.sub(r"(T|t)(T|t)(T|t)+", "tt", text)
    text = re.sub(r"(V|v)(V|v)+", "v", text)
    text = re.sub(r"(Y|y)(Y|y)(Y|y)+", "y", text)
    text = re.sub(r"plzz+", "please", text)
    text = re.sub(r"(Z|z)(Z|z)(Z|z)+", "zz", text)
    return text

def preprocess(text):
    """
    preprocess text main steps
    """
    text = remove_space(text)
    text = clean_special_punctuations(text)
    text = clean_number(text)
    text = pre_clean_rare_words(text)
    text = decontracted(text)
    text = clean_latex(text)
    text = clean_misspell(text)
    text = spacing_punctuation(text)
    text = spacing_some_connect_words(text)
    text = clean_bad_case_words(text)
    text = clean_repeat_words(text)
    text = remove_space(text)
    return text

def text_clean_wrapper(df):
    df["question_text"] = df["question_text"].apply(preprocess)
    return df


'''''''''''''''
After cleaning the text the word coverage increased dramaticly.

Embedding 	                   Original 	Text Cleaning
Glove vocab founded 	         33.92% 	72.59%
Glove vocab founded in text 	 88.20% 	99.44%
Paragram vocab founded 	         34.08% 	72.87%
Paragram vocab founded in text 	 88.21% 	99.45%
FastText vocab founded 	         31.63% 	68.32%
FastText vocab founded in text 	 87.74% 	99.38%
Google vocab founded 	         26.24% 	56.71%
Google vocab founded in text 	 87.26% 	88.01%
'''''''''''''''


'''
Finally let's fixed the dash and point spacing bug!

    spacing puncs except for '-'
    get current vocabulary, and found the words that has '-'
    check word that '-' in it whether has embedding
    spacing '-'
    fix these bug '-' words
'''

# get current vocabulary, and found the words that has '-'
cur_vocabulary = set()
for text in tqdm(train_df['question_text'].values.tolist() + test_df['question_text'].values.tolist()):
    words = text.split(' ')
    cur_vocabulary.update(set(words))

bug_punc_spacing_words_mapping = {}
for vocab in cur_vocabulary:
    if '-' in vocab:
        # whether the glove or para contain this word
        if (vocab in embed_glove or vocab.capitalize() in embed_glove or vocab.lower() in embed_glove) and \
            (vocab in embed_paragram or vocab.lower() in embed_paragram):
            bug_punc_spacing_words_mapping[f" {' - '.join(vocab.split('-'))} "] = f" {vocab} "
    
    if '.' in vocab:
        if vocab.endswith('.'):
            continue
        
        if (vocab in embed_glove or vocab.capitalize() in embed_glove or vocab.lower() in embed_glove) and \
            (vocab in embed_paragram or vocab.lower() in embed_paragram):
            bug_punc_spacing_words_mapping[f" {' . '.join(vocab.split('.'))} "] = f" {vocab} "
                                    
del bug_punc_spacing_words_mapping['  -  ']
print(f'found {len(bug_punc_spacing_words_mapping)} bug words')


'''
100%|██████████| 1362492/1362492 [00:05<00:00, 250363.15it/s]

found 18128 bug words
'''

bug_punc_spacing_words_mapping

'''
' acetyl - CoA ': ' acetyl-CoA ',
 ' 0 - 2 ': ' 0-2 ',
 ' 2 . 165 ': ' 2.165 ',
 ' micro - lending ': ' micro-lending ',
 ' ex - manager ': ' ex-manager ',
 ' 25 - 20 ': ' 25-20 ',
 ' Red - Nosed ': ' Red-Nosed ',
 ' anti - socialist ': ' anti-socialist ',
 ' bang - for - buck ': ' bang-for-buck ',
 ' fast - forward ': ' fast-forward ',
 ' non - ADD ': ' non-ADD ',
 ' post - study ': ' post-study ',
 ' escape - proof ': ' escape-proof ',
 ' hollow - point ': ' hollow-point ',
 ' Career - wise ': ' Career-wise ',
 ' grad - school ': ' grad-school ',
 ' meta - models ': ' meta-models ',
 ' self - expression ': ' self-expression ',
 ' low - ish ': ' low-ish ',
 ' pre - algebra ': ' pre-algebra ',
 ' trick - or - treater ': ' trick-or-treater ',
 ' complementarity - determining ': ' complementarity-determining ',
 ' back - test ': ' back-test ',
 ' old - line ': ' old-line ',
 ' Pre - teen ': ' Pre-teen ',
 ' ex - roommate ': ' ex-roommate ',
 ' bright - colored ': ' bright-colored ',
 ' under - exposed ': ' under-exposed ',
 ' job - seeking ': ' job-seeking ',
 ' call - off ': ' call-off ',
 ' 70 - year - old ': ' 70-year-old ',
 ' non - accredited ': ' non-accredited ',
 ' off - spinner ': ' off-spinner ',
 ' film - making ': ' film-making ',
 ' white - hat ': ' white-hat ',
 ' full - tuition ': ' full-tuition ',
 ' corporate - controlled ': ' corporate-controlled ',
 ' 17 - month - old ': ' 17-month-old ',
 ' commander - in - chief ': ' commander-in-chief ',
 ' 4 . 05 ': ' 4.05 ',
 ' computer - animated ': ' computer-animated ',
 ' semi - advanced ': ' semi-advanced ',
 ' 55 - year - old ': ' 55-year-old ',
 ' 39 . 4 ': ' 39.4 ',
 ' Pro - Israeli ': ' Pro-Israeli ',
 ' mother - child ': ' mother-child ',
 ' non - judgmental ': ' non-judgmental ',
 ' 79 . 25 ': ' 79.25 ',
 ' Math . Pow ': ' Math.Pow ',
 ' 0 . 225 ': ' 0.225 ',
 ' B . C ': ' B.C ',
 ' Jr . NTR ': ' Jr.NTR ',
 ' mothers - in - law ': ' mothers-in-law ',
 ' email - id ': ' email-id ',
 ' wild - fire ': ' wild-fire ',
 ' flim - flam ': ' flim-flam ',
 ' Indo - British ': ' Indo-British ',
 ' WordPress . org ': ' WordPress.org ',
 ' world - class ': ' world-class ',
 ' pin - points ': ' pin-points ',
 ' zig - zagging ': ' zig-zagging ',
 ' U - boat ': ' U-boat ',
 ' Co - Founder ': ' Co-Founder ',
 ' Lend - Lease ': ' Lend-Lease ',
 ' 1 . 24 ': ' 1.24 ',
 ' 13 . 70 ': ' 13.70 ',
 ' bare - minimum ': ' bare-minimum ',
 ' Mig - 21 ': ' Mig-21 ',
 ' stop - and - frisk ': ' stop-and-frisk ',
 ' SCAR - H ': ' SCAR-H ',
 '  - 4.50 ': ' -4.50 ',
 ' -4 . 50 ': ' -4.50 ',
 ' ex - Manchester ': ' ex-Manchester ',
 ' 1933 - 34 ': ' 1933-34 ',
 ' non - humans ': ' non-humans ',
 ' self - centred ': ' self-centred ',
 ' 9 - 6 ': ' 9-6 ',
 ' English - taught ': ' English-taught ',
 ' Plus - sized ': ' Plus-sized ',
 ' Dev - C ': ' Dev-C ',
 ' Latin - derived ': ' Latin-derived ',
 ' 37 - year - old ': ' 37-year-old ',
 ' suspense - thriller ': ' suspense-thriller ',
 ' BB - gun ': ' BB-gun ',
 ' V - 0 ': ' V-0 ',
 ' hard - hitting ': ' hard-hitting ',
 ' clinic . com ': ' clinic.com ',
 ' 60 . 27 ': ' 60.27 ',
 ' ninety - one ': ' ninety-one ',
 ' one - liners ': ' one-liners ',
 ' 9 . 26 ': ' 9.26 ',
 ' sec - 1 ': ' sec-1 ',
 ' non - Catholics ': ' non-Catholics ',
 ' D2 - like ': ' D2-like ',
 ' 59 . 5 ': ' 59.5 ',
 '  - 1000 ': ' -1000 ',
 ' re - exam ': ' re-exam ',
 ' lead - time ': ' lead-time ',
 ' U - Turn ': ' U-Turn ',
 ' Yahoo . com ': ' Yahoo.com ',
 ' well - set ': ' well-set ',
 ' 6 . 93 ': ' 6.93 ',
 ' Spanish - American ': ' Spanish-American ',
 ' 122 . 3 ': ' 122.3 ',
 ' hair - line ': ' hair-line ',
 ' sub - team ': ' sub-team ',
 ' reasonably - priced ': ' reasonably-priced ',
 ' 30 - year - old ': ' 30-year-old ',
 ' Extreme - Left ': ' Extreme-Left ',
 ' Smell - O - Vision ': ' Smell-O-Vision ',
 ' pre - civil ': ' pre-civil ',
 ' ON - Page ': ' ON-Page ',
 ' anti - pollution ': ' anti-pollution ',
 ' KC - 130 ': ' KC-130 ',
 ' 2 - micron ': ' 2-micron ',
 ' father - figure ': ' father-figure ',
 ' Socket . io ': ' Socket.io ',
 ' tax - cut ': ' tax-cut ',
 ' non - automated ': ' non-automated ',
 ' on - the - fly ': ' on-the-fly ',
 ' sex - role ': ' sex-role ',
 ' mid - 1970 ': ' mid-1970 ',
 ' land - owning ': ' land-owning ',
 ' lock - down ': ' lock-down ',
 ' Anti - Black ': ' Anti-Black ',
 ' medium - range ': ' medium-range ',
 ' Pre - Medical ': ' Pre-Medical ',
 ' a - 3 ': ' a-3 ',
 ' search - driven ': ' search-driven ',
 ' post - baccalaureate ': ' post-baccalaureate ',
 ' ex - post ': ' ex-post ',
 ' lesser - known ': ' lesser-known ',
 ' Indo - China ': ' Indo-China ',
 ' full - screen ': ' full-screen ',
 ' scholar - officials ': ' scholar-officials ',
 ' Jong - nam ': ' Jong-nam ',
 ' Auto - lock ': ' Auto-lock ',
 ' electric - type ': ' electric-type ',
 ' Star - Lord ': ' Star-Lord ',
 ' non - retirement ': ' non-retirement ',
 ' Happy - Ending ': ' Happy-Ending ',
 ' Sixty - four ': ' Sixty-four ',
 ' 34 . 5 ': ' 34.5 ',
 ' delta - sigma ': ' delta-sigma ',
 ' class - action ': ' class-action ',
 ' 1910 - 1945 ': ' 1910-1945 ',
 ' 13 . 80 ': ' 13.80 ',
 ' next - gen ': ' next-gen ',
 ' Copy - righted ': ' Copy-righted ',
 ' 2 - yr ': ' 2-yr ',
 ' commission - based ': ' commission-based ',
 ' 10 . 3 . 1 ': ' 10.3.1 ',
 ' non - sweet ': ' non-sweet ',
 ' Post - Gazette ': ' Post-Gazette ',
 ' 0 . 76 ': ' 0.76 ',
 ' up - coming ': ' up-coming ',
 ' 66 . 2 ': ' 66.2 ',
 ' non - existing ': ' non-existing ',
 ' 50 - 55 ': ' 50-55 ',
 ' kick - offs ': ' kick-offs ',
 ' non - malignant ': ' non-malignant ',
 ' multi - millionaire ': ' multi-millionaire ',
 ' non - transparent ': ' non-transparent ',
 ' re - enacted ': ' re-enacted ',
 ' 5 - character ': ' 5-character ',
 ' past - year ': ' past-year ',
 ' 3 - manifold ': ' 3-manifold ',
 '  - 4.4 ': ' -4.4 ',
 ' -4 . 4 ': ' -4.4 ',
 ' live - stream ': ' live-stream ',
 ' r - square ': ' r-square ',
 ' hardware - accelerated ': ' hardware-accelerated ',
 ' high - performing ': ' high-performing ',
 ' native - born ': ' native-born ',
 ' one - meter ': ' one-meter ',
 ' 9 . 6 ': ' 9.6 ',
 ' 3 - Color ': ' 3-Color ',
 ' Petropavlovsk - Kamchatsky ': ' Petropavlovsk-Kamchatsky ',
 ' non - geometric ': ' non-geometric ',
 ' 89 . 5 ': ' 89.5 ',
 ' father - and - son ': ' father-and-son ',
 ' Filipino - Canadian ': ' Filipino-Canadian ',
 ' multi - format ': ' multi-format ',
 ' fool . com ': ' fool.com ',
 ' co - sleeping ': ' co-sleeping ',
 ' non - theism ': ' non-theism ',
 ' A - 10 ': ' A-10 ',
 ' higher - ups ': ' higher-ups ',
 ' 5 . 50 ': ' 5.50 ',
 ' Console . WriteLine ': ' Console.WriteLine ',
 ' 120 - 240 ': ' 120-240 ',
 ' u - boats ': ' u-boats ',
 ' owned - and - operated ': ' owned-and-operated ',
 ' 0 . 879 ': ' 0.879 ',
 ' 80 - year - old ': ' 80-year-old ',
 ' 300 - 1000 ': ' 300-1000 ',
 ' most - viewed ': ' most-viewed ',
 ' 14.1 - inch ': ' 14.1-inch ',
 ' 14 . 1-inch ': ' 14.1-inch ',
 ' landlord - tenant ': ' landlord-tenant ',
 ' 2 . 86 ': ' 2.86 ',
 ' pro - Jewish ': ' pro-Jewish ',
 ' C - atoms ': ' C-atoms ',
 ' free - standing ': ' free-standing ',
 ' 22 . 8 ': ' 22.8 ',
 ' zip - off ': ' zip-off ',
 ' HIV - RNA ': ' HIV-RNA ',
 ' ex - spouses ': ' ex-spouses ',
 ' No - Limit ': ' No-Limit ',
 ' 3 . 41 ': ' 3.41 ',
 ' flat - Earthers ': ' flat-Earthers ',
 ' better - and ': ' better-and ',
 ' well - know ': ' well-know ',
 ' 2009 - 2011 ': ' 2009-2011 ',
 ' video - calling ': ' video-calling ',
 ' student - housing ': ' student-housing ',
 ' helium - filled ': ' helium-filled ',
 ' non - Palestinian ': ' non-Palestinian ',
 ' single - stage ': ' single-stage ',
 ' bio - technology ': ' bio-technology ',
 ' pre - 1990 ': ' pre-1990 ',
 ' wind - powered ': ' wind-powered ',
 ' 140 - 146 ': ' 140-146 ',
 ' stuck - up ': ' stuck-up ',
 ' second - highest ': ' second-highest ',
 ' sense - data ': ' sense-data ',
 ' low - poly ': ' low-poly ',
 ' low - lying ': ' low-lying ',
 ' student . I ': ' student.I ',
 ' high - street ': ' high-street ',
 ' on - ear ': ' on-ear ',
 ' under - performance ': ' under-performance ',
 ' High - Dimensional ': ' High-Dimensional ',
 ' high - density ': ' high-density ',
 ' Hardy - Weinberg ': ' Hardy-Weinberg ',
 ' non - fat ': ' non-fat ',
 ' speed - limit ': ' speed-limit ',
 ' fast - moving ': ' fast-moving ',
 ' Pre - Colonial ': ' Pre-Colonial ',
 ' B . V . Sc ': ' B.V.Sc ',
 ' click - through - rate ': ' click-through-rate ',
 ' 6 . 66 ': ' 6.66 ',
 ' One - liners ': ' One-liners ',
 ' L5 - S1 ': ' L5-S1 ',
 ' surround - sound ': ' surround-sound ',
 ' paper - 2 ': ' paper-2 ',
 ' live - streaming ': ' live-streaming ',
 ' 6 . 4 ': ' 6.4 ',
 ' anti - middle ': ' anti-middle ',
 ' 6 - 1 ': ' 6-1 ',
 ' 32 - 4 ': ' 32-4 ',
 ' 0 . 48 ': ' 0.48 ',
 ' web - developer ': ' web-developer ',
 ' ch - 1 ': ' ch-1 ',
 ' inner - child ': ' inner-child ',
 ' pre - requisites ': ' pre-requisites ',
 ' batch - oriented ': ' batch-oriented ',
 ' work - based ': ' work-based ',
 ' pop - artist ': ' pop-artist ',
 ' 2 . 24 ': ' 2.24 ',
 ' Wham - O ': ' Wham-O ',
 ' MIG - 31 ': ' MIG-31 ',
 ' toe - out ': ' toe-out ',
 ' anti - vaccine ': ' anti-vaccine ',
 ' 1846 - 1848 ': ' 1846-1848 ',
 ' two - lane ': ' two-lane ',
 ' highly - paid ': ' highly-paid ',
 ' R . I . P ': ' R.I.P ',
 ' stalker - ish ': ' stalker-ish ',
 ' well - reputed ': ' well-reputed ',
 ' Jewish - Muslim ': ' Jewish-Muslim ',
 ' flat - earthers ': ' flat-earthers ',
 ' non - philosophical ': ' non-philosophical ',
 ' Arab - looking ': ' Arab-looking ',
 ' combat - style ': ' combat-style ',
 ' 0 . 472 ': ' 0.472 ',
 ' non - genital ': ' non-genital ',
 ' Poly - vinyl ': ' Poly-vinyl ',
 ' pre - college ': ' pre-college ',
 ' multi - languages ': ' multi-languages ',
 ' e - Magazine ': ' e-Magazine ',
 ' Dr . B ': ' Dr.B ',
 ' 10 - 12th ': ' 10-12th ',
 ' semi - hollow ': ' semi-hollow ',
 ' TV - L ': ' TV-L ',
 ' C3 - PO ': ' C3-PO ',
 ' 1998 - 2004 ': ' 1998-2004 ',
 ' Nebraska - Lincoln ': ' Nebraska-Lincoln ',
 ' 52 . 4 ': ' 52.4 ',
 ' stand - by ': ' stand-by ',
 ' hard - left ': ' hard-left ',
 ' 142 . 8 ': ' 142.8 ',
 ' bail - in ': ' bail-in ',
 ' self - defending ': ' self-defending ',
 ' stand - alone ': ' stand-alone ',
 ' 62 - year - old ': ' 62-year-old ',
 ' gift - giving ': ' gift-giving ',
 ' non - synonymous ': ' non-synonymous ',
 ' d3 . js ': ' d3.js ',
 ' Half - White ': ' Half-White ',
 ' plug - in ': ' plug-in ',
 ' non - stimulating ': ' non-stimulating ',
 ' work - out ': ' work-out ',
 ' stress - strain ': ' stress-strain ',
 ' green - light ': ' green-light ',
 ' 1 . 4 . 7 ': ' 1.4.7 ',
 ' M - 14 ': ' M-14 ',
 ' AUTO - INCREMENT ': ' AUTO-INCREMENT ',
 ' co - authors ': ' co-authors ',
 ' anti - rational ': ' anti-rational ',
 ' 95 . 05 ': ' 95.05 ',
 ' K - 12 ': ' K-12 ',
 ' 1 . 0 ': ' 1.0 ',
 ' Emery - Dreifuss ': ' Emery-Dreifuss ',
 ' touchy - feely ': ' touchy-feely ',
 ' sit - in ': ' sit-in ',
 ' low - brow ': ' low-brow ',
 ' non - repairable ': ' non-repairable ',
 ' non - NHL ': ' non-NHL ',
 ' ad - blocking ': ' ad-blocking ',
 ' cone - bearing ': ' cone-bearing ',
 '  - 2.25 ': ' -2.25 ',
 ' -2 . 25 ': ' -2.25 ',
 ' guru . com ': ' guru.com ',
 ' O - Level ': ' O-Level ',
 ' 30 - day ': ' 30-day ',
 ' 3 . 33 ': ' 3.33 ',
 ' Franco - Prussian ': ' Franco-Prussian ',
 ' 94 . 33 ': ' 94.33 ',
 ' Group - A ': ' Group-A ',
 ' robot . txt ': ' robot.txt ',
 ' in - love ': ' in-love ',
 ' non - administrative ': ' non-administrative ',
 ' multi - process ': ' multi-process ',
 ' co - workers ': ' co-workers ',
 ' Black - only ': ' Black-only ',
 ' s - era ': ' s-era ',
 ' long - windedness ': ' long-windedness ',
 ' M - F ': ' M-F ',
 ' 10 - 2 ': ' 10-2 ',
 ' 7 . 50 ': ' 7.50 ',
 ' log - likelihood ': ' log-likelihood ',
 ' carbon - 12 ': ' carbon-12 ',
 ' i - j ': ' i-j ',
 ' non - African ': ' non-African ',
 ' OP - AMP ': ' OP-AMP ',
 ' Maxwell - Boltzmann ': ' Maxwell-Boltzmann ',
 ' return - on - investment ': ' return-on-investment ',
 ' M . Tech ': ' M.Tech ',
 ' g - 2 ': ' g-2 ',
 ' two - week - old ': ' two-week-old ',
 ' single - tier ': ' single-tier ',
 ' B . S . Ed ': ' B.S.Ed ',
 ' full - term ': ' full-term ',
 ' WI - FI ': ' WI-FI ',
 ' low - visibility ': ' low-visibility ',
 ' one - term ': ' one-term ',
 ' trans - sexual ': ' trans-sexual ',
 ' 6 . 33 ': ' 6.33 ',
 ' brown - nose ': ' brown-nose ',
 ' post - nominal ': ' post-nominal ',
 ' Neo - cons ': ' Neo-cons ',
 ' 34 - year - old ': ' 34-year-old ',
 ' duty - bound ': ' duty-bound ',
 ' x - 45 ': ' x-45 ',
 ' time - series ': ' time-series ',
 ' G - spot ': ' G-spot ',
 ' heavy - lift ': ' heavy-lift ',
 ' maximum - likelihood ': ' maximum-likelihood ',
 ' ultra - nationalistic ': ' ultra-nationalistic ',
 ' 1 . 13 ': ' 1.13 ',
 ' non - reader ': ' non-reader ',
 ' Six - Figure ': ' Six-Figure ',
 ' over - analyze ': ' over-analyze ',
 ' co - owning ': ' co-owning ',
 ' 22 - year - old ': ' 22-year-old ',
 ' d . a ': ' d.a ',
 ' forward - thinking ': ' forward-thinking ',
 ' b . com ': ' b.com ',
 ' post - diploma ': ' post-diploma ',
 ' Sovereign - class ': ' Sovereign-class ',
 ' p - polarized ': ' p-polarized ',
 ' 101 . 25 ': ' 101.25 ',
 ' run - in ': ' run-in ',
 ' super - powered ': ' super-powered ',
 ' home - schooled ': ' home-schooled ',
 ' near - Earth ': ' near-Earth ',
 ' dirty - minded ': ' dirty-minded ',
 ' non - contrast ': ' non-contrast ',
 ' AES - 256 ': ' AES-256 ',
 ' Jung - gu ': ' Jung-gu ',
 ' anti - homosexual ': ' anti-homosexual ',
 ' re - registered ': ' re-registered ',
 ' non - British ': ' non-British ',
 ' post - modernism ': ' post-modernism ',
 ' spatio - temporal ': ' spatio-temporal ',
 ' cut - resistant ': ' cut-resistant ',
 ' non - artists ': ' non-artists ',
 ' Co - op ': ' Co-op ',
 ' re - recording ': ' re-recording ',
 ' micro - chipping ': ' micro-chipping ',
 ' over - exertion ': ' over-exertion ',
 ' 255 . 255 . 255 . 224 ': ' 255.255.255.224 ',
 ' in - vitro ': ' in-vitro ',
 ' non - portable ': ' non-portable ',
 ' four - day ': ' four-day ',
 ' end - of - the - world ': ' end-of-the-world ',
 ' 18 - 35 ': ' 18-35 ',
 ' sit - up ': ' sit-up ',
 ' Pro - Ana ': ' Pro-Ana ',
 ' in - built ': ' in-built ',
 ' p . g ': ' p.g ',
 ' laser - cut ': ' laser-cut ',
 ' Fur - Baby ': ' Fur-Baby ',
 ' 7 . 05 ': ' 7.05 ',
 ' nanny - state ': ' nanny-state ',
 ' email . com ': ' email.com ',
 ' uh - huh ': ' uh-huh ',
 ' non - intuitive ': ' non-intuitive ',
 ' Blue - Gray ': ' Blue-Gray ',
 ' crowd - sourced ': ' crowd-sourced ',
 ' non - expert ': ' non-expert ',
 ' identity - politics ': ' identity-politics ',
 ' clothing - optional ': ' clothing-optional ',
 ' 19 - 20th ': ' 19-20th ',
 ' two - week ': ' two-week ',
 ' 35 . 75 ': ' 35.75 ',
 ' Winnie - the - Pooh ': ' Winnie-the-Pooh ',
 ' self - cultivation ': ' self-cultivation ',
 ' self - report ': ' self-report ',
 ' 1 . 797 ': ' 1.797 ',
 ' 777 - 200 ': ' 777-200 ',
 ' 3 . 1 ': ' 3.1 ',
 ' one - nation ': ' one-nation ',
 ' etc . for ': ' etc.for ',
 ' Pre - Med ': ' Pre-Med ',
 ' re - position ': ' re-position ',
 ' e - marketing ': ' e-marketing ',
 ' fail - safe ': ' fail-safe ',
 ' EU - style ': ' EU-style ',
 ' real - life ': ' real-life ',
 ' warm - blooded ': ' warm-blooded ',
 ' junk - food ': ' junk-food ',
 ' 12 . 03 ': ' 12.03 ',
 ' advertiser - friendly ': ' advertiser-friendly ',
 ' pre - capitalist ': ' pre-capitalist ',
 ' z - 2 ': ' z-2 ',
 ' non - critical ': ' non-critical ',
 ' demi - goddess ': ' demi-goddess ',
 ' 3000 - 5000 ': ' 3000-5000 ',
 ' well - defined ': ' well-defined ',
 ' Trick - or - Treating ': ' Trick-or-Treating ',
 ' anti - migrant ': ' anti-migrant ',
 ' full - grown ': ' full-grown ',
 ' hot - spot ': ' hot-spot ',
 ' Neo - Nazis ': ' Neo-Nazis ',
 ' self - soothe ': ' self-soothe ',
 ' Left - Wing ': ' Left-Wing ',
 ' 88 - 89 ': ' 88-89 ',
 ' low - density ': ' low-density ',
 ' well - paid ': ' well-paid ',
 ' 64 - year - old ': ' 64-year-old ',
 ' 47 . 74 ': ' 47.74 ',
 ' target - date ': ' target-date ',
 ' application - focused ': ' application-focused ',
 ' pre - closing ': ' pre-closing ',
 ' i - 94 ': ' i-94 ',
 ' Non - believers ': ' Non-believers ',
 ' T - 1 ': ' T-1 ',
 ' ex - slaves ': ' ex-slaves ',
 ' time - efficient ': ' time-efficient ',
 ' inter - dimensional ': ' inter-dimensional ',
 ' 92 . 25 ': ' 92.25 ',
 ' 8 - core ': ' 8-core ',
 ' Al - Aqsa ': ' Al-Aqsa ',
 ' high - skilled ': ' high-skilled ',
 ' auto - related ': ' auto-related ',
 ' L . L . Bean ': ' L.L.Bean ',
 ' micro - mini ': ' micro-mini ',
 ' am - 4 ': ' am-4 ',
 ' bait - and - switch ': ' bait-and-switch ',
 ' Counter - Strike ': ' Counter-Strike ',
 ' 1 . 41 ': ' 1.41 ',
 ' in - home ': ' in-home ',
 ' anti - circumcision ': ' anti-circumcision ',
 ' M . F . A ': ' M.F.A ',
 ' non - AP ': ' non-AP ',
 ' over - priced ': ' over-priced ',
 ' employer - paid ': ' employer-paid ',
 ' Cool - Mist ': ' Cool-Mist ',
 ' Jae - In ': ' Jae-In ',
 ' re - sending ': ' re-sending ',
 ' gun - rights ': ' gun-rights ',
 ' non - admin ': ' non-admin ',
 ' value - add ': ' value-add ',
 ' 80 . 25 ': ' 80.25 ',
 ' self - murder ': ' self-murder ',
 ' eat - out ': ' eat-out ',
 ' multi - character ': ' multi-character ',
 ' souq . com ': ' souq.com ',
 ' re - jailbreak ': ' re-jailbreak ',
 ' steely - eyed ': ' steely-eyed ',
 ' long - leg ': ' long-leg ',
 ' gun - loving ': ' gun-loving ',
 ' push - start ': ' push-start ',
 ' Pre - Calc ': ' Pre-Calc ',
 ' anti - youth ': ' anti-youth ',
 ' auto - numbering ': ' auto-numbering ',
 ' non - Armenian ': ' non-Armenian ',
 ' sub - categories ': ' sub-categories ',
 ' anti - jew ': ' anti-jew ',
 '  . 75 ': ' .75 ',
 ' Z - 80 ': ' Z-80 ',
 ' off - the - field ': ' off-the-field ',
 ' semi - nude ': ' semi-nude ',
 ' re - inventing ': ' re-inventing ',
 ' blogspot . com ': ' blogspot.com ',
 ' MS - SQL ': ' MS-SQL ',
 ' left - wing ': ' left-wing ',
 ' ex - member ': ' ex-member ',
 ' 12 - 10 ': ' 12-10 ',
 ' gun - totin ': ' gun-totin ',
 ' Mozilla . org ': ' Mozilla.org ',
 ' Bi - polar ': ' Bi-polar ',
 ' 12 . 36 ': ' 12.36 ',
 ' thirteen - year - old ': ' thirteen-year-old ',
 ' limp - dick ': ' limp-dick ',
 ' non - Anglo ': ' non-Anglo ',
 ' bi - sexual ': ' bi-sexual ',
 ' right - to - left ': ' right-to-left ',
 ' visa - free ': ' visa-free ',
 ' non - territorial ': ' non-territorial ',
 ' US - controlled ': ' US-controlled ',
 ' quarter - life ': ' quarter-life ',
 '  - 0.2 ': ' -0.2 ',
 ' -0 . 2 ': ' -0.2 ',
 ' hang - out ': ' hang-out ',
 ' by - elections ': ' by-elections ',
 ' 84 . 7 ': ' 84.7 ',
 ' less - lethal ': ' less-lethal ',
 '  . 08 ': ' .08 ',
 ' Audio - Visual ': ' Audio-Visual ',
 ' double - duty ': ' double-duty ',
 ' pre - independence ': ' pre-independence ',
 ' k - 12 ': ' k-12 ',
 ' K - 8 ': ' K-8 ',
 ' eye - opening ': ' eye-opening ',
 ' math - based ': ' math-based ',
 ' anti - infective ': ' anti-infective ',
 ' 5 . 98 ': ' 5.98 ',
 ' non - ionic ': ' non-ionic ',
 ' non - chill ': ' non-chill ',
 ' Eighty - Four ': ' Eighty-Four ',
 ' same - aged ': ' same-aged ',
 ' damage - resistant ': ' damage-resistant ',
 ' F . C ': ' F.C ',
 ' on - topic ': ' on-topic ',
 ' 10 . 000 . 000 ': ' 10.000.000 ',
 ' www . un . org ': ' www.un.org ',
 ' high - dollar ': ' high-dollar ',
 ' nutrient - dense ': ' nutrient-dense ',
 ' e - liquid ': ' e-liquid ',
 ' trick - or - treat ': ' trick-or-treat ',
 ' re - educating ': ' re-educating ',
 ' 500 - 700 ': ' 500-700 ',
 ' de - sexed ': ' de-sexed ',
 ' 2 . 345 ': ' 2.345 ',
 ' iT - A ': ' iT-A ',
 ' oft - misunderstood ': ' oft-misunderstood ',
 ' T - Rex ': ' T-Rex ',
 ' re - enactments ': ' re-enactments ',
 ' 1 . 19 ': ' 1.19 ',
 ' cross - dressers ': ' cross-dressers ',
 ' Obsessive - compulsive ': ' Obsessive-compulsive ',
 ' anti - Russian ': ' anti-Russian ',
 ' do - follow ': ' do-follow ',
 ' L - 3 ': ' L-3 ',
 ' 0 . 112 ': ' 0.112 ',
 ' T - bill ': ' T-bill ',
 ' 4 . 06 ': ' 4.06 ',
 ' MakeMyTrip . com ': ' MakeMyTrip.com ',
 ' FC - 3 ': ' FC-3 ',
 ' ill - treatment ': ' ill-treatment ',
 ' board . And ': ' board.And ',
 ' 89 . 99 ': ' 89.99 ',
 ' One - and - Twenty ': ' One-and-Twenty ',
 ' T - SQL ': ' T-SQL ',
 ' March - April ': ' March-April ',
 ' 1 . 88 ': ' 1.88 ',
 ' night - out ': ' night-out ',
 ' co - wrote ': ' co-wrote ',
 ' North - east ': ' North-east ',
 ' foo . bar ': ' foo.bar ',
 ' header . PHP ': ' header.PHP ',
 ' sea - level ': ' sea-level ',
 ' olive - skinned ': ' olive-skinned ',
 ' multi - agents ': ' multi-agents ',
 ' 1999 - 2008 ': ' 1999-2008 ',
 ' 6 - ball ': ' 6-ball ',
 ' re - bond ': ' re-bond ',
 ' free - roam ': ' free-roam ',
 ' women - hating ': ' women-hating ',
 ' state - appointed ': ' state-appointed ',
 ' whole - home ': ' whole-home ',
 ' 45 - 50 ': ' 45-50 ',
 ' fully - clothed ': ' fully-clothed ',
 ' left - behind ': ' left-behind ',
 ' hand - to - hand ': ' hand-to-hand ',
 ' number . The ': ' number.The ',
 ' united . com ': ' united.com ',
 ' Y - chromosomes ': ' Y-chromosomes ',
 ' truth - in - advertising ': ' truth-in-advertising ',
 '  - 270 ': ' -270 ',
 ' fine - tuning ': ' fine-tuning ',
 ' 5 . 60 ': ' 5.60 ',
 ' sub - saharan ': ' sub-saharan ',
 ' 146 . 5 ': ' 146.5 ',
 ' hydro - power ': ' hydro-power ',
 ' flip - out ': ' flip-out ',
 ' multi - region ': ' multi-region ',
 ' A - Way ': ' A-Way ',
 ' co - found ': ' co-found ',
 ' fan - fiction ': ' fan-fiction ',
 ' self - employment ': ' self-employment ',
 ' B1 - B2 ': ' B1-B2 ',
 ' self - educate ': ' self-educate ',
 ' 46 . 5 ': ' 46.5 ',
 ' C - Spire ': ' C-Spire ',
 ' poll - bound ': ' poll-bound ',
 ' Ing - Wen ': ' Ing-Wen ',
 ' 5 . 8 ': ' 5.8 ',
 ' pre - 1900 ': ' pre-1900 ',
 ' JAS - 39 ': ' JAS-39 ',
 ' high - precision ': ' high-precision ',
 ' back - to - work ': ' back-to-work ',
 ' pre - meds ': ' pre-meds ',
 ' windows . h ': ' windows.h ',
 ' on - set ': ' on-set ',
 ' 2 - bit ': ' 2-bit ',
 ' resource - saving ': ' resource-saving ',
 ' perpetual - motion ': ' perpetual-motion ',
 ' co - signer ': ' co-signer ',
 ' semi - automatic ': ' semi-automatic ',
 ' high - volume ': ' high-volume ',
 ' PlayStation - 3 ': ' PlayStation-3 ',
 ' time - consuming ': ' time-consuming ',
 ' in - laws ': ' in-laws ',
 ' pipe - weed ': ' pipe-weed ',
 ' Linked - In ': ' Linked-In ',
 ' fund - of - funds ': ' fund-of-funds ',
 ' semi - conductor ': ' semi-conductor ',
 ' anti - scientific ': ' anti-scientific ',
 ' self - learned ': ' self-learned ',
 ' eye - candy ': ' eye-candy ',
 ' Beer - Lambert ': ' Beer-Lambert ',
 ' times . The ': ' times.The ',
 ' Take - Two ': ' Take-Two ',
 ' 6 - 7 ': ' 6-7 ',
 ' 1 . 71 ': ' 1.71 ',
 ' shine . com ': ' shine.com ',
 ' floating - point ': ' floating-point ',
 ' Android - based ': ' Android-based ',
 ' A - levels ': ' A-levels ',
 ' per - second ': ' per-second ',
 ' O - H ': ' O-H ',
 ' Pitter - Patter ': ' Pitter-Patter ',
 ' non - adaptive ': ' non-adaptive ',
 ' user - friendly ': ' user-friendly ',
 ' readily - available ': ' readily-available ',
 ' non - cluttered ': ' non-cluttered ',
 ' companies . If ': ' companies.If ',
 ' anti - religion ': ' anti-religion ',
 ' break - even ': ' break-even ',
 ' non - associative ': ' non-associative ',
 ' tie - up ': ' tie-up ',
 ' W - 2 ': ' W-2 ',
 ' pug - nosed ': ' pug-nosed ',
 ' six - month ': ' six-month ',
 ' ex - Dictator ': ' ex-Dictator ',
 ' post - American ': ' post-American ',
 ' computer - enhanced ': ' computer-enhanced ',
 ' c - sharp ': ' c-sharp ',
 ' 172 . 25 ': ' 172.25 ',
 ' gender - reassignment ': ' gender-reassignment ',
 ' matter - antimatter ': ' matter-antimatter ',
 ' self - pollinating ': ' self-pollinating ',
 ' 0 . 55 ': ' 0.55 ',
 ' anti - Obama ': ' anti-Obama ',
 ' S - S ': ' S-S ',
 ' C . O . R . E ': ' C.O.R.E ',
 ' high - functioning ': ' high-functioning ',
 ' anti - GMO ': ' anti-GMO ',
 ' 2000 . 00 ': ' 2000.00 ',
 ' US - Vietnam ': ' US-Vietnam ',
 ' gas - filled ': ' gas-filled ',
 ' stress - less ': ' stress-less ',
 ' ground - mounted ': ' ground-mounted ',
 ' 10 - 7 ': ' 10-7 ',
 ' centuries . The ': ' centuries.The ',
 ' inflation - adjusted ': ' inflation-adjusted ',
 ' draft - dodging ': ' draft-dodging ',
 ' over - buying ': ' over-buying ',
 ' 5 . 1 . 10 ': ' 5.1.10 ',
 ' a - like ': ' a-like ',
 ' East - West ': ' East-West ',
 '  - 000 ': ' -000 ',
 ' RJ - 11 ': ' RJ-11 ',
 ' 2013 - 17 ': ' 2013-17 ',
 ' normal - range ': ' normal-range ',
 ' non - fictional ': ' non-fictional ',
 ' non - seed ': ' non-seed ',
 ' 0 . 456 ': ' 0.456 ',
 ' W - 9 ': ' W-9 ',
 ' self - gratification ': ' self-gratification ',
 ' on - stage ': ' on-stage ',
 ' big - boned ': ' big-boned ',
 ' asylum - seeking ': ' asylum-seeking ',
 ' R . M . S ': ' R.M.S ',
 ' al - Bukhari ': ' al-Bukhari ',
 ' check - up ': ' check-up ',
 ' religious - secular ': ' religious-secular ',
 ' 4 - Sided ': ' 4-Sided ',
 ' single - variable ': ' single-variable ',
 ' anti - democrat ': ' anti-democrat ',
 ' NEO - NAZIS ': ' NEO-NAZIS ',
 ' non - paranormal ': ' non-paranormal ',
 ' K - dramas ': ' K-dramas ',
 ' AK - 74 ': ' AK-74 ',
 ' history . He ': ' history.He ',
 ' non - personalized ': ' non-personalized ',
 ' re - sale ': ' re-sale ',
 ' twin - seat ': ' twin-seat ',
 ' time - varying ': ' time-varying ',
 ' 450 . 000 ': ' 450.000 ',
 ' all - rounders ': ' all-rounders ',
 ' yr - old ': ' yr-old ',
 ' 1 . 35 ': ' 1.35 ',
 ' SU - 30 ': ' SU-30 ',
 ' catch - all ': ' catch-all ',
 ' anti - progressive ': ' anti-progressive ',
 ' C - P ': ' C-P ',
 ' mini - golf ': ' mini-golf ',
 ' E . T ': ' E.T ',
 ' vice - like ': ' vice-like ',
 ' well - organised ': ' well-organised ',
 ' Solo - Baric ': ' Solo-Baric ',
 ' mid - 50 ': ' mid-50 ',
 ' E - numbers ': ' E-numbers ',
 ' a . p ': ' a.p ',
 ' 60 . 47 ': ' 60.47 ',
 ' 20 - 20 - 20 ': ' 20-20-20 ',
 ' 450 - 500 ': ' 450-500 ',
 ' a - 2 ': ' a-2 ',
 ' 63 . 2 ': ' 63.2 ',
 ' re - launching ': ' re-launching ',
 ' non - functioning ': ' non-functioning ',
 ' v - a ': ' v-a ',
 ' 4 - H ': ' 4-H ',
 ' egg - based ': ' egg-based ',
 ' T - Test ': ' T-Test ',
 ' oil - for - food ': ' oil-for-food ',
 ' be - bop ': ' be-bop ',
 ' g - 1 ': ' g-1 ',
 ' Anglo - Americans ': ' Anglo-Americans ',
 ' US - EU ': ' US-EU ',
 ' non - radioactive ': ' non-radioactive ',
 ' hitch - hiking ': ' hitch-hiking ',
 ' bucket - list ': ' bucket-list ',
 ' 1700 - 1800 ': ' 1700-1800 ',
 ' anti - thesis ': ' anti-thesis ',
 ' inter - state ': ' inter-state ',
 ' 1200 - 1300 ': ' 1200-1300 ',
 ' pre - screen ': ' pre-screen ',
 ' re - educated ': ' re-educated ',
 ' 104 . 5 ': ' 104.5 ',
 ' gravity - defying ': ' gravity-defying ',
 ' non - GM ': ' non-GM ',
 ' Add - on ': ' Add-on ',
 ' B . F . A ': ' B.F.A ',
 ' inverse - square ': ' inverse-square ',
 ' pre - market ': ' pre-market ',
 ' audible . com ': ' audible.com ',
 ' lab - grown ': ' lab-grown ',
 ' Gauss - Bonnet ': ' Gauss-Bonnet ',
 ' under - privileged ': ' under-privileged ',
 ' regular - sized ': ' regular-sized ',
 ' May - July ': ' May-July ',
 ' average - looking ': ' average-looking ',
 '  . 17 ': ' .17 ',
 ' e - class ': ' e-class ',
 ' zero - energy ': ' zero-energy ',
 ' in - demand ': ' in-demand ',
 ' 61 . 7 ': ' 61.7 ',
 '  - 9 ': ' -9 ',
 ' start . So ': ' start.So ',
 ' non - Apple ': ' non-Apple ',
 ' 2001 - 2002 ': ' 2001-2002 ',
 ' non - belligerent ': ' non-belligerent ',
 ' shaggy - haired ': ' shaggy-haired ',
 ' non - conservative ': ' non-conservative ',
 '  - 018 ': ' -018 ',
 ' Brain - Computer ': ' Brain-Computer ',
 ' Flip - flop ': ' Flip-flop ',
 ' love - story ': ' love-story ',
 ' post - 1996 ': ' post-1996 ',
 ' body - mind ': ' body-mind ',
 ' up - down ': ' up-down ',
 ' Non - Fiction ': ' Non-Fiction ',
 ' role - based ': ' role-based ',
 ' Niger - Congo ': ' Niger-Congo ',
 ' 24 - 48 ': ' 24-48 ',
 ' non - student ': ' non-student ',
 ' less - than - perfect ': ' less-than-perfect ',
 ' one - side ': ' one-side ',
 ' 2 . 12 . 4 ': ' 2.12.4 ',
 ' Brain - Dump ': ' Brain-Dump ',
 ' re - direct ': ' re-direct ',
 ' Re - testing ': ' Re-testing ',
 ' book - worm ': ' book-worm ',
 ' Chinese - style ': ' Chinese-style ',
 ' freeze - dried ': ' freeze-dried ',
 ' AT - ATs ': ' AT-ATs ',
 ' White - looking ': ' White-looking ',
 '  - 152 ': ' -152 ',
 ' She - Hulk ': ' She-Hulk ',
 ' thru - out ': ' thru-out ',
 ' self - actualization ': ' self-actualization ',
 ' medium - sized ': ' medium-sized ',
 ' acoustic - electric ': ' acoustic-electric ',
 ' Anti - Epileptic ': ' Anti-Epileptic ',
 ' early - childhood ': ' early-childhood ',
 ' reality - based ': ' reality-based ',
 ' least - significant ': ' least-significant ',
 ' non - citizen ': ' non-citizen ',
 ' non - pro ': ' non-pro ',
 ' self - learning ': ' self-learning ',
 ' fortune - telling ': ' fortune-telling ',
 ' money - less ': ' money-less ',
 ' full - strength ': ' full-strength ',
 ' 7 . 35 ': ' 7.35 ',
 ' 43 - inch ': ' 43-inch ',
 ' pro - Second ': ' pro-Second ',
 ' 17 - 22 ': ' 17-22 ',
 ' close - mindedness ': ' close-mindedness ',
 ' XE . com ': ' XE.com ',
 ' 13 - 17 ': ' 13-17 ',
 ' risk - averse ': ' risk-averse ',
 ' u - haul ': ' u-haul ',
 ' years . I ': ' years.I ',
 ' non - contact ': ' non-contact ',
 ' early - on ': ' early-on ',
 ' right - leaning ': ' right-leaning ',
 ' in - the - money ': ' in-the-money ',
 ' non - ideal ': ' non-ideal ',
 ' MX - 5 ': ' MX-5 ',
 ' note - making ': ' note-making ',
 ' sun - drenched ': ' sun-drenched ',
 ' 70 - 680 ': ' 70-680 ',
 ' Backus - Naur ': ' Backus-Naur ',
 ' J - Pop ': ' J-Pop ',
 ' 220 - age ': ' 220-age ',
 ' 190 - 193 ': ' 190-193 ',
 ' pre - Socratic ': ' pre-Socratic ',
 ' 42 - 50 ': ' 42-50 ',
 ' North - South ': ' North-South ',
 ' 16 - bit ': ' 16-bit ',
 ' sit - ins ': ' sit-ins ',
 ' N - o ': ' N-o ',
 ' well - made ': ' well-made ',
 ' non - deserving ': ' non-deserving ',
 ' 62 - 44 ': ' 62-44 ',
 ' JavaScript - based ': ' JavaScript-based ',
 ' R - 18 ': ' R-18 ',
 ' two - person ': ' two-person ',
 ' key - value ': ' key-value ',
 ' quasi - independent ': ' quasi-independent ',
 ' B - Wing ': ' B-Wing ',
 ' 300 - 500 ': ' 300-500 ',
 ' top - ten ': ' top-ten ',
 ' 30 - round ': ' 30-round ',
 ' 50 - storey ': ' 50-storey ',
 ' I - class ': ' I-class ',
 ' muscle - building ': ' muscle-building ',
 ' pre - amp ': ' pre-amp ',
 ' Alpha - GPC ': ' Alpha-GPC ',
 ' pull - start ': ' pull-start ',
 ' over - confidence ': ' over-confidence ',
 ' 1 . 07 ': ' 1.07 ',
 ' shape - shifting ': ' shape-shifting ',
 ' French - controlled ': ' French-controlled ',
 ' pre - game ': ' pre-game ',
 '  . 2 ': ' .2 ',
 ' out - of - pocket ': ' out-of-pocket ',
 ' off - campus ': ' off-campus ',
 ' 7 - year ': ' 7-year ',
 ' fear - based ': ' fear-based ',
 ' two - facedness ': ' two-facedness ',
 ' Neo - Darwinian ': ' Neo-Darwinian ',
 ' city - planning ': ' city-planning ',
 ' over - ambitious ': ' over-ambitious ',
 ' non - heat ': ' non-heat ',
 ' 2 - 105 ': ' 2-105 ',
 ' 6 - foot ': ' 6-foot ',
 ' Multi - Agent ': ' Multi-Agent ',
 ' 98 - 100 ': ' 98-100 ',
 ' long - lived ': ' long-lived ',
 ' time . He ': ' time.He ',
 ' data - driven ': ' data-driven ',
 ' 5 - in - 1 ': ' 5-in-1 ',
 ' non - conclusive ': ' non-conclusive ',
 ' A - bombs ': ' A-bombs ',
 ' father - in - Law ': ' father-in-Law ',
 ' Off - page ': ' Off-page ',
 ' old - looking ': ' old-looking ',
 ' I . R . A ': ' I.R.A ',
 ' Gai - sensei ': ' Gai-sensei ',
 ' H - Y ': ' H-Y ',
 ' ex - pat ': ' ex-pat ',
 ' 0 . 67 ': ' 0.67 ',
 ' Super - Soldier ': ' Super-Soldier ',
 ' anti - lock ': ' anti-lock ',
 ' 58 . 6 ': ' 58.6 ',
 ' Koh - i - Noor ': ' Koh-i-Noor ',
 ' x - acto ': ' x-acto ',
 ' A - j ': ' A-j ',
 ' black - tie ': ' black-tie ',
 ' 99 . 999 ': ' 99.999 ',
 ' end - of - life ': ' end-of-life ',
 ' 1 - stop ': ' 1-stop ',
 ' Korean - Canadian ': ' Korean-Canadian ',
 ' Michaelis - Menten ': ' Michaelis-Menten ',
 ' off - course ': ' off-course ',
 ' propan - 2 - ol ': ' propan-2-ol ',
 ' 35 - year - old ': ' 35-year-old ',
 ' 1945 - 47 ': ' 1945-47 ',
 ' working . I ': ' working.I ',
 ' 3 . 70 ': ' 3.70 ',
 ' C - 5 ': ' C-5 ',
 ' edge - of - the - seat ': ' edge-of-the-seat ',
 ' 1 . 009 ': ' 1.009 ',
 ' student - run ': ' student-run ',
 ' b . ed ': ' b.ed ',
 ' C - Corp ': ' C-Corp ',
 ' non - Westerners ': ' non-Westerners ',
 ' full - load ': ' full-load ',
 ' re - registration ': ' re-registration ',
 ' R - type ': ' R-type ',
 ' 26 . 4 ': ' 26.4 ',
 ' Z - 1 ': ' Z-1 ',
 ' bestbuy . com ': ' bestbuy.com ',
 ' 150 - 250 ': ' 150-250 ',
 ' al - Ghamdi ': ' al-Ghamdi ',
 ' Goodreads . com ': ' Goodreads.com ',
 ' J . D ': ' J.D ',
 ' chick - magnet ': ' chick-magnet ',
 ' one - parent ': ' one-parent ',
 ' 97 . 78 ': ' 97.78 ',
 ' c - corp ': ' c-corp ',
 ' April - May ': ' April-May ',
 ' non - autonomous ': ' non-autonomous ',
 ' 270 - 280 ': ' 270-280 ',
 ' 2006 - 2008 ': ' 2006-2008 ',
 ' self - consciousness ': ' self-consciousness ',
 ' 250 - 300 ': ' 250-300 ',
 ' back - up ': ' back-up ',
 ' Li - Young ': ' Li-Young ',
 ' network . This ': ' network.This ',
 ' per - capita ': ' per-capita ',
 ' 1 - 5 ': ' 1-5 ',
 ' well - bred ': ' well-bred ',
 ' UK - born ': ' UK-born ',
 ' low - paying ': ' low-paying ',
 ' 2 - wheeler ': ' 2-wheeler ',
 ' graft - versus - host ': ' graft-versus-host ',
 ' Chinese - Americans ': ' Chinese-Americans ',
 ' flight - path ': ' flight-path ',
 ' web - application ': ' web-application ',
 ' Anglo - Catholicism ': ' Anglo-Catholicism ',
 ' father - in - laws ': ' father-in-laws ',
 ' air - to - air ': ' air-to-air ',
 ' self - immolations ': ' self-immolations ',
 ' 20 - 56 ': ' 20-56 ',
 ' l . e ': ' l.e ',
 ' low - cost ': ' low-cost ',
 ' highly - qualified ': ' highly-qualified ',
 ' fixed - term ': ' fixed-term ',
 ' V - neck ': ' V-neck ',
 ' 2002 - 2003 ': ' 2002-2003 ',
 ' non - parasitic ': ' non-parasitic ',
 ' respect . I ': ' respect.I ',
 ' 88 . 22 ': ' 88.22 ',
 ' pro - ana ': ' pro-ana ',
 ' bio - medical ': ' bio-medical ',
 ' French - Canadian ': ' French-Canadian ',
 ' Export - Import ': ' Export-Import ',
 ' flat - roof ': ' flat-roof ',
 ' Border - Gavaskar ': ' Border-Gavaskar ',
 ' counter - strike ': ' counter-strike ',
 ' 2 . 93 ': ' 2.93 ',
 ' sex - selective ': ' sex-selective ',
 
'''



def spacing_dash_point(text):
    if '-' in text:
        text = text.replace('-', ' - ')
    if '.' in text:
        text = text.replace('.', ' . ')
    return text


train_df["question_text"] = train_df["question_text"].apply(spacing_dash_point)
test_df["question_text"] = test_df["question_text"].apply(spacing_dash_point)

def fix_dash_point_spacing_bug(text):
    for bug_dash in bug_punc_spacing_words_mapping:
        if bug_dash in text:
            text = text.replace(bug_dash, bug_punc_spacing_words_mapping[bug_dash])
    return text

def fix_dash_point_spacing_bug_wrapper(df):
    df["question_text"] = df["question_text"].apply(fix_dash_point_spacing_bug)
    return df


train_df = df_parallelize_run(train_df, fix_dash_point_spacing_bug_wrapper)
test_df = df_parallelize_run(test_df, fix_dash_point_spacing_bug_wrapper)

oov_glove, oov_paragram, oov_fasttext, oov_google = vocab_check_coverage(train_df, test_df)

'''
Glove : 
Found embeddings for 80.53% of vocab
Found embeddings for  99.68% of all text
Paragram : 
Found embeddings for 80.73% of vocab
Found embeddings for  99.68% of all text
FastText : 
Found embeddings for 74.45% of vocab
Found embeddings for  99.56% of all text
Google : 
Found embeddings for 64.14% of vocab
Found embeddings for  87.91% of all text
CPU times: user 14.1 s, sys: 616 ms, total: 14.7 s
Wall time: 14.7 s

'''

print('glove oov rate:', oov_glove['oov_rate'])
print('paragram oov rate:', oov_paragram['oov_rate'])
print('fasttext oov rate:', oov_fasttext['oov_rate'])
print('google oov rate:', oov_google['oov_rate'])

'''
glove oov rate: 0.19473270502671758
paragram oov rate: 0.19269216416429133
fasttext oov rate: 0.25549635957982325
google oov rate: 0.35857543252320423
'''

train_df.to_csv("cleaned_train.csv", index=False)
test_df.to_csv("cleaned_test.csv", index=False)

'''
After fixing the dash and point spacing bug, the coverage continue improve a lot.
Embedding 	                    Original 	Text Cleaning 	Fix punc spacing bug
Glove vocab founded 	         33.92% 	   72.59% 	        80.31%
Glove vocab founded in text 	 88.20% 	   99.44% 	        99.68%
Paragram vocab founded 	         34.08% 	   72.87% 	        80.52%
Paragram vocab founded in text 	 88.21% 	   99.45% 	        99.68%
FastText vocab founded 	         31.63% 	   68.32% 	        74.43%
FastText vocab founded in text 	 87.74% 	   99.38% 	        99.57%
Google vocab founded 	         26.24% 	   56.71% 	        64.71%
Google vocab founded in text 	 87.26% 	   88.01% 	        87.89%

'''
