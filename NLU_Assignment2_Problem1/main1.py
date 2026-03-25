import os
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')


# STEP 1: LOAD DATA

# This script trains Word2Vec embeddings on textual data collected from IIT Jodhpur sources.
# Both library implementation (Gensim) and a manual implementation (from scratch) are included
# so that their semantic behaviour and training characteristics can be compared.

data_path = "data/raw_text"

documents = []

for file in os.listdir(data_path):
    if file.endswith(".txt"):
        with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
            documents.append(f.read())

print("Total documents:", len(documents))

# STEP 2: CLEAN TEXT

# Basic text cleaning is done here to remove unwanted symbols, numbers and formatting noise.
# Since the data is collected from different webpages and documents, this step is important
# to ensure consistency of the corpus.

def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text

def remove_stopwords(tokenized_docs):

    stop_words = set(stopwords.words('english'))
    cleaned_docs = []

    for doc in tokenized_docs:
        filtered_doc = []
        for word in doc:
            if word not in stop_words:
                filtered_doc.append(word)
        cleaned_docs.append(filtered_doc)

    return cleaned_docs

clean_docs = [clean_text(doc) for doc in documents]


# STEP 3: TOKENIZATION

# Tokenization converts cleaned sentences into individual words.
# Stopwords are removed so that embeddings focus more on meaningful academic vocabulary.

tokenized_docs = [word_tokenize(doc) for doc in clean_docs]
tokenized_docs = remove_stopwords(tokenized_docs)


# STEP 4: SAVE CLEAN CORPUS


with open("cleaned_corpus.txt","w",encoding="utf-8") as f:
    for doc in clean_docs:
        f.write(doc + "\n")


# STEP 5: DATASET STATISTICS


all_tokens = []

for doc in tokenized_docs:
    all_tokens.extend(doc)

total_tokens = len(all_tokens)
vocab = set(all_tokens)

print("Total tokens:", total_tokens)
print("Vocabulary size:", len(vocab))


# WORD CLOUD

# Word cloud gives a quick visual intuition about the most frequent terms in the IITJ corpus.
# This helps in understanding whether the collected dataset reflects academic content.

freq = Counter(all_tokens)

wordcloud = WordCloud(width=800,height=400,
                      background_color='white').generate_from_frequencies(freq)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud")
plt.savefig("wordcloud.png")
plt.show()


# STEP 6: TRAIN WORD2VEC MODELS (GENSIM)

# Word cloud gives a quick visual intuition about the most frequent terms in the IITJ corpus.
# This helps in understanding whether the collected dataset reflects academic content.

print("\nTraining CBOW model...")

cbow_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=100,
    window=5,
    min_count=2,
    sg=0,
    negative=5
)

print("\nTraining Skipgram model...")

skipgram_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1,
    negative=5
)

# STEP 7: NEAREST NEIGHBORS (GENSIM)

# Cosine similarity is used to inspect semantic structure of the learned embeddings.
# Words like 'research' and 'student' are expected to retrieve academically related terms.

words = ["research","student","phd","exam"]

print("\nCBOW Nearest Words")

for w in words:
    if w in cbow_model.wv:
        print("\n",w)
        print(cbow_model.wv.most_similar(w, topn=5))

print("\nSkipgram Nearest Words")

for w in words:
    if w in skipgram_model.wv:
        print("\n",w)
        print(skipgram_model.wv.most_similar(w, topn=5))

# STEP 8: WORD2VEC FROM SCRATCH (SKIPGRAM)

# In this section, a simplified Skip-gram model is implemented manually.
# The goal is to understand the internal learning mechanism of Word2Vec
# instead of relying entirely on library implementations.

print("\nTraining Skipgram FROM SCRATCH...\n")

word_counts = Counter(all_tokens)
vocab_list = list(word_counts.keys())

word_to_idx = {w:i for i,w in enumerate(vocab_list)}
idx_to_word = {i:w for w,i in word_to_idx.items()}

V = len(vocab_list)
embed_dim = 100

W1 = np.random.randn(V, embed_dim) * 0.01
W2 = np.random.randn(embed_dim, V) * 0.01

def generate_pairs(tokenized_docs, window=2):

    pairs = []

    for doc in tokenized_docs:
        for i, word in enumerate(doc):

            target = word_to_idx[word]

            for j in range(-window, window+1):

                if j==0:
                    continue

                if i+j>=0 and i+j<len(doc):
                    context = word_to_idx[doc[i+j]]
                    pairs.append((target, context))

    return pairs

training_pairs = generate_pairs(tokenized_docs)

def neg_samples(pos, k=5):

    neg=[]
    while len(neg)<k:
        r=np.random.randint(0,V)
        if r!=pos:
            neg.append(r)
    return neg

lr=0.03
epochs=2

for ep in range(epochs):

    total_loss=0

    for target,context in training_pairs:

        h=W1[target]

        pos_score=np.dot(h,W2[:,context])
        pos_sig=1/(1+np.exp(-pos_score))

        neg_ids=neg_samples(context,5)
        neg_scores=np.dot(W2[:,neg_ids].T,h)
        neg_sig=1/(1+np.exp(-neg_scores))

        loss=-np.log(pos_sig)-np.sum(np.log(1-neg_sig))
        total_loss+=loss

        grad_pos=(pos_sig-1)

        W2[:,context]-=lr*grad_pos*h
        W1[target]-=lr*grad_pos*W2[:,context]

        for i,nid in enumerate(neg_ids):

            grad_neg=neg_sig[i]

            W2[:,nid]-=lr*grad_neg*h
            W1[target]-=lr*grad_neg*W2[:,nid]

    print("Scratch Epoch",ep,"Loss",total_loss)


# SCRATCH CBOW TRAINING

# Manual CBOW training averages context embeddings to predict the target word.
# This typically produces smoother representations but may converge slower
# due to the simplified optimization strategy used here.

print("\nTraining CBOW FROM SCRATCH...\n")

W1_cbow = np.random.randn(V, embed_dim) * 0.01
W2_cbow = np.random.randn(embed_dim, V) * 0.01

def generate_cbow_data(tokenized_docs, window=2):

    data = []

    for doc in tokenized_docs:
        for i, word in enumerate(doc):

            context = []

            for j in range(-window, window+1):
                if j == 0:
                    continue
                if i+j >= 0 and i+j < len(doc):
                    context.append(word_to_idx[doc[i+j]])

            if len(context) > 0:
                target = word_to_idx[word]
                data.append((context, target))

    return data

cbow_pairs = generate_cbow_data(tokenized_docs)

for ep in range(epochs):

    total_loss = 0

    for context, target in cbow_pairs:

        h = np.mean(W1_cbow[context], axis=0)

        score = np.dot(h, W2_cbow[:,target])
        sig = 1/(1+np.exp(-score))

        neg_ids = neg_samples(target, 5)
        neg_scores = np.dot(W2_cbow[:,neg_ids].T, h)
        neg_sig = 1/(1+np.exp(-neg_scores))

        loss = -np.log(sig) - np.sum(np.log(1-neg_sig))
        total_loss += loss

        grad = (sig - 1)

        W2_cbow[:,target] -= lr * grad * h

        for c in context:
            W1_cbow[c] -= lr * grad * W2_cbow[:,target] / len(context)

        for i,nid in enumerate(neg_ids):

            grad_neg = neg_sig[i]

            W2_cbow[:,nid] -= lr * grad_neg * h

            for c in context:
                W1_cbow[c] -= lr * grad_neg * W2_cbow[:,nid] / len(context)

    print("Scratch CBOW Epoch", ep, "Loss", total_loss)


# STEP 9: SCRATCH SIMILAR WORDS


def most_similar_scratch(word,topn=5):

    vec=W1[word_to_idx[word]]
    sims=[]

    for w in vocab_list:
        v=W1[word_to_idx[w]]
        cos=np.dot(vec,v)/(np.linalg.norm(vec)*np.linalg.norm(v)+1e-9)
        sims.append((w,cos))

    sims.sort(key=lambda x:-x[1])
    return sims[1:topn+1]

print("\nSCRATCH MODEL NEAREST WORDS")

for w in words:
    if w in word_to_idx:
        print("\n",w)
        print(most_similar_scratch(w))


def most_similar_cbow_scratch(word, topn=5):

    vec = W1_cbow[word_to_idx[word]]

    sims = []

    for w in vocab_list:
        v = W1_cbow[word_to_idx[w]]

        cos = np.dot(vec,v)/(np.linalg.norm(vec)*np.linalg.norm(v)+1e-9)

        sims.append((w,cos))

    sims.sort(key=lambda x:-x[1])

    return sims[1:topn+1]

print("\nSCRATCH CBOW NEAREST WORDS")

for w in words:
    if w in word_to_idx:
        print("\n",w)
        print(most_similar_cbow_scratch(w))



# STEP 10: EMBEDDING VISUALIZATION

# PCA projection is used to visualize high-dimensional embeddings in 2D space.
# Comparing Gensim and scratch embeddings helps in analysing clustering behaviour.

vis_words = vocab_list[:100]

# GENSIM SKIPGRAM 
vectors_gensim = []
valid_words = []

for w in vis_words:
    if w in skipgram_model.wv:
        vectors_gensim.append(skipgram_model.wv[w])
        valid_words.append(w)

pca = PCA(n_components=2)
result = pca.fit_transform(vectors_gensim)

plt.figure(figsize=(10,8))

for i, word in enumerate(valid_words):
    plt.scatter(result[i,0], result[i,1])
    plt.annotate(word,(result[i,0], result[i,1]))

plt.title("Gensim Skipgram Embeddings")
plt.savefig("embeddings_gensim.png")
plt.show()


# SCRATCH SKIPGRAM 
vectors_scratch = [W1[word_to_idx[w]] for w in vis_words]

result2 = pca.fit_transform(vectors_scratch)

plt.figure(figsize=(10,8))

for i, word in enumerate(vis_words):
    plt.scatter(result2[i,0], result2[i,1])
    plt.annotate(word,(result2[i,0], result2[i,1]))

plt.title("Scratch Skipgram Embeddings")
plt.savefig("embeddings_scratch.png")
plt.show()


# GENSIM CBOW EMBEDDING VISUALIZATION


vectors_cbow = []
valid_words_cbow = []

for w in vis_words:
    if w in cbow_model.wv:
        vectors_cbow.append(cbow_model.wv[w])
        valid_words_cbow.append(w)

pca = PCA(n_components=2)
result_cbow = pca.fit_transform(vectors_cbow)

plt.figure(figsize=(10,8))

for i, word in enumerate(valid_words_cbow):
    plt.scatter(result_cbow[i,0], result_cbow[i,1])
    plt.annotate(word,(result_cbow[i,0], result_cbow[i,1]))

plt.title("Gensim CBOW Embeddings")
plt.savefig("embeddings_cbow.png")
plt.show()
# From experimental observation, the library implementation produces more stable
# semantic neighbourhoods, while the scratch model provides conceptual clarity
# about how embedding learning actually works.