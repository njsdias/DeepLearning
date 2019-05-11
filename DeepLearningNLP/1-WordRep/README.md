## Word Vector Representations

When dealing with languages and words, we might end up classifying
texts across thousands of classes. 
Much research has been undertaken in this field
in recent years, and this has resulted in the transformation of words in
languages to the format of vectors that can be used in multiple sets of
algorithms and processes.

Here the objetive is covers in-depth the explanation of
word embeddings and their effectiveness.

### 1. Introduction to Word Embedding
Vector space models, signifying text documents and queries in the form of vectors, have
long been used for distributional semantics purposes. The representation
of words in an N-dimensional vector space by vector space models is
useful for different NLP algorithms to achieve better results, as it leads to
groupings of similar text in the new vector space.

Word embeddings models have proven to be more efficient than the
bag-of-word models or one-hot-encoding schemes, made up of sparse
vectors with a size equivalent to that of the vocabulary, used initially. 
Word embedding has replaced this concept (sparse vectors) by making use of
the surrounding words of all the individual words, using the information
present from the given text and passing it to the model. This has allowed
embedding to take the form of a dense vector, which, in a continuous
vector space, represents the projection of the individual words. 
Embedding thus refers to the coordinates of the word in the newly learned vector space.

Exampleof one-hot-enconding: 
Let’s assume that our vocabulary contains the words, Rome, Italy,
Paris, France, and country. We can make use of each of these words to
create a representation, using a one-hot scheme for all the words:

![onde-hot_WR](https://user-images.githubusercontent.com/37953610/57548346-9ecc0d80-7358-11e9-8c18-25dd6c805a70.JPG)

In a better form of representation, we can create multiple hierarchies or segments, in which the information shown by each of the
words can be assigned various weightages.

![weight_WR](https://user-images.githubusercontent.com/37953610/57548466-ea7eb700-7358-11e9-819d-1ebb1bb1e873.JPG)

The preceding vectors used for each word does signify the actual
meaning of the word and provides a better scale with which to make a
comparison across the words. The newly formed vectors are sufficiently
capable of answering the kind of relationships being held among words.

![vector_rep_WR1](https://user-images.githubusercontent.com/37953610/57548705-8a3c4500-7359-11e9-81bd-7869066cd352.JPG)


The output vectors for different words does retain the linguistic
regularities and patterns, and this is proven by the linear translations
of these patterns. For example, the result of the difference between
the vectors and the words following, vector(France) - vector(Paris) +
vector(Italy), will be close to vector(Rome)

![vector_rep_WR2](https://user-images.githubusercontent.com/37953610/57548718-96280700-7359-11e9-8fbc-f00830d65c5c.JPG)

Over time, word embeddings have emerged to become one of the
most important applications of the unsupervised learning domain. The
semantic relationships offered by word vectors have helped in the NLP
approaches of neural machine translation, information retrieval, and **question-and-answer applications.**

### 2. Word2vec

Word2vec, or word-to-vector, models were introduced by Tomas Mikolov et al ((https://arxiv.org/pdf/1301.3781.pdf)). 
It is used to learn the word embeddings, or vector representation of words.

Word2vec models make use internally of a simple neural network of a single layer and capture the weights of the hidden layer representation of words. The aim
of training the model is to learn the weights of the hidden layer, which
represents the “word embeddings.” Word2vec offers a range of models that are used to represent words
in an n-dimensional space in such a way that similar words and words
representing closer meanings are placed close to one another. The two most frequently used models are:
- skip-gram: predicts the context words using the center words 
- continuous bag-of-words (CBOW) : predicts the center
words by making use of the context or surrounding words

In comparison with the one-hot encoding, word2vec helps in reducing
the size of the encoding space and compresses the representation of words
to the desired length for the vector.

Word2vec approaches word representation on the basis of the context in which words are
presented. For example, synonyms, opposites, semantically similar
concepts, and similar words are present in similar contexts across a text
and, thus, are embedded in similar fashion, and their final embeddings lie
closer to one another. 

The next figure using the window size of 2 to pick the words from the
sentence “Machines can now recognize objects and translate speech in
real time” and training the model. We can see the procedure: 
- Machines: can - now
- can: Machines - x - now - recognize
- now: Machines - can - x - recognize - objects
- recognize: can - now - x- objects - and 
![word2vec](https://user-images.githubusercontent.com/37953610/57570462-14d48100-73fa-11e9-87a7-2ae58d2cd535.JPG)

### 3. Skip-Gram Model

A skip-gram model predicts the surrounding words by using the current
word in the sequence. The classification score of the surrounding words is
based on the syntactic relation and occurrences with the center word.

### 3.1 Model Components of Skip-Gram Architecture

Here, the input word fed as a one-hot-encoded vector and the output as a one-hot-encoded vector representing the output word.

![skipgram_arch](https://user-images.githubusercontent.com/37953610/57570581-71846b80-73fb-11e9-8c6e-8a04eab3e194.JPG)

**Hidden Layer**

The training of the neural network is done using a hidden layer, with the
count of neurons equal to the number of features or dimensions by which
we want to represent the word embedding. In the, we have
represented the hidden layer with a weight matrix having columns of 300,
equal to the number of neurons — which will be the count of the features in
the final output vector of word embedding — and rows as 100.000, which is
equal to the size of the vocabulary used to train the model.

**Output Layer**

Our main intention behind calculating the word embedding for words is
to make sure that words with similar meanings lie closer in our defined
vector space.

![skipgram_softmax](https://user-images.githubusercontent.com/37953610/57570773-9a0d6500-73fd-11e9-83de-ba52146eecce.JPG)





