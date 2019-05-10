## Word Vector Representations

When dealing with languages and words, we might end up classifying
texts across thousands of classes. 
Much research has been undertaken in this field
in recent years, and this has resulted in the transformation of words in
languages to the format of vectors that can be used in multiple sets of
algorithms and processes.

Here the objetive is covers in-depth the explanation of
word embeddings and their effectiveness.

### 1-Introduction to Word Embedding
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
