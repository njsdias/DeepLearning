### 1. Word Embeddings: GloVe and word2vec

Word embeddings are a way to transform words in text to numerical vectors so that they can be analyzed by standard machine learning algorithms that require vectors as numerical input.

**One-hot encoding** is the most basic embedding approach. One-hot encoding represents a word in the text by a vector of the size of the vocabulary, where only the entry corresponding to the word is a one and all the other entries are zero. A major problem with one-hot encoding is that there is no way to represent the similarity between words [(cat, dog), (knife, spoon)]. Similarity between vectors is computed using the dot product, which is the sum of element-wise multiplication between vector elements. In the case of one-hot encoded vectors, the dot product between any two words in a corpus is always zero.

To overcome the limitations of one-hot encoding, the NLP community has borrowed techniques from information retrieval (IR) to vectorize text using the document as the context. Notable techniques are **TF-IDF** (https://en.wikipedia.org/wiki/Tf%E2%80%93idf), **latent semantic analysis (LSA)** (https://en.wikipedia.org/wiki/Latent_semantic_analysis), and topic modeling (https://en.wikipedia.org/wiki/Topic_model).

**Word embedding** differs from previous IR-based techniques in that they use words as their context, which leads to a more natural form of semantic similarity from a human understanding perspective. Today, word embedding is the technique of choice for vectorizing text for all kinds of NLP tasks, such as text classification, document clustering, part of speech tagging, named entity recognition, sentiment analysis, and so on.

This notebook covers the **GloVe** and **word2vec** that are the word embedding tecniques that have proven more effective and have been widely adopted in the deep learning and NLP communities. For that the topics cover will be:

    Building various distributional representations of words in context
    Building models for leveraging embeddings to perform NLP tasks such as sentence parsing and sentiment analysis

### 2. Distributed representations

Consider the following pair of sentences:

    Paris is the capital of France.
    Berlin is the capital of Germany.

Even assuming you have no knowledge of world geography (or English for that matter), you would still conclude without too much effort that the word pairs (Paris, Berlin) and (France, Germany) were related in some way, and that corresponding words in each pair were related in the same way to each other, that is:

    Paris : France :: Berlin : Germany

Thus, the aim of distributed representations is to find a general transformation function 𝜑
to convert each word to its associated vector such that relations of the following form hold true: 𝜑(𝑃𝑎𝑟𝑖𝑠)−𝜑(𝐹𝑟𝑎𝑛𝑐𝑒)≈𝜑(𝐵𝑒𝑟𝑙𝑖𝑛)−𝜑(𝐺𝑒𝑟𝑚𝑎𝑛𝑦)

In other words, distributed representation aims to convert words to vectors where the similarity between the vectors correlate with the semantic similarity between the words.

The most well-known word embeddings are word2vec and GloVe.

### 3. word2vec
The models are unsupervised, taking as input a large corpus of text and producing a vector space of words. 
Comparing with one-hot enconding:
- dimensionality: lower than one-hot embedding space
- embedding space: more dense than one-hot embedding space

Architectures for word2vec:
- **Continuous Baf of Words (CBOW)**: the model predicts the current word given a window of surrounding words. In addition, the order of the context words does not influence the prediction (that is, the bag of
words assumption).
- **Skip-gram**: the model predicts the surrounding words given the center word. According to the authors, CBOW is faster but skip-gram does a better job at predicting infrequent words.

The **shallow neural networks** with the **word2vec** are the most successful techniques used in NLP.

### 4. GloVe
The **Glo**bal **Ve**ctors for word representation, or GloVe, embeddings was created by Jeffrey
Pennington, Richard Socher, and Christopher Manning (for more information refer to the
article: GloVe: Global Vectors for Word Representation, by J. Pennington, R. Socher, and C.
Manning, Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing
(EMNLP), Pp. 1532–1543, 2013). The authors describe GloVe as an **unsupervised learning algorithm**
for obtaining vector representations for words. **Training is performed on aggregated global wordword
co-occurrence statistics from a corpus**, and the resulting representations showcase interesting
linear substructures of the word vector space. GloVe differs from word2vec in that **word2vec is a predictive model while GloVe is a count-based model.

The first step is to construct a large matrix of (word, context) pairs that co-occur in the
training corpus. Each element of this matrix represents how often a word represented by the row cooccurs
in the context (usually a sequence of words) represented by the column, as shown in the
following figure:

![glove](https://user-images.githubusercontent.com/37953610/57010024-437c7b80-6bf2-11e9-8a90-62c338354c88.JPG)

The GloVe process converts the co-occurrence matrix into a pair of (word, feature) and (feature,
context) matrices. This process is known as matrix factorization and is done using stochastic
gradient descent (SGD), an iterative numerical method. Rewriting in equation form:

$/R = P = Q /approx R'$

Here, $R$ is the original co-occurrence matrix. We first populate $P$ and $Q$ with random values and
attempt to reconstruct a matrix $R'$ by multiplying them. The difference between the reconstructed
matrix $R'$ and the original matrix $R$ tells us how much we need to change the values of $P$ and $Q$ to
move $R'$ closer to $R$, to minimize the reconstruction error. This is **repeated multiple times** until the
**SGD converges** and the reconstruction error is below a specified threshold. At that point, the (word,
feature) matrix is the GloVe embedding. To speed up the process, SGD is often used in parallel mode,
as outlined in the HOGWILD! paper.

The only tool available to do this in Python is the GloVe-Python project (https://github.com/maciejkula/glove-python), which provides a toy implementation for GloVe on Python. So, the GloVe is not as mature than word2vec in Python, and for this reason is noi explored in this notebook.
