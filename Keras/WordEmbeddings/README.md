### 1. Word Embeddings: GloVe and word2vec

Word embeddings are a way to transform words in text to numerical vectors so that they can be analyzed by standard machine learning algorithms that require vectors as numerical input.

**One-hot encoding** is the most basic embedding approach. One-hot encoding represents a word in the text by a vector of the size of the vocabulary, where only the entry corresponding to the word is a one and all the other entries are zero. A major problem with one-hot encoding is that there is no way to represent the similarity between words [(cat, dog), (knife, spoon)]. Similarity between vectors is computed using the dot product, which is the sum of element-wise multiplication between vector elements. In the case of one-hot encoded vectors, the dot product between any two words in a corpus is always zero.

To overcome the limitations of one-hot encoding, the NLP community has borrowed techniques from information retrieval (IR) to vectorize text using the document as the context. Notable techniques are TF-IDF (https://en.wikipedia.org/wiki/Tf%E2%80%93idf), latent semantic analysis (LSA) (https://en.wikipedia.org/wiki/Latent_semantic_analysis), and topic modeling (https://en.wikipedia.org/wiki/Topic_model).

Word embedding differs from previous IR-based techniques in that they use words as their context, which leads to a more natural form of semantic similarity from a human understanding perspective. Today, word embedding is the technique of choice for vectorizing text for all kinds of NLP tasks, such as text classification, document clustering, part of speech tagging, named entity recognition, sentiment analysis, and so on.

This notebook covers the GloVe and word2vec that are the word embedding tecniques that have proven more effective and have been widely adopted in the deep learning and NLP communities. For that the topics cover will be:

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
