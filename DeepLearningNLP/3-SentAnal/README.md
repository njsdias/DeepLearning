### 1. Sentimental Analysis

The objetive here is the implementation of sentimental analysis from the research paper:

- A Structured Self-attentive Sentence Embedding” (https://arxiv.org/pdf/1703.03130.pdf)

which was presented at ICLR 2017 (5th International Conference on Learning Representations)
by a team of research scientists from IBM Watson and the Montreal Institute for Learning Algorithms
(MILA) of the University of Montreal (Université de Montréal).

The paper suggests a new modeling technique to extract an
interpretable sentence embedding, by introducing a **self-attention
mechanism.** The model uses a **two-dimensional matrix to represent the
sentence embedding**, in place of a vector, in which **each of the matrices
represents a different segment of the sentence.** In addition, a self-attention
mechanism and a unique regularization term are proposed.
The embedding method proposed can be visualized easily, to figure out
what specific parts of the sentence ultimately are being encoded into the
sentence embedding. The research conducted shares the performance
evaluation of the proposed model on three different types of tasks.

- Author profiling
- Sentiment classification
- Textual entailment

The model has turned out to be quite promising, compared to other
current sentence-embedding techniques, for all three of the preceding
types of tasks.

## 2. Self-Attentive Sentence Embedding

In the paper concerned uses a new self-attention mechanism that allows it to extract different aspects of the
sentence into multiple vector representations. The matrix structure, with the penalization term, gives the model a greater capacity to disentangle the latent information from the input sentence. 

Moreover, the linguistic structures are not used to guide the sentence
representation model. Additionally, using this method, one can easily
create visualizations that help in the interpretation of the learned
representations.

The **skip-thought vector** is an unsupervised learning where sentences that share
semantic and syntactic properties are mapped to similar vector
representations. For further information related to this, refer to the original
paper, available at https://arxiv.org/abs/1506.06726.
A **paragraph vector** is an unsupervised algorithm that learns fixed-length
feature representations from variable-length pieces of texts, such
as sentences, paragraphs, and documents. The algorithm represents
each document by a dense vector that is trained to predict words in the
document. Empirical results presented in the paper show that paragraph
vectors outperform bag-of-words models, as well as other techniques for
text representations. A more detailed explanation on this is included in the
original research paper, available at https://arxiv.org/abs/1405.4053.

The proposed **attention mechanism** is only performed once, and
it focuses directly on the semantics that make sense for discriminating
the targets. It is less focused on relations between words, but more so
on the semantics of the whole sentence that each word contributes to.
Computation-wise, the method scales up well with the sentence length,
as it doesn’t require the LSTM to compute an annotation vector over all its
previous words.

The proposed sentence-embedding model in “A Structured Self-attentive
Sentence Embedding” consists of two parts:

- Bidirectional LSTM
- Self-attention mechanism

The self-attention mechanism provides a set of summation weight
vectors for the LSTM hidden states. 


The set of summation weight vectors is dotted with the LSTM hidden
states, and the resulting weighted LSTM hidden states are considered as
an embedding for the sentence. It can be combined with, for example, a
multilayer perceptron (MLP), to be applied on a downstream application.

Our aim is to encode a variable-length sentence into a fixed-size
embedding. We achieve that by choosing a linear combination of the n
LSTM hidden vectors in H. Computing the linear combination requires the
self-attention mechanism. The attention mechanism takes all of the LSTM
hidden states _H_ as input and outputs a vector of weights _a_, as follows:

![softmax](https://user-images.githubusercontent.com/37953610/57781803-d0f7b980-7722-11e9-9e59-24034f587f1b.JPG)

Here, Ws1 is a weight matrix with a shape of da-by-2u, and Ws2 is a
vector of parameters with size d_a, where d_a is a hyperparameter we can set
arbitrarily. Because H is sized n-by-2u, the annotation vector _a_ will have
a size _n_. The softmax() ensures all the computed weights add up to 1. We
then add up the LSTM hidden states _H_ according to the weights provided
by _a_, to get a vector representation _m_ of the input sentence.

However, there can be multiple components in a sentence that
together form the overall semantics of it, especially for long sentences.

Thus, we must perform multiple hops of
attention. Say we want _r_ different parts to be extracted from the sentence.
For this, we extend the Ws2 into an r-by-da matrix, note it as Ws2, and the
resulting annotation vector a becomes annotation matrix A.

![matrixA](https://user-images.githubusercontent.com/37953610/57782085-6430ef00-7723-11e9-8069-f017bd3cce32.JPG)

The embedding vector _m_ then becomes an r-by-2u embedding
matrix M which is eqaul to the multiplication the matrix A (annotation) by matrix H (hidden) wich represents the _r_ weighted sum.

However, the embedding matrix M can suffer from redundancy problems. So, for encourage the diversity
of summation weight vectors across different hops of attention it is necessary a penalization term.
To evaluate the diversity is used the _Kullback Leibler divergence_ (KL) between any two of the summation weight vectors.
The KL measures the difference between two probability
distributions over the same variable x. It is related to cross entropy and
information divergence.

However, that is not very stable in this case, as, here, maximization
of a set of KL divergence is being tried (instead of minimizing only one,
which is the usual case), and as optimization of the annotation matrix
A is performed, to have a lot of sufficiently small or even zero values at
different softmax output units, the vast amount of zeros makes the training
unstable.

Thus, a new penalization term is introduced that overcomes the
previously mentioned shortcomings. Compared to the KL divergence
penalization, this term consumes only one-third of the computation.

The dot product of A and its transpose are used, subtracted from an identity
matrix (named as _Frobenius_), as a measure of redundancy.

![penalterm](https://user-images.githubusercontent.com/37953610/57784249-8462ad00-7727-11e9-9bd9-a4d158a0524d.JPG)

This penalization term, P, will be multiplied by a coefficient, and we minimize it, together with the
original loss, which is dependent on the downstream application.

The interpretation of the sentence embedding is quite straightforward,
because of the existence of annotation matrix A. For each row in the
sentence embedding matrix M, its corresponding annotation vector ai is
present. Each element in this vector corresponds to how much contribution
the LSTM hidden state of a token on that position contributes to. Thus, a
heatmap could be drawn for each row of the embedding matrix M.

In resume, the paper introduces a fixed size, matrix sentence embedding with a self-attention
mechanism, where each LSTM hidden state is only expected to provide shorter-term context information
about each word, while the higher-level semantics, which requires
longer term dependency, can be picked up directly by the attention mechanism.



