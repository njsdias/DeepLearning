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
