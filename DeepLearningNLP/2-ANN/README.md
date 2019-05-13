### 1. Deep Learning for NLP

Traditional neural networks lack the ability to capture knowledge from
previous events and pass it on to future events and related predictions. In
this chapter, we will introduce a family of neural networks that can help us
in persisting information over an extensive period.

In deep learning, all problems are generally classified into two types:

- Fixed topological structure: For images having static
data, with use cases such as image classification - using convolution neural
networks (CNNs)
- Sequential data: For text/audio with dynamic data, in
tasks related to text generation and voice recognition - using recurrent neural networks (RNNs: long short-term
memory (LSTM) methods)

RNNs and LSTM networks have applications in diverse fields, including:

- Chatbots
- Sequential pattern identification
- Image/handwriting detection
- Video and audio classification
- Sentiment analysis
- Time series modeling in finance

### 3. Recurrent Neural Networks

In RNNs the hidden states are: capable of performing such operations for two principal

- distributive by nature, store a lot of
past information and pass it on efficiently.
- updated by nonlinear methods.

By definition Recurrence is a recursive process in which a recurring function is called at each step to model the sets of temporal data.
A temporal data is any unit of data that is dependent on the
previous units of the data, particularly sequential data.
For example, a company’s share price is dependent on the prices of the share on
previous days/weeks/months/years, hence, dependence on previous
times or previous steps is important, thereby making such types of models
extremely useful.

So, the next time you see any type of data having a temporal pattern,
try using the types of models being discussed in the subsequent sections of
this chapter.

In a normal feddfoward network data is fed to it discretely, without
taking account of temporal relations. Such types of networks are useful for
discrete prediction tasks, as the features aren’t dependent on each other
temporally. 

RNNs reveals that the feedforward neural network takes decisions
based only on the current input, and an RNN takes decisions based on the
current and previous inputs and makes sure that the connections are built
across the hidden layers as well.

So, the feedforward neural networks are unsuitable for sequences, time series data, video
streaming, stock data, etc; and do not bring memory factor in modeling.

The next figure shows the conceptual differences betwwen these two types of ANN.

![fowrd_rnn](https://user-images.githubusercontent.com/37953610/57638951-c1ebfc80-75a6-11e9-96a8-85355241fd9d.JPG)

We are supposed to make use of the proper language by using its
grammar, which makes up the base rules of the language. Traditionally,
NLP tasks are extremely difficult, because of the vastness of the grammar
of different languages.
Hard-coding of the constraints with respect to each of the languages
has its own drawbacks. The Deep Learning algorithms can help us in this tasks. 
It learns the complex local structural formulation of all the languages
and uses this learning to crack the complexities present in the problem set.

RNN takes each of the words one by one and then aims to classify the given text using
word embedding: CBOW ans Skip-gram. The word2vec models aim to initialize random vectors for each word,
which are further learned to have meaningful vectors, to perform specific
tasks. The word vectors can be formed of any given dimension and are able
to encapsulate the information accordingly.

Let’s try to understand the functional process of RNNs.

![unf_rnn](https://user-images.githubusercontent.com/37953610/57639618-3a9f8880-75a8-11e9-83b6-ec2e85dd461b.JPG)

Next we describe the meaning of the variables:

- _X_t_ is input at time step t.
- _S_t_ is the hidden state at time step t. It’s the “memory”
of the network and is calculated based on the previous
hidden state and the input at the current step.
- _U_{xh} is mapping from input (x) to hidden layer (h),
hence, h × dimension (x), where the dimension of x is
the dimension of each time step input (1, in the case
of a binary summation). Refer to the U matrix in the
preceding figure.
- _W_{hh} is mapping across hidden states, hence, h × h. Refer
to the W matrix in the preceding figure.
- V_{hy} is mapping from the final hidden layer to output
y. Hence, h x dimension (y), where the dimension of
y is the dimension of the output (20, in the case of the
binary summation case considered previously). Refer
to the V matrix in the preceding figure.
- _o_t_ is the output at step t. For example, if we wanted to
predict the next word in a sentence, it would be a vector
of probabilities across our vocabulary.

