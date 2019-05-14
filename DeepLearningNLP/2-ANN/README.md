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

### 4. LSTM and GRU to solve problems of RNN

RNNs have a very peculiar problem, and it is faced because,
as the weights are propagated through time, they are multiplied recursively
in the preceding functions, thereby giving rise to the following two types of
scenarios:

- **Vanishing gradient:** If the weights are small, the
subsequent values will keep on getting smaller and
tend to ~0.

- **Exploding gradient:** If the weights are big, the final
value will approach infinity.

This occurs during backpropagation, i.e., while calculating the gradient to update
the weights, because it involves cascading of partial derivatives, and each
of the partial derivatives involves a σ term, i.e., a sigmoid neural net layer.
As the value of each of the sigmoid derivatives might become less than 1,
thereby making the overall gradient values small enough that they won’t be
able to further update the weights, that means the model will stop learning!

To solve this problem, a specific type of hidden layer, called a long
short-term memory (**LSTM**) network, and gated rectified units (**GRUs**)
are used.

An RNN can be trained in multiple ways. The output layer is the layer which give us the end results. 
This output layer can be builded as:

- take the output the last time step
- take the output of all time steps
- take the average of time steps

![multi_train_RNN](https://user-images.githubusercontent.com/37953610/57702017-12219800-7655-11e9-9c76-62ab686cc10a.JPG)

However LSTM are not perfect and they are usually slower
than other typical models. Additional with careful initialization and training, even
an RNN can deliver results similar to those of LSTM, and, too, with less
computational complexity. 
To solve the problems of LSTM networks a new mechanism named **Attention Mechanism** is growing in popularity.

**4.1. Tunning RNNs** 

RNNs are highly critical to input variables and are very receptive in nature.
A few of the important parameters in RNNs that play a major role in the
training process include the following:

- Number of hidden layers
- Number of hidden units per layer (usually chosen as same number in each layer)
- Learning rate of the optimizer
- Dropout rate (initially successful dropout in RNNs is applied to feedforward connections only, not to recurrent ones)
- Number of iterations

To check the performance of an ANN  we can:

- plot the output with validation curves and learning
curves and check for overfitting and underfitting.
- Training and testing the error at each split should be plotted, and according to the problem
we check, if it is an overfit, then we can decrease the number of hidden
layers, or hidden neurons, or add dropout, or increase the dropout rate,
and vice versa.

**4.2 Long Short-Term Memory (LSTM)**
A LSTM is a network which solves the problem of retaining information
for RNNs over longer time periods (www.bioinf.jku.at/publications/
older/2604.pdf). More specifically, it solves the gradient vanishing or gradient explosion
problem, by introducing additional gates, input and forget gates, which
allow for a better control over the gradient, by enabling what information
to preserve and what to forget, thus controlling the access of the
information to the present cell’s state, which enables better preservation of
“long-range dependencies.”

The strucutre of an LSTM network is represented in the next figure, with four interacting layer.
They have a sigmoid neural net layer, with output in [0,1] to weigh the passing limit of the component,
and a point-wise multiplication operation.
In the figure, Ci is the cell state, which is present across all
the time steps and is changed by the interactions at each of the time steps.
To retain the information flowing across an LSTM via the cell state, it has
three types of gates:

- **Input gate:** To control the contribution from a new input to the memory
- **Forget gate:** To control the limit up to which a value is pertained in the memory
- **Output gate:** To control up to what limit memory contributes in the activation block of output

![lstm_comp](https://user-images.githubusercontent.com/37953610/57703204-79d8e280-7657-11e9-8fa9-5037312721bd.JPG)

**4.3 Gated Rectified Units (GRUs)**

The GRUs is a variation of LSTM. The next picture highlight the differences.

![lstm_gru](https://user-images.githubusercontent.com/37953610/57704116-35e6dd00-7659-11e9-924d-c5b06c6dc395.JPG)

A GRU controls the flow of information like an LSTM unit but without
having to use a memory unit. It just exposes the full hidden content
without any control. Because of this the GRUs are models which usually
have lesser complexity, compared to the standard LSTM models.
It can be the reason for the rule of thumb that says the LSTM works better for bigger datasets,
while GRU works better for smaller datasets. 

## 5. Sequence-to-Sequence Models

A seq2seq model consists of two separate RNNs:

- **Encoder:**  An encoder takes the information as input in
multiple time steps and encodes the input sequence into a context vector. Bidirectional LSTMs are generally work better than anything else for almost each of the NLP tasks. The more we add bidirectional LSTMs layers, the better the result. For this is common see stacked Bidirectional Enconder.
- **Decoder:** The encoder outputs the context vector, which offers a snapshot of the
entire sequence occurring before. The context vector is used to predict the
output, by passing it to the decoder. So, the decoder takes that hidden state and decodes it into the desired output sequence.In the decoder, we have a dense layer with softmax, just as in a normal
neural network, and it is time-distributed, which means that we have one
of these for each time step.

These models are used for everything from chatboots to:

- speech-to-text 
- dialog systems 
- QnA to image captioning.

The key thing with seq2seq models is that the sequences preserve the
order of the inputs, which is not the case with basic neural nets. There’s
certainly no good way to represent the concept of time and of things
changing over time, so the seq2seq models allow us to process information
that has a time, or an order of time, element attached to it. They allow us to
preserve information that couldn’t be by a normal neural network.

The key task behind a seq2seq model is to convert a sequence into a
fixed size feature vector that encodes only the important information in the
sequence, while losing the unnecessary information.

Let’s consider the example of a basic question-and-answer system,
in which the question is “How are you?” In this case, the model takes the
sequence of words as input, so we are going to try to get each word in the
sequence into a fixed-size feature vector that can then be used to predict
the output, by model, for a structure answer. The model must remember
the important things in the first sequence and also lose any unnecessary
information in that sequence, to produce the relevant answers.

![seq2seq](https://user-images.githubusercontent.com/37953610/57705451-db02b500-765b-11e9-8e2b-8b5c0fbc6601.JPG)

## 6. Attention (Mechanism) Scoring

Basic seq2seq models work well for normal tasks on short sentences. 
normal LSTMs can remember about 30 time steps and start to drop off very quickly after it. 
The attention mechanisms perform better on the short-term length sequences and can reach a maximum length of about 50 time
steps. However, until now, thare are not models that can really go back in time and remember even a few
paragraphs.

Attention models look at the whole content shown and work out ways to
figure out which word is most important for each of the words in the text.
So, it sort of gives a score to every word in your sentence, and with that, it
is able to get a sense that there are certain words that rely on some words a
lot more than other ones.

The best way to understand
attention models is to think of them as kind of a little memory module
that basically sits above the network and then looks at the words and picks
the ones that are most important.

For example, in the second sentence the italic words were selected as the most important words in the sentence:

- Last month everyone went to the club, but I stayed at home.
- Last _month_ everyone went to the _club_, but I _stayed_ at home.

This helps in translation to different languages and for retaining context information as
well, such as the event happened “last month,” as this time information is required while doing the NLP tasks.

An attention vector (shown in Figure) helps in increasing the model’s performance, by capturing the information from the overall
input sentence at each of the steps of the decoder. This step makes sure that the decoder is not dependent only on the last decoder state but also on the combined weights of all the input states. The best technique is to use bidirectional LSTMs, along with attention on top of it, in the encoder.

![atten_score](https://user-images.githubusercontent.com/37953610/57708289-fcb26b00-7660-11e9-9f64-5483ba9f4ab7.JPG)

## 7. Teaching Forcing

To understand the process, as we train the teacher forcing model, while
doing the prediction part, we check whether every word predicted is right
and use this information while backpropagating the network. However, we
don’t feed the predicted word to the next time steps. Instead, while making
every next word prediction, we use the correct word answer of last time
step for next time step prediction. That’s why the process is called “teacher
forcing.” We are basically forcing the decoder part to not only use the
output of the last hidden state but to actually use the correct answers. This
improves the training process for **text generation** significantly. This process
is not to be followed while doing the actual scoring on the test dataset.
Make use of the learned weights for scoring step. The teacher forcing technique was developed as an alternative to
backpropagation-through-time for training an RNN.
