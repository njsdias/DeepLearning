### 1. Recurrent Neural Network — RNN

Recurrent neural networks (RNN) is a class of neural networks that exploit the sequential nature of their input. 
Such inputs could be:
- text, 
- speech, 
- time series, 
- and anything else where the occurrence of an element in the sequence is dependent on the elements that
appeared before it. 

For example, the next word in the sentence the dog... is more likely to be barks
than car, therefore, given such a sequence, an RNN is more likely to predict barks than car.

An RNN can be thought of as a graph of RNN cells, where **each cell performs the same operation on
every element** in the sequence. RNNs are very flexible and have been used to solve problems such as
-  speech recognition,
- language modeling,
- machine translation, 
- sentiment analysis, 
- and image captioning,to name a few. 

The major limitation of the SimpleRNN cell can be overcomed by two variants of SimpleRNN cell:
- Long Short Term Memory (**LSTM**) 
- Gated Recurrent Unit (**GRU**)

Both LSTM and GRU are drop-in replacements for the SimpleRNN cell, so just
replacing the RNN cell with one of these variants can often result in a major performance
improvement in your network. 

The LSTM and GRu are not the only variants but are the best choices for most of sequence problems
( An Empirical Exploration of Recurrent
Network Architectures, by R. Jozefowicz, W. Zaremba, and I. Sutskever, JMLR, 2015 and LSTM: A
Search Space Odyssey, by K. Greff, arXiv:1503.04069, 2015) 

This chapter covers the following topics:
- SimpleRNN cell
- Basic RNN implementation in Keras in generating text
- RNN topologies
- LSTM, GRU, and other RNN variants

### 2. SimpleRNN cells 
Traditional **multilayer perceptron neural networks** make the assumption that **all inputs are independent
of each other**. This assumption breaks down in the case of sequence data. In Text Mining field we examples where the first two words in the sentence affect the third. The same
idea is true of speech— if we are having a conversation in a noisy room, we can make reasonable
guesses about a word we may not have understood based on the words we have heard so far. Time series
data, such as stock prices or weather, also exhibit a dependence on past data, called the secular trend.

**RNN cells incorporate this dependence by having a hidden state, or memory**, that holds the essence of
what has been seen so far. The value of the hidden state at any point in time is a function of the value
of the hidden state at the previous time step and the value of the input at the current time step, that is

$h_t = \phi (h_{t-1},x_t)$

$h_t$ and $h_{t-1}$ are the values of the hidden states at the time steps $t$ and $t-1$ respectively, and $x_t$ is the
value of the input at time $t$. Notice that the equation is recursive, that is, $h_{t-1}$ can be represented in
terms of $h_{t-2}$ and $x_{t-1}$, and so on, until the beginning of the sequence. **This is how RNNs encode and
incorporate information from arbitrarily long sequences.**

We can also represent the RNN cell graphically as shown in the following diagram on the left. At
time $t$, the cell has an input $x_t$ and an output $y_t$. Part of the output $y_t$ (the hidden state $h_t$) is fed back
into the cell for use at a later time step $t+1$. Just as a traditional neural network's parameters are
contained in its weight matrix, **the RNN's parameters are defined by three weight matrices U, V, and
W, corresponding to the input, output, and hidden state respectively:**

![RNN-structure](https://user-images.githubusercontent.com/37953610/57179693-615b1200-6e78-11e9-8876-11657303ebce.JPG)


Another way to look at an RNN to unroll it, as shown in the preceding diagram on the right. **Unrolling
means that we draw the network out for the complete sequence.** The network shown here is a three-layer
RNN, suitable for processing three element sequences. **Notice that the weight matrices U, V,
and W are shared across the steps. This is because we are applying the same operation on different
inputs at each time step. Being able to share these weight vectors across all the time steps greatly
reduces the number of parameters that the RNN needs to learn.**


We can also describe the computations within an RNN in terms of equations. The internal state of the
RNN at a time $t$ is given by the value of the hidden vector $h_t$, which is the sum of the product of the
weight matrix $W$ and the hidden state $h_{t-1}$ at time $t-1$ and the product of the weight matrix $U$ and the
input $x_t$ at time $t$, passed through the _tanh nonlinearity_. The choice of tanh over other nonlinearities
has to do with its **second derivative decaying very slowly to zero.** This keeps the gradients in the
linear region of the activation function and **helps combat the vanishing gradient problem.** 
**The output vector $y_t$ at time $t$** is the product of the weight matrix $V$ and the hidden state $h_t$, with
**softmax** applied to the product so **the resulting vector** is a **set of output probabilities**:

$h_t = tang (W_{h_{t-1}} + U_{x_t}$
$y_t = softmax (V_{h_t} + U_{x_t}$

Keras provides the SimpleRNN (for more information refer to: https://keras.io/layers/recurrent/) recurrent
layer that incorporates all the logic we have seen so far, as well as the more advanced variants such
as LSTM and GRU. So, it is not strictly necessary to understand
how they work in order to start building with them. However, an understanding of the structure and
equations is helpful when you need to compose your own RNN to solve a given problem.

### 3. RNN topologies
The APIs for MLP (Multi Layer Perceptron) and CNN (Convolutional Neural Network) architectures are limited.Both architectures accept:
- a fixed-size tensor as input 
- and produce a fixed-size tensor as output; 
- and they perform the transformation from input to output in a fixed number of steps given by the number of layers in the model.

**RNNs don't have this limitation**. We can have sequences in the input, the output, or both. This means that RNNs can be
arranged in many ways to solve specific problems. As we have seen, **RNNs combine the input vector with the previous state vector to produce a new state vector.** This property of being able to work with sequences gives rise to a number of common topologies. Some of those are showing here.

![topologies_rnn](https://user-images.githubusercontent.com/37953610/57195531-b0be4280-6f4b-11e9-8637-fa1f17393808.JPG)

The topologie:

- **The (b) RNN topologie** could be used to **machine translation network**, which belongs 
of a general family of networks called **sequence-to-sequence** (for more information refer
to: Grammar as a Foreign Language, by O. Vinyals, Advances in Neural Information Processing
Systems, 2015). These take in a sequence and produces another sequence. In the case of machine
translation, **the input could be a sequence of English words in a sentence** and **the output could be the
words in a translated Spanish sentence**. In the case of a model that uses sequence-to-sequence to do
part-of-speech (POS) tagging, the input could be the words in a sentence and the output could be the
corresponding POS tags. It differs from the previous topology in that at certain time steps there is no
input and at others there is no output. 

- The **one-to-many network shown as (c)**, could be used for **generate a sequence of words by a image as input** (: Deep Visual-Semantic Alignments for Generating Image Descriptions, by A. Karpathy, and F. Li, Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, 2015.).

- The **many-to-one network as shown in (d)** could be a network that does
**sentiment analysis of sentences**, where the input is a sequence of words and the output is a positive or
negative sentiment ( Recursive Deep Models for Semantic
Compositionality over a Sentiment Treebank, by R. Socher, Proceedings of the Conference on
Empirical Methods in Natural Language Processing (EMNLP). Vol. 1631, 2013).

### 4- Vanishing and exploding gradients
Just like traditional neural networks, training the RNN also involves backpropagation. The difference
in this case is that since the parameters are shared by all time steps, **the gradient at each output
depends not only on the current time step, but also on the previous ones**. This process is called
_backpropagation through time_ (**BPTT**) ( Learning Internal Representations by Backpropagating errors, by G. E. Hinton, D. E. Rumelhart, and R. J. Williams, Parallel Distributed Processing: Explorations in the Microstructure of Cognition 1, 1985).

![BPTT](https://user-images.githubusercontent.com/37953610/57195771-55418400-6f4e-11e9-974b-21c470c36d66.JPG)

Consider the small three layer RNN shown in the preceding diagram. During the forward propagation
(shown by the solid lines), the network produces predictions that are compared to the labels to
compute a loss L_t at each time step. During backpropagation (shown by dotted lines), the gradients of
the loss with respect to the parameters _U_, _V_, and _W_ are computed at each time step and the parameters
are updated with the sum of the gradients.
The following equation shows the gradient of the loss with respect to **_W_, the matrix that encodes
weights for the long term dependencies.** We focus on this part of the update because it is the cause of
the vanishing and **exploding gradient problem.** The other two gradients of the loss with respect to the
matrices U and V are also summed up across all time steps in a similar way:

![first-eq](https://user-images.githubusercontent.com/37953610/57195875-32fc3600-6f4f-11e9-8f50-9684012220f9.JPG)

Let us now look at what happens to the gradient of the loss at the last time step (t=3). As you can see,
this gradient can be decomposed to a product of three sub gradients using the chain rule. The gradient
of the hidden state h2 with respect to W can be further decomposed as the sum of the gradient of each
hidden state with respect to the previous one. Finally, each gradient of the hidden state with respect to
the previous one can be further decomposed as the product of gradients of the current hidden state
against the previous one:

![second-eq](https://user-images.githubusercontent.com/37953610/57195921-97b79080-6f4f-11e9-855f-3187bd7b76f8.JPG)

Similar calculations are done to compute the gradient of losses L1 and L2 (at time steps 1 and 2) with
respect to _W_ and to sum them into the gradient update for _W_. We will not explore the math further in
this book. If you want to do so on your own, this WILDML blog post (https://goo.gl/l06lbX) has a very good
explanation of BPTT, including more detailed derivations of the mathematics behind the process.

For our purposes, **the final form of the gradient in the equation above tells us why RNNs have the
problem of vanishing and exploding gradients.** Consider the case where the individual gradients of a
hidden state with respect to the previous one is **less than one.** As we backpropagate across multiple
time steps, **the product of gradients get smaller and smaller**, leading to the **problem of vanishing
gradients**. Similarly, if the gradients are **larger than one, **the products get larger and larger, leading to
the problem of exploding gradients.

The effect of **vanishing gradients** is that the gradients from steps that are far away do **not contribute
anything to the learning process**, so the RNN ends up not learning long range dependencies. Vanishing
gradients can happen for traditional neural networks as well, **it is just more visible in case of RNNs**,
since RNNs tend to have **many more layers (time steps) over which back propagation must occur.**

**Exploding gradients are more easily detectable,** the gradients will become very large and then turn
into not a number (NaN) and the training process will crash. Exploding gradients **can be controlled**
by clipping them at a predefined threshold as discussed in the paper: On the Difficulty of Training
Recurrent Neural Networks, by R. Pascanu, T. Mikolov, and Y. Bengio, ICML, Pp 1310-1318, 2013.

While there are a few approaches to **minimize the problem of vanishing gradients**, such as proper
initialization of the W matrix, **using a ReLU** instead of tanh layers, and **pre-training the layers using
unsupervised methods**, the most popular solution is to use **the LSTM or GRU architectures.** These
architectures have been designed to deal with the vanishing gradient problem and learn long term
dependencies more effectively. 

In resume:

**Vanishing Gradients**
- they do not contribute anything to the learning process
- to solve the problem use LSTM (Long Term Short Memory) or GRU (Gated Recurrent Unit) architectures

**Exploding Gradients**
- the training process will crash
- can be controlled by clipping them at a predefined threshold 

### 5- Long short term memory : LSTM
The LSTM is a variant of RNN that is capable of **learning long term dependencies.** They work well on
a large variety of problems and are the most widely used type of RNN.
We have seen how the SimpleRNN uses the hidden state from the previous time step and the current
input in a tanh layer to implement recurrence. LSTMs also implement recurrence in a similar way,
but instead of a single tanh layer, **there are four layers interacting in a very specific way**. The
following diagram illustrates the transformations that are applied to the hidden state at time step _t_:

![lstm](https://user-images.githubusercontent.com/37953610/57196964-9f7c3280-6f59-11e9-8d7e-373a54ea3e0c.JPG)

Let us look at it component by component:
- **Internal Memory of the Unit:** It is represented by the line across the top of the diagram is the cell state c
- **Hidden State**: it is represented by the The line across the bottom.
- **Mechanish for dealing with the vanishing gradient problem:** It is represented by i: input, f: forget, o: output, and g:internal hidden gate. The values of these gates are updated throughout the training.

Let us consider the equations that show how it calculates the hidden state h_t at time _t_ from the hidden state h_{t-1} at the previous time step:

![third-eq](https://user-images.githubusercontent.com/37953610/57197066-a2c3ee00-6f5a-11e9-9e50-10531d318919.JPG)

Here i, f, and o are the input, forget, and output gates. They are computed using the same equations but
with different parameter matrices. The sigmoid function ($\sigma$) modulates the output of these gates between
zero and one, so **the output vector produced can be multiplied element-wise with another vector to
define how much of the second vector can pass through the first one.**

Functionality of the gates:
- **Forget gate (f):** it defines how much of the previous state h_{t-1} you want to allow to pass through. 
If forget gate is setting to 0 then the old memeory is ignored.
- **Input gate (i):** it defines how much of the newly computed state for the current input x_t you want to let
through. If input gate is setting to 0 then the newly computed state is ignored.
- **Output gate (o):** defines how much of the internal state you want to expose to the next
layer. 
- **Internal hidden state (g):** it is computed based on the current input x_t and the previous hidden
state h_{t-1}. Notice that the equation for g is identical to that for the SimpleRNN cell, but in this case we
will modulate the output by the output of the input gate i.
- **Hidden state (h_t) at time t**: is computed by multiplying the memory c_t with the output gate.

Given i, f, o, and g, we can now calculate the cell state c_t at time _t_ in terms of c_{t-1} at time (t-1)
multiplied by the forget gate and the state g multiplied by the input gate i. **So this is basically a way to
combine the previous memory and the new input**.

One thing to realize is that one only thing the LSTM is different than the SimpleRNN is that LSTMs are resistant to the vanishing gradient problem. You can replace an RNN cell in a network with an LSTM without worrying about any side effects. You should generally see better results along with longer training times.

For additional information related to the LSTM consult the next referencies:
- WILDML blog: has a very detailed explanation of these LSTM gates and how they work
- Christopher Olah's blog post: Understanding LSTMs where he walks you step by
step through these computations, with illustrations at each step (http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 

### 6- Gated recurrent unit: GRU
It was proposed by K. Cho, _Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine
Translation_, arXiv:1406.1078, 2014.

Characteristics:
- very similar to the LSTM 
- retains the LSTM's resistance to the vanishing gradient problem
- internal structure is simpler, and therefore is faster to train

 The gates for a GRU cell are illustrated in the following diagram. The GRU cell has two gates:
 - **update gate (z):** it defines how much previous memory to keep around.
 - **reset gate (r):** it defines how to combine the new input with the previous memory.
 
 Notice that, there is no persistent cell state distinct from the hidden state as in LSTM.
 
 ![gru](https://user-images.githubusercontent.com/37953610/57197372-64303280-6f5e-11e9-98fb-3cd0d9ebba83.JPG)
 
The following equations define the gating mechanism in a GRU:

![forth-eq](https://user-images.githubusercontent.com/37953610/57197456-6b0b7500-6f5f-11e9-8f1f-5ea448d32b96.JPG)
 
According to several empirical evaluations GRU and LSTM have comparable performance and
there is no simple way to recommend one or the other for a specific task. While GRUs are faster to
train and need less data to generalize, in situations where there is enough data, an LSTM's greater
expressive power may lead to better results. Like LSTMs, GRUs are drop-in replacements for the
SimpleRNN cell. ( R. Jozefowicz, W. Zaremba, and I. Sutskever, 2015, _An Empirical
Exploration of Recurrent Network Architectures_, JMLR, 2015 and J. Chung, 2014, _Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling_, arXiv:1412.3555. 2014))

