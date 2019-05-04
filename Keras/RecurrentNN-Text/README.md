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

$h_t = \phi (h_{t-1},x_t$

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

