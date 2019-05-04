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
— Long Short Term Memory (**LSTM**) 
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