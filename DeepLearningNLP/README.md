### 1. Information

This repository is regarding to Deep Learning following the book

![cover_book](https://user-images.githubusercontent.com/37953610/57534810-c90dd300-7338-11e9-90a9-6554f77be3b9.JPG)

### 2. Introduction 
Natural language processing (NPL) is an extremely difficult task in computer science. Natural language processing, in its simplest form, is the ability for a computer/system to truly understand human language and process it in the same way that a human does.

Languages present a wide variety of problems that vary from language to language. Structuring or extracting meaningful information from free text represents a great solution, if done in the right manner. Previously, computer scientists broke a language into its grammatical forms, such as parts of speech, phrases, etc., using complex algorithms. Today, deep learning is a key to performing the same exercises.

One of the things that turns the NLP difficult is the **ambiguity in sentences.** This may be at the word level, at the sentence level, or at the meaning level.

**Some common applications of NLP**

- Text Summarization
- Text Tagging
- Named Entity Recognition
- Chatbot
- Speech Recognition

**Common Terms Associated with Language Processing**

- Phonetics/phonology: The study of linguistic sounds and their relations to written words
- Morphology: The study of internal structures of words/composition of words
- Syntax:The study of the structural relationships among words in a sentence
- Semantics: The study of the meaning of words and how these combine to form the
meaning of sentences
- Pragmatics: Situational use of language sentences
- Discourse: A linguistic unit that is larger than a single sentence (context)

- Stemming: means removing affixes from words and returning the root word (which may not be a real word)
- Lemmatizing: is similar to stemming, but the difference is that the result of lemmatizing is a real word.

**Textblob**
It provides a simple API for diving deep into common NLP tasks, such as part-of-speech tagging, noun
phrase extraction, sentiment analysis, classification, and much more. The package offers language a translation module.
We can also use TextBlob for 

- tagging purposes: it is the process of denoting a word in a text (corpus) as corresponding to a particular part of speech.
- to deal with spelling errors

**SpaCy**

Provides very fast and accurate syntactic analysis (the fastest of any library released) and also offers named entity recognition and ready access to word vectors. **Entity recognition** is the process used to classify multiple
entities found in a text in predefined categories, such as a person, objects, location, organizations, dates, events, etc. **Word vector** refers to the mapping of the words or phrases from vocabulary to a vector of real numbers. SpaCy also offers dependency parsing, which could be further utilized to extract noun phrases from the text.

**Gensim**

It is used primarily for topic modeling and document similarity.
Gensim is most useful for tasks such as getting a word vector of a word.

Gensim offers LDA (latent dirichlet allocation), which allows

- estimation from a training corpus and
- inference of topic distribution on new, unseen documents. 

The model can also be updated with new documents for online training.

**Pattern**

It is useful for a variety of NLP tasks, such as part-of-speech taggers, n-gram searches, sentiment
analysis, and WordNet and machine learning, such as vector space
modeling, k-means clustering, Naive Bayes, K-NN, and SVM classifiers.

**Stanford CoreNLP**

Stanford CoreNLP (https://stanfordnlp.github.io/CoreNLP/) provides
the base forms of words; their parts of speech; whether they are names of
companies, people, etc.; normalizes dates, times, and numeric quantities;
marks up the structure of sentences in terms of phrases and syntactic
dependencies; indicates which noun phrases refer to the same entities;
indicates sentiment; extracts particular or open-class relations between
entity mentions; gets the quotes people said; etc.

### 3. Getting Started with NLP

- Text Search Using Regular Expressions: Regular expressions are a very useful means of searching for a particular
type of design or wordset from a given text. A regular expression (RE)
specifies a set of strings that match it. The functions in this module allow
you to check if a given string matches a particular RE (or if a given RE
matches a particular string, which comes down to the same thing).

- Text to List: You can read a text file and convert it into a list of words or list of sentences,
according to your needs.

- Preprocessing the Text: There is a large number of things you could do to preprocess the text. For
example, replacing one word with another, removing or adding some
specific type of words, etc.

- Accessing Text from the Web: A text file from a URL can be accessed using urllib.

- Removal of Stopwords: A stopword is a commonly used word (such as the) that a search engine has been programmed to ignore.

- Counter Vectorization: Counter vectorization is a SciKit-Learn library tool that takes any mass of
text and returns each unique word as a feature, with a count of the number
of times a particular word occurs in the text.

- **TF-IDF Score**: TF-IDF is an acronym of two terms: term frequency and inverse document
frequency. TF is the ratio representing the count of specific words to the
total number of words in a document. Suppose that a document contains
100 words, wherein the word happy appears five times. The term frequency
(i.e., tf ) for happy is then (5/100) = 0.05. IDF, on the other hand, is a
log ratio of the total number of documents to a document containing a
particular word. Suppose we have 10 million documents, and the word
happy appears in 1,000 of them. The inverse document frequency (i.e., idf ),
then, would be calculated as log (10,000,000/1,000) = 4. Thus, the TF-IDF
weight is the product of these quantities: 0.05 × 4 = 0.20.

- BM25, which is used to score a document on the basis of its relation to a query. BM25 ranks a
set of documents using the query terms of each of the documents,
irrespective of the relationship between the keywords of the query
within a document.

- Text Classifier: Text can be classified into various classes, such as positive and negative.
_TextBlob_ offers many such architectures.

### 4.Introduction to Deep Learning
The collection of algorithms implemented under deep learning have
similarities with the relationship between stimuli and neurons in the
human brain. 
Neural networks are a biologically inspired paradigm (imitating the
functioning of the mammalian brain) that enables a computer to learn
human faculties from observational data. They currently provide solutions
to many problems: image recognition, handwriting recognition, speech
recognition, speech analysis, and NLP.

A deep neural network is simply a feed forward neural network with multiple
hidden layers. If there are many layers in the network, then we say that the network is
deep.

### 4.1 Basic Structure of Neural Networks

The basic principle behind a neural network is a collection of basic
elements, artificial neuron or perceptron, that were first developed in the
1950s by Frank Rosenblatt. They **take several binary inputs**, x1, x2, ..., xN
and produce a **single binary output** if the sum is greater than the activation
potential. The neuron is said to **“fire”** whenever activation potential is
exceeded and **behaves as a step function**. The neurons that fire pass along
the signal to other neurons connected to their dendrites, which, in turn,
will fire, if the activation potential is exceeded, thus producing a cascading
effect.

As not all inputs have the same emphasis, **weights are attached** to each
of the inputs, x_i to allow the model to assign more importance to some
inputs. Thus, output is 1, if the weighted sum is greater than activation
potential or bias, i.e.,

![outputbias](https://user-images.githubusercontent.com/37953610/57540906-bf3e9c80-7345-11e9-8ea7-cc2e2ed09ea8.JPG)

**Sigmoid**
A modified form was created to behave more predictably, i.e., small changes in weights and bias cause only a small change in output.
There are two main modifications.

- The inputs can take on any value between 0 and 1, instead of being binary.
- To make the output behave more smoothly for given
inputs, x1, x2, …, xN, and weights. w1, w2, …, wN, and bias,
b, use the sigmoid function:

![sigmoidfunc](https://user-images.githubusercontent.com/37953610/57541133-3ffd9880-7346-11e9-9ca0-e602f2f9a48e.JPG)

![sigmoidplot](https://user-images.githubusercontent.com/37953610/57541489-1002c500-7347-11e9-9a11-200dab4ebf64.JPG)

The smoothness of the exponential function, or σ, means that small
changes in weights and bias will produce a small change in the output
from the neuron (the change could be a linear function of changes in
weights and bias).

**Rectified linear unit (ReLU)**
This keeps the activation guarded at zero. But this function shows some problems:
- when the learning rate is set to a higher value, as this triggers weight updating that doesn’t
allow the activation of the specific neurons, thereby making the gradient
of that neuron forever zero.
- the explosion of the activation function, as the input value, xj, is itself the output here

**LReLUs (Leaky ReLUs)**
These mitigate the issue of dying ReLUs by introducing a marginally reduced
slope (~0.01) for values of x less than 0. LReLUs do offer successful scenarios, although not always.

**ELU (Exponential Linear Unit)** These offer negative
values that push the mean unit activations closer to
zero, thereby speeding the learning process, by moving
the nearby gradient to the unit natural gradient. For a
better explanation of ELUs, refer to the original paper
by Djork-Arné Clevert, available at https://arxiv.
org/abs/1511.07289.

**Softmax** Also referred to as a normalized exponential
function, this transforms a set of given real values in
the range of (0,1), such that the combined sum is 1.

The layers between input and output are referred to as hidden
layers, and the density and type of connections between layers is the
configuration.

![MLPpict](https://user-images.githubusercontent.com/37953610/57542434-97513800-7349-11e9-8fa5-556935bc4df5.JPG)

References: 

- articles published by Geoffrey Hinton: (www.cs.toronto.edu/~hinton/)
- http://deeplearning.net/

### 4.2 Types of Neural Networks**
For neural networks to learn in a faster and more
efficient way, various neurons are placed in the network in such a way as to
maximize the learning of the network for the given problem. 
If the neurons are placed with connections among them taking
the form of cycles, then they form networks such as **feedback, recursive,
or recurrent neural networks.** If, however, the connections between the
neurons are acyclic, they form networks such as **feedforward neural
networks.**

**Feedforward Neural Networks**
Data movement in any feedforward neural network is
from the input layer to output layer, via present hidden layers, restricting
any kind of loops. Output from one layer serves as input
to the next layer, with restrictions on any kind of loops in the network
architecture.

![feddforwardNN](https://user-images.githubusercontent.com/37953610/57543048-290d7500-734b-11e9-980f-5109a07ec240.JPG)

**Multilayer Perceptrons**
Multilayer perceptrons (MLPs) belong to the category of feedforward
neural networks and are made up of three types of layers: an input layer,
one or more hidden layers, and a final output layer. A normal MLP has the
following properties:
- An input layer using linear functions
- Hidden layers with any number of neurons that using activation function, such as
sigmoid
- Output layer that use  an activation function giving any number of outputs

As the output given by an MLP depends only on the current input
and not on past or future inputs, so MLPs are considered apt for resolving
**classification problems**.

Following are a few of the features of network architecture that have a
direct impact on its performance:

- **Hidden layers:** These contribute to the generalization factor of the network. In most cases, a single layer is sufficient to encompass the approximation of any desired function, supported with a sufficient number of neurons.
- **Hidden neurons:** The number of neurons present across the hidden layer(s) that can be selected by using any kind of formulation. A basic rule of thumb is to select count between one and a few input units. Another means is to use cross-validation and then check the plot between the number of neurons in the hidden layer(s) and the average mean squared error (MSE) with respect to each of the combinations, finally selecting the combination with the least MSE value. It also depends on the degree of nonlinearity or the initial problem dimensionality. It is, thus, more of an adaptive process to add/delete the neurons.
- **Output nodes:** The count of output nodes is usually equal to the number of classes we want to classify the target value.
- **Activation functions:** These are applied on the inputs of individual nodes. A set of nonlinear functions are used to make the
output fall within a desired range, thereby preventing the paralysis of the network. In addition to the nonlinearity, the continuous differentiability of these functions helps in preventing the inhibition of the training of neural networks.

**Convolutional Neural Networks**
Convolutional neural networks are well adapted **for image recognition and
handwriting recognition.** Their structure is based on sampling a window
or portion of an image, detecting its features, and then using the features
to build a representation. As is evident by this description, this leads to
the use of several layers, thus these models were the first deep learning
models.

**Recurrent Neural Networks**

Recurrent neural networks are used when a data pattern changes over time.  An RNN applies the same layer to the input at each time step, using the output (i.e., the state of previous time steps as inputs). RNNs have feedback loops in which the output from the previous firing or time index T is fed as one of the inputs at time index T + 1. 

There might be cases in which the output of the neuron is fed to itself as input. As these are well-suited for applications involving sequences, they are widely used in problems related to videos, which are a time sequence of images, and for **translation purposes**, wherein understanding the next word is based on the context of the previous text. 

There are various types of RNNs:

- Encoding recurrent neural networks: This set of RNNs enables the network to take an input of the sequence form (Figure 1-12).

- Generating recurrent neural networks: Such
networks basically output a sequence of numbers or
values, like words in a sentence.

- General recurrent neural networks: These
networks are a combination of the preceding two
types of RNNs. General RNNs are used
to generate sequences and, thus, are widely used in
NLG (natural language generation) tasks.

**Encoder-Decoder Networks**
These networks use one network to create an internal
representation of the input, or to “encode” it, and that representation is
used as an input for another network to produce the output. This is useful
to go beyond a classification of the input. The final output can be in the
same modality, i.e., **language translation**, or a different modality, e.g., **text
tagging of an image**, based on concepts. For reference, one can refer to the
paper “Sequence to Sequence Learning with Neural Networks,” published
by the team at Google: (https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf).

**Recursive Neural Networks**
In a recursive neural network, a fixed set of weights is
recursively applied onto the network structure and is primarily used
to discover the hierarchy or structure of the data. Whereas an RNN is a
chain, a recursive neural network takes the form of a treelike structure.
Such networks have great use in the field of NLP, such as **to decipher
the sentiment of a sentence.** The overall sentiment is not dependent on
the individual works only, but also on the order in which the words are
syntactically grouped in the sentence.


### 4.3 Stochastic Gradient Descent
It is an iterative algorithm that **minimizes a loss function** by subsequently updating the parameters of the function.

### 4.4 Backpropagation
Understanding this algorithm would definitely give you insights into problems related to deep learning:
- learning problems
- slow learning
- exploding gradients
- diminishing gradients.

The Backpropagation help us to find the partial derivative of the loss with respect to each weight and it solves the problem of slowness of Gradient Descent. This algorithm is the workhorse of the training procedure for every deep learning algorithm. More detailed information can be found here: www.cs.toronto.edu/~hinton/backprop.html.

Backpropagation consists of two parts see the next figure):
- Forward pass, wherein we initialize the weights and
make a feedforward network to store all the values
- Backward pass, which is performed to have the
stored values update the weights

![backpropNN](https://user-images.githubusercontent.com/37953610/57545068-1c8c1b00-7351-11e9-93a4-0b2e1fb4a9f6.JPG)

Initially, all the edge weights are randomly assigned. For every input in
the training dataset, the ANN is activated, and its output is observed. This
output is compared with the desired output that we already know, and the
error is “propagated” back to the previous layer. This error is noted, and
**the weights are “adjusted” accordingly**. This process is repeated until the
output error is below a predetermined threshold.
Once the preceding algorithm terminates, we have a “learned” ANN,
which we consider to be ready to work with “new” inputs. This ANN is said
to have learned from several examples (labeled data) and from its mistakes
(error propagation).

### 5. Deep Learning Libraries
In this section we will highlight only the main featue of the libraries.

**Theano**

It is a numerical
computation library for Python with syntaxes similar to NumPy. It
is efficient at performing complex mathematical expressions with
multidimensional arrays. This makes it is a perfect choice for neural
networks. The link http://deeplearning.net/software/theano will give the
user a better idea of the various operations involved.

Many tools have been implemented on top of Theano. Principally, it
includes
• Blocks http://blocks.readthedocs.org/en/latest/
• Keras http://keras.io/
Lasagne http://lasagne.readthedocs.org/en/
latest/
• PyLearn2 http://deeplearning.net/software/
pylearn2/


**TensorFlow**

TensorFlow is an open sourced library by Google for large-scale machine
learning implementations. It has been designed in such a way that
computations on CPUs or GPU systems across a single desktop or servers
or mobile devices are catered to by a single API.

**Keras**

Keras is a highly modular neural networks library, which runs on top of
Theano or TensorFlow. Keras is one of the libraries which supports both
CNNs and RNNs and runs effortlessly on GPU and CPU.
A model is understood as a sequence or a graph of standalone,
fully configurable modules that can be plugged together with as little
restrictions as possible. In particular, neural layers, cost functions,
optimizers, initialization schemes, activation functions, regularization
schemes are all standalone modules that could be combined to create new
models.
