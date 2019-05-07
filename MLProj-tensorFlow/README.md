### 1. Purpose
This repository covers the essential of Tensorflow for Machine Learning with the objective to increase the knowledge of the best approaches in Machine Learning. For that the structure of repository follows the subjects of the book _Building Machine Learning Projects with Tensor Flow_ 

The explanations and the comments as well the code are transcripted from the book. 
So, all rights are attributed to the author of the book.

The main objective is learning with examples well explained. 

Note: Some code can be different from the original only to stay in accordance with the new features released by the new version of Tensor Flow.

### 1. Building Machine Learning Projects with Tensor Flow

This book is for **data analysts, data scientists, and researchers** who want to make the results of
their machine learning activities faster and more efficient. Those who want a crisp guide to
complex numerical computations with TensorFlow will find the book extremely helpful. This
book is also for **developers who want to implement TensorFlow in production** in various
scenarios. Some experience with C++ and Python is expected.

![book](https://user-images.githubusercontent.com/37953610/57300495-d6c02000-70ce-11e9-9d23-8f5d57468c5b.JPG)



### 2. TensorFlow

TensorFlow is an **open source software library for numerical computation** using data flow
graphs. Nodes in the graph represent mathematical operations, while the graph edges
represent the multidimensional data arrays (tensors) passed between them.

The library includes various functions that enable you to implement and explore the cutting
edge **Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN)**
architectures for image and text processing. As the complex computations are arranged in the
form of graphs, TensorFlow **can be used as a framework** that enables you to develop your
own models with ease and use them in the field of machine learning.

It is also **capable of running in** the most heterogeneous environments, from **CPUs to mobile
processors, including highly-parallel GPU computing**, and with the new serving architecture
being able to run on very complex mixes of all the named options.

The name Tensor is from **linear algebra** which is a generalization of of vectors and
matrices. So, in this sense, a vector is a 1D tensor, a matrix is a 2D tensor.
So, a tensor is just a typed, multidimensional array, with additional operations, modeled in the tensor object. 

![tensor_rank](https://user-images.githubusercontent.com/37953610/57301754-a2019800-70d1-11e9-83ae-113b0829c421.JPG)

The TensorFlow documentation uses three notational conventions to describe tensor
dimensionality: **rank, shape, and dimension number**. The following table shows how these
relate to one another:

![relations_rank](https://user-images.githubusercontent.com/37953610/57301808-bba2df80-70d1-11e9-8ba8-beca232f2333.JPG)

In addition to dimensionality, tensors have a fixed **data type**. The data type can be float, integer, string and boolean.
