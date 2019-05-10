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

-Text Classifier: Text can be classified into various classes, such as positive and negative.
_TextBlob_ offers many such architectures.
