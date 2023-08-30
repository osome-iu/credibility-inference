## Core concepts 
The Infopolluters include Population and Transformations. At a high level, a Population represents a collection of users, while a Transformation represents some action or computation you can do to this set of users. This page covers the details of populations and transformations and how you can use them.

### Population
We want a structure to:
- Keep track of user information (different metadata of users can be used for classification later, e.g: Pagerank score, embedding vector) 
- Keep track of the masked users for evaluation 

### Transformer 

Stand-alone Population objects are merely large containers of data. However features of these objectts allow us to add transformation to the users and evaluate the results, using the Transformer class. 

At a high level, a Transformer is an object that takes in a Population and gives back the same Population with some modifications done to it. In almost all cases, these modifications will take the form of changed or added metadata. For example, one kind of Transformer is the TweetEmbedding, which calculates an embedding for a user based on all the tweets they post. When you run the TweetEmbedding on a Population, it adds to each User a metadata entry called "embedded_text", whose value is the mean-pooled of all tweet embeddings. The modified Population is then returned so you can continue to do other things with it (including running other Transformers or classfication).

This design is inspired by ConvoKit and Scikit-learn 
Implementation-wise, Transformer is an `abstract class <https://docs.python.org/3/library/abc.html>`_ - that is, you
cannot directly create a Transformer object. Instead, specific Population manipulations are coded as individual classes,
each one `inheriting <https://docs.python.org/3/tutorial/classes.html#inheritance>`_ from Transformer.
If you are not super familiar with Python inheritance, don't worry - all you need to know is that each manipulation of a
Population is represented as an individual class, but these classes all "look the same" in that they have the same basic set
of functions. Specifically, they all contain a ``fit`` function and a ``transform`` function. The ``fit`` function is
used to prepare/train the Transformer object with any information it needs beforehand; for example, a Transformer that
computes bag-of-words representations of Utterances would first need to build a vocabulary. The ``transform`` function,
as its name implies, is the function that actually runs the Transformer. So in the TweetEmbedding example, to actually apply
the TweetEmbedding to a Population, you would run::

    parser.transform(Population)

Where ``parser`` is a TweetEmbedding object and ``Population`` is a Population object.

A single Transformer on its own might not do much, but because Transformers return the modified Population, you can chain
multiple Transformers together to get different user features, that can be used as inputs to classification task. For instance, after you have applied the TweetEmbedding to
your Population, you can take the modified Population and run another Transformer on it that uses the parses to perform some
more complicated task, like named entity recognition. In general, the code for chaining together arbitrary numbers of
Transformers takes the following form::

    # Assume that transformer1,transformer2,... have been previously initialized as instances of Transformer subclasses
    
    base_Population = Population(...)

    Population1 = transformer1.transform(base_Population)
    Population2 = transformer2.transform(Population1)
    Population3 = transformer3.transform(Population2)
    # ...and so on

All of the classification functionality of ConvoKit (computing linguistic coordination, finding question types, etc.) is implemented as Transformers.