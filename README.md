<p align="center">
  <img src="assets/origami_logo.jpg" style="width: 100%; height: auto;">
</p>

# ORiGAMi - Object Representation through Generative Autoregressive Modelling 

<p align="center">
| <a href=""><b>ORiGAMi Paper</b></a> | <a href=""><b>ORiGAMi Blog Post</b></a> |
</p>


## Overview

ORiGAMi is a transformer-based Machine Learning model to learn directly from semi-structured data such as JSON
or Python dictionaries. 

Typically, when working with semi-structured data in a Machine Learning context, the data needs to be flattened
into a tabular form first. This flattening can be lossy, especially in the presence of arrays and nested objects, and often requires domain expertise to extract meaningful higher-order features from the raw data. This feature extraction step is manual, slow and expensive and doesn't scale well. 

ORiGAMi is a transformer model and follows the trend of many other deep learning models by operating directly on the raw data and discovering meaningful features itself. Preprocessing is fully automated (apart from some hyper-parameters that can improve the model performance).

### Use Cases

Once an ORiGAMi model is trained on a collection of JSON objects, it can be used in several ways:

1. **Prediction**: ORiGAMi models can predict the value for any key of the dataset. This is different to typical discriminative models such as Logistic Regression or Random Forests, which have to be trained with a particular target key in mind. ORiGAMi is a generative model trained in order-agnostic fashion, and a single trained model can predict any target, given any subset of key/value pairs as input.  
2. **Autocompletion**: ORiGAMi can auto-complete partial objects based on the probabilities it has learned from the training data. This also allows it to predict complex values such as nested objects or arrays.
3. **Generation**: ORiGAMi can generate synthetic mock data by sampling from the distribution it has learned from the training data.
4. **Embeddings**: As a deep neural network, ORiGAMi creates contextualized embeddings which can be extracted at the last hidden layer. These embeddings represent the objects in latent space and can be used as inputs to other ML algorithms, for data visualization or similarity search. 

Check out the Juypter notebooks under [`./notebooks/`](./notebooks/) for examples for each of these use cases.



## Installation

To install ORiGAMi, use

```shell
pip install origami-ml
```


## Usage

ORiGAMi comes with a command line interface (CLI) and a Python SDK. 


### Usage from the Command Line

To train a model, use the `origami train` command. ORiGAMi works well with MongoDB. For example, to train a model on the `shop.orders` collection on a locally running MongoDB instance on standard port 27017, use the following command: 

```
origami train "mongodb://localhost:27017" --source-db shop --source-coll orders
```



### Usage with Python

...TBD...

```python
from origami.model import ORIGAMI
```

