# NLP Assignment: Aspect-Term Polarity Classification in Sentiment Analysis

## Student Names
The names of the students who contributed to the deliverable are:
- Bouthaina HOUASS
- Carlos SANTOS GARCIA
- Axel DARMOUNI
- Mathilde LARCHEVÊQUE

## Description of the implemented classifier 

The classification model that we used is a DistilBERT model. 

### Text pre processing 

The first step of the classification process is to transform the aspect categories into phrases: for example:
```
"LOCATION#GENERAL"     --->    "General information about location."
"RESTAURANT#GENERAL"   --->    "General information about the restaurant."
```

Then the target term is also transform into a sentence:
```
"restaurant"           --->    "Target term: restaurant."
"food"                 --->    "Target term: food."
```

Finaly the three sentences are concatenated:
```
"RESTAURANT#GENERAL | trattoria | This quaint and romantic trattoria is at the top of my Manhattan restaurant list."
                       ---> 
"General information about the restaurant. Target term: trattoria. This quaint and romantic trattoria is at the top of my Manhattan restaurant list."
```

These transformed sentences, which contains the aspect category, the target term and the sentence are given as input to the tokenizer.

### Input and Feature representation

The feature representation that we used for our model is the distlBERT tokenizer. The token are then given as input of the distilBERT model. It applies:
- lowercase the input
- basic tokenization before WordPiece
- Same default token for all out of vocabulary tokens.
- Applies token between two sentences and for padding.
- Uses classifier token (it is the first token of the sequence when built with special tokens.)

### Classification model

The DistilBERT model was proposed in the paper <ins>DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. It is a small, fast, cheap and light Transformer model</ins> trained by distilling BERT base. 

- Vocabulary size = 30522
- Maximum position embeddings = 512
- Sinusoidal position embeddings = False
- Number of layers = 6
- Number of heads = 12
- Dimension = 768
- Hidden dimension = 3072
- Dropout = 0.1
- Attention dropout = 0.1
- Activation = 'gelu'
- Standard deviation of the truncated normal initializer = 0.02

### Ressources

    @misc{https://doi.org/10.48550/arxiv.1910.01108,
    doi = {10.48550/ARXIV.1910.01108},
    url = {https://arxiv.org/abs/1910.01108},
    author = {Sanh,  Victor and Debut,  Lysandre and Chaumond,  Julien and Wolf,  Thomas},
    keywords = {Computation and Language (cs.CL),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
    title = {DistilBERT,  a distilled version of BERT: smaller,  faster,  cheaper and lighter},
    publisher = {arXiv},
    year = {2019},
    copyright = {arXiv.org perpetual,  non-exclusive license}
    }

## Accuracy on the dev dataset

