# NLP Assignment: Aspect-Term Polarity Classification in Sentiment Analysis

## Student Names

The names of the students who contributed to the deliverable are:
- Axel DARMOUNI
- Bouthaina HOUASS
- Mathilde LARCHEVÊQUE
- Carlos SANTOS GARCIA


## Description of the implemented solution 

The model that we used is a sequence-to-sequence model based on BART model for conditional generation. We use the instruction learning paradigm: we introduce positive, negative and neutral examples to the each training sample, which significantly improves the performance of our model.
This strategy is based on the paper <ins>InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis</ins> by Scaria et. al.


### Input and Feature representation

The first step of the classification process is to create the prompts. We do not use the aspect categories because it did not improve the performances.

For example, for the following data: 
```
positive	LOCATION#GENERAL	neighborhood	54:66	great food, great wine list, great service in a great neighborhood...
```
The input prompt becomes:

```
Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none.
Positive example 1-
input: With the great variety on the menu , I eat here often and never get bored. The aspect is menu.
output: positive
Positive example 2- 
input: Great food, good size menu, great service and an unpretensious setting. The aspect is food.
output: positive
Now complete the following example-
input: great food, great wine list, great service in a great neighborhood... The aspect is neighborhood. 
output:
```

This prompts are then given as input to the tokenizer. The feature representation that we used for our model is the pre-trained BART tokenizer. The tokens are then given as input of the BART model.


### Model fine tuning

The BART model is a denoising autoencoder for pretraining sequence-to-sequence models. We fine tuned it using the traindata.csv file. The fine tuning hyperparameters are:
- 2 epochs 
- batch size 8 
- Adam optimizer with 1e-5 learning rate 


## Accuracy on the dev dataset

|                        | Accuracy on dev dataset |
|------------------------|-------------------------|
| BART based model       |      87.45 (0.32)       |


### Ressources

    @misc{scaria2023instructabsa,
        title={InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis}, 
        author={Kevin Scaria and Himanshu Gupta and Siddharth Goyal and Saurabh Arjun Sawant and Swaroop Mishra and Chitta Baral},
        year={2023},
        eprint={2302.08624},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }

    @misc{lewis2019bart,
        title={BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension}, 
        author={Mike Lewis and Yinhan Liu and Naman Goyal and Marjan Ghazvininejad and Abdelrahman Mohamed and Omer Levy and Ves Stoyanov and Luke Zettlemoyer},
        year={2019},
        eprint={1910.13461},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
