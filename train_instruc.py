import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
import torch

root_path = "."    
use_mps = True if torch.has_mps else False
os.chdir(root_path)

from data_instruc import load_dataset, DatasetLoader
from bart import BartClassifier
from instruc import InstructionsHandler

model_checkpoint = 'facebook/bart-base'

id_tr_df = load_dataset("nlp_assignment/data/traindata.csv")
id_te_df = load_dataset("nlp_assignment/data/devdata.csv")

instruct_handler = InstructionsHandler()
instruct_handler.load_instruction_set1()

loader = DatasetLoader(id_tr_df, id_te_df)
if loader.train_df_id is not None:
    loader.train_df_id = loader.create_data_in_atsc_format(loader.train_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', 
                            instruct_handler.atsc['bos_instruct'], instruct_handler.atsc["delim_instruct"], instruct_handler.atsc['eos_instruct'])
if loader.test_df_id is not None:
    loader.test_df_id = loader.create_data_in_atsc_format(loader.test_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect',
                            instruct_handler.atsc['bos_instruct'], instruct_handler.atsc["delim_instruct"], instruct_handler.atsc['eos_instruct'])

# Create T5 utils object
t5_exp = BartClassifier(model_checkpoint)

# Tokenize Dataset
id_ds, id_tokenized_ds, _, _ = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs)

# Training arguments
training_args = {
    'output_dir':'.',
    'evaluation_strategy':"epoch",
    'learning_rate':1e-5,
    'per_device_train_batch_size':8,
    'per_device_eval_batch_size':16,
    'num_train_epochs':3,
    'weight_decay':0.01,
    'warmup_ratio':0.1,
    'save_strategy':'no',
    'load_best_model_at_end':False,
    'push_to_hub':False,
    'eval_accumulation_steps':1,
    'predict_with_generate':True,
    'logging_steps':1000000000,
    'use_mps_device':use_mps
}

# Train model
model_trainer = t5_exp.train(id_tokenized_ds, **training_args)

# Model inference - Trainer object - (Pass model trainer as predictor)

# Get prediction labels - Training set
id_tr_pred_labels = t5_exp.get_labels(predictor = model_trainer, tokenized_dataset = id_tokenized_ds, sample_set = 'train')
id_tr_labels = [i.strip() for i in id_ds['train']['labels']]

# Get prediction labels - Testing set
id_te_pred_labels = t5_exp.get_labels(predictor = model_trainer, tokenized_dataset = id_tokenized_ds, sample_set = 'test')
id_te_labels = [i.strip() for i in id_ds['test']['labels']]

# Compute Metrics
p, r, f1, acc = t5_exp.get_metrics(id_tr_labels, id_tr_pred_labels)
print('Train Precision: ', p)
print('Train Recall: ', r)
print('Train F1: ', f1)
print('Train Accuracy: ', acc)

p, r, f1, acc = t5_exp.get_metrics(id_te_labels, id_te_pred_labels)
print('Test Precision: ', p)
print('Test Recall: ', r)
print('Test F1: ', f1)
print('Train Accuracy: ', acc)