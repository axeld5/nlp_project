import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, BartForConditionalGeneration,
    Seq2SeqTrainingArguments, Trainer
)

class BartClassifier:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download = True)
        self.model = BartForConditionalGeneration.from_pretrained(model_checkpoint, force_download = True)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        sample['input_ids'] = self.tokenizer(sample["text"], max_length = 512, truncation = True).input_ids
        sample['labels'] = self.tokenizer(sample["labels"], max_length = 64, truncation = True).input_ids
        return sample
        
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """

        # Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
            )

        # Define trainer object
        trainer = Trainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"] if tokenized_datasets.get("test") is not None else None,
            tokenizer=self.tokenizer, 
            data_collator = self.data_collator 
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()
        return trainer

    def get_labels(self, tokenized_dataset, predictor = None, batch_size = 4, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        print('Prediction from trainer')
        pred_proba = predictor.predict(test_dataset=tokenized_dataset[sample_set]).predictions[0]
        output_ids = np.argmax(pred_proba, axis=2)
        predicted_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return predicted_output
    
    def get_metrics(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='weighted'), recall_score(y_true, y_pred, average='weighted'), \
            f1_score(y_true, y_pred, average='weighted'), accuracy_score(y_true, y_pred)
