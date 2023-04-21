import pandas as pd 
import numpy as np
import torch

from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, BartForConditionalGeneration,
    Seq2SeqTrainingArguments, Trainer
)


def load_dataset(data_path: str) -> list:
    train_df = pd.read_csv(data_path, sep="\t", header=None)
    train_df.columns = ["polarity", "aspect_category", "target_term", "character_offsets", "sentence"]

    sentence_list = train_df["sentence"].tolist()
    n_sentences = len(sentence_list)
    polarity_list = train_df["polarity"].tolist()    
    target_term_list = train_df["target_term"].tolist()
    aspect_list = []
    for i in range(n_sentences):
        aspect_dict = [{}]
        aspect_dict[0]["term"] = target_term_list[i] 
        aspect_dict[0]["polarity"] = polarity_list[i] 
        aspect_list.append(aspect_dict)
    new_df = pd.DataFrame() 
    new_df["raw_text"] = sentence_list 
    new_df["aspectTerms"] = aspect_list
    return new_df


class DatasetLoader:
    def __init__(self, train_df_id=None, test_df_id=None):
        
        self.train_df_id = train_df_id.sample(frac=1) if train_df_id is not None else train_df_id
        self.test_df_id = test_df_id

    def reconstruct_strings(self, df, col):
        """
        Reconstruct strings to dictionaries when loading csv/xlsx files.
        """
        reconstructed_col = []
        for text in df[col]:
            if text != '[]' and isinstance(text, str):
                text = text.replace('[', '').replace(']', '').replace('{', '').replace('}', '').split(", '")
                req_list = []
                for idx, pair in enumerate(text):
                    if idx%2==0:
                        reconstructed_dict = {}
                        reconstructed_dict[pair.split(':')[0].replace("'", '')] = pair.split(':')[1].replace("'", '')
                    else:
                        reconstructed_dict[pair.split(':')[0].replace("'", '')] = pair.split(':')[1].replace("'", '')
                        req_list.append(reconstructed_dict)
            else:
                req_list = text
            reconstructed_col.append(req_list)
        df[col] = reconstructed_col
        return df

    def extract_rowwise_aspect_polarity(self, df, on, key, min_val = None):
        """
        Create duplicate records based on number of aspect term labels in the dataset.
        Extract each aspect term for each row for reviews with muliple aspect term entries. 
        Do same for polarities and create new column for the same.
        """
        try:
            df.iloc[0][on][0][key]
        except:
            df = self.reconstruct_strings(df, on)

        df['len'] = df[on].apply(lambda x: len(x))
        if min_val is not None:
            df.loc[df['len'] == 0, 'len'] = min_val
        df = df.loc[df.index.repeat(df['len'])]
        df['record_idx'] = df.groupby(df.index).cumcount()
        df['aspect'] = df[[on, 'record_idx']].apply(lambda x : (x[0][x[1]][key], x[0][x[1]]['polarity']) if len(x[0]) != 0 else ('',''), axis=1)
        df['polarity'] = df['aspect'].apply(lambda x: x[1])
        df['aspect'] = df['aspect'].apply(lambda x: x[0])
        df = df.drop(['len', 'record_idx'], axis=1).reset_index(drop = True)
        return df

    def create_data_in_atsc_format(self, df, on, key, text_col, aspect_col, bos_instruction = '', 
                    asp_delim_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        df = self.extract_rowwise_aspect_polarity(df, on=on, key=key, min_val=1)
        df['text'] = df[[text_col, aspect_col]].apply(lambda x: bos_instruction + x[0] + asp_delim_instruction + x[1] + eos_instruction, axis=1)
        df = df.rename(columns = {'polarity': 'labels'})
        return df
    
    def set_data_for_training(self, tokenize_function):
        """
        Create the training and test dataset as huggingface datasets format.
        """
        # Define train and test sets
        if (self.train_df_id is not None) and (self.test_df_id is None):
            indomain_dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df_id)})
            indomain_tokenized_datasets = indomain_dataset.map(tokenize_function, batched=True)
        elif(self.train_df_id is None) and (self.test_df_id is not None):
            indomain_dataset = DatasetDict({'test': Dataset.from_pandas(self.test_df_id)})
            indomain_tokenized_datasets = indomain_dataset.map(tokenize_function, batched=True)
        elif (self.train_df_id is not None) and (self.test_df_id is not None):
            indomain_dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df_id), 'test': Dataset.from_pandas(self.test_df_id)})
            indomain_tokenized_datasets = indomain_dataset.map(tokenize_function, batched=True)
        else:
            indomain_dataset = {}
            indomain_tokenized_datasets = {}
        
        return indomain_dataset, indomain_tokenized_datasets
    

class InstructionsHandler:
    def __init__(self):
        self.atsc = {}

    def load_instruction_set1(self, ):
        
        self.atsc['bos_instruct'] = """Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
        Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none.
        Positive example 1-
        input: With the great variety on the menu , I eat here often and never get bored. The aspect is menu.
        output: positive
        Positive example 2- 
        input: Great food, good size menu, great service and an unpretensious setting. The aspect is food.
        output: positive
        Now complete the following example-
        input: """
        self.atsc['delim_instruct'] = ' The aspect is '
        self.atsc['eos_instruct'] = '.\noutput:'

    def load_instruction_set2(self, ):
        
        self.atsc['bos_instruct'] = """Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
        Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none.
        Positive example 1-
        input: With the great variety on the menu , I eat here often and never get bored. The aspect is menu.
        output: positive
        Positive example 2- 
        input: Great food, good size menu, great service and an unpretensious setting. The aspect is food.
        output: positive
        Negative example 1-
        input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it. The aspect is toast.
        output: negative
        Negative example 2-
        input: The seats are uncomfortable if you are sitting against the wall on wooden benches. The aspect is seats.
        output: negative
        Neutral example 1-
        input: This place is just fine. The aspect is place.
        output: neutral
        Neutral example 2-
        input: They wouldnt even let me finish my glass of wine before offering another. The aspect is glass of wine.
        output: neutral
        Now complete the following example-
        input: """
        self.atsc['delim_instruct'] = ' The aspect is '
        self.atsc['eos_instruct'] = '.\noutput:'

class BartClassifier:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)

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

def train_model(
    model_name: str,
    train_data_path: str,
    dev_data_path: str,
    num_epochs: int,
    lr: float,
    batch_size: int,
    training_seed: int,
    device: str
): 
    assert model_name in ("facebook/bart-base", "facebook/bart-large")
    model_checkpoint = model_name
    id_tr_df = load_dataset(train_data_path)
    id_te_df = load_dataset(dev_data_path)

    instruct_handler = InstructionsHandler()
    instruct_handler.load_instruction_set1()

    loader = DatasetLoader(id_tr_df, id_te_df)
    if loader.train_df_id is not None:
        loader.train_df_id = loader.create_data_in_atsc_format(loader.train_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect',
                                instruct_handler.atsc['bos_instruct'], instruct_handler.atsc["delim_instruct"], instruct_handler.atsc['eos_instruct'])
    if loader.test_df_id is not None:
        loader.test_df_id = loader.create_data_in_atsc_format(loader.test_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', 
                                instruct_handler.atsc['bos_instruct'], instruct_handler.atsc["delim_instruct"], instruct_handler.atsc['eos_instruct'])

    # Create Bart utils object
    bart_exp = BartClassifier(model_checkpoint)

    # Tokenize Dataset
    id_ds, id_tokenized_ds = loader.set_data_for_training(bart_exp.tokenize_function_inputs)
    # Training arguments
    training_args = {
        'output_dir':'.',
        'evaluation_strategy':"epoch",
        'learning_rate':lr,
        'per_device_train_batch_size':batch_size,
        'per_device_eval_batch_size':4,
        'num_train_epochs':num_epochs,
        'weight_decay':0.01,
        'warmup_ratio':0.1,
        'save_strategy':'no',
        'load_best_model_at_end':False,
        'push_to_hub':False,
        'eval_accumulation_steps':1,
        'predict_with_generate':True,
        'logging_steps':1000000000,
        'seed':training_seed,
    }

    if device == torch.device("cpu"):
        training_args["no_cuda"] = True 
    elif torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Train model
    model_trainer = bart_exp.train(id_tokenized_ds, **training_args)

    return bart_exp, model_trainer