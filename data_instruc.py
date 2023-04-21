import pandas as pd 
from datasets import Dataset
from datasets.dataset_dict import DatasetDict

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
    def __init__(self, train_df_id=None, test_df_id=None, train_df_ood=None, test_df_ood=None, sample_size = 1):
        
        self.train_df_id = train_df_id.sample(frac = sample_size) if train_df_id is not None else train_df_id
        self.test_df_id = test_df_id
        self.train_df_ood = train_df_ood.sample(frac = sample_size) if train_df_ood is not None else train_df_ood
        self.test_df_ood = test_df_ood

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
        df['polarity'] = df['aspect'].apply(lambda x: x[-1])
        df['aspect'] = df['aspect'].apply(lambda x: x[0])
        df = df.drop(['len', 'record_idx'], axis=1).reset_index(drop = True)
        return df

    def create_data_in_atsc_format(self, df, on, key, text_col, aspect_col, bos_instruction = '', 
                    delim_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        df = self.extract_rowwise_aspect_polarity(df, on=on, key=key, min_val=1)
        df['text'] = df[[text_col, aspect_col]].apply(lambda x: bos_instruction + x[0] + delim_instruction + x[1] + eos_instruction, axis=1)
        df = df.rename(columns = {'polarity': 'labels'})
        return df
    
    def set_data_for_training_semeval(self, tokenize_function):
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

        if (self.train_df_ood is not None) and (self.test_df_ood is None):
            other_domain_dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df_ood)})
            other_domain_tokenized_dataset = other_domain_dataset.map(tokenize_function, batched=True)
        elif (self.train_df_ood is None) and (self.test_df_ood is not None):
            other_domain_dataset = DatasetDict({'test': Dataset.from_pandas(self.test_df_ood)})
            other_domain_tokenized_dataset = other_domain_dataset.map(tokenize_function, batched=True)
        elif (self.train_df_ood is not None) and (self.test_df_ood is not None):
            other_domain_dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df_ood), 'test': Dataset.from_pandas(self.test_df_ood)})
            other_domain_tokenized_dataset = other_domain_dataset.map(tokenize_function, batched=True)
        else:
            other_domain_dataset = {}
            other_domain_tokenized_dataset = {}
        
        return indomain_dataset, indomain_tokenized_datasets, other_domain_dataset, other_domain_tokenized_dataset