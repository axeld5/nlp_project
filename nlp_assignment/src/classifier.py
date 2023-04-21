from typing import List

import numpy as np 
import torch

from train_utils import InstructionsHandler, DatasetLoader, load_dataset, train_model, BartClassifier


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
    """

    ############################################# comp

    def __init__(self, model_name="facebook/bart-base", num_epochs=2, lr=1e-5, batch_size=8):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.base_bert_exp = BartClassifier(self.model_name)
        self.training_seed = np.random.choice(10**6)

    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        self.bert_exp, self.model = train_model(
            model_name=self.model_name,
            train_data_path=train_filename,
            dev_data_path=dev_filename,
            num_epochs=self.num_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            training_seed=self.training_seed,
            device=device
        )

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        id_te_df = load_dataset(data_filename)
        instruct_handler = InstructionsHandler()
        instruct_handler.load_instruction_set1()

        loader = DatasetLoader(None, id_te_df)
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_atsc_format(loader.train_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect',
                                    instruct_handler.atsc['bos_instruct'], instruct_handler.atsc["delim_instruct"], instruct_handler.atsc['eos_instruct'])
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_atsc_format(loader.test_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', 
                                    instruct_handler.atsc['bos_instruct'], instruct_handler.atsc["delim_instruct"], instruct_handler.atsc['eos_instruct'])


        # Tokenize Dataset
        _, id_tokenized_ds = loader.set_data_for_training_semeval(self.base_bert_exp.tokenize_function_inputs)

        id_te_pred_labels = self.bert_exp.get_labels(predictor = self.model, tokenized_dataset = id_tokenized_ds, sample_set = 'test')

        return id_te_pred_labels

