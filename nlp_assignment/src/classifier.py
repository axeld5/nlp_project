from typing import List

import torch
from torch.utils.data import DataLoader
from train_utils import CustomDataset, load_dataset, train_model
from transformers import AutoTokenizer


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
    """

    ############################################# comp

    def __init__(self, model_name="bert-base-uncased", num_epochs=10, lr=5e-5, batch_size=16):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size

    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        self.model, _ = train_model(
            model_name=self.model_name,
            train_data_path=train_filename,
            dev_data_path=dev_filename,
            num_epochs=self.num_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            device=device,
        )

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        test_df = load_dataset(data_filename, with_labels=False)
        test_texts = test_df["text"].tolist()

        # Tokenizes texts
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        # Dataset and DataLoader creation
        test_dataset = CustomDataset(test_encodings, [-1] * len(test_texts))
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size)

        self.model.to(device)
        label_list = []
        label_dict = {2: "positive", 1: "neutral", 0: "negative"}
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            label_list += [label_dict[pred] for pred in predictions.cpu().numpy()]

        return label_list
