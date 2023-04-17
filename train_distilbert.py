import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_dataset(data_path: str) -> pd.DataFrame:
    train_df = pd.read_csv(data_path, sep="\t", header=None)
    train_df.columns = ["polarity", "aspect_category", "target_term", "character_offsets", "sentence"]

    # Maps aspects to sentences describing them with natural language
    asp_to_sent = {
        "AMBIENCE#GENERAL": "General information about the ambience.",
        "FOOD#QUALITY": "Food quality.",
        "SERVICE#GENERAL": "General service.",
        "FOOD#STYLE_OPTIONS": "Food style options.",
        "DRINKS#QUALITY": "Quality of the drinks.",
        "RESTAURANT#MISCELLANEOUS": "Restaurant Miscellaneous.",
        "RESTAURANT#GENERAL": "General information about the restaurant.",
        "DRINKS#PRICES": "Drink prices.",
        "FOOD#PRICES": "Food prices.",
        "LOCATION#GENERAL": "General information about location.",
        "DRINKS#STYLE_OPTIONS": "Drink style options.",
        "RESTAURANT#PRICES": "Restaurant prices.",
    }

    # Creates full sentences
    aspect_list = train_df["aspect_category"].tolist()
    aspect_sentence_list = [0] * len(aspect_list)
    for i, elem in enumerate(aspect_list):
        aspect_sentence_list[i] = asp_to_sent[elem]

    target_term_list = train_df["target_term"].tolist()
    target_sentence_list = [0] * len(target_term_list)
    for i, elem in enumerate(target_term_list):
        target_sentence_list[i] = "Target term: " + elem + "."

    sentence_list = train_df["sentence"].tolist()
    full_sentence_list = [0] * len(sentence_list)
    for i, elem in enumerate(sentence_list):
        full_sentence_list[i] = aspect_sentence_list[i] + " " + target_sentence_list[i] + " " + elem
    train_df["text"] = full_sentence_list

    # Turns target polarity labels into integers
    polarity_list = train_df["polarity"].tolist()
    label_dict = {"positive": 2, "neutral": 1, "negative": 0}
    label_list = [0] * len(polarity_list)
    for i, elem in enumerate(polarity_list):
        label_list[i] = label_dict[elem]
    train_df["label"] = label_list

    return train_df


def split_dataset(train_df, test_size=0.2):
    texts = train_df["text"].tolist()
    labels = train_df["label"].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=test_size)
    return train_texts, val_texts, train_labels, val_labels


def train_model(model_name: str, train_data_path: str, dev_data_path: str, num_epochs: int):
    assert model_name in ("distilbert-base-uncased", "bert-base-uncased")

    # Loads dataset and splits it
    train_df = load_dataset(train_data_path)
    val_df = load_dataset(dev_data_path)
    train_texts, val_texts, train_labels, val_labels = (
        train_df["text"].tolist(),
        val_df["text"].tolist(),
        train_df["label"].tolist(),
        val_df["label"].tolist(),
    )

    # Tokenizes texts
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    # Creates torch Datasets
    train_dataset = CustomDataset(train_encodings, train_labels)
    eval_dataset = CustomDataset(val_encodings, val_labels)

    # Creates torch Dataloaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16)

    # Creates model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    model.train()

    # Progress bar
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        train_loss = []
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
            progress_bar.update(1)

        ## Evaluation loop
        model.eval()
        batch_scores = []
        val_loss = []
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            val_loss.append(outputs.loss.item())

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            batch_scores.append(accuracy_score(predictions.cpu().numpy(), batch["labels"].cpu().numpy()))

        progress_bar.set_description(
            f"Epoch {epoch+1}/{num_epochs} - train_loss: {np.mean(train_loss):.4f} - val_loss: {np.mean(val_loss):.4f}"
        )
    print(f"Final val score: {np.mean(np.array(batch_scores)) :.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launches training.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model to use.")
    parser.add_argument(
        "--train_data_path", type=str, default="nlp_assignment/data/traindata.csv", help="Path to train data."
    )
    parser.add_argument(
        "--dev_data_path", type=str, default="nlp_assignment/data/devdata.csv", help="Path to eval data."
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs.")
    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        train_data_path=args.train_data_path,
        dev_data_path=args.dev_data_path,
        num_epochs=args.num_epochs,
    )
