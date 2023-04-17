import pandas as pd 
import torch
import numpy as np

from sklearn.metrics import f1_score 
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

## LOAD DATASET 

train_df = pd.read_csv("nlp_assignment/data/traindata.csv", sep="\t", header=None)
train_df.columns = ["polarity", "aspect_category", "target_term", "character_offsets", "sentence"]

## CREATE SENTENCES FOR EACH ASPECT 

asp_to_sent = {"AMBIENCE#GENERAL":"General information about the ambience.", "FOOD#QUALITY":"Food quality.", "SERVICE#GENERAL":"General service.",
                  "FOOD#STYLE_OPTIONS": "Food style options.", "DRINKS#QUALITY": "Quality of the drinks.", "RESTAURANT#MISCELLANEOUS": "Restaurant Miscellaneous.",
                  "RESTAURANT#GENERAL": "General information about the restaurant.",
                  "DRINKS#PRICES": "Drink prices.", "FOOD#PRICES": "Food prices.", "LOCATION#GENERAL": "General information about location.", 
                  "DRINKS#STYLE_OPTIONS": "Drink style options.", "RESTAURANT#PRICES": "Restaurant prices."}

## CREATE FULL SENTENCE INCLUDING ASPECT AND TARGET TERM

aspect_list = train_df["aspect_category"].tolist()
aspect_sentence_list = [0]*len(aspect_list)
for i, elem in enumerate(aspect_list):
    aspect_sentence_list[i] = asp_to_sent[elem]

target_term_list = train_df["target_term"].tolist()
target_sentence_list = [0]*len(target_term_list)
for i, elem in enumerate(target_term_list):
    target_sentence_list[i] = "Target term: " + elem + "."

sentence_list = train_df["sentence"].tolist()
full_sentence_list = [0]*len(sentence_list) 
for i, elem in enumerate(sentence_list):
    full_sentence_list[i] = aspect_sentence_list[i] + " " + target_sentence_list[i] + " " + elem 
train_df["text"] = full_sentence_list

## TURN POLARITIES INTO INTEGER LABELS 

polarity_list = train_df["polarity"].tolist()
label_dict = {"positive":2, "neutral":1, "negative":0} 
label_list = [0]*len(polarity_list)
for i, elem in enumerate(polarity_list):
    label_list[i] = label_dict[elem]
train_df["label"] = label_list

## GET TEXTS, LABELS AND TRAIN TEST SPLIT

texts = train_df["text"].tolist()
labels = train_df["label"].tolist()
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=.2)

## TOKENIZE TEXTS

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

## GET THOSE EMBEDDINGS TO TORCH DATASET AND CREATE LOADERS

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels)
eval_dataset = CustomDataset(val_encodings, val_labels)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
eval_dataloader = DataLoader(eval_dataset, batch_size=16)

## INITIATE MODEL AND TRAINING HYPERPARAMETERS

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

## TRAINING LOOP

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

## EVALUATE LOOP

model.eval()
batch_scores = []
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    batch_scores.append(f1_score(predictions.cpu().numpy(), batch["labels"].cpu().numpy(), average="weighted"))

final_score = np.mean(np.array(batch_scores))
print(final_score)