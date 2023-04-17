import pandas as pd 
import transformers

train_df = pd.read_csv("nlp_assignment/data/traindata.csv", sep="\t", header=None)
train_df.columns = ["polarity", "aspect_category", "target_term", "character_offsets", "text"]
print(train_df.head())