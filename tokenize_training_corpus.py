import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Initialize tokenizer with the specified model
model_name = "path/to/BERT_models/"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the local dataset using pandas
dataset_path = "path/to/PUBMED_title_abstracts_2020_baseline.jsonl"
df = pd.read_json(dataset_path, lines=True)

# Convert the pandas DataFrame to a Hugging Face dataset
ds = Dataset.from_pandas(df)

# Define a function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_special_tokens_mask=True
    )

# Apply the tokenize function to the dataset
tokenized_ds = ds.map(tokenize_function, batched=True)

# Save the tokenized dataset to disk
tokenized_ds.save_to_disk("path/to/tokenized_dataset/")
