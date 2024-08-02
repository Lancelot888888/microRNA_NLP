import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load the data
data_path = "path/to/dataset_NER.csv"
data = pd.read_csv(data_path)


# Preprocess the data
def preprocess_data(data, tokenizer):
    tokenized_sentences = []
    labels = []

    for _, row in data.iterrows():
        sentence = str(row['description'])
        mirna = str(row['microRNA'])
        target = str(row['Disease'])

        tokens = tokenizer.tokenize(sentence)
        token_labels = ['O'] * len(tokens)

        def get_start_index(entity, tokens):
            entity_tokens = tokenizer.tokenize(entity)
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if tokens[i:i + len(entity_tokens)] == entity_tokens:
                    return i
            return -1

        mirna_start = get_start_index(mirna, tokens)
        if (mirna_start != -1):
            token_labels[mirna_start] = 'B-microRNA'
            for j in range(mirna_start + 1, mirna_start + len(tokenizer.tokenize(mirna))):
                token_labels[j] = 'I-microRNA'

        target_start = get_start_index(target, tokens)
        if (target_start != -1):
            token_labels[target_start] = 'B-Disease'
            for j in range(target_start + 1, target_start + len(tokenizer.tokenize(target))):
                token_labels[j] = 'I-Disease'

        tokenized_sentences.append(tokens)
        labels.append(token_labels)

    return tokenized_sentences, labels


# Specify the model type
model_type = "path/to/BERT-Tokenizer"

# Tokenize and encode labels
tokenizer = AutoTokenizer.from_pretrained(model_type)
tokenized_sentences, labels = preprocess_data(data, tokenizer)


def encode_labels(labels, label_map):
    encoded_labels = []
    for label in labels:
        encoded_labels.append([label_map[l] for l in label])
    return encoded_labels


label_list = ['O', 'B-microRNA', 'I-microRNA', 'B-Disease', 'I-Disease']
label_map = {label: i for i, label in enumerate(label_list)}

input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_sentences]
attention_masks = [[float(i > 0) for i in ids] for ids in input_ids]
encoded_labels = encode_labels(labels, label_map)

# Truncate and pad sequences to the same length (maximum 256)
max_len = 256
input_ids = [ids[:max_len] + [0] * (max_len - len(ids)) for ids in input_ids]
attention_masks = [mask[:max_len] + [0] * (max_len - len(mask)) for mask in attention_masks]
encoded_labels = [label[:max_len] + [label_map['O']] * (max_len - len(label)) for label in encoded_labels]

# Convert to torch tensors
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
encoded_labels = torch.tensor(encoded_labels)


# Function to predict and decode labels
def predict_and_decode(trainer, dataset):
    predictions, labels, _ = trainer.predict(dataset)
    predictions = np.argmax(predictions, axis=2)
    decoded_predictions = decode_labels(predictions, label_map)
    decoded_labels = decode_labels(labels, label_map)
    return decoded_predictions, decoded_labels


# Convert IDs back to labels
def decode_labels(encoded_labels, label_map):
    label_map_reverse = {v: k for k, v in label_map.items()}
    return [[label_map_reverse[l] for l in label] for label in encoded_labels]


# Group predictions and labels by sentences
def group_by_sentence(predictions, labels, attention_masks):
    grouped_predictions = []
    grouped_labels = []
    for i in range(len(attention_masks)):
        sentence_predictions = []
        sentence_labels = []
        for j in range(len(attention_masks[i])):
            if attention_masks[i][j] == 0:
                break
            sentence_predictions.append(predictions[i][j])
            sentence_labels.append(labels[i][j])
        grouped_predictions.append(sentence_predictions)
        grouped_labels.append(sentence_labels)
    return grouped_predictions, grouped_labels


# Custom data collator for the trainer
def custom_data_collator(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_masks = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}


# Function to calculate average classification report
def average_classification_reports(reports):
    avg_report = defaultdict(lambda: defaultdict(float))
    for report in reports:
        for key, metrics in report.items():
            if key == 'accuracy':
                avg_report[key] += metrics
            else:
                for metric, value in metrics.items():
                    avg_report[key][metric] += value
    for key in avg_report:
        if key == 'accuracy':
            avg_report[key] /= len(reports)
        else:
            for metric in avg_report[key]:
                avg_report[key][metric] /= len(reports)
    return avg_report


# Initialize lists to store metrics
validation_reports = []
test_reports = []

# Run the process 5 times with different seeds
seeds = [42, 43, 44, 45, 46]
for random_seed in seeds:
    # Split the data
    train_size = 0.6
    val_size = 0.2

    input_ids_train, input_ids_temp, attention_masks_train, attention_masks_temp, labels_train, labels_temp = train_test_split(
        input_ids, attention_masks, encoded_labels, train_size=train_size, random_state=random_seed
    )
    input_ids_val, input_ids_test, attention_masks_val, attention_masks_test, labels_val, labels_test = train_test_split(
        input_ids_temp, attention_masks_temp, labels_temp, test_size=val_size / (1 - train_size),
        random_state=random_seed
    )

    # Create TensorDatasets
    train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    val_dataset = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    # Set up the model
    model = AutoModelForTokenClassification.from_pretrained("path/to/BERT-Model", num_labels=len(label_map))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=10,
        weight_decay=0.01,
        seed=random_seed,
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Evaluate on validation set
    val_predictions, val_labels = predict_and_decode(trainer, val_dataset)
    grouped_val_predictions, grouped_val_labels = group_by_sentence(val_predictions, val_labels,
                                                                    attention_masks_val.numpy())
    filtered_val_predictions = [[pred for pred, label in zip(pred_list, label_list) if label != 'O'] for
                                pred_list, label_list in zip(grouped_val_predictions, grouped_val_labels)]
    filtered_val_labels = [[label for label in label_list if label != 'O'] for label_list in grouped_val_labels]
    val_report = classification_report(filtered_val_labels, filtered_val_predictions, digits=4, output_dict=True)
    validation_reports.append(val_report)

    # Evaluate on test set
    test_predictions, test_labels = predict_and_decode(trainer, test_dataset)
    grouped_test_predictions, grouped_test_labels = group_by_sentence(test_predictions, test_labels,
                                                                      attention_masks_test.numpy())
    filtered_test_predictions = [[pred for pred, label in zip(pred_list, label_list) if label != 'O'] for
                                 pred_list, label_list in zip(grouped_test_predictions, grouped_test_labels)]
    filtered_test_labels = [[label for label in label_list if label != 'O'] for label_list in grouped_test_labels]
    test_report = classification_report(filtered_test_labels, filtered_test_predictions, digits=4, output_dict=True)
    test_reports.append(test_report)

# Calculate average validation metrics
avg_val_report = average_classification_reports(validation_reports)
print("Average Validation Metrics over 5 runs:")
print(pd.DataFrame(avg_val_report).transpose())

# Calculate average test metrics
avg_test_report = average_classification_reports(test_reports)
print("Average Test Metrics over 5 runs:")
print(pd.DataFrame(avg_test_report).transpose())
