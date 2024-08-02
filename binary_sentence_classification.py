import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# Load data
data_path = "path/to/dataset.csv"
hmdd4 = pd.read_csv(data_path).dropna()
positive_sentences = list(hmdd4.description)
negative_sentences = list(hmdd4.negative)
Y = [1] * len(positive_sentences) + [0] * len(negative_sentences)
X = positive_sentences + negative_sentences

# Create a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

# Function to train the model
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = []

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = loss_fn(outputs.logits.squeeze(), labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(losses)

# Function to evaluate the model
def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)
            all_labels.extend(labels.cpu().numpy())

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
            preds = np.round(preds)
            all_preds.extend(preds)

            loss = loss_fn(outputs.logits.squeeze(), labels)
            losses.append(loss.item())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return np.mean(losses), accuracy, precision, recall, f1

# Function to run the entire training and evaluation process
def run_experiment():
    # Split the data into train, validation, and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(X, Y, test_size=0.4, random_state=SEED)
    val_texts, test_texts, val_labels, test_labels = train_test_split(test_texts, test_labels, test_size=0.5, random_state=SEED)

    # Initialize model and tokenizer
    model_name = "path/to/BERT-Model"
    tokenizer = AutoTokenizer.from_pretrained("path/to/BERT-Tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # Create datasets and dataloaders
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_len=256)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_len=256)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_len=256)

    train_data_loader = DataLoader(train_dataset, batch_size=128, sampler=RandomSampler(train_dataset))
    val_data_loader = DataLoader(val_dataset, batch_size=128, sampler=SequentialSampler(val_dataset))
    test_data_loader = DataLoader(test_dataset, batch_size=128, sampler=SequentialSampler(test_dataset))

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # Define loss function
    criterion_bce = nn.BCEWithLogitsLoss()

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Set the number of epochs
    num_epochs = 3

    # Train and evaluate
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        train_loss = train_epoch(model, train_data_loader, criterion_bce, optimizer, device)
        logger.info(f'Train Loss: {train_loss}')

    val_loss, val_accuracy, val_precision, val_recall, val_f1 = eval_model(model, val_data_loader, criterion_bce, device)
    logger.info(f'Validation Loss: {val_loss}')
    logger.info(f'Validation Accuracy: {val_accuracy}')
    logger.info(f'Validation Precision: {val_precision}')
    logger.info(f'Validation Recall: {val_recall}')
    logger.info(f'Validation F1 Score: {val_f1}')

    test_loss, test_accuracy, test_precision, test_recall, test_f1 = eval_model(model, test_data_loader, criterion_bce, device)
    logger.info(f'Test Loss: {test_loss}')
    logger.info(f'Test Accuracy: {test_accuracy}')
    logger.info(f'Test Precision: {test_precision}')
    logger.info(f'Test Recall: {test_recall}')
    logger.info(f'Test F1 Score: {test_f1}')

    return {
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1
    }

# Running the experiment 5 times
results = []
for i in range(5):
    torch.manual_seed(SEED + i)  # Ensure different seed for each run
    torch.cuda.manual_seed_all(SEED + i)
    np.random.seed(SEED + i)
    result = run_experiment()
    results.append(result)

# Calculate the average and standard deviation of the results
def calculate_stats(metrics):
    avg = np.mean(metrics)
    std = np.std(metrics)
    return avg, std

val_accuracies = [r["val_accuracy"] for r in results]
val_precisions = [r["val_precision"] for r in results]
val_recalls = [r["val_recall"] for r in results]
val_f1s = [r["val_f1"] for r in results]

test_accuracies = [r["test_accuracy"] for r in results]
test_precisions = [r["test_precision"] for r in results]
test_recalls = [r["test_recall"] for r in results]
test_f1s = [r["test_f1"] for r in results]

avg_val_accuracy, std_val_accuracy = calculate_stats(val_accuracies)
avg_val_precision, std_val_precision = calculate_stats(val_precisions)
avg_val_recall, std_val_recall = calculate_stats(val_recalls)
avg_val_f1, std_val_f1 = calculate_stats(val_f1s)

avg_test_accuracy, std_test_accuracy = calculate_stats(test_accuracies)
avg_test_precision, std_test_precision = calculate_stats(test_precisions)
avg_test_recall, std_test_recall = calculate_stats(test_recalls)
avg_test_f1, std_test_f1 = calculate_stats(test_f1s)

print("Validation Set Metrics:")
print(f"Accuracy: {avg_val_accuracy:.4f} +- {std_val_accuracy:.4f}")
print(f"Precision: {avg_val_precision:.4f} +- {std_val_precision:.4f}")
print(f"Recall: {avg_val_recall:.4f} +- {std_val_recall:.4f}")
print(f"F1 Score: {avg_val_f1:.4f} +- {std_val_f1:.4f}")

print("Test Set Metrics:")
print(f"Accuracy: {avg_test_accuracy:.4f} +- {std_test_accuracy:.4f}")
print(f"Precision: {avg_test_precision:.4f} +- {std_test_precision:.4f}")
print(f"Recall: {avg_test_recall:.4f} +- {std_test_recall:.4f}")
print(f"F1 Score: {avg_test_f1:.4f} +- {std_test_f1:.4f}")
