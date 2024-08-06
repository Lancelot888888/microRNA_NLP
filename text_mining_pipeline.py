import os
import re
import glob
import fitz
import torch
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, Trainer, TrainingArguments
import warnings

warnings.filterwarnings('ignore')

# Path to the trained binary classification model
BINARY_MODEL_PATH = "./models/binary_classifier/"
# Path to the trained NER (Named Entity Recognition) model
NER_MODEL_PATH = "./models/ner_model/"

# Path to the binary classification data
BINARY_DATA_PATH = "/path/to/datasets/MTI.csv"
# Path to the named entity recognition data
NER_DATA_PATH = "/path/to/datasets/MTI_NER.csv"

# Directory containing the PDF files to be processed
PDF_DIRECTORY = "/path/to/pdfs/"
# Directory to save the annotated PDF files
OUTPUT_DIRECTORY = "/path/to/annotated_pdfs/"


# Path to the model tokenizer
TOKENIZER_PATH = "/path/to/tokenizer" # example PubMedBERT-base-uncased-abstract-fulltext/
# Path to the base model to be trained if you havent trained a model
BASE_MODEL_PATH = "/path/to/model" # example PubMedBERT-base-uncased-abstract-fulltext/

# Define keywords for filtering sentences
keywords = ['mir-', 'let-', 'lin-', 'microrna-', 'mirna-']

# Create the output directory if it does not exist
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)


# Function to extract sentences from PDFs
def extract_sentences(pdf_path):
    pdf = fitz.open(pdf_path)
    all_sentences = []

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text()
        page_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        for sentence in page_sentences:
            clean = ' '.join(sentence.replace('\n', ' ').replace('\r', ' ').split())
            if clean:
                all_sentences.append(clean)

    for i, sentence in enumerate(all_sentences):
        if "References" in sentence or "Bibliography" in sentence:
            all_sentences = all_sentences[:i]
            break

    pdf.close()
    return all_sentences


# Function to filter sentences with citations
def filter_sentences_with_citations(sentences):
    filtered_sentences = []

    author_date_pattern = re.compile(r'\(\w+ et al\.?,? \d{4}\)')
    author_date_pattern_simple = re.compile(r'\(\w+, \d{4}\)')
    numeric_pattern_brackets = re.compile(r'\[\d+(,\s?\d+)*\]')
    numeric_pattern_parentheses = re.compile(r'\(\d+(,\s?\d+)*\)')
    numeric_range_pattern_brackets = re.compile(r'\[\d+-\d+(,\s?\d+-\d+)*\]')
    numeric_range_pattern_parentheses = re.compile(r'\(\d+-\d+(,\s?\d+-\d+)*\)')

    for sentence in sentences:
        if not (author_date_pattern.search(sentence) or
                author_date_pattern_simple.search(sentence) or
                numeric_pattern_brackets.search(sentence) or
                numeric_pattern_parentheses.search(sentence) or
                numeric_range_pattern_brackets.search(sentence) or
                numeric_range_pattern_parentheses.search(sentence)):
            filtered_sentences.append(sentence)

    return filtered_sentences


# Function to filter long sentences
def filter_long_sentences(sentences, max_length=200):
    return [sentence for sentence in sentences if len(sentence.split()) <= max_length]


# Function to correct broken hyphens in sentences
def correct_broken_hyphens(sentences):
    hyphen_pattern = re.compile(r'(\w+)-\s+(\w+)')
    corrected_sentences = []

    for sentence in sentences:
        sentence = sentence.replace('\u2010', '-')
        while hyphen_pattern.search(sentence):
            sentence = hyphen_pattern.sub(r'\1\2', sentence)
        corrected_sentences.append(sentence)

    return corrected_sentences


# Function to add annotations to PDFs
def add_squiggly_and_highlight_annotations(pdf_path, sentences_with_entities, output_pdf_path):
    try:
        pdf = fitz.open(pdf_path)
    except Exception:
        return []

    annotated_sentences = set()
    failed_sentences = set()

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        for sentence, entities in sentences_with_entities:
            if sentence not in annotated_sentences:
                try:
                    text_instances = page.search_for(sentence)
                    if text_instances:
                        for inst in text_instances:
                            rect = fitz.Rect(inst)
                            page.add_squiggly_annot(rect)
                            for entity in entities:
                                try:
                                    if isinstance(entity, str) and entity.strip():
                                        entity_instances = page.search_for(entity)
                                        for entity_inst in entity_instances:
                                            if rect.contains(fitz.Rect(entity_inst)):
                                                entity_rect = fitz.Rect(entity_inst)
                                                page.add_highlight_annot(entity_rect)
                                except Exception:
                                    continue
                        annotated_sentences.add(sentence)
                    else:
                        failed_sentences.add(sentence)
                except Exception:
                    failed_sentences.add(sentence)

    try:
        pdf.save(output_pdf_path)
    except Exception:
        pass
    finally:
        pdf.close()
        return list(failed_sentences)


# Function for binary classification model inference
def classify_sentences(sentences):
    tokenizer = AutoTokenizer.from_pretrained(BINARY_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BINARY_MODEL_PATH)
    model.eval()

    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.sigmoid(logits).numpy()

    positive_sentences = []
    for sentence, probability in zip(sentences, probabilities):
        if probability[0] > 0.9:
            positive_sentences.append(sentence)

    return positive_sentences


# Function for NER model inference
def extract_entities_from_sentences(sentences):
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
    model.eval()

    def extract_entities(predictions, inputs):
        tokens = inputs['input_ids'].squeeze().tolist()
        labels = predictions['logits']
        token_predictions = torch.argmax(labels, dim=2).squeeze().tolist()

        tokenized_sentence = tokenizer.convert_ids_to_tokens(tokens)
        word_ids = inputs.word_ids()

        entities = []
        current_entity = ""
        current_label = None
        previous_word_id = -1

        for token, pred, word_id in zip(tokenized_sentence, token_predictions, word_ids):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue  # Skip special tokens

            label = model.config.id2label[pred]

            if word_id is None:
                continue

            if label != 'LABEL_0':
                if word_id != previous_word_id or current_label != label:
                    if current_entity:
                        entities.append((current_entity, current_label))
                    current_entity = token if not token.startswith("##") else token[2:]
                    current_label = label
                else:
                    if token.startswith("##"):
                        current_entity += token[2:]
                    else:
                        current_entity += token
            else:
                if current_entity:
                    entities.append((current_entity, current_label))
                    current_entity = ""
                    current_label = None

            previous_word_id = word_id

        if current_entity:
            entities.append((current_entity, current_label))

        # Merge subwords into original words based on their labels
        merged_entities = []
        i = 0
        while i < len(entities):
            entity, label = entities[i]
            # Merge consecutive LABEL_2 entities
            if label == 'LABEL_2':
                while i + 1 < len(entities) and entities[i + 1][1] == 'LABEL_2':
                    next_entity, _ = entities[i + 1]
                    entity += next_entity
                    i += 1
                merged_entities.append((entity, label))
            elif label == 'LABEL_4':
                while i + 1 < len(entities) and entities[i + 1][1] == 'LABEL_4':
                    next_entity, _ = entities[i + 1]
                    entity += next_entity
                    i += 1
                merged_entities.append((entity, label))
            else:
                merged_entities.append((entity, label))
            i += 1

        # Further merge LABEL_1 with LABEL_2 and LABEL_3 with LABEL_4
        final_entities = []
        i = 0
        while i < len(merged_entities):
            entity, label = merged_entities[i]
            if label == 'LABEL_1' and i + 1 < len(merged_entities) and merged_entities[i + 1][1] == 'LABEL_2':
                next_entity, _ = merged_entities[i + 1]
                final_entities.append((entity + next_entity, label))
                i += 2
            elif label == 'LABEL_3' and i + 1 < len(merged_entities) and merged_entities[i + 1][1] == 'LABEL_4':
                next_entity, _ = merged_entities[i + 1]
                final_entities.append((entity + next_entity, label))
                i += 2
            else:
                final_entities.append((entity, label))
                i += 1

        return final_entities

    sentences_with_entities = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)

        entities = extract_entities(outputs, inputs)
        reassembled_entities = [entity for entity, label in entities]
        sentences_with_entities.append((sentence, reassembled_entities))

    return sentences_with_entities


# Function to save results to CSV
def save_results_to_csv(sentences_with_entities, csv_path):
    with open(csv_path, mode='w', encoding='utf-8-sig', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Sentence', 'Entity 1', 'Entity 2'])
        for sentence, entities in sentences_with_entities:
            entities += [""] * (2 - len(entities))  # Ensure there are exactly two columns for entities
            csv_writer.writerow([sentence, entities[0], entities[1]])


# Function to load data for binary classification
def load_binary_data():
    hmdd4 = pd.read_csv(BINARY_DATA_PATH).dropna()
    positive_sentences = list(hmdd4.Positive)
    negative_sentences = list(hmdd4.Negative)
    Y = [1] * len(positive_sentences) + [0] * len(negative_sentences)
    X = positive_sentences + negative_sentences
    return X, Y


# Custom dataset class for binary classification
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


# Function to train binary classifier
def train_binary_classifier():
    X, Y = load_binary_data()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_PATH, num_labels=1)

    dataset = CustomDataset(X, Y, tokenizer, max_len=256)
    data_loader = DataLoader(dataset, batch_size=16, sampler=RandomSampler(dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        losses = []
        for d in tqdm(data_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits.squeeze(), labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.save_pretrained(BINARY_MODEL_PATH)
    tokenizer.save_pretrained(BINARY_MODEL_PATH)


# Function to load NER data
def load_ner_data():
    return pd.read_csv(NER_DATA_PATH)


# Preprocess data for NER
def preprocess_ner_data(data, tokenizer):
    tokenized_sentences = []
    labels = []

    for _, row in data.iterrows():
        sentence = str(row['Sentence'])
        mirna = str(row['microRNA'])
        target = str(row['Disease/Target'])

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


# Encode labels
def encode_labels(labels, label_map):
    encoded_labels = []
    for label in labels:
        encoded_labels.append([label_map[l] for l in label])
    return encoded_labels


# Decode labels
def decode_labels(encoded_labels, label_map):
    label_map_reverse = {v: k for k, v in label_map.items()}
    return [[label_map_reverse[l] for l in label] for label in encoded_labels]


# Train NER model
def train_ner_model():
    data = load_ner_data()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenized_sentences, labels = preprocess_ner_data(data, tokenizer)

    label_list = ['O', 'B-microRNA', 'I-microRNA', 'B-Disease', 'I-Disease']
    label_map = {label: i for i, label in enumerate(label_list)}

    input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_sentences]
    attention_masks = [[float(i > 0) for i in ids] for ids in input_ids]
    encoded_labels = encode_labels(labels, label_map)

    max_len = 256
    input_ids = [ids[:max_len] + [0] * (max_len - len(ids)) for ids in input_ids]
    attention_masks = [mask[:max_len] + [0] * (max_len - len(mask)) for mask in attention_masks]
    encoded_labels = [label[:max_len] + [label_map['O']] * (max_len - len(label)) for label in encoded_labels]

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    encoded_labels = torch.tensor(encoded_labels)

    dataset = TensorDataset(input_ids, attention_masks, encoded_labels)
    data_loader = DataLoader(dataset, batch_size=16, sampler=RandomSampler(dataset))

    model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL_PATH, num_labels=len(label_map))

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="no",
        logging_strategy="no",  # Disable logging
        save_strategy="no",  # Disable saving
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        num_train_epochs=10,
        weight_decay=0.01,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data]),
                                    'labels': torch.stack([f[2] for f in data])},
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(NER_MODEL_PATH)
    tokenizer.save_pretrained(NER_MODEL_PATH)


# Function to load models
def load_models():
    binary_model = AutoModelForSequenceClassification.from_pretrained(BINARY_MODEL_PATH)
    binary_tokenizer = AutoTokenizer.from_pretrained(BINARY_MODEL_PATH)
    ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
    return binary_model, binary_tokenizer, ner_model, ner_tokenizer


# Process all PDFs in a directory
def process_pdfs_in_directory(directory_path):
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    for filename in tqdm(pdf_files, desc="Processing PDFs", unit="pdf"):
        pdf_path = os.path.join(directory_path, filename)
        sentences = extract_sentences(pdf_path)
        sentences = correct_broken_hyphens(sentences)
        sentences = [sentence for sentence in sentences if
                     any(re.search(keyword, sentence, re.IGNORECASE) for keyword in keywords)]
        sentences = filter_sentences_with_citations(sentences)
        sentences = filter_long_sentences(sentences)

        positive_sentences = classify_sentences(sentences)
        sentences_with_entities = extract_entities_from_sentences(positive_sentences)

        output_pdf_path = os.path.join(OUTPUT_DIRECTORY, f"{os.path.splitext(filename)[0]}_annotated.pdf")
        failed_sentences = add_squiggly_and_highlight_annotations(pdf_path, sentences_with_entities, output_pdf_path)

        csv_path = os.path.join(OUTPUT_DIRECTORY, f"{os.path.splitext(filename)[0]}.csv")
        save_results_to_csv(sentences_with_entities, csv_path)
    print('all done!')


# Train models if not available
if not os.path.exists(BINARY_MODEL_PATH):
    train_binary_classifier()

if not os.path.exists(NER_MODEL_PATH):
    train_ner_model()

# Process PDFs in the directory
process_pdfs_in_directory(PDF_DIRECTORY)
