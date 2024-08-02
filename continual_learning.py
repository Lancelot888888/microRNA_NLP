# This code is partially taken from https://github.com/nlpie-research/Compact-Biomedical-Transformers/blob/main/BioDistilBERT-Training.py
# We would like to extend our sincerest gratitude towards them

from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback, ProgressCallback
from datasets import load_from_disk
import torch

# Load the tokenized dataset from disk
tokenized_ds = load_from_disk("path/to/tokenized_dataset/") 

# Define the model path
model_path = "path/to/model/"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer/")
model = AutoModelForMaskedLM.from_pretrained(model_path)

# Print the tokenizer configuration
print(tokenizer)

# Count and print the number of trainable parameters
count = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(f"Trainable Parameters: {count / 1e6} million")

# Define the data collator for masked language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    return_tensors="pt"
)

# Set up logging
save_path = "path/to/save_model/"

# Initialize logging file
try:
    with open(save_path + "logs.txt", "w+") as f:
        f.write("")
except Exception as e:
    print(f"Error initializing log file: {e}")

# Define custom callback for logging
class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)  # Remove unnecessary log information
        if state.is_local_process_zero:
            print(logs)
            with open(save_path + "logs.txt", "a+") as f:
                f.write(str(logs) + "\n")

# Define training arguments
training_arguments = TrainingArguments(
    output_dir=save_path + "checkpoints",
    logging_steps=500,
    overwrite_output_dir=True,
    save_steps=2500,
    num_train_epochs=1,  # Set the desired number of epochs
    learning_rate=1e-4,
    lr_scheduler_type="linear",
    warmup_steps=10000,
    per_device_train_batch_size=24,  # Adjust if not using 8 GPUs
    weight_decay=0.01,
    save_total_limit=5,
    remove_unused_columns=True,
)

# Initialize Trainer and start training
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_ds,
    data_collator=data_collator,
    callbacks=[ProgressCallback(), CustomCallback()],
)

# Train the model
trainer.train()

# Save the final model
trainer.save_model(save_path + "final/model/")
