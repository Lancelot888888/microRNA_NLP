import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, ProgressCallback, TrainerCallback, DataCollatorForLanguageModeling
from transformers.modeling_outputs import MaskedLMOutput
from datasets import load_from_disk

# Load the tokenized dataset from disk
tokenized_ds = load_from_disk("path/to/tokenized_dataset/")


# Function to initialize the half-sized student model
def initializeStudent(save_path, teacher_model_name):
    # Load the teacher model
    teacher_model = AutoModelForMaskedLM.from_pretrained(teacher_model_name)

    # Create a new configuration based on the teacher model's configuration
    distil_model_config = teacher_model.config.to_dict()
    # Reduce the number of hidden layers by half
    distil_model_config["num_hidden_layers"] //= 2

    # Create a new model using the modified configuration
    distillation_model = AutoModelForMaskedLM.from_config(AutoModelForMaskedLM.from_dict(distil_model_config))

    # Copy embeddings from the teacher model to the student model
    distillation_model.embeddings = teacher_model.embeddings

    # Copy every second layer from the teacher model to the student model
    for index, layer in enumerate(distillation_model.encoder.layer):
        distillation_model.encoder.layer[index] = teacher_model.encoder.layer[2 * index + 1]

    # Save the initialized student model
    distillation_model.save_pretrained(save_path)
    return save_path


# Initialize the half-sized student model
model_name = "path/to/teacher_model/"
initializeStudent("path/to/save_student_model/", model_name)

# Load the initialized student model and the teacher model
student = AutoModelForMaskedLM.from_pretrained("path/to/save_student_model/")
teacher = AutoModelForMaskedLM.from_pretrained(model_name)

# Freeze the teacher model parameters to avoid updating them during training
for param in teacher.parameters():
    param.requires_grad = False


# Define a distillation wrapper class to compute losses and perform forward pass
class DistillationWrapper(nn.Module):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.attention_loss = nn.KLDivLoss(reduction="mean")  # Loss for attention maps
        self.hidden_loss = nn.CosineEmbeddingLoss(reduction="mean")  # Loss for hidden states
        self.output_loss = nn.KLDivLoss(reduction="batchmean")  # Loss for output logits
        self.temperature = 1.0  # Temperature for distillation

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get the student model's outputs
        student_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                       output_hidden_states=True, output_attentions=True, **kwargs)

        # Get the teacher model's outputs without computing gradients
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask,
                                           output_hidden_states=True, output_attentions=True, **kwargs)

        # Extract the necessary outputs from both models
        s_attentions = student_outputs.attentions
        t_attentions = [att.detach() for att in teacher_outputs.attentions]
        s_hiddens = student_outputs.hidden_states[1:]
        t_hiddens = [hidden.detach() for hidden in teacher_outputs.hidden_states[1:]]
        s_logits = student_outputs.logits
        t_logits = teacher_outputs.logits.detach()

        # Compute individual losses
        att_loss = self.compute_attention_loss(s_attentions, t_attentions, attention_mask)
        hidden_loss = self.compute_hidden_loss(s_hiddens, t_hiddens, attention_mask)
        output_loss = self.compute_output_loss(s_logits, t_logits, labels)

        # Combine the individual losses with specific weights
        total_loss = student_outputs.loss + 3.0 * (att_loss + hidden_loss) + 5.0 * output_loss
        return MaskedLMOutput(loss=total_loss, logits=student_outputs.logits,
                              hidden_states=student_outputs.hidden_states, attentions=student_outputs.attentions)

    def compute_output_loss(self, s_logits, t_logits, labels):
        # Mask out the padding tokens for loss computation
        mask = (labels > -1).unsqueeze(-1).expand_as(s_logits).bool()
        s_logits_slct = torch.masked_select(s_logits, mask).view(-1, s_logits.size(-1))
        t_logits_slct = torch.masked_select(t_logits, mask).view(-1, s_logits.size(-1))

        # Compute the output logits loss using KL divergence
        output_loss = self.output_loss(F.log_softmax(s_logits_slct / self.temperature, dim=-1),
                                       F.softmax(t_logits_slct / self.temperature, dim=-1)) * (self.temperature ** 2)
        return output_loss

    def compute_attention_loss(self, s_attentions, t_attentions, attention_mask):
        total_loss = None
        for index, s_map in enumerate(s_attentions):
            # Extract the corresponding teacher attention map
            t_map = t_attentions[2 * index + 1]

            # Create a mask for the attention loss
            att_loss_mask = attention_mask.unsqueeze(1).repeat(1, s_map.size(1), 1).unsqueeze(-1).expand_as(
                s_map).bool()
            s_map_slct = torch.masked_select(s_map, att_loss_mask).view(-1, s_map.size(-1)) + 1e-12
            t_map_slct = torch.masked_select(t_map, att_loss_mask).view(-1, s_map.size(-1)) + 1e-12

            # Compute the attention loss using KL divergence
            att_loss = self.attention_loss(torch.log(s_map_slct), t_map_slct)
            total_loss = att_loss if total_loss is None else total_loss + att_loss
        return total_loss

    def compute_hidden_loss(self, s_hiddens, t_hiddens, attention_mask):
        total_loss = None
        for index in range(len(s_hiddens)):
            s_hidden_states = s_hiddens[index]
            t_hidden_states = t_hiddens[2 * index + 1]

            # Create a mask for the hidden states loss
            mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states).bool()
            s_hidden_states_slct = torch.masked_select(s_hidden_states, mask).view(-1, s_hidden_states.size(-1))
            t_hidden_states_slct = torch.masked_select(t_hidden_states, mask).view(-1, t_hidden_states.size(-1))

            # Create a target tensor for cosine embedding loss
            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)

            # Compute the hidden states loss using cosine embedding loss
            hidden_loss = self.hidden_loss(s_hidden_states_slct, t_hidden_states_slct, target)
            total_loss = hidden_loss if total_loss is None else total_loss + hidden_loss
        return total_loss


# Instantiate the distillation model
model = DistillationWrapper(student=student, teacher=teacher)

# Count and print the number of trainable parameters
count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {count / 1e6} million")

# Data collator for language modeling, including masking
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer/")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
                                                return_tensors="pt")

# Training arguments setup
save_path = "path/to/save_model/"
training_arguments = TrainingArguments(
    output_dir=save_path + "checkpoints",
    logging_steps=250,
    overwrite_output_dir=True,
    save_steps=2500,
    num_train_epochs=1,  # Set the desired number of epochs
    learning_rate=5e-4,
    lr_scheduler_type="linear",
    warmup_steps=5000,
    per_device_train_batch_size=24,  # Adjusted from per_gpu_train_batch_size to per_device_train_batch_size
    weight_decay=1e-4,
    save_total_limit=5,
    remove_unused_columns=True,
)


# Custom callback to log training progress
class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)  # Remove unnecessary log information
        if state.is_local_process_zero:
            print(logs)
            with open(save_path + "logs.txt", "a+") as f:
                f.write(str(logs) + "\n")


# Trainer setup
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_ds,
    data_collator=data_collator,
    callbacks=[ProgressCallback(), CustomCallback()],
)

# Train the model
trainer.train()

# Save the final model checkpoint
trainer.save_model(save_path + "final/rawModel/")


# Function to load and save the pretrained model
def load_and_save_pretrained(model, checkpoint_path, save_path):
    print(model.load_state_dict(torch.load(checkpoint_path)))  # Load the model state from checkpoint
    model.student.save_pretrained(save_path)  # Save the student model in the desired format
    return model


# Load the raw model and save it as the final model
load_and_save_pretrained(model, save_path + "final/rawModel/pytorch_model.bin", save_path + "final/model/")