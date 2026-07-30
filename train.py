# Install required libraries if you haven't already:
# !pip install transformers datasets evaluate accelerate torchvision scikit-learn

import numpy as np
import torch
import torch.nn as nn
import evaluate
from datasets import load_dataset
from torchvision.transforms import v2
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)

# ==========================================
# 1. ENVIRONMENT & DEVICE CONFIGURATION
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Optimizes Tensor Core math for the Ada Lovelace (L4) architecture
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

# ==========================================
# 2. DATASET LOADING & SPLITTING
# ==========================================
print("Loading HAM10000 dataset from Hugging Face Hub...")
# This defines the 'dataset' object by fetching it directly from the cloud Hub
raw_dataset = load_dataset("mrtg/ham10000", split="train")

# Split the dataset into 80% train and 20% test (validation)
dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
print(f"Dataset loaded. Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

# Extract class names and label mappings
labels = dataset["train"].features["label"].names
num_labels = len(labels)
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

# ==========================================
# 3. COMPUTE CLASS WEIGHTS FOR LOSS
# ==========================================
train_labels = dataset["train"]["label"]
class_counts = np.bincount(train_labels, minlength=num_labels)
total_samples = len(train_labels)

# Compute inverse frequency weights to combat severe class imbalance
class_weights = total_samples / (num_labels * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Calculated class weights for loss function: {class_weights}")

# ==========================================
# 4. PREPROCESSING & 448x448 MEDICAL AUGMENTATIONS
# ==========================================
model_checkpoint = "facebook/resnext50_32x4d"

# Force the processor to scale up target shapes to 448x448
image_processor = AutoImageProcessor.from_pretrained(
    model_checkpoint,
    size={"height": 448, "width": 448}
)

# Deep augmentation pipeline to prevent overfitting on skin textures/backgrounds
augmentations = v2.Compose([
    v2.RandomResizedCrop(size=(448, 448), scale=(0.85, 1.0), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

def train_transform(examples):
    images = [augmentations(img.convert("RGB")) for img in examples["image"]]
    inputs = image_processor(images, return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

def val_transform(examples):
    inputs = image_processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

# Apply transforms to the dataset splits dynamically to save system memory
dataset["train"].set_transform(train_transform)
dataset["test"].set_transform(val_transform)

# ==========================================
# 5. INITIALIZE MODEL & CUSTOM LOSS TRAINER
# ==========================================
print(f"Initializing ResNeXt model from checkpoint: {model_checkpoint}")
model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True  # Safely replaces ImageNet head with 7 skin lesion classes
)

# Custom Trainer subclass to inject class-weighted Cross-Entropy loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ==========================================
# 6. EVALUATION METRICS CONFIGURATION
# ==========================================
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
    # Macro F1 tracks your average performance across all 7 classes independently
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]

    return {"accuracy": acc, "macro_f1": f1}

# ==========================================
# 7. TRAINING ARGUMENTS (OPTIMIZED FOR L4)
# ==========================================
training_args = TrainingArguments(
    output_dir="./resnext50-skin-cancer-448",
    per_device_train_batch_size=32,        # Tailored for 448px tensors inside 24GB VRAM
    per_device_eval_batch_size=32,
    learning_rate=4e-5,                    # Stable learning rate for CNN backbone adjustment
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",      # Prioritize checkpoint with the best rare-class recall

    bf16=True,                             # Leverages native bfloat16 mixed-precision on L4
    dataloader_num_workers=2,              # Accelerates multi-threaded image loading
    report_to="none"
)

# ==========================================
# 8. EXECUTE TRAINING RUN
# ==========================================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

print("Starting training pipeline...")
trainer.train()
