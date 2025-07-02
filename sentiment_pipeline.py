# sentiment_pipeline.py

import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, pipeline
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load dataset
dataset = load_dataset('imdb')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize_function(example):
    return tokenizer(example['text'], truncation=True)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split dataset
small_train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(2000))
small_test_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(1000))

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Save model
model.save_pretrained("sentiment-bert")
tokenizer.save_pretrained("sentiment-bert")

# Inference pipeline
def predict_sentiment(text):
    classifier = pipeline("sentiment-analysis", model="sentiment-bert", tokenizer="sentiment-bert")
    return classifier(text)

# Example usage
if __name__ == "__main__":
    result = predict_sentiment("This movie was great!")
    print(result)