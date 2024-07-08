import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

# Load train.tsv
train_data = pd.read_csv("train-1.tsv", sep="\t")
test_data = pd.read_csv("test_YourLastName_UID-1.tsv", sep="\t")

# Define tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

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
            'label': torch.tensor(label, dtype=torch.long)
        }

# Split train data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_data['content'].values,
    train_data['label'].values,
    test_size=0.1,
    random_state=42
)

# Define training parameters
MAX_LEN = 128
BATCH_SIZE = 16

# Create datasets and dataloaders
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
criterion = nn.CrossEntropyLoss()

# Function for calculating accuracy
def accuracy(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat)

# Function for training
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch_idx, batch in enumerate(iterator):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = outputs.logits

        loss = criterion(preds, labels)

        acc = accuracy(preds.detach().cpu().numpy(), labels.detach().cpu().numpy())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(iterator)}: Loss: {loss.item():.4f} | Accuracy: {acc * 100:.2f}%')

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Function for evaluation
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits

            loss = criterion(preds, labels)

            acc = accuracy(preds.detach().cpu().numpy(), labels.detach().cpu().numpy())

            epoch_loss += loss.item()
            epoch_acc += acc

            all_preds.extend(np.argmax(preds.detach().cpu().numpy(), axis=1).tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    f1 = f1_score(all_labels, all_preds, average='binary')  # Change average parameter as needed
    print(f'\t Val. F1: {f1:.2f}')  # Print F1 score here

    return epoch_loss / len(iterator), epoch_acc / len(iterator), f1

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train the model
N_EPOCHS = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    print(f'Epoch {epoch+1}/{N_EPOCHS}')
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc, valid_f1 = evaluate(model, val_loader, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'bert_sentiment_model.pt')

    print(f'Epoch: {epoch + 1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% | Val. F1: {valid_f1:.2f}')

# Load the trained model
model.load_state_dict(torch.load('bert_sentiment_model.pt'))
model.eval()

# Define test dataset
test_texts = test_data['content'].values
test_dataset = SentimentDataset(test_texts, np.zeros(len(test_texts)), tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Make predictions on the test set
predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = outputs.logits

        predictions.extend(np.argmax(preds.detach().cpu().numpy(), axis=1).tolist())

# Replace default values (-1) of the label column with predictions
test_data['label'] = predictions

# Compute F1 score on the test data
true_labels = test_data['label']

tf1 = f1_score(true_labels, predictions, average='binary')
print(f"Test F1 score: {tf1}")

# Save the test data with predictions
test_data.to_csv("test_YourLastName_UID_predictions.tsv", sep="\t", index=False)
