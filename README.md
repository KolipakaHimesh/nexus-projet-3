import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split

# Assuming you have a dataset with text and corresponding sentiment labels
class CustomDataset(Dataset):
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
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Function to train the sentiment analysis model
def train_model(train_dataloader, val_dataloader, model, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # Evaluation on validation data
        model.eval()
        val_loss = 0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct_preds += torch.sum(preds == labels).item()
                total_preds += len(labels)

        val_accuracy = correct_preds / total_preds
        print(f'Epoch {epoch + 1}/{epochs}, '
              f'Training Loss: {loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.4f}')


# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2) # Assuming binary classification

# Assuming you have loaded your dataset into `texts` and `labels`
texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2)

# Define dataset and dataloaders
train_dataset = CustomDataset(texts_train, labels_train, tokenizer, max_len=128)
val_dataset = CustomDataset(texts_val, labels_val, tokenizer, max_len=128)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Send model to device
model.to(device)

# Define optimizer and hyperparameters
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
epochs = 3

# Train the model
train_model(train_dataloader, val_dataloader, model, optimizer, epochs)
