import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW, AutoModelForMaskedLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Prepare dataset class
class MisspelledDataset(Dataset):
    def __init__(self, misspelled_correct_pairs, tokenizer, max_length=128):
        self.pairs = misspelled_correct_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        misspelled, correct = self.pairs[idx]
        sentence = f"I have a {misspelled}."
        # Mask the misspelled word
        masked_sentence = sentence.replace(misspelled, '[MASK]')

        # Tokenize the masked sentence
        encoding = self.tokenizer(masked_sentence, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        # Tokenize the correct word for the label
        correct_encoding = self.tokenizer(correct, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        # Return tokenized inputs and labels
        return {
            'input_ids': encoding.input_ids.squeeze(),
            'attention_mask': encoding.attention_mask.squeeze(),
            'labels': correct_encoding.input_ids.squeeze()
        }

# Load dataset
def load_dataset(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Drop rows with NaN values
    df = df.dropna()

    # Convert columns to strings just in case
    df['misspelled'] = df['misspelled'].astype(str)
    df['correct'] = df['correct'].astype(str)

    # Create pairs
    dataset_pairs = list(zip(df['misspelled'], df['correct']))
    return dataset_pairs


# Train model
def train_model(dataset_pairs):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Split dataset
    train_pairs, val_pairs = train_test_split(dataset_pairs, test_size=0.2, random_state=42)

    # Create DataLoader
    train_dataset = MisspelledDataset(train_pairs, tokenizer)
    val_dataset = MisspelledDataset(val_pairs, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 10

    # Tracking metrics
    train_losses, val_losses, f1_scores = [], [], []

    best_f1 = 0
    patience = 3
    early_stopping_counter = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss, predictions, targets = 0, [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy().flatten())
                targets.extend(labels.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate F1 Score
        f1 = f1_score(targets, predictions, average='weighted')
        f1_scores.append(f1)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, F1 Score: {f1}")

        # Early Stopping
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained("best_spell_corrector_bert")
            tokenizer.save_pretrained("best_spell_corrector_bert")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

    # Plot Metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(f1_scores) + 1), f1_scores, label='F1 Score', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score Over Epochs')

    plt.tight_layout()
    plt.savefig('bert_experiment_results.png')
    plt.show()

if __name__ == "__main__":
    dataset_pairs = load_dataset('birkbeck_dataset.csv')
    train_model(dataset_pairs)
