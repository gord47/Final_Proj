import os
import kagglehub
import torch

import pandas as pd
import torch.nn as nn
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from model import HateSpeechClassifier
from utils import HateSpeechDataset, clean_text, collate_fn


def main():
    path = kagglehub.dataset_download("waalbannyantudre/hate-speech-detection-curated-dataset")

    hate_speech_data = pd.read_csv(os.path.join(path, "HateSpeechDatasetBalanced.csv"))
    print(hate_speech_data.head())
    print(hate_speech_data['Label'].value_counts())

    hate_speech_data['Cleaned_Content'] = hate_speech_data['Content'].apply(clean_text)
    train_data, test_data = train_test_split(hate_speech_data, test_size=0.2, random_state=42, shuffle=True, stratify=hate_speech_data['Label'])
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_dataset = HateSpeechDataset(tokenizer=tokenizer, data=train_data)
    test_dataset = HateSpeechDataset(tokenizer=tokenizer, data=test_data)
    batch_size = 64
    num_epochs = 3
    dataloader_train = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2,
                                collate_fn=collate_fn,
                                pin_memory=True)

    dataloader_test = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=2,
                                collate_fn=collate_fn,
                                pin_memory=True)
    model = HateSpeechClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    num_training_steps = len(dataloader_train) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': 2e-5},
        {'params': model.classifier.parameters(), 'lr': 5e-4}
    ], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    loss_fn = nn.BCEWithLogitsLoss()

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    count_train = 0
    count_test = 0
    for epoch in range(num_epochs):
        print(f"epoch {epoch+1}/{num_epochs}")
        model.train()
        correct_train = 0
        total_train = 0
        total_train_loss = 0
        for i, batch in enumerate(tqdm(dataloader_train)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Squeeze the attention mask to remove the extra dimension if present
            if attention_mask.dim() > 2:
                attention_mask = attention_mask.squeeze(1)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            # train_losses.append(loss.item())
            preds = (torch.sigmoid(outputs) > 0.5).long()  # threshold at 0.5
            preds = preds.view(-1)
            labels = labels.view(-1)
            correct_train += (preds == labels.long()).sum().item()
            total_train += labels.size(0)
            count_train += 1

        avg_train_loss = total_train_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)

        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()
        total_test_loss = 0
        correct_test = 0
        total_test = 0
        preds = list()
        truths = list()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader_test)):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # If attention_mask has more than 2 dimensions, squeeze it
                if attention_mask.dim() > 2:
                    attention_mask = attention_mask.squeeze()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.squeeze(), labels.float())
                total_test_loss += loss.item()
                # test_losses.append(loss.item())
                preds = (torch.sigmoid(outputs) > 0.5).long()
                preds = preds.view(-1)
                correct_test += (preds == labels.long()).sum().item()
                total_test += labels.size(0)
                count_test += 1

        avg_test_loss = total_test_loss / len(dataloader_test)
        test_losses.append(avg_test_loss)
        test_accuracy = correct_test / total_test
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}")
        print(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    
    torch.save(model.state_dict(), 'hate_speech_model_weights.pth')


if __name__ == "__main__":
    main()
