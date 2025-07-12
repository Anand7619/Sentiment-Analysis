import argparse
import json
import os
import random
import numpy as np
import torch
import time

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TFAutoModelForSequenceClassification,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    return Dataset.from_list(lines)


def label_to_id(example):
    label = example["label"]
    if isinstance(label, str):
        label = label.lower()
        if label == "positive":
            example["label"] = 1
        elif label == "negative":
            example["label"] = 0
        else:
            raise ValueError(f"Unknown label: {label}")
    return example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", required=True)
    parser.add_argument("-epochs", type=int, default=3)
    parser.add_argument("-lr", type=float, required=True)
    parser.add_argument("--framework", choices=["pt", "tf"], default="pt")
    args = parser.parse_args()

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"🔁 Loading data from {args.data}")
    dataset = load_data(args.data).map(label_to_id)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    dataset = dataset.map(tokenize, batched=True)

    if args.framework == "pt":
        run_pytorch(dataset, tokenizer, args)
    else:
        import tensorflow as tf
        run_tensorflow(dataset, tokenizer, args)


def run_pytorch(dataset, tokenizer, args):
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_scheduler

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.train()

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_training_steps = args.epochs * len(dataloader)
    scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    print("🚀 Starting PyTorch training...")
    start = time.time()

    for epoch in range(args.epochs):
        print(f"📘 Epoch {epoch+1}/{args.epochs}")
        for batch in dataloader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"]
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        print(f"✅ Finished epoch {epoch+1}")

    end = time.time()
    print(f"⏱️ PyTorch training completed in {end - start:.2f} seconds")

    model.save_pretrained("./backend/model")
    tokenizer.save_pretrained("./backend/model")
    print("✅ PyTorch model saved to ./model")


def run_tensorflow(dataset, tokenizer, args):
    import tensorflow as tf

    tf_dataset = dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],
        shuffle=True,
        batch_size=8,
    )

    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print("🚀 Starting TensorFlow training...")
    start = time.time()
    model.fit(tf_dataset, epochs=args.epochs)
    end = time.time()
    print(f"⏱️ TensorFlow training completed in {end - start:.2f} seconds")

    model.save_pretrained("./backend/model")
    tokenizer.save_pretrained("./backend/model")
    print("✅ TensorFlow model saved to ./model")


if __name__ == "__main__":
    main()
