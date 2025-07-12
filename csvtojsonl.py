import csv
import json

input_path = "data/balanced_sentiment_dataset.csv"
output_path = "data/balanced_sentiment_dataset.jsonl"

label_map = {
    "1": "positive",
    "0": "negative"
}

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    reader = csv.reader(infile)
    for row in reader:
        if len(row) < 2:
            continue  # skip malformed lines
        label, text = row[0].strip(), row[1].strip()
        label_str = label_map.get(label)
        if label_str:
            obj = {"text": text, "label": label_str}
            outfile.write(json.dumps(obj) + "\n")
