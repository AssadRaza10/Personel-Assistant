from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

model_name = "google/flan-t5-base"
model_path = "./models/flan_t5"

print("Downloading FLAN-T5 model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

os.makedirs(model_path, exist_ok=True)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print("Model saved locally!")