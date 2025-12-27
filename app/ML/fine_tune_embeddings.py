import os, json, random
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

TRAIN_CSV = os.getenv("EMB_TRAIN_CSV", "app/data/train_data.csv")
BASE_MODEL = os.getenv("EMB_BASE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
OUTPUT_DIR = os.getenv("EMB_OUTPUT_DIR", "app/embedding_model_custom")

df = pd.read_csv(TRAIN_CSV)
train_examples = [InputExample(texts=[row["text1"], row["text2"]], label=float(row["label"])) for _, row in df.iterrows()]
random.shuffle(train_examples)

split = int(0.9 * len(train_examples))
train_data = train_examples[:split]
dev_data = train_examples[split:]

model = SentenceTransformer(BASE_MODEL)
train_dl = DataLoader(train_data, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Evaluador simple: correlación Spearman en el dev set
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_data, name="dev")

model.fit(
    train_objectives=[(train_dl, train_loss)],
    epochs=4,
    warmup_steps=max(10, len(train_dl)//10),
    output_path=OUTPUT_DIR,
    evaluator=evaluator,
    evaluation_steps=max(50, len(train_dl)),
    show_progress_bar=True
)

print(f"✅ Modelo guardado en: {OUTPUT_DIR}")
