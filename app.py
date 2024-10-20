import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, \
    DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score
import pandas as pd
import shap
import string
import numpy as np
import optuna
import os

app = Flask(__name__)


splits = {
    'train': 'split/train-00000-of-00001.parquet',
    'validation': 'split/validation-00000-of-00001.parquet',
    'test': 'split/test-00000-of-00001.parquet'
}


train_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["train"])
validation_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["validation"])
test_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["test"])


num_labels = len(train_df['label'].unique())
print(f"Number of unique labels: {num_labels}")


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)


stop_words = set(["the", "is", "in", "and", "to", "a", "of", "for", "on", "with", "at", "by", "an", "be", "this"])
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


X_train = train_df['text'].apply(preprocess_text)
y_train = train_df['label'] - train_df['label'].min()


train_encodings = bert_tokenizer(list(X_train), truncation=True, padding=True, max_length=128)


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'f1': f1_score(labels, predictions, average='weighted')}


def model_init(model_type):
    try:
        if model_type == "bert":
            return BertForSequenceClassification.from_pretrained('./bert_model_trained', num_labels=num_labels)
        else:
            return DistilBertForSequenceClassification.from_pretrained('./distilbert_model_trained', num_labels=num_labels)
    except OSError:
        print("Trained model not found. Using default pre-trained model.")
        if model_type == "bert":
            return BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        else:
            return DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)


def objective(trial, model_type):
    model = model_init(model_type)
    tokenizer = bert_tokenizer if model_type == "bert" else distilbert_tokenizer

    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(validation_df['text'].apply(preprocess_text)), truncation=True, padding=True, max_length=128)

    train_dataset = EmotionDataset(train_encodings, y_train.tolist())
    val_dataset = EmotionDataset(val_encodings, (validation_df['label'] - validation_df['label'].min()).tolist())

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-5)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32])
    num_train_epochs = trial.suggest_int('num_train_epochs', 2, 3)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_f1"]


best_trials = {"bert": None, "distilbert": None}

def run_tuning(model_type):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, model_type), n_trials=3)

    best_trial = study.best_trial
    best_trials[model_type] = best_trial

    if model_type == "bert":
        if not os.path.exists("./bert_model_trained"):
            os.makedirs("./bert_model_trained")
        bert_model.config.update(best_trial.params)
        bert_model.save_pretrained('/app/models/bert_model_trained')
    else:
        if not os.path.exists("./distilbert_model_trained"):
            os.makedirs("./distilbert_model_trained")
        distilbert_model.config.update(best_trial.params)
        distilbert_model.save_pretrained('/app/models/distilbert_model_trained')

    print(f"Best trial for {model_type}:")
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

run_tuning("bert")
run_tuning("distilbert")


@app.route('/model', methods=['GET'])
def get_model_info():
    class_distribution = y_train.value_counts().to_dict()
    model_info = {
        "bert_model": "BERT base uncased",
        "distilbert_model": "DistilBERT base uncased",
        "class_distribution": class_distribution,
        "bert_best_hyperparameters": best_trials["bert"].params if best_trials["bert"] else "Not tuned yet",
        "distilbert_best_hyperparameters": best_trials["distilbert"].params if best_trials["distilbert"] else "Not tuned yet",
        "bert_best_f1": best_trials["bert"].value if best_trials["bert"] else "Not available",
        "distilbert_best_f1": best_trials["distilbert"].value if best_trials["distilbert"] else "Not available"
    }
    return jsonify(model_info)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = preprocess_text(data['text'])

    bert_inputs = bert_tokenizer(text, return_tensors='pt')
    distilbert_inputs = distilbert_tokenizer(text, return_tensors='pt')

    bert_output = bert_model(**bert_inputs)
    distilbert_output = distilbert_model(**distilbert_inputs)

    bert_pred = torch.argmax(bert_output.logits, dim=1).item()
    distilbert_pred = torch.argmax(distilbert_output.logits, dim=1).item()

    explainer = shap.Explainer(bert_model, tokenizer=bert_tokenizer)
    shap_values = explainer([text])

    shap.summary_plot(shap_values, show=False)

    response = {
        "bert_prediction": bert_pred,
        "distilbert_prediction": distilbert_pred,
        "shap_summary": "SHAP plot generated successfully."
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
