"""
MahaNER Training and Evaluation Script
--------------------------------------
Fine-tunes transformer models (e.g., L3Cube-MahaBERT, IndicBERT, or XLM-RoBERTa)
on the Marathi Named Entity Recognition dataset (L3Cube-MahaNER or IndicNER).

Features:
- Subword tokenization and word-level label alignment with `word_ids`.
- Masking of special tokens and subsequent subtokens with label ID -100.
- Sequence labeling evaluation with Seqeval (Precision, Recall, F1).
- Model and tokenizer serialization for downstream inference.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Any

import numpy as np

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("MahaNER-Trainer")

# Canonical label definitions for Marathi NER (L3Cube-MahaNER / IndicNER scheme)
CANONICAL_LABELS = [
    "O",
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
    "B-MISC",
    "I-MISC"
]
LABEL2ID = {label: i for i, label in enumerate(CANONICAL_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(CANONICAL_LABELS)}

# Synthetic sample dataset used as a fallback if remote dataset cannot be downloaded
FALLBACK_DATASET = {
    "train": [
        {
            "tokens": ["छत्रपती", "शिवाजी", "महाराज", "यांनी", "रायगडावर", "हिंदवी", "स्वराज्याची", "स्थापना", "केली", "."],
            "ner_tags": [1, 2, 2, 0, 3, 7, 8, 0, 0, 0]  # B-PER, I-PER, I-PER, O, B-LOC, B-MISC, I-MISC, O, O, O
        },
        {
            "tokens": ["महाराष्ट्राचे", "मुख्यमंत्री", "देवेंद्र", "फडणवीस", "यांनी", "मुंबई", "येथे", "नवीन", "मेट्रो", "मार्गाची", "घोषणा", "केली", "."],
            "ner_tags": [3, 0, 1, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0]
        },
        {
            "tokens": ["सचिन", "तेंडुलकरने", "वानखेडे", "स्टेडियमवर", "आपला", "शेवटचा", "आंतरराष्ट्रीय", "क्रिकेट", "सामना", "खेळला", "."],
            "ner_tags": [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0]
        },
        {
            "tokens": ["भारतीय", "अंतराळ", "संशोधन", "संस्था", "(", "ISRO", ")", "ने", "श्रीहरिकोटा", "येथून", "चांद्रयान", "मोहिमेचे", "प्रक्षेपण", "केले", "."],
            "ner_tags": [5, 6, 6, 6, 0, 5, 0, 0, 3, 0, 7, 0, 0, 0, 0]
        },
        {
            "tokens": ["पुणे", "विद्यापीठात", "सावित्रीबाई", "फुले", "यांच्या", "पुतळ्याचे", "अनावरण", "झाले", "."],
            "ner_tags": [3, 4, 1, 2, 0, 0, 0, 0, 0]
        },
        {
            "tokens": ["टाटा", "मोटर्सने", "पुण्यात", "नवीन", "ईव्ही", "प्रकल्प", "सुरू", "केला", "."],
            "ner_tags": [5, 6, 3, 0, 7, 0, 0, 0, 0]
        }
    ],
    "validation": [
        {
            "tokens": ["लता", "मंगेशकर", "यांचा", "जन्म", "इंदूर", "येथे", "झाला", "."],
            "ner_tags": [1, 2, 0, 0, 3, 0, 0, 0]
        },
        {
            "tokens": ["रिझर्व्ह", "बँक", "ऑफ", "इंडियाचे", "मुख्यालय", "मुंबई", "येथे", "आहे", "."],
            "ner_tags": [5, 6, 6, 6, 0, 3, 0, 0, 0]
        }
    ]
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Marathi Named Entity Recognition Model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="l3cube-pune/marathi-bert-v2",
        help="Base model checkpoint (e.g. l3cube-pune/marathi-bert-v2, ai4bharat/indic-bert, or xlm-roberta-base)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="l3cube-pune/maha-ner",
        help="HuggingFace dataset repository name or local path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./saved_marathi_ner_model",
        help="Directory where trained model checkpoints and tokenizer will be saved"
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Train batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Eval batch size per device")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Peak learning rate for AdamW")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for linear schedule")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length after subword tokenization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dry_run", action="store_true", help="Execute a quick single-step dry run on fallback data")
    return parser.parse_args()


def load_ner_dataset(dataset_name: str, dry_run: bool = False):
    """
    Loads the Marathi NER dataset from HuggingFace Hub, or uses fallback data if offline/dry_run.
    """
    from datasets import Dataset, DatasetDict

    if dry_run:
        logger.info("Running in dry-run mode. Using local Marathi NER sample dataset.")
        train_ds = Dataset.from_list(FALLBACK_DATASET["train"])
        val_ds = Dataset.from_list(FALLBACK_DATASET["validation"])
        return DatasetDict({"train": train_ds, "validation": val_ds}), LABEL2ID, ID2LABEL

    try:
        from datasets import load_dataset
        logger.info(f"Attempting to download/load dataset '{dataset_name}' from Hugging Face Hub...")
        raw_dataset = load_dataset(dataset_name)
        logger.info(f"Successfully loaded dataset: {raw_dataset}")

        # Extract label list from dataset features if available
        features = raw_dataset["train"].features
        if "ner_tags" in features and hasattr(features["ner_tags"].feature, "names"):
            label_list = features["ner_tags"].feature.names
            label2id = {label: i for i, label in enumerate(label_list)}
            id2label = {i: label for i, label in enumerate(label_list)}
            logger.info(f"Dataset labels: {label_list}")
            return raw_dataset, label2id, id2label
        return raw_dataset, LABEL2ID, ID2LABEL

    except Exception as exc:
        logger.warning(
            f"Could not load dataset '{dataset_name}' from Hugging Face Hub ({exc}). "
            f"Falling back to local high-fidelity Marathi demonstration dataset."
        )
        train_ds = Dataset.from_list(FALLBACK_DATASET["train"])
        val_ds = Dataset.from_list(FALLBACK_DATASET["validation"])
        return DatasetDict({"train": train_ds, "validation": val_ds}), LABEL2ID, ID2LABEL


def tokenize_and_align_labels(examples, tokenizer, label2id: Dict[str, int], max_length: int = 128):
    """
    Aligns word-level Marathi NER labels with subword tokens generated by WordPiece or SentencePiece.
    Assigns -100 to special tokens ([CLS], [SEP], etc.) and non-initial subwords.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length"
    )

    labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Special tokens are marked with word_idx = None -> map to -100
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword token for this word
                curr_label = label_seq[word_idx]
                if isinstance(curr_label, str):
                    label_id = label2id.get(curr_label, 0)
                else:
                    label_id = curr_label
                label_ids.append(label_id)
            else:
                # Subsequent subwords within the same word: mask with -100 to avoid double counting
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def build_compute_metrics(id2label: Dict[int, str]):
    """
    Constructs the seqeval metrics evaluation function for Hugging Face Trainer.
    """
    from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Filter out special tokens (-100)
        true_predictions = [
            [id2label[p_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
            for prediction, label in zip(predictions, labels)
        ]

        prec = precision_score(true_labels, true_predictions, zero_division=0)
        rec = recall_score(true_labels, true_predictions, zero_division=0)
        f1 = f1_score(true_labels, true_predictions, zero_division=0)

        report = classification_report(true_labels, true_predictions, zero_division=0)
        logger.info("\nSeqeval Classification Report:\n" + report)

        return {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

    return compute_metrics


def main():
    args = parse_arguments()
    logger.info("=== Starting MahaNER Training Script ===")
    logger.info(f"Arguments: {args}")

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
        set_seed
    )

    set_seed(args.seed)

    # 1. Load Dataset
    raw_dataset, label2id, id2label = load_ner_dataset(args.dataset_name, dry_run=args.dry_run)
    logger.info(f"Using {len(label2id)} classes: {label2id}")

    # 2. Load Tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # 3. Preprocess and Align Labels
    logger.info("Tokenizing and aligning dataset labels...")
    tokenized_datasets = raw_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id, args.max_length),
        batched=True,
        remove_columns=raw_dataset["train"].column_names
    )

    # 4. Load Model
    logger.info(f"Loading base model: {args.model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # 5. Define Training Arguments
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch" if "validation" in tokenized_datasets else "no",
        save_strategy="epoch" if "validation" in tokenized_datasets else "no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10 if not args.dry_run else 1,
        save_total_limit=2,
        load_best_model_at_end=True if "validation" in tokenized_datasets else False,
        metric_for_best_model="f1" if "validation" in tokenized_datasets else None,
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
    )

    # 6. Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 7. Trainer Setup
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation", None),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(id2label) if "validation" in tokenized_datasets else None,
    )

    # 8. Run Training
    logger.info("Starting training loop...")
    train_result = trainer.train()
    metrics = train_result.metrics
    logger.info(f"Training metrics: {metrics}")

    # 9. Evaluation
    if "validation" in tokenized_datasets:
        logger.info("Evaluating on validation set...")
        eval_metrics = trainer.evaluate()
        logger.info(f"Eval metrics: {eval_metrics}")

    # 10. Save Model and Tokenizer
    logger.info(f"Saving final model and tokenizer to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Model saved successfully! Ready for inference in MahaNER app.")


if __name__ == "__main__":
    main()
