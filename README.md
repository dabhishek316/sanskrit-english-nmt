# Sanskrit → English Neural Machine Translation

Custom Transformer sequence-to-sequence model for Sanskrit-to-English translation, built from scratch in PyTorch (no `nn.Transformer`, no pre-trained translation models, no external APIs).

**NLU Assignment 2 — Abhishek Das (G25AIT1005)**

## Approach

- **Tokenization:** SentencePiece BPE, separate 6k vocabularies for Sanskrit (Devanagari) and English, trained on the training split only
- **Model:** 3-layer encoder + 3-layer decoder Transformer, d_model 256, 4 heads, FFN 1024, pre-norm residuals, sinusoidal positional encoding, weight-tied output embeddings — **8.6M parameters total**
- **Training:** AdamW + label smoothing 0.1, linear warmup → inverse-sqrt LR decay, 70 epochs, best checkpoint selected by dev BLEU
- **Decoding:** batched beam search (beam 4) with GNMT length penalty

## Results (test set, 1000 sentences)

| Metric | Score |
|---|---|
| BLEU (NLTK default, no weights) | 0.1109 |
| BERTScore F1 (rescale_with_baseline=True) | 0.3464 |
| Inference time (RTX 4070, beam 4) | 19.84 s |
| Trainable parameters | 8,602,624 |

## How to run

Open `Sanskrit_English_NMT.ipynb`, place the six dataset CSVs
(`train/dev/test` × `sa/en`) in the same folder, and Run All.
The first cell installs all dependencies. Outputs: `submission.csv`,
`results.json`, `training_curves.png`, `examples.csv`, `work/best_model.pt`.

For inference on a new Sanskrit CSV with a trained checkpoint, set
`PRIVATE_SA = "your_file.csv"` in the last cell and run it — no retraining needed.

## Notes

- Only the provided course dataset was used for training (not included in this repo).
- BERTScore internally uses pre-trained `roberta-large` — for evaluation only, never for translation.
