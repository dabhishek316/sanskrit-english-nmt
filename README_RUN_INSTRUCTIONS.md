# Sanskrit→English NMT — Run Instructions

## Tonight (before 4 July 11:59 PM)

1. **Run the notebook on your RTX 4070** (`Sanskrit_English_NMT.ipynb`)
   - Put the 6 CSV files in the same folder as the notebook (or set `CFG.data_dir`).
   - Run all cells top to bottom. Training ≈ 5–10 min on the 4070; BERTScore's first run
     downloads roberta-large (~1.4 GB) once.
   - Outputs produced automatically: `submission.csv`, `results.json`,
     `training_curves.png`, `examples.csv`, `work/best_model.pt`.

2. **Fill the report** (`Report_Sanskrit_NMT.docx`)
   - Replace every yellow-highlighted bracket with numbers from `results.json` /
     the notebook printout.
   - Insert `training_curves.png` in Section 4 (Insert → Picture).
   - Paste 6 rows from `examples.csv` into the Section 5 table and write 2–3 lines
     of error analysis based on what you actually see.
   - Add your name + roll number at the top. Export as PDF (File → Save As → PDF).

3. **Package the ZIP**: `submission.csv` + report PDF + the `.ipynb`
   (with outputs saved — run "Restart & Run All" once so cell outputs are embedded).

4. **GitHub**: push the notebook + README to a public repo (rule requirement).

## Evaluation day (5 July)

- Keep `work/best_model.pt` and the two `.model` SentencePiece files.
- Use the last notebook cell: set `PRIVATE_SA = "private_sa.csv"` and run — it
  reuses the trained checkpoint, prints inference time + parameter count, and
  writes `submission_private.csv`. No retraining needed.

## If dev BLEU looks low after 40 epochs

Quick levers (one line each in `CFG`):
- `epochs = 70` (still fast on the 4070)
- `d_model = 320`, `d_ff = 1280` (a bit more capacity; params still ~13M)
- `beam_size = 6` (slower inference, usually +0.2–0.5 BLEU)

## Compliance notes (already handled in code/report)

- Custom seq2seq built from scratch — no `nn.Transformer`, no external APIs.
- Only the provided dataset is used for training.
- BERTScore's internal roberta-large is disclosed in the report (evaluation only).
- BLEU = default NLTK `corpus_bleu`, no weights; BERTScore F1 with
  `rescale_with_baseline=True` — exactly as the assignment specifies.
