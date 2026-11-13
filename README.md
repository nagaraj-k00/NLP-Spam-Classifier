# NLP Spam Email Classifier

Deep-learning spam classifier built with TensorFlow/Keras — compares five recurrent architectures (SimpleRNN, LSTM, GRU, Bidirectional LSTM, Bidirectional GRU) on the same labelled email dataset.

**Status:** Completed (June 2025 – September 2025)
**Tech stack:** Python · TensorFlow · Keras · NLTK · Scikit-learn · Pandas

## Result

- **98%+ accuracy** on the held-out test set (best models), benchmarked across all five architectures
- Full evaluation per model: classification report, confusion matrix, and accuracy/loss curves, with false positives kept in check

## What's inside

`NLP_Spam_classifier.ipynb` — a single end-to-end notebook:

1. **EDA** — class balance, dataset shape, null checks on the `emails.csv` dataset (columns: `Text`, `Spam`)
2. **Preprocessing pipeline** — text cleaning (URL/HTML removal), stop-word removal, Porter stemming, Bag-of-Words exploration
3. **Sequence prep** — Keras Tokenizer, padding to max length 200, 80/20 train–test split
4. **Model comparison** — Embedding + SimpleRNN, LSTM, GRU, Bidirectional LSTM, and Bidirectional GRU, each trained for 5 epochs with Adam (lr 0.001)
5. **Evaluation & prediction** — per-model classification reports and confusion matrices, a combined accuracy comparison chart, and a `predict_spam` helper for classifying new messages

## How to run

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow
```

1. Place your dataset as `emails.csv` next to the notebook (expected columns: `Text`, `Spam`). The original dataset is not included in this repo.
2. Open `NLP_Spam_classifier.ipynb` in Jupyter / VS Code / Google Colab.
3. Run all cells top to bottom — the notebook downloads the NLTK `punkt` and `stopwords` corpora itself on first run.

## What I built

- Built an NLP spam-detection system with TensorFlow/Keras sequence models, reaching 98%+ test accuracy
- Engineered the preprocessing pipeline (cleaning, stop-word removal, stemming, tokenization, padding)
- Benchmarked five RNN-family architectures side by side instead of reporting a single model
- Evaluated with classification reports and confusion matrices to keep false positives low

---
*By Nagaraj Kamatagi — Final-year B.E. CSE (2027), BGMIT Mudhol (VTU). Open to fresher / trainee Java backend & full-stack roles.*
