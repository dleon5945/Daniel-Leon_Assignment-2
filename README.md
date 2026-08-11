# Assignment-2
This repo contains Assignment 2: Sentiment Classification with Neural Language Models.

Binary sentiment classifier for movie reviews (0 = negative, 1 = positive), built on
a frozen pretrained language model with a trained linear classification head.

# Results
Accuracy on the public test set was 0.7650, and balanced accuracy was also 0.7650.

Confusion matrix: `[[141, 59], [35, 165]]`

Cross validated balanced accuracy on the 240 training examples was 0.7889.

# Approach
Each review is tokenized with DistilBERT's tokenizer and split into non-overlapping. 512 token chunks: 210 of 240 training reviews (87.5%) exceed the 512-token limit, so truncation would discard most of a typical review. Every chunk is encoded by a frozen distilbert-base-uncased model, and the chunk CLS vectors are averaged into a single 768-dimensional representation per review.

A linear head (768 to 2) is trained on those representations with weighte cross-entropy to compensate for the 180/60 class imbalance. The decision threshold is selected by cross-validation on out-of-fold predictions.

The encoder is never updated; only the head's 1,538 parameters are trained.
## Setup
'''
pip install -r requirements.txt
'''

No GPU needed, ran only on CPU

## Running the test
Open `stage1_notebook.ipynb` and run all cells from the top. The data is read from
`data/train.csv` and `data/public_test.csv` with relative paths, so run it from the
repo root.

It takes about 6 minutes. Most of that is the two embedding passes over all 640
reviews, one for the main model and one for the chunk overlap comparison. The encoder
weights download from Hugging Face the first time.

Running the notebook regenerates `public_test_predictions.csv` and everything in
`model_checkpoint/`.

## Predicting from the checkpoint

`model_checkpoint/` has two files:

- `head.pt` — the trained weights for the linear head
- `config.json` — encoder name, hidden dimension, chunk length, overlap, and the
  decision threshold

The threshold is saved in the checkpoint instead of hardcoded in the notebook, because
you can't reproduce the predictions without it.

To predict without retraining:

1. Read `config.json`
2. Build `nn.Linear(config["hidden_dim"], 2)` and load `head.pt` into it
3. Load the encoder from `config["encoder"]` and set it to eval mode
4. Chunk and embed the reviews using `config["max_length"]` and `config["overlap"]`
5. Apply softmax and compare the positive probability to `config["threshold"]`

The encoder is referenced by name instead of being saved in the repo, since its
weights are unchanged from the public version. Only the trained head and the config
get saved.

There's a cell in the notebook that does this reload and checks that the predictions
come out identical.

Lastly, The cached embeddings (`*.npz`) are gitignored since the notebook regenerates them.
