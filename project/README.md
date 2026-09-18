# Classical Music Era Audio Branch

This implements the fixed audio -> Transformer -> VAE branch. The local official
`MTT_musicnn` source exposes `penultimate` from its 200-unit backend dense layer;
the 50 sigmoid tag probabilities are never used. Each 12-second clip is loaded at
16 kHz, padded when short, and split into four non-overlapping 3-second windows.

`AudioRepresentationModel` accepts cached musicnn penultimate features shaped
`[batch, 4, 200]`. For raw windows, inject a PyTorch `feature_extractor` that
returns that shape; its parameters are automatically frozen by default and are
trainable when `freeze_musicnn=False`. The checked-in official implementation is
TensorFlow 1-style, so it cannot provide gradients through a PyTorch backward
pass. Its checkpoint can still be used to generate the cached penultimate
features, while a compatible differentiable adapter can be injected for
fine-tuning.

For inference with the original checkpoint, use
`TensorFlowMusicNNExtractor(musicnn_root=workspace / "musicnn")` as the
`feature_extractor`. This requires a separate Python environment with a
Python 3.10 or 3.11 interpreter and the packages in
`requirements-musicnn-tf.txt`; TensorFlow is not currently available in the
workspace's Python 3.13 environment. The extractor
uses the official 16 kHz, 96-mel, 512-FFT, 256-hop preprocessing and returns
200-D `penultimate` features, never the 50 tag probabilities. It is not
autograd-compatible by design.

## Era classifier

After training the audio branch, fit
`project.classifiers.RandomForestEraClassifier` on the detached VAE latent
vectors (`z`, 64-D by default). Set `feature_name="transformer_embedding"` to
use the 128-D Transformer representation instead. The Random Forest is a
separate scikit-learn classifier and does not alter the fixed neural
architecture.

The supplied `Data/features_30_sec.csv` and `features_3_sec.csv` files contain
GTZAN **genre** labels, not historical-era labels. Train the available baseline
with:

```text
python -m project.training.train_random_forest Data/features_30_sec.csv --output genre_random_forest.pkl
```

Use a dataset with era labels for the actual era-classification experiment.

The checkpoint directory must contain the actual TensorFlow weight shard files
(`.index` and `.data-*`), not only the `checkpoint` metadata file. The adapter
handles the included metadata's original Linux path and its empty local
checkpoint prefix.

Run from the workspace root after installing dependencies:

```text
python -m pytest project/tests
```
