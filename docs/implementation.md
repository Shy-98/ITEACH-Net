# Implementation details

## Model

The input dimensions are audio 512, text 1024, and video 1024. Each modality is projected to hidden size 128. Both networks use eight attention heads, a feed-forward expansion of four, and GELU.

- Teacher: three Transformer layers, overall dropout 0.5.
- Student: four NAS layers, overall dropout 0.
- ECCE: one independent seven-tap temporal kernel per modality and network, shared over the 128 hidden channels and reused across layers. The kernel is implemented with `Conv1d(1, 1, 7, padding=3, bias=False)` and softmax-normalized weights. ECCE input and convolution-output dropout are both 0.5.
- ECCE uses inclusive interval means, a maximum distance of 30, and scale 1, without extra normalization or activation. Its contribution is added to attention scores. Only attention keys are masked for padding; padded hidden positions are otherwise processed.
- The student Router computes a weighted mixture of attention, a time-axis MLP, average pooling, and max pooling. All four operations participate in the forward pass.

The shared, normalized convolution kernel is an implementation choice in this code. The manuscript does not explicitly specify kernel sharing or this normalization, so these details alone do not establish an exact historical implementation.

## Data format

`IEMOCAP_features_raw_4way.pkl` contains the six-tuple `(videoIDs, videoLabels, videoSpeakers, videoSentence, trainVid, testVid)`. The loader combines `trainVid` and `testVid` and splits dialogues by session for five-fold evaluation. Integer labels are retained as supplied.

Audio and text directories contain `<utterance-id>*.npy` files. Visual features are stored under `<utterance-id>*/compress_F.npy` and `compress_M.npy`. F/M identify the two speakers. Sequence features are averaged over time; batches use zero padding. Feature extraction is performed separately from this package.

## Training

The teacher receives complete features and the student receives masked features. Their two classification cross-entropy losses are averaged. Distillation adds hidden-state MSE from the three layers nearest the classifier, with weights 0.5, 0.1, and 0.05. Teacher targets are detached; hidden losses use only valid utterances missing at least one modality.

Default training uses batch size 32, 100 epochs, and AdamW:

| Parameter group | Learning rate | Weight decay |
| --- | --- | --- |
| Non-Router | 0.0005 | 0.00001 |
| Router | 0.005 | 0.00001 |

Each run initializes its own seed and requires a new or empty output directory.

## Missing modalities

- `constant-R`: a fixed nominal missing rate R in [0, 0.7].
- `random`: each training batch draws a continuous rate from U(0, 0.7).
- `progressive`: the rate is `min(0.1 * (epoch_index // 10), 0.7)`, using zero-based epoch indices.

The finite-count sampler preserves at least one of the three modalities. Rates above 2/3 therefore saturate at an actual missing fraction of 2/3 across sampled cells. Host and guest masks are sampled separately, including padded cells; reported actual missing fractions count valid utterances only.

## Checkpoint selection

Constant training evaluates the held-out test session once per epoch at its training rate. Random and Progressive training evaluate it once at each of 0.0, 0.1, …, 0.7 per epoch. Selection uses the constant-rate WAF or the equally weighted eight-rate mean WAF, respectively. A strictly higher score replaces `best.pt`; ties keep the earliest epoch.

The selected epoch's WAF, accuracy, and actual missing fraction at every evaluated rate are stored in the history, checkpoint metadata, and final summary. Dynamic strategies report that epoch's entire curve. Individual rates do not select separate epochs. This protocol selects on the test set; there is no separate validation selection.

Each rate evaluation samples a fresh mask. The optional `evaluate` command remeasures a saved checkpoint with a specified mask seed (default 7000). That additional measurement can differ from the selected epoch's recorded results.

```bash
python -m iteach_iemocap.evaluate \
  --data-root /path/to/IEMOCAP \
  --checkpoint runs/random/seed100/fold1/best.pt \
  --output-dir runs/random/seed100/fold1/reevaluation \
  --fold 1 --mask-seed 7000
```

## Parameter counts

For four classes, the feature dimensions above, and a maximum dialogue length of 110:

| Network | All trainable parameters |
| --- | ---: |
| Teacher | 2,512,028 |
| Student | 3,538,940 |
| Joint model | 6,050,968 |

These counts include input projections, ECCE, encoder and fusion layers, classifiers, and the student's Router and time-axis MLP. The student's count changes with the maximum dialogue length. Dropout contributes no parameters. Each training run records its own counts in `optimizer.json`.

The manuscript's 2.19M / 3.88M counts use an unresolved historical counting scope. An earlier implementation with ordinary biased `Conv1d(128, 128, 7)` has 2,856,455 teacher and 3,883,367 student parameters under the same dimensions. Changing the convolution changes the architecture and parameter count; it cannot be described solely as correcting an omitted module in the historical table.
