# Few-Shot Learning with Prototypical Networks

A PyTorch coursework implementation of episodic few-shot classification.
Experiments compare convolutional and residual backbones, one-shot/five-shot
settings, data augmentation, and character-domain transfer.

## Experiment scope

- CUB bird classification with Conv4, Conv6, ResNet10, ResNet18, and ResNet34.
- Omniglot classification and Omniglot-to-EMNIST transfer.
- Five-way episodes with configurable support examples.
- Feature-based evaluation and an optional adaptation comparison.

Results and the original discussion are included in
[RelazioneEdoardoReML.pdf](RelazioneEdoardoReML.pdf) and
[PresentazioneEdoardoReML.pptx](PresentazioneEdoardoReML.pptx).
Historical reported accuracies are not new measurements of this fork.

## Runtime and data

Use a compatible Python 3 environment with PyTorch, torchvision, NumPy, Pillow,
h5py, and Matplotlib. The source uses CUDA directly; the training and evaluation
entry points are **not CPU-ready without modification**.
No pinned dependency manifest is included.

`train.py`, `save_features.py`, and `test.py` each set
`CUDA_VISIBLE_DEVICES = "1"` internally. Change those source settings to select
your available GPU; setting only the shell environment is overwritten.

Dataset directories are defined in [configs.py](configs.py):

```text
datasets/CUB/
datasets/omniglot/
datasets/emnist/
```

Split JSON files contain `label_names`, `image_names`, and `image_labels`.
Ensure their image paths resolve on your machine: split metadata is not a
substitute for the image datasets. Use the datasets under their applicable terms.

## Training and evaluation workflow

1. Prepare the datasets and check the paths in `configs.py` and their JSON splits.
2. Configure the dataset, backbone, ways, shots, augmentation, and GPU in
   `train.py`. These are source variables, not CLI flags.
3. Run training from the repository root:

   ```sh
   python train.py
   ```

4. Match the same dataset, backbone, ways, shots, augmentation, and GPU settings
   in `save_features.py`, then extract features:

   ```sh
   python save_features.py
   ```

5. Prepare features for every configuration evaluated by `test.py`, then run:

   ```sh
   python test.py
   ```

Training saves `best_model.tar` and periodic checkpoints below `checkpoints/`.
Feature extraction writes HDF5 files under the matching `features/` hierarchy.
The full evaluation script expects a matrix of experiments, not just the
checkpoint from one default training run. Align or restrict its experiment
calls when reproducing only a subset.

The current training defaults use CUB, Conv6, five-way one-shot episodes, and
augmentation. Automatic epoch selection uses 600 epochs for one-shot and 400
for five-shot training; this is not a quick smoke test.

## Repository guide

| Path | Purpose |
| --- | --- |
| [backbone.py](backbone.py) | Convolutional and residual feature extractors |
| [protonet.py](protonet.py) | Prototype-based classification and episode loops |
| [data](data) | Image transformations, episode sampling, and feature loading |
| [train.py](train.py) | Training and checkpoint selection |
| [save_features.py](save_features.py) | Checkpoint-to-HDF5 feature extraction |
| [test.py](test.py) | Repeated episode evaluation, tables, and plots |

Checkpoints are not comprehensively included because of their size.
Load only trusted model files. Historical Python/PyTorch API assumptions may
need adaptation on current runtimes.

## References and attribution

The original README credits **Edoardo Re, 2021**, and identifies these sources:

- [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/),
  [Omniglot](https://github.com/brendenlake/omniglot), and
  [EMNIST data used by the original project](https://github.com/NanqingD/DAOSL).
- [Data/backbone reference](https://github.com/facebookresearch/low-shot-shrink-hallucinate).
- [Prototypical Networks reference](https://github.com/jakesnell/prototypical-networks).
- [Experiment structure reference](https://github.com/wyharveychen/CloserLookFewShot).

This repository is a fork of
[edoardore/FewShotProtoNet](https://github.com/edoardore/FewShotProtoNet).
Original credit, component licenses, and history are preserved.

Discuss substantial changes in an issue and include appropriate tests.
