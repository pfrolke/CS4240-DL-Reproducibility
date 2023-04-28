
# Reproduction of Cross-Encoder for Unsupervised Gaze Representation Learning

This repository contains the code for a PyTorch reproduction of the Cross-Encoder from the paper Cross-Encoder for "Unsupervised Gaze Representation Learning", by Sun et al.

Read the [blog post](https://hackmd.io/@oi0lLFG7QOCwvkntOV5dHA/SyB1gpYmh) for more information.

## Authors

- [@pfrolke](https://www.github.com/pfrolke)
- [@irethans](https://www.github.com/irethans)
- [@gadmiraal](https://www.github.com/gadmiraal)
## Installation

To install this project's dependencies run

```bash
  pip install -r requirements.txt
```

Then, save the Columbia Gaze dataset to a `data` folder. To preprocess the dataset and start training the Cross-Encoder run

```bash
  python train_cross_encoder.py
```
