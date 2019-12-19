garbled neural net experiments
==============================
This repo contains an implementation of convolutional neural networks using arithmetic
garbled circuits, via [fancy-garbling](https://github.com/spaceships/fancy-garbling). 
It contains the models we ran our experiments on in the paper.

to run a benchmark on a neural network:
---------------------------------------
```
cargo run --release -- bench neural_nets/DINN_30
```

For more information:

```
cargo run -- help
```
