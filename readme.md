garbled neural net experiments
==============================
This repo contains an implementation of convolutional neural networks using arithmetic
garbled circuits, via [fancy-garbling](https://github.com/GaloisInc/swanky). 
It contains the models we ran our experiments on in our paper, [Garbled Neural Networks
Are Practical](https://eprint.iacr.org/2019/338).

The high-level idea is that we use JSON output of `tensorflow` models to build neural
network layers as a garbled circuit. The `Garbler` either hard codes the weights and
biases as public values or as secret garbler input wires, depending on how the circuit is
configured. Public weights is much, much cheaper to run.  Finally, the `Evaluator` is
started on another thread, and the `Garbler` passes the garbled circuit incrementally
through a channel as it is created. The `Evaluator` evaluates it using the test input - in
our examples this is always an image.

The `neural_nets` directory contains the trained neural networks we used in the paper. 
To run an experiment, you simply point the binary rust program to the directory you want
and give it a command on what kind of test you would like to run.

Generally, the `Garbler` does not know how large to make the integers. Integers must be
large enough to avoid overflow. But the smaller they are, the better the performance.
Therefore, we have the `bitwidth` command to run on a particular neural network. This will
evaluate the neural network on all the test data and return the maximum bitwith necessary
for each layer. You can then use this information to customize the bitwith for each layer
when using other commands (using the `-w` argument).

If you aren't familiar with rust, the getting started page is
[here](https://www.rust-lang.org/learn/get-started).

*Note on requirements*: 
This project uses nightly `rustc` due to use of atomic operations.
We recommend using [rustup](https://rustup.rs/) to configure nightly rust.
Fancy Garbling also requires AESNI, so requires a processor that supports that
instruction.

usage
-----
```
cargo run --relase -- [FLAGS] [OPTIONS] <DIR> [SUBCOMMAND]

FLAGS:
    -b, --boolean    runs in boolean mode
    -h, --help       Prints help information
    -s, --secret     use secret weights
    -V, --version    Prints version information

OPTIONS:
    -w, --bitwidth <bitwidth>            comma separated bitwidths to use for each layer (last number is replicated)
                                         [default: 15]
    -a, --accuracy <default-accuracy>    default accuracy for activations and max (overridden by specific accuracy
                                         settings) [default: 100%]
        --max <max-accuracy>             accuracy of max
    -n <NUM>                             number of tests to run
        --relu <relu-accuracy>           accuracy of relu
        --sign <sign-accuracy>           accuracy of sign

ARGS:
    <DIR>    Sets the neural network directory to use

SUBCOMMANDS:
    bitwidth    Evaluate the neural net to find the maximum bitwidth needed for each layer
    direct      Evaluate the given neural net directly over i64 values
    dummy       Test the accuracy of the fancy encoding of the neural network
    bench       Benchmark garbling and evaluating the neural network
    help        Prints this message or the help of the given subcommand(s)
```

for example: to run a benchmark on a neural network:
----------------------------------------------------
```
cargo run --release -- neural_nets/DINN_30 bench
```
