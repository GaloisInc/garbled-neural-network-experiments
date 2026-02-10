//! The lowest level of the Neural Network abstraction is a Layer. We describe how to
//! evaluate a NN Layer polymorphically by encoding basic operations - adding, encoding,
//! max, etc - in a struct, which could be Fancy - Arithemtic or Boolean - or plaintext in
//! the clear. We even use the same struct to evaluate the maximum bitwith. The upshot is
//! that we only have to say how to evaluate each kind of layer once (in `Layer::eval`),
//! minimizing NN evaluation bugs.

use crate::util;
use fancy_garbling::{
    BinaryBundle, BinaryGadgets, CrtBundle, CrtGadgets, Fancy, FancyInput, HasModulus,
};
use fancy_garbling::{FancyArithmetic, util as numbers};
use itertools::iproduct;
use ndarray::Array3;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use swanky_channel::Channel;

/// The accuracy of each kind of activation function.
#[derive(Clone, Debug)]
pub struct Accuracy {
    pub(crate) relu: String,
    pub(crate) sign: String,
    pub(crate) max: String,
}

/// A layer of a [`crate::NeuralNet`].
///
/// Each layer optionally contains weights and biases. If they are not present,
/// the weights and biases are treated as secret (garbler inputs).
#[derive(Clone)]
pub enum Layer {
    Dense {
        weights: Vec<Array3<Option<i64>>>,
        biases: Vec<Option<i64>>,
        activation: String,
    },

    Convolutional {
        filters: Vec<Array3<Option<i64>>>,
        biases: Vec<Option<i64>>,
        input_shape: (usize, usize, usize),
        kernel_shape: (usize, usize, usize),
        stride: (usize, usize),
        activation: String,
        pad: bool,
    },

    MaxPooling2D {
        input_shape: (usize, usize, usize),
        stride: (usize, usize),
        size: (usize, usize),
        pad: bool,
    },

    Flatten {
        input_shape: (usize, usize, usize),
        output_shape: (usize, usize, usize),
    },

    Activation {
        // Layer just does activations, one per input
        activation: String,
        input_shape: (usize, usize, usize),
    },
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Layer::Dense { .. } => write!(f, "Dense"),
            Layer::Convolutional { .. } => write!(f, "Convolutional"),
            Layer::MaxPooling2D { .. } => write!(f, "MaxPooling2D"),
            Layer::Flatten { .. } => write!(f, "Flatten"),
            Layer::Activation { .. } => write!(f, "Activation"),
        }
    }
}

impl std::fmt::Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Layer::Dense { activation, .. } => {
                let (x, _, _) = self.output_dims();
                write!(f, "Dense[{}] activation={}", x, activation)
            }
            Layer::Convolutional {
                kernel_shape,
                stride,
                filters,
                activation,
                ..
            } => write!(
                f,
                "Conv[{}] activation={} stride={:?} kernel_shape={:?}",
                filters.len(),
                activation,
                stride,
                kernel_shape
            ),
            Layer::MaxPooling2D { stride, size, .. } => {
                write!(f, "MaxPooling2D stride={:?} size={:?}", stride, size)
            }
            Layer::Flatten { .. } => write!(f, "Flatten"),
            Layer::Activation { activation, .. } => write!(f, "Activation {}", activation),
        }
    }
}

/// NeuralNetOps encodes the particular way that we evaluate a neural net - whether it is
/// directly over `i64` or as an arithmetic circuit, or whatever. The first argument to
/// these functions could be a `Fancy` object.
#[allow(clippy::type_complexity)]
struct NeuralNetOps<B, T> {
    // Encode a constant.
    enc: Box<dyn Fn(&mut B, i64, &mut Channel) -> T>,
    // Encode a secret.
    sec: Box<dyn Fn(&mut B, Option<i64>, &mut Channel) -> T>,
    // Add two values.
    add: Box<dyn Fn(&mut B, &T, &T, &mut Channel) -> T>,
    // Scalar multiplication.
    cmul: Box<dyn Fn(&mut B, &T, i64, &mut Channel) -> T>,
    // Apply secret weight to an input.
    proj: Box<dyn Fn(&mut B, &T, Option<i64>, &mut Channel) -> T>,
    // Maximum of a slice of encodings.
    max: Box<dyn Fn(&mut B, &[T], &mut Channel) -> T>,
    // Activation function chosen based on string name.
    act: Box<dyn Fn(&mut B, &str, &T, &mut Channel) -> T>,
    // Encode a zero value.
    zero: Box<dyn Fn(&mut B, &mut Channel) -> T>,
}

impl Layer {
    /// Returns (height, width, depth).
    pub fn input_dims(&self) -> (usize, usize, usize) {
        match self {
            Layer::Dense { weights, .. } => weights.iter().next().map_or((0, 0, 0), |w0| w0.dim()),
            Layer::Convolutional { input_shape, .. } => *input_shape,
            Layer::MaxPooling2D { input_shape, .. } => *input_shape,
            Layer::Flatten { input_shape, .. } => *input_shape,
            Layer::Activation { input_shape, .. } => *input_shape,
        }
    }

    /// Get the number of items in the input.
    pub fn input_size(&self) -> usize {
        let (x, y, z) = self.input_dims();
        x * y * z
    }

    /// Get the dimensions of the output in (height, width, depth).
    pub fn output_dims(&self) -> (usize, usize, usize) {
        match self {
            Layer::Dense { biases, .. } => (biases.len(), 1, 1),

            Layer::Convolutional {
                input_shape,
                kernel_shape,
                stride,
                filters,
                pad,
                ..
            } => {
                let (height, width, _) = input_shape;
                let (ker_height, ker_width, _) = kernel_shape;
                let (stride_y, stride_x) = stride;

                if *pad {
                    (*height, *width, filters.len())
                } else {
                    (
                        (height - ker_height) / stride_y + 1,
                        (width - ker_width) / stride_x + 1,
                        filters.len(),
                    )
                }
            }

            Layer::MaxPooling2D {
                input_shape,
                stride,
                size,
                pad,
            } => {
                let (height, width, depth) = input_shape;
                let (pool_height, pool_width) = size;
                let (stride_y, stride_x) = stride;

                if *pad {
                    *input_shape
                } else {
                    (
                        (height - pool_height) / stride_y + 1,
                        (width - pool_width) / stride_x + 1,
                        *depth,
                    )
                }
            }

            Layer::Flatten { output_shape, .. } => *output_shape,

            Layer::Activation { input_shape, .. } => *input_shape,
        }
    }

    /// Get the number of items in the output.
    pub fn output_size(&self) -> usize {
        let (x, y, z) = self.output_dims();
        x * y * z
    }

    /// Evaluate this layer in plaintext while finding the max value on a wire.
    pub fn max_bitwidth(
        &self,
        input: &Array3<i64>,
        _: usize,
        channel: &mut Channel,
    ) -> (Array3<i64>, i64) {
        let max_atomic: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
        let thread_atomic = max_atomic.clone();
        let store_max_base = Arc::new(move |x: i64| {
            thread_atomic.fetch_max(x.unsigned_abs() as usize, Ordering::SeqCst);
        });

        let store_max = store_max_base.clone();
        let enc = move |_: &mut usize, x: i64, _: &mut Channel| {
            store_max(x);
            x
        };

        let store_max = store_max_base.clone();
        let proj = move |_: &mut usize, inp: &i64, opt_w: Option<i64>, _: &mut Channel| {
            if let Some(w) = opt_w {
                let x = w * inp;
                store_max(x);
                x
            } else {
                *inp
            }
        };

        let store_max = store_max_base.clone();
        let add = move |_: &mut usize, x: &i64, y: &i64, _: &mut Channel| {
            let res = x + y;
            store_max(res);
            res
        };

        let store_max = store_max_base.clone();
        let cmul = move |_: &mut usize, x: &i64, y: i64, _: &mut Channel| {
            let res = x * y;
            store_max(res);
            res
        };

        let store_max = store_max_base.clone();
        let max = move |_: &mut usize, xs: &[i64], _: &mut Channel| {
            xs.iter()
                .map(|&x| {
                    store_max(x);
                    x
                })
                .max()
                .unwrap()
        };

        let act = |_: &mut usize, a: &str, x: &i64, _: &mut Channel| match a {
            "sign" => {
                if *x >= 0 {
                    1
                } else {
                    -1
                }
            }
            "relu" => std::cmp::max(*x, 0),
            "id" => *x,
            act => panic!("unsupported activation {}", act),
        };

        let ops = NeuralNetOps {
            enc: Box::new(enc),
            sec: Box::new(move |_, _, _| 0),
            add: Box::new(add),
            cmul: Box::new(cmul),
            proj: Box::new(proj),
            max: Box::new(max),
            act: Box::new(act),
            zero: Box::new(|_, _| 0),
        };

        let layer_output = self.eval(&mut 0, input, &ops, false, channel);
        let max_val = max_atomic.load(Ordering::SeqCst) as i64;
        (layer_output, max_val)
    }

    /// Evaluate this layer in plaintext.
    pub fn as_plaintext(
        &self,
        input: &Array3<i64>,
        _: usize,
        channel: &mut Channel,
    ) -> Array3<i64> {
        let ops = NeuralNetOps {
            enc: Box::new(|_, x, _| x),
            sec: Box::new(|_, _, _| panic!("secret not supported for plaintext eval")),
            add: Box::new(|_, x, y, _| x + y),
            cmul: Box::new(|_, x, y, _| x * y),
            proj: Box::new(|_, _, _, _| panic!("secret not supported for plaintext eval")),
            max: Box::new(|_, xs, _| *xs.iter().max().unwrap()),
            act: Box::new(|_, a, x, _| match a {
                "sign" => {
                    if *x >= 0 {
                        1
                    } else {
                        -1
                    }
                }
                "relu" => std::cmp::max(*x, 0),
                "id" => *x,
                act => panic!("unsupported activation {}", act),
            }),
            zero: Box::new(|_, _| 0),
        };

        self.eval(&mut 0, input, &ops, false, channel)
    }

    /// Perform an arithmetic fancy computation for this layer
    #[allow(clippy::too_many_arguments)]
    pub fn as_arith<W, F>(
        &self,
        b: &mut F,
        q: u128, // input modulus
        output_mod: u128,
        input: &Array3<CrtBundle<W>>,
        _: usize,
        secret_weights: bool,
        secret_weights_owned: bool,
        accuracy: &Accuracy,
        channel: &mut Channel,
    ) -> Array3<CrtBundle<W>>
    where
        W: Clone + HasModulus,
        F: Fancy<Item = W>
            + FancyInput<Item = W>
            + FancyArithmetic<Item = W>
            + CrtGadgets<Item = W>,
    {
        let relu_accuracy = accuracy.relu.clone();
        let sign_accuracy = accuracy.sign.clone();
        let max_accuracy = accuracy.max.clone();
        let output_ps = numbers::factor(output_mod);
        let ops = NeuralNetOps {
            enc: Box::new(move |b: &mut F, x, channel| {
                b.crt_constant_bundle(util::to_mod_q(x, q), q, channel)
                    .unwrap()
            }),

            sec: if secret_weights_owned {
                Box::new(move |b: &mut F, opt_x, channel| {
                    b.crt_encode(util::to_mod_q(opt_x.unwrap(), q), q, channel)
                        .expect("error encoding secret CRT value")
                })
            } else {
                Box::new(move |b: &mut F, _, channel| {
                    b.crt_receive(q, channel)
                        .expect("error receiving secret CRT value")
                })
            },

            add: Box::new(
                move |b: &mut F, x: &CrtBundle<W>, y: &CrtBundle<W>, _: &mut Channel| {
                    b.crt_add(x, y)
                },
            ),

            cmul: Box::new(move |b: &mut F, x: &CrtBundle<W>, y, _: &mut Channel| {
                b.crt_cmul(x, util::to_mod_q(y, q))
            }),

            proj: Box::new(move |b: &mut F, inp, opt_w, channel| {
                if let Some(w) = opt_w {
                    // convert the weight to crt mod q
                    let ws = util::to_mod_q_crt(w, q);
                    CrtBundle::new(
                        inp.wires()
                            .iter()
                            .zip(ws.iter())
                            .map(|(wire, weight)| {
                                let q = wire.modulus();
                                let tab = (0..q).map(|x| x * weight % q).collect::<Vec<_>>();
                                // project each input x to x*w
                                b.proj(wire, q, Some(tab), channel).unwrap()
                            })
                            .collect::<Vec<_>>(),
                    )
                } else {
                    CrtBundle::new(
                        inp.wires()
                            .iter()
                            .map(|wire| {
                                // project the input, without knowing the weight
                                b.proj(wire, wire.modulus(), None, channel).unwrap()
                            })
                            .collect::<Vec<_>>(),
                    )
                }
            }),
            max: Box::new(move |b: &mut F, xs: &[CrtBundle<W>], channel| {
                b.crt_max(xs, &max_accuracy, channel).unwrap()
            }),
            act: Box::new(
                move |b: &mut F, a: &str, x: &CrtBundle<W>, channel| match a {
                    "sign" => b
                        .crt_sgn(x, &sign_accuracy, Some(&output_ps), channel)
                        .unwrap(),
                    "relu" => b
                        .crt_relu(x, &relu_accuracy, Some(&output_ps), channel)
                        .unwrap(),
                    "id" => x.clone(),
                    act => panic!("unsupported activation {}", act),
                },
            ),
            zero: Box::new(move |b: &mut F, channel: &mut Channel| {
                b.crt_constant_bundle(0, q, channel).unwrap()
            }),
        };

        self.eval(b, input, &ops, secret_weights, channel)
    }

    /// Perform a binary fancy computation for this layer
    #[allow(clippy::too_many_arguments)]
    pub fn as_binary<W, F>(
        &self,
        b: &mut F,
        nbits: usize,
        _: usize,
        input: &Array3<BinaryBundle<W>>,
        _: usize,
        secret_weights: bool,
        secret_weights_owned: bool,
        channel: &mut Channel,
    ) -> Array3<BinaryBundle<W>>
    where
        W: Clone + HasModulus,
        F: Fancy<Item = W> + FancyInput<Item = W> + BinaryGadgets<Item = W>,
    {
        let ops = NeuralNetOps {
            enc: Box::new(move |b: &mut F, x, channel: &mut Channel| {
                let twos = util::i64_to_twos_complement(x, nbits);
                b.bin_constant_bundle(twos, nbits, channel).unwrap()
            }),

            sec: Box::new(move |b: &mut F, opt_x, channel: &mut Channel| {
                if secret_weights_owned {
                    let xbits = util::i64_to_twos_complement(opt_x.unwrap(), nbits);
                    b.bin_encode(xbits, nbits, channel)
                        .expect("error encoding binary secret value")
                } else {
                    b.bin_receive(nbits, channel)
                        .expect("error receiving binary secret value")
                }
            }),

            add: Box::new(
                move |b: &mut F,
                      x: &BinaryBundle<W>,
                      y: &BinaryBundle<W>,
                      channel: &mut Channel| {
                    b.bin_addition_no_carry(x, y, channel).unwrap()
                },
            ),

            cmul: Box::new(
                move |b: &mut F, x: &BinaryBundle<W>, y, channel: &mut Channel| {
                    b.bin_cmul(x, util::i64_to_twos_complement(y, nbits), nbits, channel)
                        .unwrap()
                },
            ),

            proj: Box::new(move |b: &mut F, inp, opt_w, channel: &mut Channel| {
                // ignore the input weight - it needs to be a garbler input
                let weight_bits = opt_w.map(|w| util::i64_to_twos_complement(w, nbits));
                let w = if secret_weights_owned {
                    b.bin_encode(weight_bits.unwrap(), nbits, channel)
                        .expect("could not encode binary secret")
                } else {
                    b.bin_receive(nbits, channel)
                        .expect("could not receive binary secret")
                };
                b.bin_multiplication_lower_half(inp, &w, channel).unwrap()
            }),

            max: Box::new(
                move |b: &mut F, xs: &[BinaryBundle<W>], channel: &mut Channel| {
                    b.bin_max(xs, channel).unwrap()
                },
            ),

            act: Box::new(
                move |b: &mut F, a: &str, x: &BinaryBundle<W>, channel: &mut Channel| match a {
                    "sign" => {
                        let sign = x.wires().last().unwrap();
                        let neg1 = (1 << nbits) - 1;
                        b.bin_multiplex_constant_bits(sign, 1, neg1, nbits, channel)
                            .unwrap()
                    }
                    "relu" => {
                        let sign = x.wires().last().unwrap();
                        let zeros = b.bin_constant_bundle(0u128, nbits, channel).unwrap();
                        b.bin_multiplex(sign, x, &zeros, channel).unwrap()
                    }
                    "id" => x.clone(),
                    act => panic!("unsupported activation {}", act),
                },
            ),

            zero: Box::new(move |b: &mut F, channel: &mut Channel| {
                b.bin_constant_bundle(0u128, nbits, channel).unwrap()
            }),
        };
        self.eval(b, input, &ops, secret_weights, channel)
    }

    /// Polymorphic evaluation so we can run on `i64` directly as well as use this
    /// function to build `Circuit`s.
    fn eval<T, B>(
        &self,
        b: &mut B,
        input: &Array3<T>,
        ops: &NeuralNetOps<B, T>,
        secret_weights: bool,
        channel: &mut Channel,
    ) -> Array3<T>
    where
        T: Clone,
    {
        assert_eq!(self.input_dims(), input.dim());
        let (height, width, depth) = self.input_dims();

        let mut output: Array3<Option<T>> = Array3::default(self.output_dims());
        let nouts = self.output_size();

        match self {
            Layer::Dense {
                weights,
                biases,
                activation,
            } => {
                for neuron in 0..nouts {
                    let mut x = if secret_weights {
                        (ops.sec)(b, biases[neuron], channel)
                    } else {
                        (ops.enc)(
                            b,
                            biases[neuron].expect("biases required for evaluation"),
                            channel,
                        )
                    };

                    for i in 0..height {
                        for j in 0..width {
                            for k in 0..depth {
                                let prod = if secret_weights {
                                    (ops.proj)(
                                        b,
                                        &input[(i, j, k)],
                                        weights[neuron][(i, j, k)],
                                        channel,
                                    )
                                } else {
                                    let w = weights[neuron][(i, j, k)].expect(
                                        "Dense layer eval: weights required for evaluation",
                                    );
                                    (ops.cmul)(b, &input[(i, j, k)], w, channel)
                                };
                                x = (ops.add)(b, &x, &prod, channel);
                            }
                        }
                    }

                    let z = (ops.act)(b, activation, &x, channel);
                    output[(neuron, 0, 0)] = Some(z);
                }
            }

            Layer::Convolutional {
                filters,
                biases,
                kernel_shape,
                stride,
                activation,
                pad,
                ..
            } => {
                let (kheight, kwidth, kdepth) = *kernel_shape;
                let (stride_y, stride_x) = *stride;

                let zero_rows = if *pad {
                    (stride_y - 1) * height + kheight - stride_y
                } else {
                    0
                };
                let zero_cols = if *pad {
                    (stride_x - 1) * width + kwidth - stride_x
                } else {
                    0
                };

                let shift_y = ((zero_rows as f32) / 2.0).floor() as usize;
                let shift_x = ((zero_cols as f32) / 2.0).floor() as usize;

                for filterno in 0..filters.len() {
                    let mut h = 0;
                    while stride_y * h <= height - kheight + zero_rows {
                        // && h < oheight { // drop indices that dont make sense (padding=valid)

                        let mut w = 0;
                        while stride_x * w <= width - kwidth + zero_cols {
                            // && w < owidth {

                            let mut x = if secret_weights {
                                (ops.sec)(b, biases[filterno], channel)
                            } else {
                                (ops.enc)(b, biases[filterno].expect("no bias"), channel)
                            };

                            for i in 0..kheight {
                                let idx_y = stride_y * h + i;
                                for j in 0..kwidth {
                                    let idx_x = stride_x * w + j;
                                    for k in 0..kdepth {
                                        let pad_condition = *pad
                                            && ((idx_y < shift_y || idx_x < shift_x)
                                                || (idx_y >= height + shift_y
                                                    || idx_x >= width + shift_x));

                                        let input_val = if pad_condition {
                                            &(ops.zero)(b, channel)
                                        } else {
                                            &input[(idx_y - shift_y, idx_x - shift_x, k)]
                                        };

                                        let prod = if secret_weights {
                                            (ops.proj)(
                                                b,
                                                input_val,
                                                filters[filterno][(i, j, k)],
                                                channel,
                                            )
                                        } else {
                                            (ops.cmul)(
                                                b,
                                                input_val,
                                                filters[filterno][(i, j, k)].expect("no weight"),
                                                channel,
                                            )
                                        };
                                        x = (ops.add)(b, &x, &prod, channel);
                                    }
                                }
                            }

                            let z = (ops.act)(b, activation, &x, channel);
                            assert!(output[(h, w, filterno)].is_none());
                            output[(h, w, filterno)] = Some(z);
                            w += 1;
                        }
                        h += 1;
                    }
                }
            }

            Layer::MaxPooling2D {
                stride, size, pad, ..
            } => {
                let (pheight, pwidth) = *size;
                let (stride_y, stride_x) = *stride;

                let zero_rows = if *pad {
                    (stride_y - 1) * height + pheight - stride_y
                } else {
                    0
                };
                let zero_cols = if *pad {
                    (stride_x - 1) * width + pwidth - stride_x
                } else {
                    0
                };

                let shift_y = ((zero_rows as f32) / 2.0).floor() as usize;
                let shift_x = ((zero_cols as f32) / 2.0).floor() as usize;

                // create windows
                let mut windows = Vec::new();
                let mut y = 0;
                while stride_y * y <= height - pheight + zero_rows {
                    let mut x = 0;
                    while stride_x * x <= width - pwidth + zero_cols {
                        for z in 0..depth {
                            let mut vals = Vec::with_capacity(pheight * pwidth);
                            for h in 0..pheight {
                                let idx_y = stride_y * y + h;
                                for w in 0..pwidth {
                                    let idx_x = stride_x * x + w;

                                    let pad_condition = *pad
                                        && ((idx_y < shift_y || idx_x < shift_x)
                                            || (idx_y >= height + shift_y
                                                || idx_x >= width + shift_x));

                                    let val = if pad_condition {
                                        (ops.zero)(b, channel).clone()
                                    } else {
                                        input[(idx_y - shift_y, idx_x - shift_x, z)].clone()
                                    };

                                    vals.push(val);
                                }
                            }
                            windows.push(((y, x, z), vals));
                        }
                        x += 1;
                    }
                    y += 1;
                }

                for (coordinate, window) in windows.into_iter() {
                    let val = (ops.max)(b, &window, channel);
                    output[coordinate] = Some(val);
                }
            }

            Layer::Flatten { output_shape, .. } => {
                output = input.map(|v| Option::Some(v.clone()));
                output = output.into_shape(*output_shape).unwrap();
            }

            Layer::Activation { activation, .. } => {
                let coordinates = iproduct!(0..height, 0..width, 0..depth).collect::<Vec<_>>();
                for c in coordinates.into_iter() {
                    let z = (ops.act)(b, activation, &input[c], channel);
                    output[c] = Some(z);
                }
            }
        }

        for (coordinate, val) in output.indexed_iter() {
            if val.is_none() {
                println!("{}: uninitialized output at {:?}", self, coordinate);
                println!("exiting...");
                std::process::exit(1);
            }
        }

        output.mapv(|elem| {
            elem.unwrap_or_else(|| {
                println!("{}: uninitialized output", self);
                println!("exiting...");
                std::process::exit(1);
            })
        })
    }
}
