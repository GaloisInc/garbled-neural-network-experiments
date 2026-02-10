use colored::*;
use fancy_garbling::FancyInput;
use fancy_garbling::dummy::Dummy;
use fancy_garbling::util as numbers;
use itertools::Itertools;
use ndarray::Array3;
use std::time::Duration;
use std::time::Instant;
use swanky_channel::Channel;

use crate::layer::Accuracy;
use crate::neural_net::NeuralNet;
use crate::util;

/// test the artihmetic encoding of the neural network using Dummy
pub fn arith_accuracy_test(
    nn: &NeuralNet,
    images: &[Array3<i64>],
    labels: &[Vec<i64>],
    bitwidth: &[usize],
    secret_weights: bool,
    accuracy: &Accuracy,
) {
    println!("{}", "* running circuit accuracy evaluation".green());

    let moduli = bitwidth
        .iter()
        .map(|&b| numbers::modulus_with_width(b as u32))
        .collect_vec();

    let qfirst = *moduli.first().unwrap();
    let qlast = *moduli.last().unwrap();

    let mut errors = 0;

    let mut total_time = Instant::now();

    for (img_num, img) in images.iter().enumerate() {
        println!(
            "(avg {:?}) [{} errors ({:.2}%)] ",
            if img_num > 0 {
                total_time.elapsed() / img_num as u32
            } else {
                Duration::ZERO
            },
            errors,
            100.0 * (1.0 - errors as f32 / img_num as f32)
        );

        let (start, outs) = Channel::with(std::io::empty(), |channel| {
            // create a new dummy with the image as the input
            let mut dummy = Dummy::new();
            let inp = img
                .iter()
                .map(|&x| {
                    dummy
                        .crt_encode(util::to_mod_q(x, qfirst), qfirst, channel)
                        .unwrap()
                })
                .collect_vec();

            // evaluate the fancy computation using the dummy
            let start = Instant::now();
            let outs = nn.eval_arith(
                &mut dummy,
                &inp,
                &moduli,
                8,
                secret_weights,
                true,
                accuracy,
                channel,
            );
            Ok((start, outs))
        })
        .unwrap();
        total_time += start.elapsed();

        // decode the output back to i64
        let res = outs
            .iter()
            .map(|out| {
                let vals = &out.iter().map(|v| v.val()).collect_vec();
                util::from_mod_q_crt(vals, qlast)
            })
            .collect_vec();

        if util::index_of_max(&res) != util::index_of_max(&labels[img_num]) {
            errors += 1;
        }
    }

    println!(
        "errors: {}/{}. accuracy: {:.2}%",
        errors,
        images.len(),
        100.0 * (1.0 - errors as f32 / images.len() as f32)
    );
}

/// test the boolean encoding of the neural network using dummy
pub fn boolean_accuracy_test(
    nn: &NeuralNet,
    images: &[Array3<i64>],
    labels: &[Vec<i64>],
    bitwidth: &[usize],
    secret_weights: bool,
) {
    println!(
        "{}",
        "* running circuit accuracy evaluation (binary)".green()
    );

    let mut errors = 0;

    let first_layer_nbits = *bitwidth.first().unwrap();

    let mut total_time = Instant::now();

    for (img_num, img) in images.iter().enumerate() {
        println!(
            "(avg {:?}) [{} errors ({:.2}%)] ",
            if img_num > 0 {
                total_time.elapsed() / img_num as u32
            } else {
                Duration::ZERO
            },
            errors,
            100.0 * (1.0 - errors as f32 / img_num as f32)
        );

        let (start, outs) = Channel::with(std::io::empty(), |channel| {
            // create a new dummy with the image as the input
            let mut dummy = Dummy::new();

            // encode the image in twos complement
            let inp = img
                .iter()
                .map(|&x| {
                    let bits = util::i64_to_twos_complement(x, first_layer_nbits);
                    dummy.bin_encode(bits, first_layer_nbits, channel).unwrap()
                })
                .collect_vec();

            // evaluate the fancy computation using the dummy
            let start = Instant::now();
            let outs =
                nn.eval_boolean(&mut dummy, &inp, bitwidth, 8, secret_weights, true, channel);
            Ok((start, outs))
        })
        .unwrap();
        total_time += start.elapsed();

        // decode the output back to i64
        let res = outs
            .iter()
            .map(|out| {
                let vals = &out.iter().map(|v| v.val()).collect_vec();
                util::i64_from_bits(vals)
            })
            .collect_vec();

        if util::index_of_max(&res) != util::index_of_max(&labels[img_num]) {
            errors += 1;
        }
    }

    println!(
        "errors: {}/{}. accuracy: {:.2}%",
        errors,
        images.len(),
        100.0 * (1.0 - errors as f32 / images.len() as f32)
    );
}
