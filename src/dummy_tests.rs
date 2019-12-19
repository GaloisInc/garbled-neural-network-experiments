use colored::*;
use fancy_garbling::dummy::Dummy;
use fancy_garbling::util as numbers;
use fancy_garbling::FancyInput;
use itertools::Itertools;
use ndarray::Array3;

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

    let mut mb = pbr::MultiBar::new();
    let mut p1 = mb.create_bar(images.len() as u64);
    p1.message("Test ");
    let mut p2 = mb.create_bar(nn.nlayers() as u64);
    std::thread::spawn(move || mb.listen());
    let mut total_time = time::Duration::zero();

    for (img_num, img) in images.iter().enumerate() {
        p1.inc();
        p1.message(&format!(
            "(avg {}ms) [{} errors ({:.2}%)] ",
            if img_num > 0 {
                total_time.num_milliseconds() / img_num as i64
            } else {
                0
            },
            errors,
            100.0 * (1.0 - errors as f32 / img_num as f32)
        ));

        // create a new dummy with the image as the input
        let mut dummy = Dummy::new();
        let inp = img
            .iter()
            .map(|&x| dummy.crt_encode(util::to_mod_q(x, qfirst), qfirst).unwrap())
            .collect_vec();

        // evaluate the fancy computation using the dummy
        let start = time::PreciseTime::now();
        let outs = nn.eval_arith(
            &mut dummy,
            &inp,
            &moduli,
            Some(&mut p2),
            8,
            secret_weights,
            true,
            accuracy,
        );
        let end = time::PreciseTime::now();
        total_time = total_time + start.to(end);

        // decode the output back to i64
        let res = outs
            .iter()
            .map(|out| {
                let vals = &out.iter().map(|v| v.val()).collect_vec();
                util::from_mod_q_crt(&vals, qlast)
            })
            .collect_vec();

        if util::index_of_max(&res) != util::index_of_max(&labels[img_num]) {
            errors += 1;
        }
    }

    p1.finish();
    p2.finish();

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

    let mut mb = pbr::MultiBar::new();
    let mut p1 = mb.create_bar(images.len() as u64);
    p1.message("Test ");
    let mut p2 = mb.create_bar(nn.nlayers() as u64);
    std::thread::spawn(move || mb.listen());
    let mut total_time = time::Duration::zero();

    for (img_num, img) in images.iter().enumerate() {
        p1.inc();
        p1.message(&format!(
            "(avg {} ms) [{} errors ({:.2}%)] ",
            if img_num > 0 {
                total_time.num_milliseconds() / img_num as i64
            } else {
                0
            },
            errors,
            100.0 * (1.0 - errors as f32 / img_num as f32)
        ));

        // create a new dummy with the image as the input
        let mut dummy = Dummy::new();

        // encode the image in twos complement
        let inp = img
            .iter()
            .map(|&x| {
                let bits = util::i64_to_twos_complement(x, first_layer_nbits);
                dummy.bin_encode(bits, first_layer_nbits).unwrap()
            })
            .collect_vec();

        // evaluate the fancy computation using the dummy
        let start = time::PreciseTime::now();
        let outs = nn.eval_boolean(
            &mut dummy,
            &inp,
            bitwidth,
            Some(&mut p2),
            8,
            secret_weights,
            true,
        );
        let end = time::PreciseTime::now();
        total_time = total_time + start.to(end);

        // decode the output back to i64
        let res = outs
            .iter()
            .map(|out| {
                let vals = &out.iter().map(|v| v.val()).collect_vec();
                util::i64_from_bits(&vals)
            })
            .collect_vec();

        if util::index_of_max(&res) != util::index_of_max(&labels[img_num]) {
            errors += 1;
        }
    }

    p1.finish();
    p2.finish();

    println!(
        "errors: {}/{}. accuracy: {:.2}%",
        errors,
        images.len(),
        100.0 * (1.0 - errors as f32 / images.len() as f32)
    );
}
