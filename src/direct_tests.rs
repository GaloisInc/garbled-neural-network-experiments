use std::time::{Duration, Instant};

use colored::*;
use ndarray::Array3;

use crate::neural_net::NeuralNet;
use crate::util;

/// Test the neural network directly over i64 values.
pub fn direct_test(nn: &NeuralNet, inputs: &[Array3<i64>], labels: &[Vec<i64>]) {
    println!("{}", "* running plaintext accuracy evaluation".green());

    let mut errors = 0;
    let mut img_num = 0;

    let mut total_time = Instant::now();

    for (img, label) in inputs.iter().zip(labels.iter()) {
        println!(
            "Testing (avg {:?}) [{} errors ({:.2}%)] ",
            if img_num > 0 {
                total_time.elapsed() / img_num
            } else {
                Duration::ZERO
            },
            errors,
            100.0 * (1.0 - errors as f32 / img_num as f32)
        );

        let start = Instant::now();
        let res = nn.eval_plaintext(img).iter().cloned().collect::<Vec<_>>();
        total_time += start.elapsed();

        if util::index_of_max(&res) != util::index_of_max(label) {
            errors += 1;
        }

        img_num += 1;
    }

    println!(
        "errors: {}/{}. accuracy: {}%\n",
        errors,
        img_num,
        100.0 * (1.0 - errors as f32 / img_num as f32)
    );
}

/// Test the neural network directly over i64 values, printing output.
pub fn direct_debug(nn: &NeuralNet, inputs: &[Array3<i64>], labels: &[Vec<i64>]) {
    println!("{}", "* running plaintext debug evaluation".green());

    let mut errors = 0;
    let mut img_num = 0;

    for (img, label) in inputs.iter().zip(labels.iter()) {
        println!(
            "{}",
            format!(
                "test {}/{} [{} errors ({:.2}%)] ",
                img_num + 1,
                inputs.len(),
                errors,
                100.0 * (1.0 - errors as f32 / img_num as f32)
            )
            .yellow()
        );

        let res = nn.eval_plaintext(img).iter().cloned().collect::<Vec<_>>();

        println!("label:  {:?}", label);
        println!("result: {:?}", res);

        if util::index_of_max(&res) != util::index_of_max(label) {
            println!("{}", "error".red());
            errors += 1;
        } else {
            println!("{}", "correct".green());
        }

        img_num += 1;
    }

    println!(
        "errors: {}/{}. accuracy: {}%\n",
        errors,
        img_num,
        100.0 * (1.0 - errors as f32 / img_num as f32)
    );
}
