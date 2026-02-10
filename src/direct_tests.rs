use colored::*;
use itertools::Itertools;
use ndarray::Array3;

use crate::neural_net::NeuralNet;
use crate::util;

/// Test the neural network directly over i64 values.
pub fn direct_test(nn: &NeuralNet, inputs: &[Array3<i64>], labels: &[Vec<i64>]) {
    println!("{}", "* running plaintext accuracy evaluation".green());

    let mut errors = 0;
    let mut img_num = 0;

    let mut total_time = time::Duration::zero();

    let mut pb = pbr::ProgressBar::new(inputs.len() as u64);
    for (img, label) in inputs.iter().zip(labels.iter()) {
        pb.inc();
        pb.message(&format!(
            "Testing (avg {}ms) [{} errors ({:.2}%)] ",
            if img_num > 0 {
                total_time.num_milliseconds() / img_num as i64
            } else {
                0
            },
            errors,
            100.0 * (1.0 - errors as f32 / img_num as f32)
        ));

        let start = time::PreciseTime::now();
        let res = nn.eval_plaintext(img).iter().cloned().collect_vec();
        let end = time::PreciseTime::now();
        total_time = total_time + start.to(end);

        if util::index_of_max(&res) != util::index_of_max(label) {
            errors += 1;
        }

        img_num += 1;
    }

    pb.finish_println(&format!(
        "errors: {}/{}. accuracy: {}%\n",
        errors,
        img_num,
        100.0 * (1.0 - errors as f32 / img_num as f32)
    ));
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

        let res = nn.eval_plaintext(img).iter().cloned().collect_vec();

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
