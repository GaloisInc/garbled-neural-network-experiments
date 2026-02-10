pub mod direct_tests;
pub mod dummy_tests;
pub mod garbling_benches;
pub mod layer;
pub mod neural_net;
pub mod util;

use clap::{App, Arg, ArgMatches, Error, ErrorKind, SubCommand};
use colored::*;
use itertools::Itertools;
use ndarray::Array3;
use serde_json::{self, Value};
use swanky_channel::Channel;

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::layer::Accuracy;
use crate::neural_net::NeuralNet;

static VERSION: &str = "0.1.0";

fn is_dir(s: String) -> Result<(), String> {
    let dir = Path::new(&s);
    if !dir.is_dir() {
        Err(String::from("The value is not a valid directory"))
    } else {
        Ok(())
    }
}

fn is_positive(s: String) -> Result<(), String> {
    if s.parse::<u64>().is_ok() {
        return Ok(());
    } else if let Ok(f) = s.parse::<f64>() {
        if f > 0_f64 {
            return Ok(());
        }
    }
    Err(String::from("Not a positive number"))
}

fn get_arg_usize(matches: &ArgMatches, s: &str) -> usize {
    let v = matches.value_of(s).unwrap();
    v.parse::<usize>().unwrap()
}

pub fn main() {
    let matches = App::new("Garbled Neural Net Experiment Launcher")
        .version(VERSION)
        .author("Brent Carmer <bcarmer@galois.com>")
        .about("Runs experiments for (fancy) garbling neural nets")
        .global_setting(clap::AppSettings::ColoredHelp)

        .arg(Arg::with_name("DIR")
            .help("Sets the neural network directory to use")
            .required(true)
            .validator(is_dir)
            .index(1))

        .arg(Arg::with_name("bitwidth")
            .short("w")
            .long("bitwidth")
            .help("comma separated bitwidths to use for each layer (last number is replicated)")
            .default_value("15")
            .global(true))

        .arg(Arg::with_name("boolean")
            .short("b")
            .long("boolean")
            .help("runs in boolean mode")
            .global(true))

        .arg(Arg::with_name("secret")
            .short("s")
            .long("secret")
            .help("use secret weights")
            .global(true))

        .arg(Arg::with_name("ntests")
            .short("n")
            .help("number of tests to run")
            .takes_value(true)
            .validator(is_positive)
            .value_name("NUM")
            .global(true))

        .arg(Arg::with_name("default-accuracy")
            .long("accuracy")
            .short("a")
            .help("default accuracy for activations and max (overridden by specific accuracy settings)")
            .takes_value(true)
            .global(true)
            .default_value("100%"))

        .arg(Arg::with_name("relu-accuracy")
            .long("relu")
            .help("accuracy of relu")
            .takes_value(true)
            .global(true))

        .arg(Arg::with_name("sign-accuracy")
             .long("sign")
             .help("accuracy of sign")
             .takes_value(true)
             .global(true))

        .arg(Arg::with_name("max-accuracy")
             .long("max")
             .help("accuracy of max")
             .takes_value(true)
             .global(true))

        .subcommand(SubCommand::with_name("bitwidth")
            .about("Evaluate the neural net to find the maximum bitwidth needed for each layer")
            .display_order(1))

        .subcommand(SubCommand::with_name("direct")
            .about("Evaluate the given neural net directly over i64 values")
            .display_order(2)
            .arg(Arg::with_name("debug")
                .short("d")
                .help("show output of each evaluation")))

        .subcommand(SubCommand::with_name("dummy")
            .about("Test the accuracy of the fancy encoding of the neural network")
            .display_order(3)
            .arg(Arg::with_name("debug")
                .short("d")
                .help("show output of each evaluation")))

        .subcommand(SubCommand::with_name("bench")
            .about("Benchmark garbling and evaluating the neural network")
            .arg(Arg::with_name("niters")
                .long("niters")
                .help("number of iterations to run")
                .default_value("1")
                .validator(is_positive))
            .arg(Arg::with_name("nthreads")
                .short("t")
                .help("number of threads used by fancy objects")
                .default_value("8")
                .validator(is_positive))
            .display_order(4))

        .get_matches();

    ////////////////////////////////////////////////////////////////////////////////
    // read tests, labels, and neural net from DIR
    let dir = Path::new(matches.value_of("DIR").unwrap());
    let ntests = matches.value_of("ntests").map(|s| s.parse().unwrap());

    let model_path = dir.join(Path::new("model.json"));
    if !model_path.is_file() {
        Error::exit(&Error::with_description(
            "Given directory does not contain 'model.json'",
            ErrorKind::InvalidValue,
        ));
    }

    let weights_path = dir.join(Path::new("weights.json"));
    if !weights_path.is_file() {
        Error::exit(&Error::with_description(
            "Given directory does not contain 'weights.json'",
            ErrorKind::InvalidValue,
        ));
    }

    print!("reading model...");
    std::io::stdout().flush().unwrap();
    let nn = NeuralNet::from_json(model_path.to_str().unwrap(), weights_path.to_str().unwrap());
    println!("finished");
    // always print info
    nn.print_info();

    let mut tests_path = dir.join(Path::new("tests.json"));
    if !tests_path.is_file() {
        tests_path = dir.join(Path::new("tests.csv"));
        if !tests_path.is_file() {
            Error::exit(&Error::with_description(
                "Given directory contains neither 'tests.json' nor 'tests.csv'",
                ErrorKind::InvalidValue,
            ));
        }
    }

    print!("reading tests...");
    std::io::stdout().flush().unwrap();
    let tests = read_tests(tests_path.to_str().unwrap(), ntests);
    println!("finished");

    let mut labels_path = dir.join(Path::new("labels.json"));
    if !labels_path.is_file() {
        labels_path = dir.join(Path::new("labels.csv"));
        if !labels_path.is_file() {
            Error::exit(&Error::with_description(
                "Given directory contains neither 'labels.json' nor 'labels.csv'",
                ErrorKind::InvalidValue,
            ));
        }
    }

    print!("reading labels...");
    std::io::stdout().flush().unwrap();
    let labels = read_labels(labels_path.to_str().unwrap());
    println!("finished");

    ////////////////////////////////////////////////////////////////////////////////
    // read global options

    let is_secret = matches.is_present("secret");

    let default_accuracy = matches.value_of("default-accuracy").unwrap();
    let accuracy = &Accuracy {
        relu: matches
            .value_of("relu-accuracy")
            .unwrap_or(default_accuracy)
            .to_string(),
        sign: matches
            .value_of("sign-accuracy")
            .unwrap_or(default_accuracy)
            .to_string(),
        max: matches
            .value_of("max-accuracy")
            .unwrap_or(default_accuracy)
            .to_string(),
    };

    ////////////////////////////////////////////////////////////////////////////////
    // compute bitwidth

    // parse bitwidth argument
    let mut bitwidth = matches
        .value_of("bitwidth")
        .unwrap()
        .split(",")
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .expect("bitwidth: expected number")
        })
        .collect_vec();

    // pad the end with the last value
    bitwidth.resize(nn.nlayers() + 1, *bitwidth.last().unwrap());

    assert!(bitwidth[0] != 0, "you need bits for the input, dude");

    // replace 0s with the previous value
    for i in 1..bitwidth.len() {
        if bitwidth[i] == 0 {
            bitwidth[i] = bitwidth[i - 1];
        }
    }

    println!("bitwidth: {:?}", bitwidth);
    println!(
        "nprimes: {:?}",
        bitwidth
            .iter()
            .map(|&w| fancy_garbling::util::primes_with_width(w as u32).len())
            .collect_vec()
    );

    ////////////////////////////////////////////////////////////////////////////////
    // run benches and tests

    if matches.subcommand_matches("bitwidth").is_some() {
        println!("{}", "* computing bitwidth for each layer".green());
        let nbits = Channel::with(std::io::empty(), |channel| {
            Ok(nn.max_bitwidth(&tests, channel))
        })
        .unwrap();
        for (layerno, nbits) in nbits.into_iter().enumerate() {
            println!("Layer {}: {} bits", layerno, nbits);
        }
    } else if let Some(matches) = matches.subcommand_matches("direct") {
        if matches.is_present("debug") {
            direct_tests::direct_debug(&nn, &tests, &labels);
        } else {
            direct_tests::direct_test(&nn, &tests, &labels);
        }
    } else if let Some(matches) = matches.subcommand_matches("dummy") {
        let is_secret = matches.is_present("secret");
        if matches.is_present("boolean") {
            dummy_tests::boolean_accuracy_test(&nn, &tests, &labels, &bitwidth, is_secret);
        } else {
            dummy_tests::arith_accuracy_test(&nn, &tests, &labels, &bitwidth, is_secret, accuracy);
        }
    } else if let Some(matches) = matches.subcommand_matches("bench") {
        let niters = get_arg_usize(matches, "niters");
        let nthreads = get_arg_usize(matches, "nthreads");
        garbling_benches::bench(
            &nn,
            &bitwidth,
            niters,
            is_secret,
            matches.is_present("boolean"),
            nthreads,
            accuracy,
        );
    } else {
        Error::exit(&Error::with_description(
            "no command given! try \"help\"",
            ErrorKind::EmptyValue,
        ));
    }
}

fn read_tests(filename: &str, num: Option<usize>) -> Vec<Array3<i64>> {
    if filename.ends_with(".csv") {
        // Note: csv can be at most 1-dimensional, if each image gets its own line
        let iter = util::get_lines(filename).map(|line| {
            let data = line
                .unwrap()
                .split(",")
                .map(|s| s.parse().expect("couldn't parse!"))
                .collect_vec();
            Array3::from_shape_vec((data.len(), 1, 1), data).expect("couldn't create array!")
        });

        if let Some(n) = num {
            iter.take(n).collect()
        } else {
            iter.collect()
        }
    } else if filename.ends_with(".json") {
        let file = File::open(filename).expect("couldn't open file!");
        let obj: Value = serde_json::from_reader(file).expect("couldn't parse json!");
        let iter = obj
            .as_array()
            .unwrap()
            .iter()
            .map(crate::util::value_to_array3);

        if let Some(n) = num {
            iter.take(n).collect()
        } else {
            iter.collect()
        }
    } else {
        panic!("unsupported filetype: \"{}\"", filename);
    }
}

fn read_labels(filename: &str) -> Vec<Vec<i64>> {
    if filename.ends_with(".csv") {
        util::get_lines(filename)
            .map(|line| {
                line.unwrap()
                    .split(",")
                    .map(|s| s.parse().expect("couldn't parse!"))
                    .collect()
            })
            .collect()
    } else if filename.ends_with(".json") {
        let file = File::open(filename).expect("couldn't open file!");
        let obj: Value = serde_json::from_reader(file).expect("couldn't parse json!");

        obj.as_array()
            .unwrap()
            .iter()
            .map(|val| {
                val.as_array()
                    .unwrap()
                    .iter()
                    .map(|val| val.as_i64().unwrap())
                    .collect()
            })
            .collect()
    } else {
        panic!("unsupported filetype: \"{}\"", filename);
    }
}
