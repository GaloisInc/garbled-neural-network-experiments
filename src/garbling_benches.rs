use crate::layer::Accuracy;
use crate::neural_net::NeuralNet;
use colored::*;
use fancy_garbling::dummy::Dummy;
use fancy_garbling::informer::Informer;
use fancy_garbling::util as numbers;
use fancy_garbling::AllWire;
use fancy_garbling::FancyInput;
use fancy_garbling::WireMod2;
use itertools::Itertools;
use pbr::Pipe;
use swanky_aes_rng::AesRng;
use swanky_channel::Channel;
use swanky_ot_alsz_kos::alsz;
use swanky_twopac::semihonest::{Evaluator, Garbler};
use time::{Duration, PreciseTime};

/// Run benchmarks on the given neural network and its associated parameters.
pub fn bench(
    nn: &NeuralNet,
    bitwidth: &[usize],
    niters: usize,
    secret_weights: bool,
    binary: bool,
    nthreads: usize,
    accuracy: &Accuracy,
) {
    println!("{}", "* running garble/eval benchmark".green());

    // generate moduli for the given bitwidth
    let moduli = bitwidth
        .iter()
        .map(|&b| numbers::modulus_with_width(b as u32))
        .collect_vec();

    println!("{}", "* computing fancy computation info".green());

    ////////////////////////////////////////////////////////////////////////////////
    // run the neural network with Informer
    let mut pb = pbr::ProgressBar::new(nn.nlayers() as u64);
    let mut informer = Informer::new(Dummy::new());

    if binary {
        Channel::with(std::io::empty(), |channel| {
            let inps = (0..nn.num_inputs())
                .map(|_| informer.bin_encode(0, bitwidth[0], channel).unwrap())
                .collect_vec();

            nn.eval_boolean(
                &mut informer,
                &inps,
                bitwidth,
                Some(&mut pb),
                16,
                secret_weights,
                true,
                channel,
            );
            Ok(())
        })
        .unwrap();
    } else {
        Channel::with(std::io::empty(), |channel| {
            let inps = (0..nn.num_inputs())
                .map(|_| informer.crt_encode(0, moduli[0], channel).unwrap())
                .collect_vec();

            nn.eval_arith(
                &mut informer,
                &inps,
                &moduli,
                Some(&mut pb),
                16,
                secret_weights,
                true,
                accuracy,
                channel,
            );
            Ok(())
        })
        .unwrap();
    }
    pb.finish();
    println!("{}", informer.stats());

    ////////////////////////////////////////////////////////////////////////////////
    // bench streaming

    println!(
        "{}",
        "* benchmarking garbler streaming to evaluator".green()
    );

    let mb = pbr::MultiBar::new();
    let mut p1 = mb.create_bar(niters as u64);
    p1.message("Test ");
    let mut p2 = mb.create_bar(nn.nlayers() as u64);
    // let mb_handle = std::thread::spawn(move || mb.listen());

    let mut total_time = Duration::zero();

    for _ in 0..niters {
        p1.inc();

        let start = PreciseTime::now();

        swanky_channel::local::local_channel_pair(
            |channel| {
                if binary {
                    let mut gb: Garbler<_, alsz::Sender, WireMod2> =
                        Garbler::new(channel, AesRng::new()).unwrap();
                    let inps = gb
                        .bin_receive_many(nn.num_inputs(), bitwidth[0], channel)
                        .unwrap();
                    nn.eval_boolean::<_, _, Pipe>(
                        &mut gb,
                        &inps,
                        bitwidth,
                        None,
                        nthreads,
                        secret_weights,
                        true,
                        channel,
                    );
                } else {
                    let mut gb: Garbler<_, alsz::Sender, AllWire> =
                        Garbler::new(channel, AesRng::new()).unwrap();
                    let inps = gb
                        .crt_receive_many(nn.num_inputs(), moduli[0], channel)
                        .unwrap();
                    nn.eval_arith::<_, _, Pipe>(
                        &mut gb,
                        &inps,
                        &moduli,
                        None,
                        nthreads,
                        secret_weights,
                        true,
                        accuracy,
                        channel,
                    );
                }
                Ok(())
            },
            |channel| {
                if binary {
                    let mut ev: Evaluator<AesRng, alsz::Receiver, WireMod2> =
                        Evaluator::new(channel, AesRng::new()).unwrap();
                    let inps = ev
                        .bin_encode_many(&vec![0; nn.num_inputs()], bitwidth[0], channel)
                        .unwrap();
                    nn.eval_boolean::<_, _, Pipe>(
                        &mut ev,
                        &inps,
                        bitwidth,
                        Some(&mut p2),
                        nthreads,
                        secret_weights,
                        false,
                        channel,
                    );
                } else {
                    let mut ev: Evaluator<AesRng, alsz::Receiver, AllWire> =
                        Evaluator::new(channel, AesRng::new()).unwrap();

                    let inps = ev
                        .crt_encode_many(&vec![0; nn.num_inputs()], moduli[0], channel)
                        .unwrap();
                    nn.eval_arith::<_, _, Pipe>(
                        &mut ev,
                        &inps,
                        &moduli,
                        Some(&mut p2),
                        nthreads,
                        secret_weights,
                        false,
                        accuracy,
                        channel,
                    );
                }

                Ok(())
            },
        )
        .unwrap();

        let end = PreciseTime::now();
        total_time = total_time + start.to(end);
    }

    p1.finish();
    p2.finish();

    // mb_handle.join().unwrap();

    total_time = total_time / niters as i32;
    println!("streaming took {} ms", total_time.num_milliseconds());
}
