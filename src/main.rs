#[allow(unused)]
use rand::{distributions::WeightedIndex, prelude::Distribution, thread_rng, Rng};
use std::convert::TryInto;
use std::hint::black_box;
use std::time::{Duration, Instant};
use wasm_flow_free::{reservoir_sample, Board};
#[allow(unused)]
fn gen_boards(size: usize, num_iters: u128) {
    println!("generating boards of size {}x{}...", size, size);
    let mut rng = thread_rng();
    let mut total_time = 0;
    for _ in 0..num_iters {
        let start = Instant::now();

        let _board = black_box(Board::gen_filled_board(size, size, &mut rng));
        // println!("elapsed: {:.2?}", start.elapsed());
        total_time += start.elapsed().as_nanos();
    }
    println!(
        "average of {:?} ms/iter",
        Duration::from_nanos((total_time / num_iters).try_into().unwrap())
    );
}
fn test_reservoir_sample(num_iters: u128) {
    const NUM_WEIGHTS: usize = 10;
    let mut chosen = [0u128; NUM_WEIGHTS];
    let weights: [f32; NUM_WEIGHTS] = [1.0; NUM_WEIGHTS];
    println!("choosing from {} choices", NUM_WEIGHTS);
    let mut total_time = 0;
    let start = Instant::now();
    let mut rng = thread_rng();
    for _ in 0..num_iters {
        // let mut possible = rng.gen::<u16>() & ((1 << NUM_WEIGHTS) - 1);
        // while possible == 0 {
        //     possible = rng.gen::<u16>() & ((1 << NUM_WEIGHTS) - 1);
        // }
        let possible = std::hint::black_box((1 << NUM_WEIGHTS) - 1);
        let choices_indices = (0..10).filter(|i| possible & (1 << i) != 0);
        let choices_indices: Vec<usize> = black_box(choices_indices.collect());
        let weights: Vec<f32> = choices_indices.iter().map(|&i| weights[i]).collect();
        let dist = black_box(WeightedIndex::new(weights).unwrap());
        let res = Some(choices_indices[dist.sample(&mut rng)]);
        // let res = black_box(reservoir_sample(choices_indices, &weights, &mut rng));
        match res {
            Some(res) => chosen[res] += 1,
            None => panic!("brokey"),
        }
        // println!("elapsed: {:.2?}", start.elapsed());
    }
    total_time += start.elapsed().as_nanos();
    println!(
        "average of {:?}/iter",
        Duration::from_nanos((total_time / num_iters).try_into().unwrap())
    );
    println!(
        "for a total of {:?}",
        Duration::from_nanos(total_time.try_into().unwrap())
    );
    println!("actual: {:?}", chosen);
    let weights_sum: f32 = weights.iter().sum();
    let expected: Vec<u128> = weights
        .iter()
        .map(|f| (num_iters as f32 * (f / weights_sum)) as u128)
        .collect();
    println!("expected : {:?}", expected);
}

fn main() {
    gen_boards(40, 200);
    // test_reservoir_sample(1_000_000);
}
