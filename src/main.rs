use std::hint::black_box;
use std::time::Instant;
use wasm_flow_free::{Board, FlowDir};
fn main() {
    /*
    const BOARD_SIZE: usize = 120;
    const NUM_ITERS: u128 = 4;
    println!("generating boards of size {}x{}...", BOARD_SIZE, BOARD_SIZE);
    let mut total_time = 0;
    for _ in 0..NUM_ITERS {
        let start = Instant::now();
        let _board = black_box(Board::gen_filled_board(BOARD_SIZE, BOARD_SIZE));
        // println!("elapsed: {:.2?}", start.elapsed());
        total_time += start.elapsed().as_nanos();
    }
    println!(
        "average of {}.{} ms/iter",
        total_time / NUM_ITERS / 1_000_000,
        total_time % 1_000_000
    );
    */
    #[rustfmt::skip]
    const WEIGHTS: [f32; 10] = Board::WEIGHTS;
    let mut dist = [0; 10];
    for _ in 0..100 {
        let grid = Board::wfc_gen_dirty(40, 40);
        for val in grid {
            dist[FlowDir::ALL_DIRS.iter().position(|v| *v == val).unwrap()] += 1;
        }
    }
    let total = dist.iter().sum::<i32>();
    println!(
        "dist : {:?}",
        dist.iter()
            .map(|&ct| ct as f32 / total as f32)
            .collect::<Vec<f32>>()
    );
    let total_weight = WEIGHTS.iter().sum::<f32>();
    println!(
        "weight: {:?}",
        WEIGHTS
            .iter()
            .map(|&ct| ct as f32 / total_weight)
            .collect::<Vec<f32>>()
    );
}
