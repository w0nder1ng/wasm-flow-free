use std::hint::black_box;
use std::time::Instant;
use wasm_flow_free::{Board, FlowDir};
fn main() {
    const BOARD_SIZE: usize = 40;
    const NUM_ITERS: u128 = 200;
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
}
