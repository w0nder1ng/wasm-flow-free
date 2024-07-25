use std::time::Instant;
use wasm_flow_free::Board;
fn main() {
    const BOARD_SIZE: usize = 9;
    let start = Instant::now();
    println!("generating board of size {}x{}...", BOARD_SIZE, BOARD_SIZE);
    for _ in 0..1000 {
        let board = Board::gen_filled_board(BOARD_SIZE, BOARD_SIZE);
        println!(
            "{}",
            wasm_flow_free::FlowDir::grid_str(board.as_dirs_grid(), BOARD_SIZE)
        );
    }
    // println!("elapsed: {:.2?}, {:?}", start.elapsed(), board.fills[0]);
}
