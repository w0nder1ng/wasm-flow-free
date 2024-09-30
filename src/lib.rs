mod board;
mod fill;
mod flow;
pub use board::Board;
use fill::Fill;
use flow::{Flow, FlowDir};
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
};
use std::{
    collections::{HashMap, HashSet},
    convert::TryInto,
    sync::{LazyLock, Mutex},
};
type Pcg = rand_pcg::Pcg32;

// pub use board:::Board;
use wasm_bindgen::prelude::*;
const SPRITE_SIZE: usize = 40;
const SPRITE_SCALE: usize = 1;
const FLOW_SIZE: usize = SPRITE_SIZE * SPRITE_SCALE;
const BORDER_SIZE: usize = 1;
const BORDER_FILL: u16 = 0xccc;
static ENTROPY_SOURCE: LazyLock<Mutex<Pcg>> =
    LazyLock::new(|| Mutex::new(Pcg::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7)));

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

fn is_nbr(pos_a: usize, pos_b: usize, width: usize) -> bool {
    (pos_a % width).abs_diff(pos_b % width) + (pos_a / width).abs_diff(pos_b / width) == 1
}
pub fn reservoir_sample(
    mut to_sample: impl Iterator<Item = usize>,
    weights: &[f32],
    rng: &mut impl Rng,
) -> Option<usize> {
    debug_assert!(weights.iter().all(|&f| f > 0.0));
    let mut current_item = to_sample.next()?;
    let mut total = weights[current_item];
    for item in to_sample {
        let w = weights[item];
        total += w;
        let val = rng.gen_range(0.0..=total);
        if val < w {
            current_item = item;
        }
    }
    Some(current_item)
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
enum WfcStepError {
    ChoiceUnsuccessful,
    NoMoreChoices,
}
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum InvalidBoardError {
    UnpairedDots,
    Loop,
    SelfAdjacentPath,
    DisconnectedFlow,
    DifferentColorConnection,
    UnmatchedConnection,
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct Canvas {
    board: Board,
    solved_board: Option<Board>,
    pix_buf: Vec<u8>,
    rendered_flow_cache: HashMap<Fill, [u32; FLOW_SIZE * FLOW_SIZE]>,
    current_pos: Option<(usize, usize)>,
    is_mouse_down: bool,
    rng: Pcg,
}

impl Default for Canvas {
    fn default() -> Self {
        let mut rng = ENTROPY_SOURCE.lock().unwrap();
        let seed: u64 = rng.gen();
        let stream: u64 = rng.gen();
        Self {
            rng: Pcg::new(seed, stream),
            board: Default::default(),
            solved_board: None,
            pix_buf: Default::default(),
            rendered_flow_cache: Default::default(),
            current_pos: Default::default(),
            is_mouse_down: Default::default(),
        }
    }
}

#[wasm_bindgen]
impl Canvas {
    fn set_pix(buf: &mut [u8], loc: usize, color: u32) {
        buf[loc * 4..loc * 4 + 4].copy_from_slice(&color.to_be_bytes());
    }

    fn unpack_color(input: u16) -> u32 {
        let r = (((input >> 8) & 0xf) << 4) as u32;
        let g = (((input >> 4) & 0xf) << 4) as u32;
        let b = ((input & 0xf) << 4) as u32;
        const CENTER_NUMS: u32 = 0x07070700;
        r << 24 | g << 16 | b << 8 | 0xff | CENTER_NUMS
    }

    pub fn new(width: usize, height: usize) -> Self {
        console_error_panic_hook::set_once();
        let (width, height) = (width, height);
        let board = Board::new(width, height);
        let total_width = board.width * FLOW_SIZE + (board.width - 1) * BORDER_SIZE;
        let total_height = board.height * FLOW_SIZE + (board.height - 1) * BORDER_SIZE;
        let mut pix_buf = (0..(total_width * total_height * 4))
            .map(|i| if i % 4 == 3 { 0xff } else { 0 })
            .collect::<Vec<u8>>();

        for y in 0..total_height {
            for x in 0..total_width {
                if x % (FLOW_SIZE + BORDER_SIZE) >= FLOW_SIZE
                    || y % (FLOW_SIZE + BORDER_SIZE) >= FLOW_SIZE
                {
                    Self::set_pix(
                        &mut pix_buf,
                        y * total_width + x,
                        Canvas::unpack_color(BORDER_FILL),
                    );
                }
            }
        }
        Self {
            board,
            pix_buf,
            ..Default::default()
        }
    }
    fn render_flow(&mut self, fill: &Fill, board_x: usize, board_y: usize) {
        self.rendered_flow_cache
            .entry(fill.clone())
            .or_insert_with(|| {
                let mut prerendered = [0u32; FLOW_SIZE * FLOW_SIZE];
                let sprite: &[u8; SPRITE_SIZE * SPRITE_SIZE] = match fill.flow {
                    Flow::Dot => match fill.dirs {
                        FlowDir::Detached => include_bytes!("sprites/0"),
                        FlowDir::Left => include_bytes!("sprites/1"),
                        FlowDir::Right => include_bytes!("sprites/2"),
                        FlowDir::Up => include_bytes!("sprites/3"),
                        FlowDir::Down => include_bytes!("sprites/4"),
                        invalid => panic!("Invalid flow for dot: {:?}", invalid),
                    },
                    Flow::Empty => match fill.dirs {
                        FlowDir::Detached => include_bytes!("sprites/5"),
                        invalid => panic!("Invalid flow for empty: {:?}", invalid),
                    },
                    Flow::Line => match fill.dirs {
                        FlowDir::Left => include_bytes!("sprites/6"),
                        FlowDir::Right => include_bytes!("sprites/7"),
                        FlowDir::Up => include_bytes!("sprites/8"),
                        FlowDir::Down => include_bytes!("sprites/9"),
                        FlowDir::RightUp => include_bytes!("sprites/10"),
                        FlowDir::RightDown => include_bytes!("sprites/11"),
                        FlowDir::LeftDown => include_bytes!("sprites/12"),
                        FlowDir::LeftUp => include_bytes!("sprites/13"),
                        FlowDir::UpDown => include_bytes!("sprites/14"),
                        FlowDir::LeftRight => include_bytes!("sprites/15"),
                        invalid => panic!("Invalid flow for line: {:?}", invalid),
                    },
                };
                for y in 0..FLOW_SIZE {
                    for x in 0..FLOW_SIZE {
                        let scale = sprite[y / SPRITE_SCALE * SPRITE_SIZE + x / SPRITE_SCALE]
                            as f32
                            / 255.0;

                        let pix_r = ((((fill.color >> 8) & 0xf) as f32 * scale) as u32) << 4;
                        let pix_g = ((((fill.color >> 4) & 0xf) as f32 * scale) as u32) << 4;
                        let pix_b = (((fill.color & 0xf) as f32 * scale) as u32) << 4;

                        let color = pix_r << 24 | pix_g << 16 | pix_b << 8 | 0xff;
                        prerendered[y * FLOW_SIZE + x] = color;
                    }
                }
                prerendered
            });
        let cached_render = self.rendered_flow_cache[fill];
        // let cached_render = [0xff0000ff; FLOW_SIZE * FLOW_SIZE];
        let start_x = board_x * (FLOW_SIZE + BORDER_SIZE);
        let start_y = board_y * (FLOW_SIZE + BORDER_SIZE);
        for y in 0..FLOW_SIZE {
            for x in 0..FLOW_SIZE {
                let color = cached_render[y * FLOW_SIZE + x];
                let pos = (start_y + y) * (self.canvas_width()) + start_x + x;
                Self::set_pix(&mut self.pix_buf, pos, color);
            }
        }
    }
    pub fn render(&mut self) {
        for y in 0..self.board.height {
            for x in 0..self.board.width {
                let fill = self.board.fills[y * self.board.width + x].clone();
                self.render_flow(&fill, x, y);
            }
        }
    }

    pub fn get_pix_buf(&self) -> *const u8 {
        self.pix_buf.as_ptr()
    }

    pub fn width(&self) -> usize {
        self.board.width
    }

    pub fn height(&self) -> usize {
        self.board.height
    }
    pub fn canvas_height(&self) -> usize {
        self.board.height * FLOW_SIZE + (self.board.height - 1) * BORDER_SIZE
    }
    pub fn canvas_width(&self) -> usize {
        self.board.width * FLOW_SIZE + (self.board.width - 1) * BORDER_SIZE
    }

    fn box_at(&self, x: usize, y: usize) -> Option<Vec<usize>> {
        let x_pos = x % (FLOW_SIZE + BORDER_SIZE);
        let y_pos = y % (FLOW_SIZE + BORDER_SIZE);
        if x_pos >= FLOW_SIZE || y_pos >= FLOW_SIZE {
            return None;
        }
        Some([x / (FLOW_SIZE + BORDER_SIZE), y / (FLOW_SIZE + BORDER_SIZE)].into())
    }

    fn try_vec_to_tup(&self, pos: Vec<i32>) -> Option<(usize, usize)> {
        if pos.len() != 2 || pos[0] < 0 || pos[1] < 0 {
            return None;
        }
        let board_pos = self.box_at(pos[0] as usize, pos[1] as usize)?;
        let (x, y) = (board_pos[0], board_pos[1]);
        if !((0..self.board.width).contains(&x) && (0..self.board.height).contains(&y)) {
            return None;
        }
        Some((x, y))
    }
    pub fn handle_md(&mut self, pos: Vec<i32>) {
        let Some((x, y)) = self.try_vec_to_tup(pos) else {
            return;
        };

        let fill_pos = x + y * self.board.width;
        match self.board.fills[fill_pos].flow {
            Flow::Dot => {
                self.board.clear_pipe(fill_pos);
                self.current_pos = Some((x, y));
            }
            Flow::Line => {
                if self.board.fills[fill_pos].dirs.num_connections() == 1 {
                    self.current_pos = Some((x, y));
                };
            }
            Flow::Empty => (),
        };
        self.is_mouse_down = true;
    }
    pub fn handle_mu(&mut self) {
        // self.current_pos = None;
        self.is_mouse_down = false;
    }
    pub fn handle_mm(&mut self, pos: Vec<i32>) {
        if !self.is_mouse_down {
            return;
        }
        let Some((current_x, current_y)) = self.current_pos else {
            return;
        };
        let Some((x, y)) = self.try_vec_to_tup(pos) else {
            return;
        };
        let diff_x = current_x.abs_diff(x);
        let diff_y = current_y.abs_diff(y);
        let changed_x = match x > current_x {
            true => current_x + 1,
            false => current_x - 1,
        };
        let changed_y = match y > current_y {
            true => current_y + 1,
            false => current_y - 1,
        };
        let (x, y) = match diff_x + diff_y {
            0 => return, // no change in position, so we're done
            1 => (x, y), // if diff is only one, we don't have to actually choose
            _ => {
                if diff_x >= diff_y
                    && (self.board.is_attached_nbrs(
                        changed_x + self.board.width * current_y,
                        current_x + self.board.width * current_y,
                    ) || self.board.get_fill(changed_x, current_y).flow == Flow::Empty)
                {
                    (changed_x, current_y)
                } else {
                    (current_x, changed_y)
                }
            }
        };
        // avoid attempting to connect when other index is out of bounds
        if current_x.abs_diff(x) > 1 || current_y.abs_diff(y) > 1 {
            return;
        }
        let pos_a = current_x + current_y * self.board.width;
        let pos_b = x + y * self.board.width;
        if self.board.fills[pos_a].flow != Flow::Empty {
            // lazily evaluated, so either one or the other will happen
            let res = self.board.add_connection(pos_a, pos_b)
                || self.board.remove_connection(pos_a, pos_b);
            let fill_at_b = &self.board.fills[pos_b];
            let at_max_connections = match fill_at_b.flow {
                Flow::Dot => fill_at_b.dirs.num_connections() == 1,
                Flow::Line => fill_at_b.dirs.num_connections() == 2,
                Flow::Empty => true,
            };
            if res {
                // if the new spot isn't all the way connected, force the mouse to disconnect
                self.current_pos = if at_max_connections {
                    None
                } else {
                    Some((x, y))
                };
            }
        }
    }
    pub fn handle_keypress(&mut self, keypress: String) {
        let delta = match keypress.chars().next() {
            Some('a') => FlowDir::Left,
            Some('d') => FlowDir::Right,
            Some('w') => FlowDir::Up,
            Some('s') => FlowDir::Down,
            _ => return,
        };
        let Some((current_x, current_y)) = self.current_pos else {
            return;
        };
        let pos_a = current_x + current_y * self.board.width;
        let Some(pos_b) = delta.nbr_of(pos_a, self.board.width, self.board.height) else {
            return;
        };
        if self.board.fills[pos_a].flow != Flow::Empty {
            // lazily evaluated, so either one or the other will happen
            let res = self.board.add_connection(pos_a, pos_b)
                || self.board.remove_connection(pos_a, pos_b);
            let fill_at_b = &self.board.fills[pos_b];
            let at_max_connections = match fill_at_b.flow {
                Flow::Dot => fill_at_b.dirs.num_connections() == 1,
                Flow::Line => fill_at_b.dirs.num_connections() == 2,
                Flow::Empty => true,
            };
            if res {
                // if the new spot isn't all the way connected, force the mouse to disconnect
                self.current_pos = if at_max_connections {
                    None
                } else {
                    Some((pos_b % self.board.width, pos_b / self.board.width))
                };
            }
        }
    }

    pub fn check_all_connected(&self) -> bool {
        self.board.check_all_connected()
    }

    pub fn game_won(&self) -> bool {
        self.board.fully_filled() && self.board.check_all_connected()
    }

    pub fn write_board(&self) -> Vec<u8> {
        self.board.write_board()
    }

    pub fn read_board(&mut self, serialized: &[u8]) {
        let board = Board::read_board(serialized);
        self.board = board.unwrap_or_else(|| Board::new(9, 9));
    }

    pub fn gen_filled_board(width: usize, height: usize) -> Self {
        let mut canvas = Canvas::new(width, height);
        canvas.board = Board::gen_filled_board(width, height, &mut canvas.rng);
        canvas.solved_board = Some(canvas.board.clone());
        canvas
    }
    pub fn clear_board(&mut self) {
        let width = self.board.width;
        let height = self.board.height;
        for x in 0..width {
            for y in 0..height {
                let is_dot = self.board.get_fill(x, y).flow == Flow::Dot;
                self.board.set_fill(
                    x,
                    y,
                    Fill::new(
                        if is_dot {
                            self.board.get_fill(x, y).color
                        } else {
                            0x000
                        },
                        if is_dot { Flow::Dot } else { Flow::Empty },
                        FlowDir::Detached,
                    ),
                );
            }
        }
    }
    // wasm-bindgen doesn't seem to export clone, so I'm adding this
    pub fn get_cloned(&self) -> Self {
        self.clone()
    }
    pub fn gen_new_board(width: usize, height: usize) -> Self {
        let mut canvas = Canvas::gen_filled_board(width, height);
        canvas.clear_board();
        canvas
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.write_board()
    }

    pub fn from_bytes(&mut self, board: &[u8]) {
        assert_eq!(board.len() % 2, 0, "invalid length");

        let stated_w = u32::from_be_bytes(board[12..12 + 4].try_into().unwrap());
        let stated_h = u32::from_be_bytes(board[12 + 4..12 + 4 + 4].try_into().unwrap());

        if stated_w != self.board.width as u32 || stated_h != self.board.height as u32 {
            self.resize(stated_w as usize, stated_h as usize);
        }

        self.read_board(board);
    }
    pub fn hint(&mut self) {
        // TODO: fix the utterly awful mess that is this function
        if let Some(ref mut solved_board) = self.solved_board {
            // fix all incorrect squares
            for fill_pos in 0..solved_board.fills.len() {
                let unsolved = &self.board.fills[fill_pos];
                if unsolved.flow == Flow::Dot {
                    // log(&format!("called at dot pos {fill_pos}"));
                    let (_, end) = self.board.flood_search(fill_pos, |entry| {
                        self.board.fills[entry] != solved_board.fills[entry]
                    });
                    let end_fill = &self.board.fills[end];
                    let end_fill_solved = &solved_board.fills[end];
                    if end_fill.color != end_fill_solved.color
                        || match end_fill.dirs.num_connections() {
                            1 => !end_fill_solved.dirs.is_connected(end_fill.dirs),
                            2 => end_fill_solved.dirs != end_fill.dirs,
                            _ => false,
                        }
                    {
                        // TODO: discriminate incorrect vs funky
                        self.board.clear_pipe(fill_pos);
                        if self.current_pos.is_some() {
                            self.current_pos =
                                Some((fill_pos % self.board.width, fill_pos % self.board.width));
                        }
                    }
                }
            }
            // since everything is correct, we can definitely add one connection

            let Some(free) = self
                .current_pos
                .and_then(|(x, y)| Some(x + y * self.board.width))
                .filter(|pos| {
                    let fill = &self.board.fills[*pos];
                    let threshold = match fill.flow {
                        Flow::Dot => 0,
                        Flow::Line => 1,
                        Flow::Empty => 999, // impossible, so this should never get called
                    };
                    fill.dirs.num_connections() == threshold
                })
                .or_else(|| {
                    (0..self.board.fills.len())
                        .filter(|i| {
                            self.board.fills[*i].color == solved_board.fills[*i].color
                                && solved_board.fills[*i].dirs.num_connections()
                                    - self.board.fills[*i].dirs.num_connections()
                                    == 1
                        })
                        .choose(&mut self.rng)
                })
            else {
                return;
            };
            let nbr_dir = solved_board.fills[free]
                .dirs
                .remove_connection(self.board.fills[free].dirs);
            let nbr = nbr_dir
                .nbr_of(free, self.board.width, self.board.height)
                .expect("guaranteed to have a neighbor");
            let old_dirs = self.board.fills[nbr].dirs;
            self.board.fills[nbr] = solved_board.fills[nbr].clone();
            self.board.fills[nbr].dirs = old_dirs
                .try_connect(nbr_dir.rev())
                .expect("should not be attached already (nbr)");
            self.board.fills[free].dirs = self.board.fills[free]
                .dirs
                .try_connect(nbr_dir)
                .expect("should not be attached already");
            if Some((free % self.board.width, free / self.board.width)) == self.current_pos {
                self.current_pos = Some((nbr % self.board.width, nbr / self.board.width));
            }
        }
    }
    pub fn resize(&mut self, width: usize, height: usize) {
        if self.board.width == width && self.board.height == height {
            return;
        }
        let new_canvas = Canvas::new(width, height);
        *self = new_canvas;
    }

    pub fn add_dot_at(&mut self, x: usize, y: usize, color: u16) {
        if (x >= self.board.width) || (y >= self.board.height) {
            return;
        }
        let old_fill = self.board.get_fill(x, y);
        if old_fill.dirs.num_connections() >= 2 {
            return;
        }
        self.board
            .set_fill(x, y, Fill::new(color, Flow::Dot, old_fill.dirs));
    }

    pub fn remove_dot_at(&mut self, x: usize, y: usize) {
        self.board.clear_pipe(y * self.board.width + x);
        self.board.set_fill(x, y, Default::default());
    }

    pub fn remap_color_palette(&mut self, new_palette: Option<Vec<u16>>) {
        let current_palette = self
            .board
            .fills
            .iter()
            .filter(|fill| fill.flow == Flow::Dot)
            .map(|fill| fill.color)
            .collect::<HashSet<u16>>();
        let num_colors = current_palette.len();
        let new_palette = new_palette.unwrap_or_else(|| {
            get_color_palette(num_colors, &mut self.rng).expect("too many colors")
        });
        if new_palette.len() < num_colors {
            return;
        }
        for fill in self.board.fills.iter_mut() {
            match fill.flow {
                Flow::Empty => (),
                _ => {
                    let new_color = new_palette[current_palette
                        .iter()
                        .position(|&x| x == fill.color)
                        .unwrap()];
                    fill.color = new_color;
                }
            };
        }
    }
}

fn get_color_palette(size: usize, rng: &mut impl Rng) -> Option<Vec<u16>> {
    // log(&format!("palette size: {}", size));
    const CLASSIC_COLORS: [u16; 16] = [
        0xf00, 0xff0, 0x13f, 0x0a0, 0xa33, 0xfa0, 0x0ff, 0xf0c, 0x808, 0xfff, 0xaaa, 0x0f0, 0xbb6,
        0x008, 0x088, 0xf19,
    ];
    if size <= CLASSIC_COLORS.len() {
        let mut res = CLASSIC_COLORS[0..size].to_vec();
        res.shuffle(rng);
        Some(res)
    } else {
        let mut colors = Vec::with_capacity(size);

        let h_step = 360.0 / ((size + 1) as f32);
        // log(&format!("h_step: {}", h_step));

        for i in 0..size {
            for s_val in [0.5, 0.95] {
                let (h, s, l) = (h_step * (i as f32), s_val, 0.5);
                // log(&format!("h{} s{} l{}", h, s, l));
                colors.push(hsl_to_rgb16(h, s, l));
            }
        }
        colors.sort_unstable();
        colors.dedup();

        if colors.len() < size {
            colors = Vec::with_capacity(11 * 11 * 11);
            for r in 4..=15 {
                for g in 4..=15 {
                    for b in 4..=15 {
                        colors.push(r << 8 | g << 4 | b);
                    }
                }
            }
            // log(&format!("silly path: {}", colors.len()));
        }
        colors.shuffle(rng);
        colors.truncate(size);
        if colors.len() < size {
            None
        } else {
            Some(colors)
        }
    }
}
#[wasm_bindgen]
pub fn seed_rng(data: &[u8]) {
    let mut rng = ENTROPY_SOURCE.lock().unwrap();
    assert_eq!(data.len(), 8);
    let data_raw: u64 = data
        .iter()
        .enumerate()
        .map(|(i, dat)| (*dat as u64) << (8 * i))
        .sum();
    *rng = Pcg::new(data_raw, 0xa02bdbf7bb3c0a7);
}
fn hsl_to_rgb16(h: f32, s: f32, l: f32) -> u16 {
    // https://en.wikipedia.org/wiki/HSL_and_HSV#HSL_to_RGB
    let h = h.clamp(0.0, 360.0);
    let s = s.clamp(0.0, 1.0);
    let l = l.clamp(0.0, 1.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let (r1, g1, b1) = match h_prime {
        0.0..1.0 => (c, x, 0.0),
        1.0..=2.0 => (x, c, 0.0),
        2.0..=3.0 => (0.0, c, x),
        3.0..=4.0 => (0.0, x, c),
        4.0..=5.0 => (x, 0.0, c),
        5.0..=6.0 => (c, 0.0, x),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    let (r, g, b) = (r1 + m, g1 + m, b1 + m);
    // log(&format!("{} {} {}", r, g, b));
    ((r * 16.0) as u16) << 8 | ((g * 16.0) as u16) << 4 | ((b * 16.0) as u16)
}
