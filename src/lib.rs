mod utils;
use bitvec::prelude::*;
use rand::{
    distributions::{Distribution, WeightedIndex},
    seq::IteratorRandom,
    thread_rng, Rng,
};
use std::{
    collections::HashSet,
    convert::{TryFrom, TryInto},
    f32::consts::E,
};
use wasm_bindgen::prelude::*;
const SPRITE_SIZE: i32 = 40;
const SPRITE_SCALE: i32 = 1;
const FLOW_SIZE: i32 = SPRITE_SIZE * SPRITE_SCALE;
const BORDER_SIZE: i32 = 1;
const BORDER_FILL: u16 = 0xccc;

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum FlowDir {
    #[default]
    Detached = 0b0000,
    Left = 0b0001,
    Right = 0b0010,
    Up = 0b0100,
    Down = 0b1000,
    LeftRight = 0b0011,
    UpDown = 0b1100,
    LeftUp = 0b0101,
    LeftDown = 0b1001,
    RightUp = 0b0110,
    RightDown = 0b1010,
}

impl FlowDir {
    pub const fn num_connections(self) -> usize {
        (self as u8).count_ones() as usize
    }
    pub fn try_connect(self, other_direction: FlowDir) -> Result<Self, ()> {
        if other_direction.num_connections() != 1 {
            return Err(());
        }
        FlowDir::try_from((self as u8 | other_direction as u8) as u32)
    }
    pub fn is_connected(self, other_direction: FlowDir) -> bool {
        if other_direction.num_connections() != 1 {
            panic!("should only be L/R/U/D");
        }
        self as u8 & other_direction as u8 != 0
    }
    pub fn remove_connection(self, other_direction: FlowDir) -> Self {
        if other_direction.num_connections() != 1 {
            return self;
        }
        FlowDir::try_from((self as u8 & !(other_direction as u8)) as u32).unwrap()
    }

    const fn mask(self) -> u32 {
        1 << match self {
            FlowDir::Left => 0,
            FlowDir::Right => 1,
            FlowDir::Up => 2,
            FlowDir::Down => 3,
            FlowDir::LeftRight => 4,
            FlowDir::LeftUp => 5,
            FlowDir::LeftDown => 6,
            FlowDir::RightUp => 7,
            FlowDir::RightDown => 8,
            FlowDir::UpDown => 9,
            FlowDir::Detached => 10,
        }
    }
    pub fn rev(self) -> Self {
        match self {
            FlowDir::Left => FlowDir::Right,
            FlowDir::Right => FlowDir::Left,
            FlowDir::Up => FlowDir::Down,
            FlowDir::Down => FlowDir::Up,
            _ => panic!("can only call with L/R/U/D"),
        }
    }
    #[rustfmt::skip]
    pub fn allowed_adjacent(self, other_direction: FlowDir) -> u32 {
        const LEFT: u32 = 0b0001110001;
        const RIGHT: u32 = 0b0110010010;
        const UP: u32 = 0b1010100100;
        const DOWN: u32 = 0b1101001000;

        const UNCONNECTED_ARR: [u32; 4] = [!RIGHT, !LEFT, !DOWN, !UP];
        const LEFT_ARR: [u32; 4] = [RIGHT & !FlowDir::Right.mask(), !LEFT, !DOWN, !UP];
        const RIGHT_ARR: [u32; 4] = [!RIGHT, LEFT & !FlowDir::Left.mask(), !DOWN, !UP];
        const UP_ARR: [u32; 4] = [!RIGHT, !LEFT, DOWN & !FlowDir::Down.mask(), !UP];
        const DOWN_ARR: [u32; 4] = [!RIGHT, !LEFT, !DOWN, UP & !FlowDir::Up.mask()];
        const LEFTRIGHT_ARR: [u32; 4] = [RIGHT, LEFT, !DOWN, !UP];
        const LEFTUP_ARR: [u32; 4] = [RIGHT & !FlowDir::RightUp.mask(), !LEFT, DOWN & !FlowDir::LeftDown.mask(), !UP];
        const LEFTDOWN_ARR: [u32; 4] = [RIGHT & !FlowDir::RightDown.mask(), !LEFT, !DOWN, UP & !FlowDir::LeftUp.mask()];
        const RIGHTUP_ARR: [u32; 4] = [!RIGHT, LEFT & !FlowDir::LeftUp.mask(), DOWN & !FlowDir::RightDown.mask(), !UP];
        const RIGHTDOWN_ARR: [u32; 4] = [!RIGHT, LEFT & !FlowDir::LeftDown.mask(), !DOWN, UP & !FlowDir::RightUp.mask()];
        const UPDOWN_ARR: [u32; 4] = [!RIGHT, !LEFT, DOWN, UP];

        if other_direction.num_connections() != 1 {
            panic!("should only be L/R/U/D");
        }
        // left, right, up, down
        let idx = match other_direction {
            FlowDir::Left => 0,
            FlowDir::Right => 1,
            FlowDir::Up => 2,
            FlowDir::Down => 3,
            _ => panic!("must only have one connection"),
        };

        (match self {
            FlowDir::Detached => UNCONNECTED_ARR,
            FlowDir::Left => LEFT_ARR,
            FlowDir::Right => RIGHT_ARR,
            FlowDir::Up => UP_ARR,
            FlowDir::Down => DOWN_ARR,
            FlowDir::LeftRight => LEFTRIGHT_ARR,
            FlowDir::LeftUp => LEFTUP_ARR,
            FlowDir::LeftDown => LEFTDOWN_ARR,
            FlowDir::RightUp => RIGHTUP_ARR,
            FlowDir::RightDown => RIGHTDOWN_ARR,
            FlowDir::UpDown => UPDOWN_ARR,
        })[idx]
    }
}
impl TryFrom<u32> for FlowDir {
    type Error = ();
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0b0000 => Ok(FlowDir::Detached),
            0b0001 => Ok(FlowDir::Left),
            0b0010 => Ok(FlowDir::Right),
            0b0100 => Ok(FlowDir::Up),
            0b1000 => Ok(FlowDir::Down),
            0b0011 => Ok(FlowDir::LeftRight),
            0b1100 => Ok(FlowDir::UpDown),
            0b0101 => Ok(FlowDir::LeftUp),
            0b1001 => Ok(FlowDir::LeftDown),
            0b0110 => Ok(FlowDir::RightUp),
            0b1010 => Ok(FlowDir::RightDown),
            _ => Err(()),
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]

pub enum Flow {
    Dot = 2,
    Line = 1,
    #[default]
    Empty = 0,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Fill {
    color: u16,
    flow: Flow,
    dirs: FlowDir,
    // left, right, up, down
}

impl Fill {
    pub fn new(color: u16, flow: Flow, dirs: FlowDir) -> Self {
        Self { color, flow, dirs }
    }
}
// impl Default for Fill {
//     fn default() -> Self {
//         Fill::new(0x000, Default::default(), Default::default())
//     }
// }
#[derive(Clone, Debug)]
pub struct Board {
    pub width: i32,
    pub height: i32,
    pub fills: Vec<Fill>,
}

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

fn softmax(ins: Vec<f32>) -> Vec<f32> {
    let to_the_power: Vec<f32> = ins.iter().copied().map(|val| E.powf(val)).collect();
    let total: f32 = to_the_power.iter().sum();
    to_the_power
        .iter()
        .copied()
        .map(|val| val / total)
        .collect()
}
impl Board {
    pub fn new(width: i32, height: i32) -> Self {
        if width <= 0 || height <= 0 || width * height < 2 {
            panic!("board too small")
        }
        let fills = vec![Default::default(); (width * height) as usize];
        Self {
            width,
            height,
            fills,
        }
    }
    pub fn gen_filled_board(width: i32, height: i32) -> Self {
        // 0: left, 1: right, 2: up, 3: down, 4: left-right, 5: left-up,
        // 6: left-down, 7: right-up, 8: right-down, 9: up-down, 10: unconnected
        let mut board = Self::new(width, height);
        let mut rng = thread_rng();
        const WEIGHTS: [f32; 10] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0];
        let grid_w = (width + 2) as usize;
        let grid_h = (height + 2) as usize;
        let mut to_fill: Vec<Option<FlowDir>> = Vec::new();
        for num_iters in 0.. {
            if num_iters > 1_000 {
                panic!("failed to generate");
            }
            to_fill = vec![None; (grid_w * grid_h) as usize];
            let mut all_candidates: Vec<u32> = vec![0b1111111111; grid_w * grid_h];
            let mut num_open: i32 = all_candidates.len() as i32;
            for i in 0..to_fill.len() {
                // set all edges to FlowDir::Unconnected
                if i % grid_w == 0 {
                    all_candidates[i + 1] &= FlowDir::Detached.allowed_adjacent(FlowDir::Right);
                } else if i % grid_w == (grid_w - 1) {
                    all_candidates[i - 1] &= FlowDir::Detached.allowed_adjacent(FlowDir::Left);
                } else if i / grid_w == 0 {
                    all_candidates[i + grid_w] &= FlowDir::Detached.allowed_adjacent(FlowDir::Down);
                } else if i / grid_w == (grid_h - 1) {
                    all_candidates[i - grid_w] &= FlowDir::Detached.allowed_adjacent(FlowDir::Up);
                } else {
                    continue;
                }
                to_fill[i] = Some(FlowDir::Detached);
                num_open -= 1;
            }

            // let mut ct = 0;
            // idx placed, possible w/o choice, old neighbor possible
            let mut rollbacks: Vec<(usize, u32, [Option<u32>; 4])> = Vec::new();
            while num_open > 0 {
                let candidate = all_candidates
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| to_fill[*i].is_none())
                    .min_by(|(_, a), (_, b)| a.count_ones().cmp(&b.count_ones()))
                    .map(|(index, _)| index)
                    .unwrap();
                // 0: left, 1: right, 2: up, 3: down, 4: left-right, 5: left-up,
                // 6: left-down, 7: right-up, 8: right-down, 9: up-down
                let possible: u32 = all_candidates[candidate];
                #[rustfmt::skip]
                let nbrs = [candidate - 1, candidate + 1, candidate - grid_w, candidate + grid_w];
                let dirs = [FlowDir::Left, FlowDir::Right, FlowDir::Up, FlowDir::Down];

                if possible != 0 {
                    // log(&format!("0b{:010b}", possible));
                    let choices_indices: Vec<usize> =
                        (0..10).filter(|i| possible & (1 << i) != 0).collect();
                    let weights = softmax(choices_indices.iter().map(|&i| WEIGHTS[i]).collect());
                    let dist = WeightedIndex::new(weights).unwrap();
                    let choice = choices_indices[dist.sample(&mut rng)];
                    rollbacks.push((
                        candidate,
                        possible & !(1 << choice),
                        nbrs.iter()
                            .map(|&i| {
                                if to_fill[i].is_none() {
                                    Some(all_candidates[i])
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<Option<u32>>>()
                            .try_into()
                            .unwrap(),
                    ));
                    to_fill[candidate] = Some(
                        [
                            FlowDir::Left,
                            FlowDir::Right,
                            FlowDir::Up,
                            FlowDir::Down,
                            FlowDir::LeftRight,
                            FlowDir::LeftUp,
                            FlowDir::LeftDown,
                            FlowDir::RightUp,
                            FlowDir::RightDown,
                            FlowDir::UpDown,
                        ][choice],
                    );
                    num_open -= 1;
                    for (&possible_neighbor, &change_dir) in nbrs.iter().zip(dirs.iter()) {
                        let dir = to_fill[candidate].unwrap();
                        all_candidates[possible_neighbor] &= dir.allowed_adjacent(change_dir);
                    }
                } else {
                    let (pos, possible_without_choice, old_nbrs) = rollbacks.pop().unwrap();
                    if possible_without_choice == 0 {
                        break;
                    }
                    let old_nbr_idx = [pos - 1, pos + 1, pos - grid_w, pos + grid_w];
                    all_candidates[pos] = possible_without_choice;
                    to_fill[pos] = None;
                    for (i, &idx) in old_nbr_idx.iter().enumerate() {
                        match old_nbrs[i] {
                            Some(possible) => all_candidates[idx] = possible,
                            None => (),
                        }
                    }
                    num_open += 1;
                }
            }
            if !to_fill.contains(&None) {
                log(&format!("finished in {} iterations", num_iters));
                break;
            }
        }
        // divvy into paths and break up loops
        // TODO: break up loops/self-intersections
        let mut paths: Vec<Vec<usize>> = Vec::new();
        let mut checked = bitvec![0; grid_w*grid_h];
        let dirs = [FlowDir::Left, FlowDir::Right, FlowDir::Up, FlowDir::Down];

        for (i, val) in to_fill.iter().enumerate() {
            if let Some(FlowDir::Detached) = val {
                checked.set(i, true);
            }
        }
        while checked.count_zeros() > 0 {
            let mut pos_iter = checked.iter().enumerate().filter(|(i, b)| {
                !**b && to_fill[*i].unwrap_or(FlowDir::Detached).num_connections() == 1
            });
            let mut pos = match pos_iter.next() {
                Some((pos, _)) => pos,
                None => break,
            };
            assert!(to_fill[pos].unwrap().num_connections() == 1);
            let mut intermediate: Vec<usize> = Vec::new();

            while !checked[pos] {
                checked.set(pos, true);
                intermediate.push(pos);
                let nbrs = [pos - 1, pos + 1, pos - grid_w, pos + grid_w];
                if let Some(flow_dir) = to_fill[pos] {
                    for (&dir, nbr) in dirs.iter().zip(nbrs.iter()) {
                        if flow_dir.is_connected(dir) && !checked[*nbr] {
                            pos = *nbr;
                            continue;
                        }
                    }
                }
            }
            paths.push(intermediate);
        }
        // TODO: make this apply to all loops
        // this breaks up only one loop
        // hopefully shouldn't be now
        while checked.count_zeros() > 0 {
            // log("found loop!");
            let start_pos = checked
                .iter()
                .enumerate()
                .filter(|(_, b)| !**b)
                .choose(&mut rng);
            let start_pos = start_pos.unwrap().0;
            let mut chosen_pos = vec![start_pos];
            let mut chosen_dirs = Vec::new();

            while !checked[*chosen_pos.last().unwrap()] {
                let pos = *chosen_pos.last().unwrap();
                checked.set(pos, true);
                let nbrs = [pos - 1, pos + 1, pos - grid_w, pos + grid_w];
                let dirs = [FlowDir::Left, FlowDir::Right, FlowDir::Up, FlowDir::Down];
                let first = nbrs
                    .iter()
                    .zip(dirs.iter())
                    .filter(|(nbr, delta)| {
                        !checked[**nbr] && to_fill[pos].unwrap().is_connected(**delta)
                    })
                    .next();
                let Some(first) = first else {
                    break;
                };
                chosen_dirs.push(*first.1);
                chosen_pos.push(*first.0);
            }
            let last_pos = *chosen_pos.last().unwrap();
            let last_nbrs = [
                last_pos - 1,
                last_pos + 1,
                last_pos - grid_w,
                last_pos + grid_w,
            ];
            let dirs = [FlowDir::Left, FlowDir::Right, FlowDir::Up, FlowDir::Down];
            let last_dir = *dirs
                .iter()
                .zip(last_nbrs.iter())
                .filter(|(_, nbr)| **nbr == start_pos)
                .next()
                .unwrap()
                .0;
            to_fill[last_pos] = Some(to_fill[last_pos].unwrap().remove_connection(last_dir));
            to_fill[start_pos] = Some(
                to_fill[start_pos]
                    .unwrap()
                    .remove_connection(last_dir.rev()),
            );

            let middle = chosen_pos.len() / 2;
            let middle_pos_a = chosen_pos[middle];
            let middle_pos_b = chosen_pos[middle + 1];
            let middle_dir = chosen_dirs[middle];
            to_fill[middle_pos_a] =
                Some(to_fill[middle_pos_a].unwrap().remove_connection(middle_dir));
            to_fill[middle_pos_b] = Some(
                to_fill[middle_pos_b]
                    .unwrap()
                    .remove_connection(middle_dir.rev()),
            );
            let mut old_vec = chosen_pos;
            let new_split = old_vec.split_off(middle + 1);
            paths.push(old_vec);
            paths.push(new_split);
        }
        for i in 0..to_fill.len() {
            to_fill[i] = Some(to_fill[i].unwrap_or(FlowDir::Detached));
        }
        for grid_r in 1..grid_w - 1 {
            for grid_c in 1..grid_h - 1 {
                let board_pos = (grid_c - 1) + (grid_r - 1) * (width as usize);
                board.fills[board_pos] = match to_fill[grid_c + grid_r * grid_w] {
                    Some(dirs) => Fill::new(
                        0xfff,
                        match dirs.num_connections() {
                            1 => Flow::Dot,
                            2 => Flow::Line,
                            0 => Flow::Empty,
                            _ => panic!("illegal number of connections when generating board"),
                        },
                        dirs,
                    ),
                    None => Default::default(),
                };
            }
        }
        let palette = get_color_palette(paths.len() as i32);
        for (i, path) in paths.iter().enumerate() {
            for pos in path {
                let board_pos = pos - grid_w - 2 * (pos / grid_w - 1) - 1;
                board.fills[board_pos].color = palette[i];
            }
        }

        board
    }

    pub fn get_fill(&self, x: i32, y: i32) -> Fill {
        self.fills[(y * self.width + x) as usize]
    }

    pub fn set_fill(&mut self, x: i32, y: i32, fill: Fill) {
        self.fills[(y * self.width + x) as usize] = fill;
    }
    fn adjacent(&self, pos_a: i32, pos_b: i32) -> bool {
        pos_a >= 0
            && pos_a < self.width * self.height
            && pos_b >= 0
            && pos_b < self.width * self.height
            && ((pos_a % self.width) - (pos_b % self.width)).abs() <= 1
            && ((pos_a / self.width) as i32 - (pos_b / self.width) as i32).abs() <= 1
    }

    fn flood_search(&self, pos: i32) -> (BitVec, i32) {
        // traverse from a starting dot, returning when another dot is reached or the path runs out
        // TODO: replace with a bitvec
        let mut explored = bitvec![0; (self.width * self.height) as usize]; //vec![false; (self.width * self.height) as usize];
        match self.fills[pos as usize].flow {
            Flow::Dot => (),
            _ => {
                return (explored, pos);
            }
        };
        let deltas = [-1, 1, -self.width, self.width];
        let fill_start = self.fills[pos as usize];

        let mut to_explore: Vec<i32> = Vec::new();
        let mut last_entry = pos;
        to_explore.push(pos);
        while let Some(entry) = to_explore.pop() {
            if explored[entry as usize] {
                continue;
            }
            explored.set(entry as usize, true);
            last_entry = entry;
            if entry != pos {
                if let Flow::Dot = self.fills[entry as usize].flow {
                    break;
                }
            }
            let this_fill = self.fills[entry as usize];
            for (i, &delta) in deltas.iter().enumerate() {
                if (this_fill.dirs as u8) & (1 << i) == 0 {
                    continue;
                }
                let new_pos = entry + delta;
                if new_pos < 0
                    || new_pos >= self.width * self.height
                    || !self.adjacent(entry, new_pos)
                    || explored[new_pos as usize]
                    || fill_start.color != self.fills[new_pos as usize].color
                {
                    continue;
                }
                // if not connected
                let my_pos = match delta {
                    -1 => 0,                        // left
                    1 => 1,                         // right
                    _ if delta == -self.width => 2, // up
                    _ if delta == self.width => 3,  // down
                    _ => panic!("Invalid delta"),
                };

                let their_pos = [1, 0, 3, 2][my_pos as usize]; // opposite of my_pos
                if (this_fill.dirs as u8) & (1 << my_pos) == 0
                    && (self.fills[new_pos as usize].dirs as u8) & (1 << their_pos) == 0
                {
                    continue;
                }

                to_explore.push(new_pos);
            }
        }
        (explored, last_entry)
    }
    pub fn add_connection(&mut self, pos_a: i32, pos_b: i32) -> bool {
        if !self.adjacent(pos_a, pos_b) {
            return false;
        }
        let fill_a = self.fills[pos_a as usize];
        let fill_b = self.fills[pos_b as usize];
        // precondition: A already is a line/dot
        if let Flow::Empty = fill_a.flow {
            return false;
        } else if let Flow::Empty = fill_b.flow {
            // if B is empty, we'll give it the same color as A
        } else if fill_a.color != fill_b.color {
            return false;
        }

        // dots should only have one connection, lines can have two, empty will have one
        for fill in [fill_a, fill_b] {
            if fill.dirs.num_connections()
                >= match fill.flow {
                    Flow::Empty => 1,
                    Flow::Dot => 1,
                    Flow::Line => 2,
                }
            {
                return false;
            }
        }

        let delta = pos_b - pos_a;
        let my_change_direction = match delta {
            -1 => FlowDir::Left,                       // left
            1 => FlowDir::Right,                       // right
            _ if delta == -self.width => FlowDir::Up,  // up
            _ if delta == self.width => FlowDir::Down, // down
            _ => panic!("Invalid delta"),
        };
        let their_change_direction = my_change_direction.rev();

        // actually add connection between A and B
        let new_fill_a = Fill {
            color: fill_a.color,
            flow: fill_a.flow,
            dirs: fill_a.dirs.try_connect(my_change_direction).unwrap(),
        };
        let new_fill_b = Fill {
            color: match fill_b.flow {
                Flow::Empty => fill_a.color,
                _ => fill_b.color,
            },
            flow: match fill_b.flow {
                Flow::Empty => Flow::Line,
                _ => fill_b.flow,
            },
            dirs: fill_b.dirs.try_connect(their_change_direction).unwrap(),
        };
        self.fills[pos_a as usize] = new_fill_a;
        self.fills[pos_b as usize] = new_fill_b;
        true
    }
    // for consistency with add_connection, pos_a will be the one that gets removed
    pub fn remove_connection(&mut self, pos_a: i32, pos_b: i32) -> bool {
        if !self.adjacent(pos_a, pos_b) {
            return false;
        }
        let fill_a = self.fills[pos_a as usize];
        let fill_b = self.fills[pos_b as usize];
        // precondition: both are not empty, fill_a is not a dot, and both are same color
        match fill_a.flow {
            Flow::Empty => return false,
            Flow::Dot => return false,
            _ => (),
        };
        match fill_b.flow {
            Flow::Empty => return false,
            _ => (),
        };
        if fill_a.color != fill_b.color {
            return false;
        }
        let delta = pos_b - pos_a;
        let my_change_direction = match delta {
            -1 => FlowDir::Left,                       // left
            1 => FlowDir::Right,                       // right
            _ if delta == -self.width => FlowDir::Up,  // up
            _ if delta == self.width => FlowDir::Down, // down
            _ => panic!("Invalid delta"),
        };
        let their_change_direction = my_change_direction.rev();
        if !fill_a.dirs.is_connected(my_change_direction)
            || !fill_b.dirs.is_connected(their_change_direction)
        {
            return false;
        }
        // actually remove connection between A and B
        let new_dirs = fill_a.dirs.remove_connection(my_change_direction);
        let new_fill_a = Fill {
            color: fill_a.color,
            flow: if new_dirs.num_connections() > 0 {
                Flow::Line
            } else {
                Flow::Empty
            },
            dirs: new_dirs,
        };
        let new_fill_b = Fill {
            color: match fill_b.flow {
                Flow::Empty => fill_a.color,
                _ => fill_b.color,
            },
            flow: match fill_b.flow {
                Flow::Empty => Flow::Line,
                _ => fill_b.flow,
            },
            dirs: fill_b.dirs.remove_connection(their_change_direction),
        };
        self.fills[pos_a as usize] = new_fill_a;
        self.fills[pos_b as usize] = new_fill_b;

        true
    }

    /*

    */
    pub fn clear_pipe(&mut self, pos: i32) {
        let (visited, _) = self.flood_search(pos);
        for (i, was_visited) in visited.iter().enumerate() {
            if *was_visited {
                if let Flow::Dot = self.fills[i].flow {
                    self.fills[i].dirs = FlowDir::Detached;
                } else {
                    self.fills[i] = Default::default();
                }
            }
        }
    }

    pub fn is_connected(&self, pos_a: i32, pos_b: i32) -> bool {
        let (_, final_pos) = self.flood_search(pos_a);
        pos_b == final_pos
    }

    fn check_all_connected(&self) -> bool {
        // sizes are too small for this to matter performance-wise,
        // but if it ends up being an issue, then use a hashmap
        for start in 0..(self.width * self.height) {
            let fill_start = self.fills[start as usize];
            if let Flow::Dot = fill_start.flow {
                for end in 0..(self.width * self.height) {
                    let fill_end = self.fills[end as usize];
                    if let Flow::Dot = fill_end.flow {
                        if start != end && fill_start.color == fill_end.color {
                            if !self.is_connected(start, end) {
                                return false;
                            }
                        }
                    }
                }
            }
        }

        true
    }

    fn fully_filled(&self) -> bool {
        for i in 0..(self.width * self.height) as usize {
            if let Flow::Empty = self.fills[i].flow {
                return false;
            }
        }
        true
    }

    /*
    u32 width
    u32 height

    [u32; width*height] fill:
        bits 0-3: dirs (left, right, up, down)
        bits 4-5: flow type (Dot = 2, Line = 1, Empty = 0)
        bits 6-17: color (4 bits per color)
        bits 17-31: reserved for future use

    */
    fn write_board(&self) -> Vec<u8> {
        let mut serialized: Vec<u8> = Vec::new();
        serialized.extend(self.width.to_be_bytes().iter());
        serialized.extend(self.height.to_be_bytes().iter());
        for fill in &self.fills {
            let mut fill_data: u32 = 0;
            fill_data |= (fill.dirs as u8) as u32;
            fill_data |= match fill.flow {
                Flow::Empty => 0 << 4,
                Flow::Line => 1 << 4,
                Flow::Dot => 2 << 4,
            };
            fill_data |= (fill.color as u32 & 0xfff) << 6;
            serialized.extend(fill_data.to_be_bytes().iter());
        }
        serialized
    }

    fn read_board(serialized: &[u32]) -> Self {
        let width = serialized[0].try_into().expect("Board too wide");
        let height = serialized[1].try_into().expect("Board too high");
        let mut board = Board::new(width, height);
        if width * height != (serialized.len() - 2) as i32 {
            panic!("Invalid board size");
        }
        for i in 2..serialized.len() {
            let dirs = FlowDir::try_from(serialized[i] & 0xf).unwrap();
            let flow = match (serialized[i] >> 4) & 0b11 {
                0 => Flow::Empty,
                1 => Flow::Line,
                2 => Flow::Dot,
                _ => panic!("Invalid flow type"),
            };
            let color = (serialized[i] >> 6) & 0xfff;
            board.fills[i - 2] = Fill::new(color as u16, flow, dirs);
        }
        board
    }
}

#[wasm_bindgen]
pub struct Canvas {
    board: Board,
    pix_buf: Vec<u8>,
}

#[wasm_bindgen]
impl Canvas {
    fn set_pix(buf: &mut [u8], loc: usize, color: u32) {
        buf[loc * 4] = ((color >> 24) & 0xff) as u8; // red
        buf[loc * 4 + 1] = ((color >> 16) & 0xff) as u8; // green
        buf[loc * 4 + 2] = ((color >> 8) & 0xff) as u8; // blue
        buf[loc * 4 + 3] = (color & 0xff) as u8; // alpha should just be 255 hopefully
    }

    fn unpack_color(input: u16) -> u32 {
        let r = (((input >> 8) & 0xf) << 4) as u32;
        let g = (((input >> 4) & 0xf) << 4) as u32;
        let b = (((input >> 0) & 0xf) << 4) as u32;
        r << 24 | g << 16 | b << 8 | 0xff
    }

    pub fn new(width: i32, height: i32) -> Self {
        utils::set_panic_hook();
        let board = Board::new(width, height);
        let total_width = board.width * FLOW_SIZE + (board.width - 1) * BORDER_SIZE;
        let total_height = board.height * FLOW_SIZE + (board.height - 1) * BORDER_SIZE;
        let mut pix_buf = (0..((total_width * total_height * 4) as usize))
            .map(|i| if i % 4 == 3 { 0xff } else { 0 })
            .collect::<Vec<u8>>();

        for y in 0..total_height {
            for x in 0..total_width {
                if x % (FLOW_SIZE + BORDER_SIZE) >= FLOW_SIZE
                    || y % (FLOW_SIZE + BORDER_SIZE) >= FLOW_SIZE
                {
                    Self::set_pix(
                        &mut pix_buf,
                        (y * total_width + x) as usize,
                        Canvas::unpack_color(BORDER_FILL),
                    );
                }
            }
        }
        // board.set_fill(0, 0, Fill::new(0xf00, Flow::Dot, [false; 4]));
        // board.set_fill(7, 7, Fill::new(0xf00, Flow::Dot, [false; 4]));
        // board.set_fill(1, 0, Fill::new(0x0f0, Flow::Dot, [false; 4]));
        // board.set_fill(7, 6, Fill::new(0x0f0, Flow::Dot, [false; 4]));
        Self { board, pix_buf }
    }
    fn render_flow(&mut self, fill: Fill, x: i32, y: i32) {
        let sprite: &[u8; (SPRITE_SIZE * SPRITE_SIZE) as usize] = match fill.flow {
            Flow::Dot => match fill.dirs {
                FlowDir::Detached => include_bytes!("sprites/0"),
                FlowDir::Left => include_bytes!("sprites/1"),
                FlowDir::Right => include_bytes!("sprites/2"),
                FlowDir::Up => include_bytes!("sprites/3"),
                FlowDir::Down => include_bytes!("sprites/4"),
                _ => panic!("Invalid flow"),
            },
            Flow::Empty => include_bytes!("sprites/5"),
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
                invalid => panic!("Invalid flow {:?}", invalid),
            },
        };
        let start_x = x * (FLOW_SIZE + BORDER_SIZE);
        let start_y = y * (FLOW_SIZE + BORDER_SIZE);
        for y in 0..FLOW_SIZE {
            for x in 0..FLOW_SIZE {
                let scale = sprite[((y / SPRITE_SCALE) * SPRITE_SIZE + x / SPRITE_SCALE) as usize]
                    as f32
                    / 255.0;

                let pix_r = ((((fill.color >> 8) & 0xf) as f32 * scale) as u32) << 4;
                let pix_g = ((((fill.color >> 4) & 0xf) as f32 * scale) as u32) << 4;
                let pix_b = ((((fill.color >> 0) & 0xf) as f32 * scale) as u32) << 4;

                let color = pix_r << 24 | pix_g << 16 | pix_b << 8 | 0xff;
                let pos = ((start_y + y) * (self.canvas_width()) + start_x + x) as usize;
                Self::set_pix(&mut self.pix_buf, pos, color);
            }
        }
    }
    pub fn render(&mut self) {
        for y in 0..self.board.height {
            for x in 0..self.board.width {
                let fill = self.board.fills[(y * self.board.width + x) as usize];
                self.render_flow(fill, x, y);
            }
        }
    }

    pub fn get_pix_buf(&self) -> *const u8 {
        self.pix_buf.as_ptr() as *const u8
    }

    pub fn width(&self) -> i32 {
        self.board.width
    }

    pub fn height(&self) -> i32 {
        self.board.height
    }
    pub fn canvas_height(&self) -> i32 {
        self.board.height * FLOW_SIZE + (self.board.height - 1) * BORDER_SIZE
    }
    pub fn canvas_width(&self) -> i32 {
        self.board.width * FLOW_SIZE + (self.board.width - 1) * BORDER_SIZE
    }

    pub fn box_at(&self, x: i32, y: i32) -> Option<Vec<i32>> {
        let x_pos = x / (FLOW_SIZE + BORDER_SIZE);
        let y_pos = y / (FLOW_SIZE + BORDER_SIZE);
        if x_pos >= FLOW_SIZE || y_pos >= FLOW_SIZE {
            return None;
        }
        Some([x / (FLOW_SIZE + BORDER_SIZE), y / (FLOW_SIZE + BORDER_SIZE)].into())
    }

    pub fn box_md(&self, x: i32, y: i32) -> Option<Vec<i32>> {
        let x_pos = x / (FLOW_SIZE + BORDER_SIZE);
        let y_pos = y / (FLOW_SIZE + BORDER_SIZE);
        if x_pos >= FLOW_SIZE || y_pos >= FLOW_SIZE {
            return None;
        }
        let pos: i32 = x_pos + y_pos * self.board.width;
        let relevant_fill = self.board.fills[pos as usize];
        if let Flow::Line = relevant_fill.flow {
            if relevant_fill.dirs.num_connections() != 1 {
                return None;
            }
        }
        Some([x / (FLOW_SIZE + BORDER_SIZE), y / (FLOW_SIZE + BORDER_SIZE)].into())
    }
    pub fn clear_pipe(&mut self, pos: Vec<i32>) {
        if pos.len() != 2 {
            return;
        }
        let pos = pos[1] * self.board.width + pos[0];
        self.board.clear_pipe(pos);
    }

    pub fn add_connection(&mut self, pos: Vec<i32>, delta: i32) -> bool {
        if pos.len() != 2 {
            return false;
        }
        let pos = pos[1] * self.board.width + pos[0];
        let other_pos = pos
            + match delta {
                0 => -1,
                1 => 1,
                2 => -self.board.width,
                3 => self.board.width,
                _ => return false,
            };
        self.board.add_connection(pos, other_pos)
    }
    pub fn remove_connection(&mut self, pos: Vec<i32>, delta: i32) -> bool {
        if pos.len() != 2 {
            return false;
        }
        let pos = pos[1] * self.board.width + pos[0];
        let other_pos = pos
            + match delta {
                0 => -1,
                1 => 1,
                2 => -self.board.width,
                3 => self.board.width,
                _ => return false,
            };
        self.board.remove_connection(pos, other_pos)
    }

    pub fn check_all_connected(&self) -> bool {
        self.board.check_all_connected()
    }

    pub fn game_won(&self) -> bool {
        self.board.check_all_connected() && self.board.fully_filled()
    }

    pub fn write_board(&self) -> Vec<u8> {
        self.board.write_board()
    }

    pub fn read_board(&mut self, serialized: &[u32]) {
        let board = Board::read_board(serialized);
        self.board = board;
    }

    pub fn gen_filled_board(width: i32, height: i32) -> Self {
        let mut canvas = Canvas::new(width, height);
        canvas.board = Board::gen_filled_board(width, height);
        canvas
    }

    pub fn gen_new_board(width: i32, height: i32) -> Self {
        let mut canvas = Canvas::gen_filled_board(width, height);
        for x in 0..width {
            for y in 0..height {
                let is_dot = match canvas.board.get_fill(x, y).flow {
                    Flow::Dot => true,
                    _ => false,
                };
                canvas.board.set_fill(
                    x,
                    y,
                    Fill::new(
                        if is_dot {
                            canvas.board.get_fill(x, y).color
                        } else {
                            0x000
                        },
                        if is_dot { Flow::Dot } else { Flow::Empty },
                        FlowDir::Detached,
                    ),
                );
            }
        }
        canvas
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.write_board()
    }

    pub fn from_bytes(&mut self, board: &[u8]) {
        if (board.len()) % 4 != 0 {
            return;
        }
        if u32::from_be_bytes(board[0..4].try_into().unwrap()) != self.board.width as u32
            || u32::from_be_bytes(board[4..8].try_into().unwrap()) != self.board.height as u32
        {
            return;
        }

        self.read_board(
            (0..board.len())
                .step_by(4)
                .map(|i| u32::from_be_bytes(board[i..(i + 4)].try_into().unwrap()))
                .collect::<Vec<u32>>()
                .as_slice(),
        );
    }

    pub fn resize(&mut self, width: i32, height: i32) {
        let new_canvas = Canvas::new(width, height);
        self.board = new_canvas.board;
        self.pix_buf = new_canvas.pix_buf;
    }

    pub fn add_dot_at(&mut self, x: i32, y: i32, color: u16) {
        if (x < 0 || x >= self.board.width) || (y < 0 || y >= self.board.height) {
            return;
        }
        let old_fill = self.board.get_fill(x, y);
        if old_fill.dirs.num_connections() >= 2 {
            return;
        }
        self.board
            .set_fill(x, y, Fill::new(color, Flow::Dot, old_fill.dirs));
    }

    pub fn remove_dot_at(&mut self, x: i32, y: i32) {
        self.board.clear_pipe(y * self.board.width + x);
        self.board.set_fill(x, y, Default::default());
    }

    pub fn remap_color_palette(&mut self, new_palette: Option<Vec<u16>>) {
        let current_palette = self
            .board
            .fills
            .iter()
            .filter(|fill| match fill.flow {
                Flow::Dot => true,
                _ => false,
            })
            .map(|fill| fill.color)
            .collect::<HashSet<u16>>();
        let num_colors = current_palette.len();
        let new_palette = match new_palette {
            Some(palette) => palette,
            None => get_color_palette(num_colors as i32),
        };
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

#[wasm_bindgen]
pub fn get_color_palette(size: i32) -> Vec<u16> {
    const CLASSIC_COLORS: [u16; 8] = [0xf00, 0xff0, 0x13f, 0x0a0, 0xa33, 0xfa0, 0x0ff, 0xf0c];
    const CONTRAST_COLORS: [u16; 25] = [
        0xfaf, 0x7d, 0x930, 0x405, 0x053, 0x2c4, 0xfc9, 0x888, 0x9fb, 0x870, 0x9c0, 0xc08, 0x038,
        0xfa0, 0xfab, 0x460, 0xf01, 0x5ff, 0x098, 0xef6, 0x70f, 0x900, 0xff8, 0xff0, 0xf50,
    ];
    let mut rng = thread_rng();
    if size < CLASSIC_COLORS.len() as i32 {
        CLASSIC_COLORS[0..size as usize].to_vec()
    } else if size < CONTRAST_COLORS.len() as i32 {
        CONTRAST_COLORS
            .iter()
            .choose_multiple(&mut rng, size as usize)
            .iter()
            .map(|&&x| x)
            .collect()
    } else {
        let mut colors = Vec::with_capacity(size as usize);
        loop {
            for _ in colors.len()..size as usize {
                let red = rng.gen_range(4..16);
                let green = rng.gen_range(4..16);
                let blue = rng.gen_range(4..16);
                colors.push((red << 8) | (green << 4) | blue);
            }
            colors.sort();
            colors.dedup();
            if colors.len() == size as usize {
                break;
            }
        }
        colors
    }
}
