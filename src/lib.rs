mod utils;
use bitvec::prelude::*;
use rand::{
    distributions::{Distribution, WeightedIndex},
    seq::SliceRandom,
    thread_rng,
};
use std::{
    collections::{HashMap, HashSet},
    convert::{TryFrom, TryInto},
    f32::consts::E,
    io::Cursor,
};
use wasm_bindgen::prelude::*;
const SPRITE_SIZE: usize = 40;
const SPRITE_SCALE: usize = 1;
const FLOW_SIZE: usize = SPRITE_SIZE * SPRITE_SCALE;
const BORDER_SIZE: usize = 1;
const BORDER_FILL: u16 = 0xccc;

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
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
    pub const NBR_DIRS: [FlowDir; 4] = [FlowDir::Left, FlowDir::Right, FlowDir::Up, FlowDir::Down];
    pub const ALL_DIRS: [FlowDir; 11] = [
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
        FlowDir::Detached,
    ];
    pub fn grid_str(grid: Vec<FlowDir>, width: usize) -> String {
        let char_map = |dir| match dir {
            FlowDir::Left => '<',
            FlowDir::Right => '>',
            FlowDir::Up => '^',
            FlowDir::Down => 'v',
            FlowDir::LeftRight => '-',
            FlowDir::LeftUp => '⅃',
            FlowDir::LeftDown => '⅂',
            FlowDir::RightUp => 'L',
            FlowDir::RightDown => 'f',
            FlowDir::UpDown => '|',
            FlowDir::Detached => ' ',
        };
        let mut res = String::new();
        for (pos, c) in grid.iter().enumerate() {
            res.push(char_map(*c));
            if pos % width == width - 1 {
                res.push('\n');
            }
        }
        res
    }
    pub const fn num_connections(self) -> usize {
        (self as u8).count_ones() as usize
    }
    pub fn try_connect(self, other_direction: FlowDir) -> Option<Self> {
        if other_direction.num_connections() != 1 {
            return None;
        }
        FlowDir::try_from((self as u8 | other_direction as u8) as u32).ok()
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
    const fn nbr_of(self, index: usize, width: usize, height: usize) -> Option<usize> {
        match self {
            FlowDir::Left => {
                if index % width == 0 {
                    None
                } else {
                    Some(index - 1)
                }
            }
            FlowDir::Right => {
                if index % width == width - 1 {
                    None
                } else {
                    Some(index + 1)
                }
            }
            FlowDir::Up => {
                if index / width == 0 {
                    None
                } else {
                    Some(index - width)
                }
            }
            FlowDir::Down => {
                if index / width == height - 1 {
                    None
                } else {
                    Some(index + width)
                }
            }
            _ => panic!("tried to get neighbor for non-cardinal direction"),
        }
    }
    fn change_dir(pos_a: usize, pos_b: usize, width: usize, height: usize) -> Option<Self> {
        for dir in Self::NBR_DIRS {
            if let Some(nbr) = dir.nbr_of(pos_a, width, height) {
                if nbr == pos_b {
                    return Some(dir);
                }
            }
        }
        None
    }
    const fn mask(self) -> u32 {
        #[rustfmt::skip]
        let res = 1 << match self {
        FlowDir::Left => 0,     FlowDir::Right => 1,     FlowDir::Up => 2,
        FlowDir::Down => 3,     FlowDir::LeftRight => 4, FlowDir::LeftUp => 5,
        FlowDir::LeftDown => 6, FlowDir::RightUp => 7,   FlowDir::RightDown => 8,
        FlowDir::UpDown => 9,   FlowDir::Detached => 10,
        };
        res
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
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[repr(u8)]

pub enum Flow {
    Dot = 2,
    Line = 1,
    #[default]
    Empty = 0,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
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
    pub width: usize,
    pub height: usize,
    pub fills: Vec<Fill>,
}

impl Default for Board {
    fn default() -> Self {
        Self::new(9, 9)
    }
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
    pub fn new(width: usize, height: usize) -> Self {
        if width * height < 2 {
            panic!("board too small")
        }
        let fills = vec![Default::default(); width * height];
        Self {
            width,
            height,
            fills,
        }
    }
    pub fn as_dirs_grid(&self) -> Vec<FlowDir> {
        self.fills.iter().map(|val| val.dirs).collect()
    }
    fn wfc_gen_dirty(width: usize, height: usize) -> Vec<FlowDir> {
        let mut rng = thread_rng();
        const WEIGHTS: [f32; 10] = [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.5, 1.5];
        let grid_w = width + 2;
        let grid_h = height + 2;
        let mut to_fill: Vec<Option<FlowDir>> = Vec::new();
        for num_iters in 0.. {
            if num_iters > 1_000 {
                panic!("failed to generate");
            }
            to_fill = vec![None; grid_w * grid_h];
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
                let dirs = FlowDir::NBR_DIRS;

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
                    to_fill[candidate] = Some(FlowDir::ALL_DIRS[choice]);
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
                        if let Some(possible) = old_nbrs[i] {
                            all_candidates[idx] = possible
                        }
                    }
                    num_open += 1;
                }
            }
            if !to_fill.contains(&None) {
                // log(&format!("finished in {} iterations", num_iters));
                break;
            }
        }
        let no_borders: Option<Vec<FlowDir>> = to_fill
            .into_iter()
            .filter(|dir| Some(FlowDir::Detached) != *dir)
            .collect();
        no_borders.expect("only breaks from loop if no None values")
    }

    fn get_paths(dirs_grid: &[FlowDir], width: usize, height: usize) -> Vec<Vec<usize>> {
        // NOTE: this is no longer in sorted order, so we have to think about that
        let mut paths: Vec<Vec<usize>> = Vec::new();
        let mut checked = bitvec![0; (width*height)];
        while checked.count_zeros() != 0 {
            let mut to_visit = vec![checked.leading_ones()];
            let mut intermediate: Vec<usize> = Vec::new();
            while let Some(pos) = to_visit.pop() {
                if checked[pos] {
                    continue;
                }
                checked.set(pos, true);
                intermediate.push(pos);
                let flow_dir = dirs_grid[pos];
                for dir in FlowDir::NBR_DIRS {
                    if let Some(nbr) = dir.nbr_of(pos, width, height) {
                        if flow_dir.is_connected(dir) && !checked[nbr] {
                            to_visit.push(nbr);
                        }
                    }
                }
            }
            paths.push(intermediate)
        }
        paths
    }
    fn split_loops_and_intersects(
        dirs_grid: &mut [FlowDir],
        paths: &mut Vec<Vec<usize>>,
        width: usize,
        height: usize,
    ) {
        // log("entered loop splitter");
        let mut created_paths = Vec::new();
        for path in &mut *paths {
            if path
                .iter()
                .map(|&pos| dirs_grid[pos])
                .any(|dir| dir.num_connections() == 1)
            {
                continue; // the path has an endpoint
            }
            // log("loop found!");
            // because it's a dfs on a loop, the elements must be in order
            let path_start = (0, 1);
            let path_middle = (path.len() / 2, path.len() / 2 + 1);
            for pair in [path_start, path_middle] {
                let break_0_a = path[pair.0];
                let break_0_b = path[pair.1];
                let dir_connected_in =
                    FlowDir::change_dir(break_0_a, break_0_b, width, height).unwrap();
                dirs_grid[break_0_a] = dirs_grid[break_0_a].remove_connection(dir_connected_in);
                dirs_grid[break_0_b] =
                    dirs_grid[break_0_b].remove_connection(dir_connected_in.rev());
            }
            let new_path_section: Vec<usize> =
                path.drain((path_start.1)..=(path_middle.0)).collect();
            created_paths.push(new_path_section);
        }
        paths.append(&mut created_paths);
        // log("entered path fixer");
        'outer: for _num_iters in 0.. {
            for path in &mut *paths {
                let mut val = path.iter().enumerate().filter(|(_, pos)| {
                    FlowDir::NBR_DIRS.iter().any(|dir| {
                        !dirs_grid[**pos].is_connected(*dir)
                            && dir
                                .nbr_of(**pos, width, height)
                                .is_some_and(|nbr| path.contains(&nbr))
                    })
                });
                if val.next().is_some() {
                    let start = *path
                        .iter()
                        .find(|pos| dirs_grid[**pos].num_connections() == 1)
                        .unwrap();
                    let mut queue = vec![start];
                    let mut visited = bitvec![0; dirs_grid.len()];
                    // println!("{}: {:?}", start, path);
                    // println!("{}", FlowDir::grid_str(dirs_grid.clone(), width));
                    loop {
                        let pos = *queue.last().unwrap();
                        visited.set(pos, true);
                        let next = FlowDir::NBR_DIRS
                            .iter()
                            .filter_map(|dir| {
                                if dirs_grid[pos].is_connected(*dir) {
                                    dir.nbr_of(pos, width, height).and_then(|val| {
                                        match visited[val] {
                                            true => None,
                                            false => Some(val),
                                        }
                                    })
                                } else {
                                    None
                                }
                            })
                            .next();
                        if next.is_none() {
                            break;
                        }
                        let next = next.unwrap();
                        queue.push(next);
                    }
                    let break_pos = queue.len() / 2;
                    let break_start = queue[break_pos - 1];
                    let break_end = queue[break_pos];

                    let dir_connected_in =
                        FlowDir::change_dir(break_start, break_end, width, height).unwrap();
                    dirs_grid[break_start] =
                        dirs_grid[break_start].remove_connection(dir_connected_in);
                    dirs_grid[break_end] =
                        dirs_grid[break_end].remove_connection(dir_connected_in.rev());

                    let new_vec = queue.split_off(break_pos);
                    path.clear();
                    path.append(&mut queue);
                    created_paths.push(new_vec);
                    continue 'outer;
                }
            }
            break;
        }
        paths.append(&mut created_paths);
        // log("exited loop/intersect");
    }

    pub fn gen_filled_board(width: usize, height: usize) -> Self {
        // 0: left, 1: right, 2: up, 3: down, 4: left-right, 5: left-up,
        // 6: left-down, 7: right-up, 8: right-down, 9: up-down, 10: unconnected
        let mut board = Self::new(width, height); // guarantees width/height are positive
        let (width, height) = (width, height);
        let mut dirs_dirty = Self::wfc_gen_dirty(width, height);
        let mut paths = Self::get_paths(&dirs_dirty, width, height);
        Self::split_loops_and_intersects(&mut dirs_dirty, &mut paths, width, height);
        // for i in 0..dirs_dirty.len() {
        for (i, &dir) in dirs_dirty.iter().enumerate() {
            board.fills[i] = Fill::new(
                0xfff,
                match dir.num_connections() {
                    1 => Flow::Dot,
                    2 => Flow::Line,
                    0 => Flow::Empty,
                    _ => panic!("illegal number of connections when generating board"),
                },
                dir,
            )
        }
        let palette = get_color_palette(paths.len() as i32);
        for (i, path) in paths.iter().enumerate() {
            for pos in path {
                board.fills[*pos].color = palette[i];
            }
        }

        board
    }
    fn verify_valid(&self) -> bool {
        // TODO: implement logic, make an error type, potentially do error correction for basic errors
        // no more than 2 dots of the same color
        // no connections which aren't a pair
        // no connections to a different color
        // no loops
        // no paths that are next to themselves
        // no flows that aren't connected to a dot
        true
    }
    pub fn get_fill(&self, x: usize, y: usize) -> Fill {
        self.fills[y * self.width + x]
    }

    pub fn set_fill(&mut self, x: usize, y: usize, fill: Fill) {
        self.fills[y * self.width + x] = fill;
    }
    fn adjacent(&self, pos_a: usize, pos_b: usize) -> bool {
        pos_a < self.width * self.height
            && pos_b < self.width * self.height
            && ((pos_a % self.width).abs_diff(pos_b % self.width)) <= 1
            && (pos_a / self.width).abs_diff(pos_b / self.width) <= 1
    }

    fn flood_search(&self, pos: usize) -> (BitVec, usize) {
        // traverse from a starting dot, returning when another dot is reached or the path runs out
        let mut explored = bitvec![0; (self.width * self.height)]; //vec![false; (self.width * self.height) as usize];
        match self.fills[pos].flow {
            Flow::Dot => (),
            _ => {
                return (explored, pos);
            }
        };
        let start_color = self.fills[pos].color;

        let mut to_explore: Vec<usize> = Vec::new();
        let mut last_entry = pos;
        to_explore.push(pos);
        while let Some(entry) = to_explore.pop() {
            if explored[entry] {
                continue;
            }
            explored.set(entry, true);
            last_entry = entry;
            if entry != pos {
                if let Flow::Dot = self.fills[entry].flow {
                    break;
                }
            }
            let this_fill = self.fills[entry];
            for nbr in FlowDir::NBR_DIRS.iter().copied() {
                if !this_fill.dirs.is_connected(nbr) {
                    continue;
                }
                let Some(new_pos) = nbr.nbr_of(entry, self.width, self.height) else {
                    continue;
                };
                if explored[new_pos] || start_color != self.fills[new_pos].color {
                    continue;
                }
                // already checked this_fill for connection
                if !self.fills[new_pos].dirs.is_connected(nbr.rev()) {
                    continue;
                }

                to_explore.push(new_pos);
            }
        }
        (explored, last_entry)
    }
    pub fn add_connection(&mut self, pos_a: usize, pos_b: usize) -> bool {
        if !self.adjacent(pos_a, pos_b) {
            return false;
        }
        let fill_a = self.fills[pos_a];
        let fill_b = self.fills[pos_b];
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

        let delta = pos_b.abs_diff(pos_a);
        let my_change_direction = match delta {
            1 if pos_a > pos_b => FlowDir::Left,
            1 if pos_a < pos_b => FlowDir::Right,
            _ if delta == self.width && pos_a > pos_b => FlowDir::Up,
            _ if delta == self.width && pos_a < pos_b => FlowDir::Down,
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
        self.fills[pos_a] = new_fill_a;
        self.fills[pos_b] = new_fill_b;
        true
    }
    // for consistency with add_connection, pos_a will be the one that gets removed
    pub fn remove_connection(&mut self, pos_a: usize, pos_b: usize) -> bool {
        if !self.adjacent(pos_a, pos_b) {
            return false;
        }
        let fill_a = self.fills[pos_a];
        let fill_b = self.fills[pos_b];
        // precondition: A already is a line/dot
        if Flow::Empty == fill_a.flow || Flow::Empty == fill_b.flow || fill_a.color != fill_b.color
        {
            return false;
        }

        let delta = pos_b.abs_diff(pos_a);
        let my_change_direction = match delta {
            1 if pos_a > pos_b => FlowDir::Left,
            1 if pos_a < pos_b => FlowDir::Right,
            _ if delta == self.width && pos_a > pos_b => FlowDir::Up,
            _ if delta == self.width && pos_a < pos_b => FlowDir::Down,
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
        self.fills[pos_a] = new_fill_a;
        self.fills[pos_b] = new_fill_b;

        true
    }

    /*

    */
    pub fn clear_pipe(&mut self, pos: usize) {
        let (visited, _) = self.flood_search(pos);
        // log(&format!("{:?}", visited));
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

    pub fn is_connected(&self, pos_a: usize, pos_b: usize) -> bool {
        let (_, final_pos) = self.flood_search(pos_a);
        pos_b == final_pos
    }

    fn check_all_connected(&self) -> bool {
        // sizes are too small for this to matter performance-wise,
        // but if it ends up being an issue, then use a hashmap
        for start in 0..(self.width * self.height) {
            let fill_start = self.fills[start];
            if let Flow::Dot = fill_start.flow {
                for end in 0..(self.width * self.height) {
                    let fill_end = self.fills[end];
                    if let Flow::Dot = fill_end.flow {
                        if start != end
                            && fill_start.color == fill_end.color
                            && !self.is_connected(start, end)
                        {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    fn fully_filled(&self) -> bool {
        for i in 0..(self.width * self.height) {
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
        bits 0-3: dirs/type (left, right, up, down)
        if the dirs have 3 or 4 high bits, the value is a dot w/ negated directions
        bits 4-15: color (4 bits per color)
    */
    fn write_board(&self) -> Vec<u8> {
        let mut serialized: Vec<u8> = Vec::new();
        serialized.extend(self.width.to_be_bytes().iter());
        serialized.extend(self.height.to_be_bytes().iter());
        for fill in &self.fills {
            let mut fill_data: u16 = 0;
            let mut dirs_info = fill.dirs as u8;
            match dirs_info.count_ones() {
                0..2 => match fill.flow {
                    Flow::Dot => dirs_info = !dirs_info & 0xf,
                    _ => (),
                },
                _ => (),
            };
            fill_data |= dirs_info as u16;
            fill_data |= (fill.color as u16 & 0xfff) << 4;
            serialized.extend(fill_data.to_be_bytes().iter());
        }
        log(&format!("serialized {}", serialized.len()));
        serialized
    }

    fn read_board(serialized: &[u8]) -> Option<Self> {
        let width: usize = {
            let w: &[u8; 4] = serialized[0..4].try_into().ok()?;
            usize::from_be_bytes(*w)
        };
        let height: usize = {
            let h: &[u8; 4] = serialized[4..8].try_into().ok()?;
            usize::from_be_bytes(*h) as usize
        };
        let mut board = Board::new(width, height);
        if width * height != (serialized.len() - 8) / 2 {
            return None;
        }
        for i in (8..serialized.len()).step_by(2) {
            let packed = u16::from_be_bytes(serialized[i..i + 2].try_into().ok()?);
            let mut dirs = packed & 0xf;
            let flow = match dirs.count_ones() {
                0 => Flow::Empty,
                1..=2 => Flow::Line,
                3..=4 => {
                    dirs = !dirs & 0xf;
                    Flow::Dot
                }
                _ => {
                    return None;
                }
            };
            let color = packed >> 4;
            board.fills[(i - 8) / 2] =
                Fill::new(color as u16, flow, FlowDir::try_from(dirs as u32).ok()?);
        }

        if !Self::verify_valid(&board) {
            None
        } else {
            Some(board)
        }
    }
}

#[wasm_bindgen]
#[derive(Default)]
pub struct Canvas {
    board: Board,
    pix_buf: Vec<u8>,
    rendered_flow_cache: HashMap<Fill, [u32; FLOW_SIZE * FLOW_SIZE]>,
    current_pos: Option<(usize, usize)>,
    is_mouse_down: bool,
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
        r << 24 | g << 16 | b << 8 | 0xff
    }

    pub fn new(width: usize, height: usize) -> Self {
        utils::set_panic_hook();
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
        // board.set_fill(0, 0, Fill::new(0xf00, Flow::Dot, [false; 4]));
        // board.set_fill(7, 7, Fill::new(0xf00, Flow::Dot, [false; 4]));
        // board.set_fill(1, 0, Fill::new(0x0f0, Flow::Dot, [false; 4]));
        // board.set_fill(7, 6, Fill::new(0x0f0, Flow::Dot, [false; 4]));
        Self {
            board,
            pix_buf,
            ..Default::default()
        }
    }
    fn render_flow(&mut self, fill: Fill, board_x: usize, board_y: usize) {
        if !self.rendered_flow_cache.contains_key(&fill) {
            let mut prerendered = [0u32; FLOW_SIZE * FLOW_SIZE];
            let sprite: &[u8; SPRITE_SIZE * SPRITE_SIZE] = match fill.flow {
                Flow::Dot => match fill.dirs {
                    FlowDir::Detached => include_bytes!("sprites/0"),
                    FlowDir::Left => include_bytes!("sprites/1"),
                    FlowDir::Right => include_bytes!("sprites/2"),
                    FlowDir::Up => include_bytes!("sprites/3"),
                    FlowDir::Down => include_bytes!("sprites/4"),
                    invalid => panic!("Invalid flow {:?}", invalid),
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
            for y in 0..FLOW_SIZE {
                for x in 0..FLOW_SIZE {
                    let scale =
                        sprite[y / SPRITE_SCALE * SPRITE_SIZE + x / SPRITE_SCALE] as f32 / 255.0;

                    let pix_r = ((((fill.color >> 8) & 0xf) as f32 * scale) as u32) << 4;
                    let pix_g = ((((fill.color >> 4) & 0xf) as f32 * scale) as u32) << 4;
                    let pix_b = (((fill.color & 0xf) as f32 * scale) as u32) << 4;

                    let color = pix_r << 24 | pix_g << 16 | pix_b << 8 | 0xff;
                    prerendered[y * FLOW_SIZE + x] = color;
                }
            }
            self.rendered_flow_cache.insert(fill, prerendered);
        }
        let cached_render = self.rendered_flow_cache[&fill];
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
                let var_name = y * self.board.width;
                let fill = self.board.fills[var_name + x];
                self.render_flow(fill, x, y);
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

    pub fn box_at(&self, x: usize, y: usize) -> Option<Vec<usize>> {
        let x_pos = x % (FLOW_SIZE + BORDER_SIZE);
        let y_pos = y % (FLOW_SIZE + BORDER_SIZE);
        if x_pos >= FLOW_SIZE || y_pos >= FLOW_SIZE {
            return None;
        }
        Some([x / (FLOW_SIZE + BORDER_SIZE), y / (FLOW_SIZE + BORDER_SIZE)].into())
    }

    fn vec_to_tup(&self, pos: Vec<i32>) -> Option<(usize, usize)> {
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
        let Some((x, y)) = self.vec_to_tup(pos) else {
            return;
        };

        let fill_pos = x + y * self.board.width;
        match self.board.get_fill(x, y).flow {
            Flow::Dot => {
                self.board.clear_pipe(fill_pos);
                self.current_pos = Some((x, y));
            }
            Flow::Line => {
                if self.board.fills[fill_pos].dirs.num_connections() == 1 {
                    self.current_pos = Some((x, y))
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
        let Some((x, y)) = self.vec_to_tup(pos) else {
            return;
        };
        let total_diff = current_x.abs_diff(x) + current_y.abs_diff(y);
        if total_diff != 1 {
            return;
        }
        let pos_a = current_x + current_y * self.board.width;
        let pos_b = x + y * self.board.width;
        if self.board.fills[pos_a].flow != Flow::Empty {
            // lazily evaluated, so either one or the other will happen
            let res = self.board.add_connection(pos_a, pos_b)
                || self.board.remove_connection(pos_a, pos_b);
            let fill_at_b = self.board.fills[pos_b];
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
            let fill_at_b = self.board.fills[pos_b];
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
    pub fn box_md(&self, x: usize, y: usize) -> Option<Vec<usize>> {
        let x_pos = x / (FLOW_SIZE + BORDER_SIZE);
        let y_pos = y / (FLOW_SIZE + BORDER_SIZE);
        if x_pos >= FLOW_SIZE || y_pos >= FLOW_SIZE {
            return None;
        }
        let pos = x_pos + y_pos * self.board.width;
        let relevant_fill = self.board.fills[pos];
        if let Flow::Line = relevant_fill.flow {
            if relevant_fill.dirs.num_connections() != 1 {
                return None;
            }
        }
        Some([x / (FLOW_SIZE + BORDER_SIZE), y / (FLOW_SIZE + BORDER_SIZE)].into())
    }
    pub fn clear_pipe(&mut self, pos: Vec<i32>) {
        let Some((x, y)) = self.vec_to_tup(pos) else {
            return;
        };
        let pos = y * self.board.width + x;
        self.board.clear_pipe(pos);
    }

    pub fn add_connection(&mut self, pos: Vec<i32>, delta: i32) -> bool {
        let Some((x, y)) = self.vec_to_tup(pos) else {
            return false;
        };
        let delta = match delta {
            0 => FlowDir::Left,
            1 => FlowDir::Right,
            2 => FlowDir::Up,
            3 => FlowDir::Down,
            _ => return false,
        };
        let pos = x + self.board.width * y;
        let Some(other_pos) = delta.nbr_of(pos, self.board.width, self.board.height) else {
            return false;
        };
        self.board.add_connection(pos, other_pos)
    }
    pub fn remove_connection(&mut self, pos: Vec<i32>, delta: i32) -> bool {
        let Some((x, y)) = self.vec_to_tup(pos) else {
            return false;
        };
        let delta = match delta {
            0 => FlowDir::Left,
            1 => FlowDir::Right,
            2 => FlowDir::Up,
            3 => FlowDir::Down,
            _ => return false,
        };
        let pos = x + self.board.width * y;
        let Some(other_pos) = delta.nbr_of(pos, self.board.width, self.board.height) else {
            return false;
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

    pub fn read_board(&mut self, serialized: &[u8]) {
        let board = Board::read_board(serialized);
        self.board = board.unwrap_or_else(|| Board::new(9, 9));
    }

    pub fn gen_filled_board(width: usize, height: usize) -> Self {
        let mut canvas = Canvas::new(width, height);
        canvas.board = Board::gen_filled_board(width, height);
        canvas
    }

    pub fn gen_new_board(width: usize, height: usize) -> Self {
        let mut canvas = Canvas::gen_filled_board(width, height);
        for x in 0..width {
            for y in 0..height {
                let is_dot = canvas.board.get_fill(x, y).flow == Flow::Dot;
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

    pub fn to_png(&self) -> Vec<u8> {
        let mut to_ret = Vec::new();
        let ref mut buf = Cursor::new(&mut to_ret);
        let mut encoder =
            png::Encoder::new(buf, (self.board.width * 2) as u32, self.board.height as u32);
        encoder.set_color(png::ColorType::Indexed);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Best);
        let palette = get_color_palette(256);
        let plte: Vec<u8> = palette
            .iter()
            .copied()
            .map(|i| Canvas::unpack_color(i))
            .flat_map(|i| (1..4).map(move |pos| ((i >> (pos * 8)) & 0xff) as u8))
            .collect();
        encoder.set_palette(plte);
        {
            let mut pix_writer = encoder.write_header().unwrap();
            pix_writer
                .write_image_data(&self.write_board()[8..])
                .unwrap();
        }
        to_ret
    }
    pub fn from_png(&mut self, png: &[u8]) -> Vec<usize> {
        let mut decoder = png::Decoder::new(Cursor::new(png));
        let info = decoder.read_header_info().unwrap();
        let w = info.width;
        let h = info.height;
        // log(&format!("{w}x{h}"));
        let mut data_reader = decoder.read_info().unwrap();
        self.resize((w / 2) as usize, h as usize);
        let mut pix = vec![0; (w * h + 8) as usize];
        pix[0..4].copy_from_slice(&(w / 2).to_be_bytes());
        pix[4..8].copy_from_slice(&h.to_be_bytes());
        data_reader.next_frame(&mut pix[8..]).unwrap();
        self.from_bytes(&pix);
        vec![self.board.width, self.board.height]
        // log(&format!("{}", pix.iter().map(|&v| v as u32).sum::<u32>()));
    }
    pub fn from_bytes(&mut self, board: &[u8]) {
        if (board.len()) % 2 != 0 {
            panic!("invalid length");
            // return;
        }
        let stated_w = u32::from_be_bytes(board[0..4].try_into().unwrap());
        let stated_h = u32::from_be_bytes(board[4..8].try_into().unwrap());

        if stated_w != self.board.width as u32 || stated_h != self.board.height as u32 {
            panic!(
                "board size should be {}x{}, was {}x{}",
                self.board.width, self.board.height, stated_w, stated_h
            );
            // return;
        }

        self.read_board(board);
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        if self.board.width == width && self.board.height == height {
            return;
        }
        let new_canvas = Canvas::new(width, height);
        self.board = new_canvas.board;
        self.pix_buf = new_canvas.pix_buf;
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
    assert!(size > 0);
    // log(&format!("palette size: {}", size));
    const CLASSIC_COLORS: [u16; 16] = [
        0xf00, 0xff0, 0x13f, 0x0a0, 0xa33, 0xfa0, 0x0ff, 0xf0c, 0x808, 0xfff, 0xaaa, 0x0f0, 0xbb6,
        0x008, 0x088, 0xf19,
    ];
    let mut rng = thread_rng();
    if size < CLASSIC_COLORS.len() as i32 {
        let mut res = CLASSIC_COLORS[0..size as usize].to_vec();
        res.shuffle(&mut rng);
        res
    } else {
        let mut colors = Vec::with_capacity(size as usize);

        let h_step = 360.0 / ((size + 1) as f32);
        // log(&format!("h_step: {}", h_step));

        for i in 0..size {
            for s_val in (3..=9).step_by(6) {
                let (h, s, l) = (h_step * (i as f32), (s_val as f32) / 10.0, 0.5);
                // log(&format!("h{} s{} l{}", h, s, l));
                colors.push(hsl_to_rgb16(h, s, l));
            }
        }
        colors.sort_unstable();
        colors.dedup();
        colors.shuffle(&mut rng);

        if colors.len() < size as usize {
            colors = Vec::with_capacity(11 * 11 * 11);
            for r in 4..16 {
                for g in 4..16 {
                    for b in 4..16 {
                        colors.push(r << 8 | g << 4 | b);
                    }
                }
            }
            // log(&format!("silly path: {}", colors.len()));
            colors.shuffle(&mut rng);
        }
        colors.truncate(size as usize);
        assert_eq!(size, colors.len() as i32, "could not create enough colors");
        colors
    }
}

fn hsl_to_rgb16(h: f32, s: f32, l: f32) -> u16 {
    // https://en.wikipedia.org/wiki/HSL_and_HSV#HSL_to_RGB
    let h = h.clamp(0.0, 360.0);
    let s = s.clamp(0.0, 1.0);
    let l = l.clamp(0.0, 1.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match h_prime {
        0.0..=1.001 => (c, x, 0.0),
        1.0..=2.001 => (x, c, 0.0),
        2.0..=3.001 => (0.0, c, x),
        3.0..=4.001 => (0.0, x, c),
        4.0..=5.001 => (x, 0.0, c),
        5.0..=6.001 => (c, 0.0, x),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    let (r, g, b) = (r1 + m, g1 + m, b1 + m);
    // log(&format!("{} {} {}", r, g, b));
    ((r * 16.0) as u16) << 8 | ((g * 16.0) as u16) << 4 | ((b * 16.0) as u16)
}
