use bitvec::{bitvec, vec::BitVec};
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
};
use std::{
    collections::{HashMap, HashSet},
    convert::{TryFrom, TryInto},
    fmt::{Display, Error, Formatter, Write},
    io::{Cursor, Read},
    sync::{LazyLock, Mutex},
};
type Pcg = rand_pcg::Pcg32;
use wasm_bindgen::prelude::*;
const SPRITE_SIZE: usize = 40;
const SPRITE_SCALE: usize = 1;
const FLOW_SIZE: usize = SPRITE_SIZE * SPRITE_SCALE;
const BORDER_SIZE: usize = 1;
const BORDER_FILL: u16 = 0xccc;
const FLOW_FREE_HEADER: &[u8; 8] = b"FFBOARD\x97";
const BOARD_FORMAT_VERSION: u32 = 0x01;
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

impl Display for FlowDir {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.write_char(match self {
            FlowDir::Left => '<',
            FlowDir::Right => '>',
            FlowDir::Up => '^',
            FlowDir::Down => 'v',
            FlowDir::LeftRight => '═',
            FlowDir::LeftUp => '╝',
            FlowDir::LeftDown => '╗',
            FlowDir::RightUp => '╚',
            FlowDir::RightDown => '╔',
            FlowDir::UpDown => '║',
            FlowDir::Detached => ' ',
        })
    }
}
impl FlowDir {
    pub const NBR_DIRS: [FlowDir; 4] = [FlowDir::Left, FlowDir::Right, FlowDir::Up, FlowDir::Down];
    #[rustfmt::skip]
    pub const ALL_DIRS: [FlowDir; 11] = [
        FlowDir::Left, FlowDir::Right, FlowDir::Up, FlowDir::Down,
        FlowDir::LeftRight,
        FlowDir::LeftUp, FlowDir::LeftDown, FlowDir::RightUp, FlowDir::RightDown,
        FlowDir::UpDown,
        FlowDir::Detached,
    ];
    pub const fn num_connections(self) -> usize {
        (self as u8).count_ones() as usize
    }
    pub fn try_connect(self, other_direction: FlowDir) -> Option<Self> {
        assert_eq!(
            other_direction.num_connections(),
            1,
            "other direction must only have one connection"
        );
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
    fn change_dir(pos_a: usize, pos_b: usize, width: usize) -> Option<Self> {
        let delta = pos_b.abs_diff(pos_a);
        match delta {
            1 if pos_a > pos_b => Some(FlowDir::Left),
            1 if pos_a < pos_b => Some(FlowDir::Right),
            _ if delta == width && pos_a > pos_b => Some(FlowDir::Up),
            _ if delta == width && pos_a < pos_b => Some(FlowDir::Down),
            _ => None,
        }
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
    pub fn flow_type(self) -> Flow {
        match self.num_connections() {
            1 => Flow::Dot,
            2 => Flow::Line,
            0 => Flow::Empty,
            _ => panic!("illegal number of connections when generating board"),
        }
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

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Fill {
    color: u16,
    flow: Flow,
    dirs: FlowDir,
}

impl Fill {
    pub fn new(color: u16, flow: Flow, dirs: FlowDir) -> Self {
        Self { color, flow, dirs }
    }
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

impl Display for Board {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        for (i, fill) in self.fills.iter().enumerate() {
            fill.dirs.fmt(f)?;
            if i % self.width == self.width - 1 {
                f.write_char('\n')?;
            }
        }
        Ok(())
    }
}
impl Board {
    const DOT_WEIGHT: f32 = 0.1;
    const TURN_WEIGHT: f32 = 1.0;
    const LINE_WEIGHT: f32 = 1.5;
    #[rustfmt::skip]
    pub const WEIGHTS: [f32; 10] = [
        Self::DOT_WEIGHT, Self::DOT_WEIGHT,Self::DOT_WEIGHT,Self::DOT_WEIGHT,
        Self::LINE_WEIGHT,
        Self::TURN_WEIGHT, Self::TURN_WEIGHT, Self::TURN_WEIGHT,Self::TURN_WEIGHT,
        Self::LINE_WEIGHT,
    ];

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

    /// take one step in the wave function collapse algorithm.
    /// if it succeeded, return the filled-in index and set things as appropriate.
    /// if it failed, return None and undo all changes, removing the invalid choice from the possibilities
    fn wfc_step(
        to_fill: &mut [Option<FlowDir>],
        all_candidates: &mut [u32],
        grid_w: usize,
        grid_h: usize,
        rng: &mut impl Rng,
    ) -> Result<usize, WfcStepError> {
        let mut min_entropy = 11; // # of possibilities + 1
        let mut choices = Vec::new();
        for (i, item) in all_candidates.iter().enumerate() {
            if to_fill[i].is_none() {
                match item.count_ones() {
                    val if val < min_entropy => {
                        min_entropy = val;
                        choices.clear();
                        choices.push(i);
                    }
                    val if val == min_entropy => {
                        choices.push(i);
                    }
                    _ => (),
                }
            }
        }
        let candidate = *choices.choose(rng).unwrap();
        // 0: left, 1: right, 2: up, 3: down, 4: left-right, 5: left-up,
        // 6: left-down, 7: right-up, 8: right-down, 9: up-down
        let possible: u32 = all_candidates[candidate];

        if possible != 0 {
            let choices_indices = (0..10).filter(|i| possible & (1 << i) != 0);
            let choice = reservoir_sample(choices_indices, &Self::WEIGHTS, rng).expect("brokey");

            to_fill[candidate] = Some(FlowDir::ALL_DIRS[choice]);
            all_candidates[candidate] = 1 << choice;

            let permissible = |could_be: u32, delta: FlowDir| {
                let mut capable = 0;
                for i in 0..10 {
                    if could_be & (1 << i) != 0 {
                        capable |= FlowDir::ALL_DIRS[i].allowed_adjacent(delta);
                    }
                }
                capable
            };
            let mut to_check = vec![candidate];
            let mut former_values: HashMap<usize, u32> = HashMap::new();
            former_values.insert(candidate, possible ^ (1 << choice));
            while let Some(item) = to_check.pop() {
                for &change_dir in FlowDir::NBR_DIRS
                    .iter()
                    .filter(|dir| to_fill[dir.nbr_of(item, grid_w, grid_h).unwrap()].is_none())
                {
                    let possible_nbr = change_dir.nbr_of(item, grid_w, grid_h).unwrap();
                    let old_possible = all_candidates[possible_nbr];
                    let new_possible = permissible(all_candidates[item], change_dir);
                    if new_possible & old_possible != old_possible {
                        let _ = former_values
                            .entry(possible_nbr)
                            .or_insert(all_candidates[possible_nbr]);
                        all_candidates[possible_nbr] &= new_possible;
                        to_check.push(possible_nbr);
                    }
                }
            }
            if former_values
                .keys()
                .any(|updated_val| all_candidates[*updated_val] == 0)
            {
                // because there are no options, we've made a contradiction
                // undo all changes and return the appropriate error
                for (k, v) in former_values {
                    all_candidates[k] = v;
                }
                to_fill[candidate] = None;
                if possible.count_ones() == 1 {
                    Err(WfcStepError::NoMoreChoices)
                } else {
                    Err(WfcStepError::ChoiceUnsuccessful)
                }
            } else {
                Ok(candidate)
            }
        } else {
            panic!("no possibilities found, which should already be handled");
        }
    }
    pub fn wfc_gen_dirty(width: usize, height: usize, rng: &mut impl Rng) -> Vec<FlowDir> {
        const MAX_ITERS: i32 = 1_000;
        let grid_w = width + 2;
        let grid_h = height + 2;
        let mut to_fill: Vec<Option<FlowDir>> = Vec::new();
        'attempts: for num_iters in 0.. {
            if num_iters >= MAX_ITERS {
                // equivalent to panic-ing, but won't crash the rest of the program hopefully
                return vec![FlowDir::UpDown; width * height];
            }
            to_fill = vec![None; grid_w * grid_h];
            let mut all_candidates: Vec<u32> = vec![(1 << 10) - 1; grid_w * grid_h];
            assert!(all_candidates.iter().all(|val| val.count_ones() == 10));
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
            while num_open > 0 {
                let res = Self::wfc_step(&mut to_fill, &mut all_candidates, grid_w, grid_h, rng);
                match res {
                    Ok(_candidate) => num_open -= 1,
                    Err(WfcStepError::ChoiceUnsuccessful) => {}
                    Err(WfcStepError::NoMoreChoices) => continue 'attempts,
                }
            }
            if !to_fill.contains(&None) {
                // log(&format!("finished in {num_iters} iterations"));
                break;
            }
        }
        let no_borders: Option<Vec<FlowDir>> = to_fill
            .into_iter()
            .filter(|dir| *dir != Some(FlowDir::Detached))
            .collect();
        no_borders.expect("only breaks from loop if no None values")
    }
    pub fn from_dirs_grid(dirs_grid: &[FlowDir], width: usize, height: usize) -> Self {
        let mut to_ret = Self::new(width, height);
        for (i, dir) in dirs_grid.iter().enumerate() {
            to_ret.fills[i].dirs = *dir;
        }
        let paths = Board::get_paths(dirs_grid, width, height);
        let rng = &mut Pcg::new(0, 0);
        let palette = get_color_palette(paths.len(), rng)
            .unwrap_or_else(|| vec![rng.gen::<u16>() & 0xfff; paths.len()]);
        for (i, path) in paths.iter().enumerate() {
            for pos in path {
                to_ret.fills[*pos].color = palette[i];
            }
        }
        to_ret
    }
    fn to_dirs_grid(&self) -> Vec<FlowDir> {
        self.fills.iter().map(|fill| fill.dirs).collect()
    }
    fn get_paths(dirs_grid: &[FlowDir], width: usize, height: usize) -> Vec<Vec<usize>> {
        // TODO: make this work for non-full boards
        // TODO: this is not right :anguish:
        let mut paths: Vec<Vec<usize>> = Vec::new();
        let mut checked = bitvec![0; (width*height)];
        while checked.count_zeros() != 0 {
            let first_unchecked = checked
                .iter()
                .enumerate()
                .filter(|(i, val)| !(**val) && dirs_grid[*i].num_connections() == 1)
                .map(|(i, _)| i)
                .next()
                .unwrap_or_else(|| checked.leading_ones());

            if dirs_grid[first_unchecked] == FlowDir::Detached {
                checked.set(first_unchecked, true);
                continue;
            }

            let mut to_visit = vec![first_unchecked];
            let mut intermediate: Vec<usize> = Vec::new();
            while let Some(pos) = to_visit.pop() {
                if checked[pos] {
                    continue;
                }
                checked.set(pos, true);
                intermediate.push(pos);
                let flow_dir = dirs_grid[pos];
                to_visit.extend(FlowDir::NBR_DIRS.iter().filter_map(|dir| {
                    dir.nbr_of(pos, width, height)
                        .filter(|nbr| flow_dir.is_connected(*dir) && !checked[*nbr])
                }));
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
            let path_start = (path.len() - 1, 0);
            let path_middle = (path.len() / 2, path.len() / 2 + 1);
            for pair in [path_start, path_middle] {
                let break_0_a = path[pair.0];
                let break_0_b = path[pair.1];
                let dir_connected_in = FlowDir::change_dir(break_0_a, break_0_b, width).unwrap();
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
                let self_intersect_pos = path.iter().find(|pos| {
                    FlowDir::NBR_DIRS.iter().any(|dir| {
                        dir.nbr_of(**pos, width, height)
                            .filter(|_| !dirs_grid[**pos].is_connected(*dir))
                            .is_some_and(|nbr| path.contains(&nbr))
                    })
                });
                if let Some(_self_intersect_pos) = self_intersect_pos {
                    // log(&format!(
                    //     "found a self-intersect with path {path:?} at position {}",
                    //     self_intersect_pos
                    // ));
                    // paths MUST be in sorted order :)
                    let break_pos = path.len() / 2;
                    let break_start = path[break_pos - 1];
                    let break_end = path[break_pos];
                    let dir_connected_in =
                        FlowDir::change_dir(break_start, break_end, width).unwrap();
                    dirs_grid[break_start] =
                        dirs_grid[break_start].remove_connection(dir_connected_in);
                    dirs_grid[break_end] =
                        dirs_grid[break_end].remove_connection(dir_connected_in.rev());
                    created_paths.push(path.split_off(break_pos));
                    continue 'outer;
                }
            }
            if !created_paths.is_empty() {
                paths.append(&mut created_paths);
                continue 'outer;
            }
            break;
        }

        // log("exited loop/intersect");
    }
    fn join_adjacent_paths(
        dirs_grid: &mut [FlowDir],
        paths: &mut Vec<Vec<usize>>,
        width: usize,
        height: usize,
    ) {
        let mut bitvec_path_b_set = bitvec![0; width*height];
        for i_a in 0..paths.len() {
            for i_b in 0..paths.len() {
                let path_a = &paths[i_a];
                let path_b = &paths[i_b];
                if path_a.len() < 2 || path_b.len() < 2 {
                    continue;
                }
                let mut pair = None;
                for a in [path_a[0], *path_a.last().unwrap()] {
                    for b in [path_b[0], *path_b.last().unwrap()] {
                        if is_nbr(a, b, width) {
                            pair = Some((a, b));
                        }
                    }
                }
                let Some((a, b)) = pair else {
                    continue;
                };
                assert_eq!(
                    dirs_grid[a].num_connections(),
                    1,
                    "A must have one connection"
                );
                assert_eq!(
                    dirs_grid[b].num_connections(),
                    1,
                    "B must have one connection"
                );
                // log(&format!("trying with {a}, {b}"));
                // empty out the set
                bitvec_path_b_set.set_elements(0);
                for i in path_b.iter().copied() {
                    bitvec_path_b_set.set(i, true);
                }
                if !path_a.iter().copied().any(|idx| {
                    FlowDir::NBR_DIRS
                        .iter()
                        .copied()
                        .flat_map(|dir| {
                            dir.nbr_of(idx, width, height)
                                .filter(|_| !dirs_grid[idx].is_connected(dir))
                        })
                        .any(|val| !(idx == a && val == b) && bitvec_path_b_set[val])
                }) {
                    let dir = FlowDir::change_dir(a, b, width)
                        .expect("a and b must be adjacent to be nbrs");
                    dirs_grid[a] = dirs_grid[a]
                        .try_connect(dir)
                        .expect("change direction/connection should be checked (a)");
                    dirs_grid[b] = dirs_grid[b]
                        .try_connect(dir.rev())
                        .expect("change direction/connection should be checked (b)");
                    let one_end = path_a[if path_a[0] == a { path_a.len() - 1 } else { 0 }];
                    assert_eq!(
                        dirs_grid[one_end].num_connections(),
                        1,
                        "board: {:?}, path: {:?}",
                        Self::from_dirs_grid(dirs_grid, width, height).write_board(),
                        path_a
                    );
                    let mut queue = vec![one_end];
                    let mut visited = bitvec![0; dirs_grid.len()];
                    loop {
                        let pos = *queue.last().unwrap();
                        visited.set(pos, true);
                        let next = FlowDir::NBR_DIRS
                            .iter()
                            .filter_map(|dir| {
                                dir.nbr_of(pos, width, height)
                                    .filter(|_| dirs_grid[pos].is_connected(*dir))
                                    .filter(|val| !visited[*val])
                            })
                            .next();
                        let Some(next) = next else {
                            break;
                        };
                        queue.push(next);
                    }
                    paths[i_a] = std::mem::take(&mut queue);
                    paths[i_b].clear();
                }
            }
        }
        paths.retain(|val| !val.is_empty());
    }
    pub fn gen_filled_board(width: usize, height: usize, rng: &mut impl Rng) -> Self {
        // 0: left, 1: right, 2: up, 3: down, 4: left-right, 5: left-up,
        // 6: left-down, 7: right-up, 8: right-down, 9: up-down, 10: unconnected
        let mut board = Self::new(width, height); // guarantees width/height are positive

        let (width, height) = (width, height);
        let mut dirs_dirty = Self::wfc_gen_dirty(width, height, rng);

        let mut paths = Self::get_paths(&dirs_dirty, width, height);
        // log(&format!("{:?}", paths));
        Self::split_loops_and_intersects(&mut dirs_dirty, &mut paths, width, height);
        Self::join_adjacent_paths(&mut dirs_dirty, &mut paths, width, height);
        for (i, &dir) in dirs_dirty.iter().enumerate() {
            board.fills[i] = Fill::new(0xfff, dir.flow_type(), dir)
        }

        let palette = get_color_palette(paths.len(), rng)
            .unwrap_or_else(|| vec![rng.gen::<u16>() & 0xfff; paths.len()]);
        for (i, path) in paths.iter().enumerate() {
            for pos in path {
                board.fills[*pos].color = palette[i];
            }
        }

        board
    }
    pub fn verify_valid(&self) -> Result<(), InvalidBoardError> {
        let as_grid = self.to_dirs_grid();
        let paths = Self::get_paths(&as_grid, self.width, self.height);
        for path in paths.iter() {
            let dot_pos = path.iter().any(|pos| self.fills[*pos].flow == Flow::Dot);
            let crosses_self = path.iter().copied().any(|idx| {
                FlowDir::NBR_DIRS
                    .iter()
                    .copied()
                    .filter_map(|dir| {
                        dir.nbr_of(idx, self.width, self.height)
                            .filter(|_| !as_grid[idx].is_connected(dir))
                    })
                    .any(|val| path.contains(&val))
            });
            if !dot_pos {
                let has_unlinked_connection = path
                    .iter()
                    .any(|pos| self.fills[*pos].dirs.num_connections() != 2);
                return match has_unlinked_connection {
                    false => Err(InvalidBoardError::Loop),
                    true => Err(InvalidBoardError::DisconnectedFlow),
                };
            }
            if crosses_self {
                return Err(InvalidBoardError::SelfAdjacentPath);
            }
        }
        let mut colors_map: HashMap<u16, usize> = HashMap::new();
        for (idx, fill) in self.fills.iter().enumerate() {
            if let Some(error) = FlowDir::NBR_DIRS
                .iter()
                .copied()
                .filter_map(|dir| {
                    dir.nbr_of(idx, self.width, self.height)
                        .filter(|_| as_grid[idx].is_connected(dir))
                        .and_then(|nbr| {
                            if self.fills[idx].color != self.fills[nbr].color {
                                Some(InvalidBoardError::DifferentColorConnection)
                            } else if !self.fills[nbr].dirs.is_connected(dir.rev()) {
                                Some(InvalidBoardError::UnmatchedConnection)
                            } else {
                                None
                            }
                        })
                })
                .next()
            {
                return Err(error);
            }

            if fill.flow == Flow::Dot {
                *colors_map.entry(fill.color).or_insert(0) += 1;
            }
        }
        if colors_map.iter().any(|(_k, v)| *v != 2) {
            return Err(InvalidBoardError::UnpairedDots);
        }
        return Ok(());
    }
    pub fn get_fill(&self, x: usize, y: usize) -> &Fill {
        &self.fills[y * self.width + x]
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

    /// Traverse from a starting pos, returning when `break_condition(pos)` returns true or the path runs out.
    /// Note that this now allows non-dot starting places, but the traversal does not guarantee a particular direction
    fn flood_search(&self, pos: usize, break_condition: impl Fn(usize) -> bool) -> (BitVec, usize) {
        let mut explored = bitvec![0; (self.width * self.height)];
        let start_color = self.fills[pos].color;

        let mut to_explore: Vec<usize> = vec![pos];
        let mut last_entry = pos;
        while let Some(entry) = to_explore.pop() {
            if explored[entry] {
                panic!("entry was previously explored; should have been checked before pushing as option");
            }
            explored.set(entry, true);
            last_entry = entry;
            if break_condition(entry) {
                break;
            }
            let this_fill = &self.fills[entry];
            for nbr in FlowDir::NBR_DIRS.iter().copied() {
                let Some(new_pos) = nbr.nbr_of(entry, self.width, self.height) else {
                    continue;
                };
                if !this_fill.dirs.is_connected(nbr)
                    || !self.fills[new_pos].dirs.is_connected(nbr.rev())
                    || explored[new_pos]
                    || start_color != self.fills[new_pos].color
                {
                    continue;
                }
                // we only want to traverse in one direction, so break
                to_explore.push(new_pos);
                break;
            }
        }
        (explored, last_entry)
    }
    pub fn add_connection(&mut self, pos_a: usize, pos_b: usize) -> bool {
        if !self.adjacent(pos_a, pos_b) {
            return false;
        }
        let fill_a = &self.fills[pos_a];
        let fill_b = &self.fills[pos_b];
        // precondition: A already is a line/dot
        if let Flow::Empty = fill_a.flow {
            return false;
        }

        // if B is empty, we'll give it the same color as A
        if Flow::Empty != fill_b.flow && fill_a.color != fill_b.color {
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
        let my_change_direction = FlowDir::change_dir(pos_a, pos_b, self.width)
            .expect("method should only be called with valid delta");
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
        let fill_a = &self.fills[pos_a];
        let fill_b = &self.fills[pos_b];
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
            _ => panic!("Invalid delta: {} -> {}", pos_a, pos_b),
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
        let (visited, _end) = self.flood_search(pos, |entry| {
            entry != pos && self.fills[entry].flow == Flow::Dot
        });
        // log(&format!("ended at {end} from {pos}"));
        // log(&format!("visited {visited:?}"));
        for (i, was_visited) in visited.iter().enumerate() {
            if *was_visited {
                // log(&format!("removing {i}"));
                if let Flow::Dot = self.fills[i].flow {
                    self.fills[i].dirs = FlowDir::Detached;
                } else {
                    self.fills[i] = Default::default();
                }
            }
        }
    }

    pub fn is_connected(&self, pos_a: usize, pos_b: usize) -> bool {
        let (_, final_pos) = self.flood_search(pos_a, |entry| {
            entry != pos_a && self.fills[entry].flow == Flow::Dot
        });
        pos_b == final_pos
    }

    pub fn is_attached_nbrs(&self, pos_a: usize, pos_b: usize) -> bool {
        is_nbr(pos_a, pos_b, self.width)
            && self.fills[pos_a]
                .dirs
                .is_connected(FlowDir::change_dir(pos_a, pos_b, self.width).unwrap())
            && self.fills[pos_b]
                .dirs
                .is_connected(FlowDir::change_dir(pos_b, pos_a, self.width).unwrap())
    }
    fn check_all_connected(&self) -> bool {
        let mut map = HashMap::new();
        for i in 0..(self.width * self.height) {
            if self.fills[i].flow == Flow::Dot {
                map.entry(self.fills[i].color)
                    .or_insert_with(Vec::new)
                    .push(i);
            }
        }
        for endpoints in map.values() {
            if endpoints.len() != 2 {
                return false;
            }
            if !self.is_connected(endpoints[0], endpoints[1]) {
                return false;
            }
        }
        true
    }

    fn fully_filled(&self) -> bool {
        for i in 0..(self.width * self.height) {
            if self.fills[i].flow == Flow::Empty {
                return false;
            }
        }
        true
    }

    /*
    u32 width
    u32 height

    [u16; width*height] fill:
        bits 0-3: dirs/type (left, right, up, down)
        if the dirs have 3 or 4 high bits, the value is a dot w/ bitwise-not directions
        bits 4-15: color (4 bits per color)
    */
    fn write_board(&self) -> Vec<u8> {
        let mut serialized: Vec<u8> = Vec::new();
        serialized.extend(FLOW_FREE_HEADER);
        serialized.extend(BOARD_FORMAT_VERSION.to_be_bytes());
        serialized.extend((self.width as u32).to_be_bytes().iter());
        serialized.extend((self.height as u32).to_be_bytes().iter());
        for fill in &self.fills {
            let mut fill_data: u16 = 0;
            let mut dirs_info = fill.dirs as u8;
            if fill.flow == Flow::Dot {
                dirs_info = !dirs_info & 0xf
            }
            fill_data |= dirs_info as u16;
            fill_data |= (fill.color & 0xfff) << 4;
            serialized.extend(fill_data.to_be_bytes().iter());
        }
        serialized
    }

    fn read_board(serialized: &[u8]) -> Option<Self> {
        let mut cursor = Cursor::new(serialized);
        let mut scratch_buf = [0u8; 8];
        // read the header
        cursor.read_exact(&mut scratch_buf).ok()?;
        if &scratch_buf != FLOW_FREE_HEADER {
            return None;
        }
        // read version number
        cursor.read_exact(&mut scratch_buf[..4]).ok()?;
        if scratch_buf[..4] != BOARD_FORMAT_VERSION.to_be_bytes() {
            return None;
        }
        let width: usize = {
            cursor.read_exact(&mut scratch_buf[..4]).ok()?;
            let w: &[u8; 4] = scratch_buf[..4].try_into().ok()?;
            u32::from_be_bytes(*w) as usize
        };
        let height: usize = {
            cursor.read_exact(&mut scratch_buf[..4]).ok()?;
            let h: &[u8; 4] = scratch_buf[..4].try_into().ok()?;
            u32::from_be_bytes(*h) as usize
        };
        let mut board = Board::new(width, height);
        let pos: usize = cursor.position().try_into().ok()?;
        if 2 * width * height != serialized.len() - pos {
            return None;
        }
        for (i, b) in serialized[pos..].chunks(2).enumerate() {
            let packed = u16::from_be_bytes(b.try_into().ok()?);
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
            board.fills[i] = Fill::new(color, flow, FlowDir::try_from(dirs as u32).ok()?);
        }
        Some(board)
        // if !Self::verify_valid(&board) {
        //     None
        // } else {
        //     Some(board)
        // }
    }
}

#[wasm_bindgen]
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
                        // log(&format!("chose to clear pipe at position {fill_pos}"));
                        self.board.clear_pipe(fill_pos);
                        if self.current_pos.is_some() {
                            self.current_pos =
                                Some((fill_pos % self.board.width, fill_pos % self.board.width));
                        }
                    }
                }
                /*
                match unsolved.flow {
                    Flow::Empty => (),
                    Flow::Dot => {
                        // TODO: fix this
                        // it currently doesn't check that the whole path is correct

                        if solved_board.fills[end].flow != Flow::Dot {
                            // TODO: make this clear less vehemently?
                        }
                    }
                    Flow::Line => {
                        if unsolved.dirs.num_connections() == 1 {
                            let (mut whole_path, dot_end) =
                                self.board.flood_search(fill_pos, |entry| {
                                    entry != fill_pos && self.board.fills[entry].flow == Flow::Dot
                                });

                            if self.board.fills[dot_end].flow == Flow::Dot {
                                // TODO: this seems incorrect... fix?

                                let (correct_positions, end_incorrect) =
                                    self.board.flood_search(dot_end, |entry| {
                                        self.board.fills[entry] != solved_board.fills[entry]
                                    });
                                // TODO: fix meeee
                                let incorrect_dirs = self.board.fills[end_incorrect].dirs;
                                let correct_dirs = solved_board.fills[end_incorrect].dirs;
                                if incorrect_dirs != correct_dirs
                                    && !(incorrect_dirs.num_connections() == 1
                                        && correct_dirs.is_connected(incorrect_dirs))
                                {
                                    // TODO: what the heck is happening with this entire block of code?
                                    // the path is actually incorrect, we didn't just hit the end

                                    // store only incorrect positions as true in whole_path
                                    whole_path ^= &correct_positions;
                                    whole_path.set(end_incorrect, true);
                                    let last_correct =
                                        FlowDir::NBR_DIRS
                                            .iter()
                                            .flat_map(|dir| {
                                                dir.nbr_of(
                                                    end_incorrect,
                                                    self.board.width,
                                                    self.board.height,
                                                )
                                                .filter(|nbr| {
                                                    self.board.fills[*nbr]
                                                        .dirs
                                                        .is_connected(dir.rev())
                                                })
                                            })
                                            .find(|nbr| correct_positions[*nbr])
                                            .expect("must have had a nbr to reach during search");
                                    let change_dir = FlowDir::change_dir(
                                        last_correct,
                                        end_incorrect,
                                        self.board.width,
                                    )
                                    .expect("must be nbrs because of search");

                                    self.board.fills[last_correct].dirs = self.board.fills
                                        [last_correct]
                                        .dirs
                                        .remove_connection(change_dir);
                                    for (i, val) in whole_path.iter().enumerate() {
                                        if *val {
                                            self.board.fills[i] = Default::default();
                                        }
                                    }
                                }
                            }
                        }
                    }
                };*/
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
