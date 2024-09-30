use std::{
    collections::HashMap,
    fmt::{Display, Formatter, Write},
    io::{Cursor, Read},
};

use bitvec::{bitvec, vec::BitVec};
use rand::{seq::SliceRandom, Rng};

use crate::{
    fill::Fill,
    flow::{Flow, FlowDir},
    get_color_palette, is_nbr, reservoir_sample, InvalidBoardError, Pcg, WfcStepError,
};

const FLOW_FREE_HEADER: &[u8; 8] = b"FFBOARD\x97";
const BOARD_FORMAT_VERSION: u32 = 0x01;

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
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
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
                        .filter_map(|dir| {
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
    pub fn flood_search(&self, pos: usize, break_condition: impl Fn(usize) -> bool) -> (BitVec, usize) {
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
    pub fn check_all_connected(&self) -> bool {
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

    pub fn fully_filled(&self) -> bool {
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
    pub fn write_board(&self) -> Vec<u8> {
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

    pub fn read_board(serialized: &[u8]) -> Option<Self> {
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
