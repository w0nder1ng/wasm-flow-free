use std::fmt::{Display, Formatter, Write};


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
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
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
        pub const fn nbr_of(self, index: usize, width: usize, height: usize) -> Option<usize> {
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
        pub fn change_dir(pos_a: usize, pos_b: usize, width: usize) -> Option<Self> {
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