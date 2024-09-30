use crate::flow::Flow;
use crate::flow::FlowDir;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Fill {
    pub color: u16,
    pub flow: Flow,
    pub dirs: FlowDir,
}

impl Fill {
    pub fn new(color: u16, flow: Flow, dirs: FlowDir) -> Self {
        Self { color, flow, dirs }
    }
}
