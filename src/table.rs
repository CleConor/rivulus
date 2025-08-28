use crate::column::Column;

pub struct Table {
    pub schema: [&'static str; 3],
    pub cols: [Column; 3],
    pub len: usize,
}
