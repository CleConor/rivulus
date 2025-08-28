use crate::column::{Column, ColumnView};

pub struct Batch<'a> {
    pub schema: &'a [&'static str],
    pub cols: Vec<ColumnView<'a>>,
    pub len: usize,
}

pub struct OwnedBatch {
    pub schema: Vec<&'static str>,
    pub cols: Vec<Column>,
    pub len: usize, //selected row
}

impl OwnedBatch {
    pub fn as_view(&self) -> Batch<'_> {
        debug_assert_eq!(self.schema.len(), self.cols.len());
        debug_assert!(self.cols.iter().all(|c| c.len() == self.len));

        let mut cols = Vec::with_capacity(self.cols.len());

        for col in &self.cols {
            cols.push(col.slice_view(0..self.len));
        }

        Batch {
            schema: &self.schema[..],
            cols,
            len: self.len,
        }
    }
}
