use crate::batch::Batch;
use crate::table::Table;

pub struct Scan<'a> {
    table: &'a Table,
    pos: usize,
    chunk: usize, // default 8192 ?
}

impl<'a> Scan<'a> {
    pub fn new(table: &'a Table) -> Self {
        Self {
            table,
            pos: 0,
            chunk: 8192,
        }
    }
    pub fn with_chunk(table: &'a Table, chunk: usize) -> Self {
        assert!(chunk > 0);
        Self {
            table,
            pos: 0,
            chunk,
        }
    }
    pub fn next_batch(&mut self) -> Option<Batch<'a>> {
        if self.pos >= self.table.len {
            return None;
        }

        let start = self.pos;
        let end = (start + self.chunk).min(self.table.len);
        let len = end - start;

        let mut cols = Vec::with_capacity(self.table.cols.len());

        for col in &self.table.cols {
            cols.push(col.slice_view(start..end));
        }

        self.pos = end;

        debug_assert!(cols.iter().all(|c| c.len() == len));
        Some(Batch {
            schema: &self.table.schema,
            cols,
            len,
        })
    }
}
