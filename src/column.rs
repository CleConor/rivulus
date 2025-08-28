pub enum Column {
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

pub enum ColumnView<'a> {
    I64(&'a [i64]),
    Bool(&'a [bool]),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::I64(v) => v.len(),
            Column::Bool(v) => v.len(),
        }
    }
    pub fn slice_view(&self, range: std::ops::Range<usize>) -> ColumnView<'_> {
        match &self {
            Column::I64(v) => ColumnView::I64(&v[range]),
            Column::Bool(v) => ColumnView::Bool(&v[range]),
        }
    }
}
impl<'a> ColumnView<'a> {
    pub fn len(&self) -> usize {
        match self {
            ColumnView::I64(v) => v.len(),
            ColumnView::Bool(v) => v.len(),
        }
    }
}
