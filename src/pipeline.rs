use crate::batch::OwnedBatch;
use crate::errors::EvalError;
use crate::expr::Expr;
use crate::ops::filter::{filter_batch, FilterScratch, FilterStrategy};
use crate::ops::project::{project_batch, ProjItem, ProjectScratch};
use crate::scan::Scan;
use crate::table::Table;

pub struct Pipeline<'a> {
    scan: Scan<'a>,
    predicate: Expr,
    strategy: FilterStrategy,
    items: Vec<ProjItem>,
    filter_scratch: FilterScratch,
    project_scratch: ProjectScratch,
}

impl<'a> Pipeline<'a> {
    pub fn new(
        table: &'a Table,
        predicate: Expr,
        strategy: FilterStrategy,
        items: Vec<ProjItem>,
    ) -> Self {
        Self {
            scan: Scan::new(table),
            predicate,
            strategy,
            items,
            filter_scratch: FilterScratch::new(),
            project_scratch: ProjectScratch::new(),
        }
    }

    pub fn with_chunk(
        table: &'a Table,
        chunk: usize,
        predicate: Expr,
        strategy: FilterStrategy,
        items: Vec<ProjItem>,
    ) -> Self {
        Self {
            scan: Scan::with_chunk(table, chunk),
            predicate,
            strategy,
            items,
            filter_scratch: FilterScratch::new(),
            project_scratch: ProjectScratch::new(),
        }
    }

    pub fn next(&mut self) -> Result<Option<OwnedBatch>, EvalError> {
        let Some(in_batch) = self.scan.next_batch() else {
            return Ok(None);
        };

        debug_assert_eq!(in_batch.schema.len(), in_batch.cols.len());
        debug_assert!(in_batch.cols.iter().all(|c| c.len() == in_batch.len));

        let filtered = filter_batch(
            &in_batch,
            &self.predicate,
            self.strategy,
            &mut self.filter_scratch,
        )?;

        let view = filtered.as_view();

        let projected = project_batch(&view, &self.items, &mut self.project_scratch)?;

        Ok(Some(projected))
    }
}
