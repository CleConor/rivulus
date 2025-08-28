mod errors;
mod column;
mod batch;
mod table;
mod scan;
mod expr;
mod ops;
mod pipeline;

// Re-exports for bins and tests
pub use column::{Column, ColumnView};
pub use batch::{Batch, OwnedBatch};
pub use table::Table;
pub use scan::Scan;
pub use errors::EvalError;
pub use expr::Expr;
pub use ops::filter::{
    build_indices, eval_predicate_mask, filter_batch, filter_with_indices, filter_with_mask,
    FilterScratch, FilterStrategy,
};
pub use ops::project::{project_batch, ProjItem, ProjectScratch};
pub use pipeline::Pipeline;
