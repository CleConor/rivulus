use crate::batch::{Batch, OwnedBatch};
use crate::column::Column;
use crate::errors::EvalError;
use crate::expr::Expr;

#[derive(Clone, Debug)]
pub enum ProjItem {
    Keep {
        name: &'static str,
        alias: Option<&'static str>,
    },
    ComputeI64 {
        name: &'static str,
        expr: Expr,
    },
    ComputeBool {
        name: &'static str,
        expr: Expr,
    },
}

pub struct ProjectScratch {
    pub tmp_i64: Vec<i64>,
    pub tmp_bool: Vec<bool>,
}

impl ProjectScratch {
    pub fn new() -> Self {
        Self {
            tmp_i64: Vec::new(),
            tmp_bool: Vec::new(),
        }
    }
}

pub fn project_batch<'a>(
    batch: &Batch<'a>,
    items: &[ProjItem],
    scratch: &mut ProjectScratch,
) -> Result<OwnedBatch, EvalError> {
    debug_assert_eq!(batch.schema.len(), batch.cols.len());

    let out_cols = items.len();
    let mut schema: Vec<&'static str> = Vec::with_capacity(out_cols);
    let mut cols: Vec<Column> = Vec::with_capacity(out_cols);

    for proj in items {
        match proj {
            ProjItem::Keep { name, alias } => {
                let mut idx = usize::MAX;
                for i in 0..batch.schema.len() {
                    if batch.schema[i] == *name {
                        idx = i;
                        break;
                    }
                }

                if idx == usize::MAX {
                    return Err(EvalError::UnknownColumn(name.to_string()));
                }

                match &batch.cols[idx] {
                    crate::column::ColumnView::I64(v) => cols.push(Column::I64(v.to_vec())),
                    crate::column::ColumnView::Bool(v) => cols.push(Column::Bool(v.to_vec())),
                }

                schema.push(alias.unwrap_or(name));
            }
            ProjItem::ComputeI64 { name, expr } => {
                expr.eval_i64(batch, &mut scratch.tmp_i64)?;
                cols.push(Column::I64(scratch.tmp_i64.clone()));
                schema.push(name);
            }
            ProjItem::ComputeBool { name, expr } => {
                expr.eval_bool(batch, &mut scratch.tmp_bool)?;
                cols.push(Column::Bool(scratch.tmp_bool.clone()));
                schema.push(name);
            }
        }
    }

    debug_assert_eq!(schema.len(), cols.len());
    debug_assert!(cols.iter().all(|c| c.len() == batch.len));

    Ok(OwnedBatch {
        schema,
        cols,
        len: batch.len,
    })
}
