use crate::batch::{Batch, OwnedBatch};
use crate::column::{Column, ColumnView};
use crate::errors::EvalError;
use crate::expr::Expr;

#[derive(Copy, Clone)]
pub enum FilterStrategy {
    Mask,
    Indices,
}

//Buffer to avoid reallocations
pub struct FilterScratch {
    pub mask: Vec<bool>,
    pub idx: Vec<usize>,
    pub tmp_i64_a: Vec<i64>,
    pub tmp_i64_b: Vec<i64>,
}

impl FilterScratch {
    pub fn new() -> Self {
        Self {
            mask: Vec::new(),
            idx: Vec::new(),
            tmp_i64_a: Vec::new(),
            tmp_i64_b: Vec::new(),
        }
    }
}

pub fn eval_predicate_mask<'a>(
    predicate: &Expr,
    batch: &Batch<'a>,
    out_mask: &mut Vec<bool>,
) -> Result<(), EvalError> {
    debug_assert_eq!(batch.schema.len(), batch.cols.len());

    predicate.eval_bool(batch, out_mask)?;

    debug_assert_eq!(out_mask.len(), batch.len);

    Ok(())
}

pub fn build_indices(mask: &[bool], out_idx: &mut Vec<usize>) {
    out_idx.clear();
    //let true_count = mask.iter().filter(|&&b| b).count();
    //out_idx.reserve(true_count);

    for i in 0..mask.len() {
        if mask[i] {
            out_idx.push(i);
        }
    }
}

pub fn filter_with_mask<'a>(batch: &Batch<'a>, mask: &[bool]) -> OwnedBatch {
    debug_assert_eq!(batch.schema.len(), batch.cols.len());
    debug_assert_eq!(mask.len(), batch.len);

    let n_cols = batch.cols.len();
    let true_count = mask.iter().filter(|&&b| b).count();

    let mut cols = Vec::with_capacity(n_cols);
    for i in 0..n_cols {
        match &batch.cols[i] {
            ColumnView::I64(v) => {
                let mut tmp_v = Vec::with_capacity(true_count);
                for row in 0..mask.len() {
                    if mask[row] {
                        tmp_v.push(v[row]);
                    }
                }
                cols.push(Column::I64(tmp_v));
            }
            ColumnView::Bool(v) => {
                let mut tmp_v = Vec::with_capacity(true_count);
                for row in 0..mask.len() {
                    if mask[row] {
                        tmp_v.push(v[row]);
                    }
                }
                cols.push(Column::Bool(tmp_v));
            }
        }
    }

    let owned = OwnedBatch {
        schema: batch.schema.to_vec(),
        cols,
        len: true_count,
    };

    debug_assert!(owned.schema.len() == owned.cols.len());

    owned
}

pub fn filter_with_indices<'a>(batch: &Batch<'a>, idx: &[usize]) -> OwnedBatch {
    debug_assert_eq!(batch.schema.len(), batch.cols.len());
    debug_assert!(idx.iter().all(|&i| i < batch.len));

    let n_cols = batch.cols.len();
    let mut cols = Vec::with_capacity(n_cols);

    for i in 0..n_cols {
        match &batch.cols[i] {
            ColumnView::I64(v) => {
                let mut tmp_v = Vec::with_capacity(idx.len());
                for &row in idx {
                    tmp_v.push(v[row]);
                }
                cols.push(Column::I64(tmp_v));
            }
            ColumnView::Bool(v) => {
                let mut tmp_v = Vec::with_capacity(idx.len());
                for &row in idx {
                    tmp_v.push(v[row]);
                }
                cols.push(Column::Bool(tmp_v));
            }
        }
    }

    let owned = OwnedBatch {
        schema: batch.schema.to_vec(),
        cols,
        len: idx.len(),
    };

    debug_assert!(owned.schema.len() == owned.cols.len());

    owned
}

pub fn filter_batch<'a>(
    batch: &Batch<'a>,
    predicate: &Expr,
    strategy: FilterStrategy,
    scratch: &mut FilterScratch,
) -> Result<OwnedBatch, EvalError> {
    eval_predicate_mask(predicate, batch, &mut scratch.mask)?;

    match strategy {
        FilterStrategy::Mask => Ok(filter_with_mask(batch, &scratch.mask)),
        FilterStrategy::Indices => {
            build_indices(&scratch.mask, &mut scratch.idx);
            Ok(filter_with_indices(batch, &scratch.idx))
        }
    }
}
