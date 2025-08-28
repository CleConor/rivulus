use crate::batch::Batch;
use crate::column::ColumnView;
use crate::errors::EvalError;

#[derive(Clone, Debug)]
pub enum Expr {
    Col(&'static str),
    LitI64(i64),
    Gt(Box<Expr>, Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
}

impl Expr {
    pub fn eval_i64<'a>(&self, batch: &Batch<'a>, out: &mut Vec<i64>) -> Result<(), EvalError> {
        match self {
            Expr::Col(name) => {
                let mut pos = usize::MAX;
                for i in 0..batch.schema.len() {
                    if batch.schema[i] == *name {
                        pos = i;
                        break;
                    }
                }

                if pos == usize::MAX {
                    return Err(EvalError::UnknownColumn(name.to_string()));
                }

                let col = &batch.cols[pos];
                match col {
                    ColumnView::I64(slice) => {
                        let len = slice.len();
                        out.resize(len, 0);

                        out.copy_from_slice(&slice);

                        Ok(())
                    }
                    ColumnView::Bool(_) => Err(EvalError::TypeMismatch("expected Int64 column")),
                }
            }
            Expr::LitI64(v) => {
                out.resize(batch.len, 0);
                out.fill(*v);

                Ok(())
            }
            Expr::Gt(_, _) => Err(EvalError::TypeMismatch(
                "Greater than does not produce Int64",
            )),
            Expr::Add(l, r) => {
                let mut buf_a = Vec::new();
                let mut buf_b = Vec::new();

                l.eval_i64(batch, &mut buf_a)?;
                r.eval_i64(batch, &mut buf_b)?;

                debug_assert!(buf_a.len() == buf_b.len() && buf_a.len() == batch.len);

                out.resize(buf_a.len(), 0);

                for i in 0..buf_a.len() {
                    out[i] = buf_a[i] + buf_b[i];
                }

                Ok(())
            }
        }
    }

    pub fn eval_bool<'a>(&self, batch: &Batch<'a>, out: &mut Vec<bool>) -> Result<(), EvalError> {
        match self {
            Expr::Col(name) => {
                let mut pos = usize::MAX;
                for i in 0..batch.schema.len() {
                    if batch.schema[i] == *name {
                        pos = i;
                        break;
                    }
                }

                if pos == usize::MAX {
                    return Err(EvalError::UnknownColumn(name.to_string()));
                }

                let col = &batch.cols[pos];
                match col {
                    ColumnView::I64(_) => Err(EvalError::TypeMismatch("expected Bool column")),
                    ColumnView::Bool(slice) => {
                        let len = slice.len();
                        out.resize(len, false);

                        out.copy_from_slice(&slice);

                        Ok(())
                    }
                }
            }
            Expr::LitI64(_) => Err(EvalError::TypeMismatch("expected Bool, got I64 literal")),
            Expr::Gt(l, r) => {
                let mut buf_a = Vec::new();
                let mut buf_b = Vec::new();

                l.eval_i64(batch, &mut buf_a)?;
                r.eval_i64(batch, &mut buf_b)?;

                debug_assert!(buf_a.len() == buf_b.len() && buf_a.len() == batch.len);

                out.resize(buf_a.len(), false);

                for i in 0..buf_a.len() {
                    out[i] = buf_a[i] > buf_b[i];
                }

                Ok(())
            }
            Expr::Add(_, _) => Err(EvalError::TypeMismatch("Add does not produce Bool")),
        }
    }
}
