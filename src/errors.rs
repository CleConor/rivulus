#[derive(Debug)]
pub enum EvalError {
    UnknownColumn(String),
    TypeMismatch(&'static str),
}
