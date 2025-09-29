use super::schema::DataType;
use std::sync::Arc;

pub mod bitmap;
pub mod boolean;
pub mod null;
pub mod primitive;
pub mod string;

pub trait Array: Send + Sync + std::fmt::Debug {
    fn len(&self) -> usize;
    fn data_type(&self) -> &DataType;
    fn null_count(&self) -> usize;
    fn slice(&self, offset: usize, length: usize) -> ArrayRef;
    fn as_any(&self) -> &dyn std::any::Any;
}

pub type ArrayRef = Arc<dyn Array>;

pub use boolean::{BooleanArray, BooleanArrayBuilder, BooleanArrayIter};
pub use null::{NullArray, NullArrayBuilder, NullArrayIter};
pub use primitive::{PrimitiveArray, PrimitiveArrayBuilder, PrimitiveArrayIter};
pub use string::{StringArray, StringArrayIter, StringBuilder};
