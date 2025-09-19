use super::schema::DataType;
use std::sync::Arc;

pub mod bitmap;
pub mod primitive;

pub trait Array: Send + Sync + std::fmt::Debug {
    fn len(&self) -> usize;
    fn data_type(&self) -> &DataType;
    fn null_count(&self) -> usize;
    fn slice(&self, offset: usize, length: usize) -> ArrayRef;
    fn as_any(&self) -> &dyn std::any::Any;
}

pub type ArrayRef = Arc<dyn Array>;

//pub(crate) use bitmap::BitMap;

pub use primitive::PrimitiveArray;
// pub use string::StringArray;
// pub use boolean::BooleanArray;
// pub use null::NullArray;
