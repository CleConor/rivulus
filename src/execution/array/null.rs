use super::super::schema::DataType;
use super::{Array, ArrayRef};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct NullArray {
    length: usize,
    offset: usize,
}

impl NullArray {
    pub fn new(length: usize) -> Self {
        Self { length, offset: 0 }
    }

    pub fn value(&self, index: usize) -> Option<()> {
        assert!(index < self.length, "Index {} out of bounds", index);
        None
    }

    pub fn iter(&self) -> NullArrayIter {
        NullArrayIter {
            array: self,
            index: 0,
        }
    }

    pub fn is_null(&self, index: usize) -> bool {
        assert!(index < self.length, "Index {} out of bounds", index);
        true
    }

    pub fn is_valid(&self, _index: usize) -> bool {
        false
    }

    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl Array for NullArray {
    fn len(&self) -> usize {
        self.length
    }

    fn data_type(&self) -> &DataType {
        &DataType::Null
    }

    fn null_count(&self) -> usize {
        self.length
    }

    fn slice(&self, offset: usize, length: usize) -> ArrayRef {
        assert!(offset + length <= self.length, "Slice out of bounds");

        Arc::new(NullArray {
            length,
            offset: self.offset + offset,
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct NullArrayIter<'a> {
    array: &'a NullArray,
    index: usize,
}

impl<'a> Iterator for NullArrayIter<'a> {
    type Item = Option<()>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.array.length {
            None
        } else {
            self.index += 1;
            Some(None)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.length - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for NullArrayIter<'a> {}

pub struct NullArrayBuilder {
    length: usize,
}

impl NullArrayBuilder {
    pub fn new() -> Self {
        Self { length: 0 }
    }

    pub fn with_capacity(_capacity: usize) -> Self {
        Self { length: 0 }
    }

    pub fn append_null(&mut self) {
        self.length += 1;
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn finish(self) -> NullArray {
        NullArray::new(self.length)
    }
}

impl Default for NullArrayBuilder {
    fn default() -> Self {
        Self::new()
    }
}

//generated tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let array = NullArray::new(5);
        assert_eq!(array.len(), 5);
        assert_eq!(array.data_type(), &DataType::Null);
        assert_eq!(array.null_count(), 5);
        assert_eq!(array.offset, 0);
    }

    #[test]
    fn test_empty_array() {
        let array = NullArray::new(0);
        assert_eq!(array.len(), 0);
        assert_eq!(array.null_count(), 0);
    }

    #[test]
    fn test_value_access() {
        let array = NullArray::new(3);

        for i in 0..3 {
            assert_eq!(array.value(i), None);
            assert!(array.is_null(i));
            assert!(!array.is_valid(i));
        }
    }

    #[test]
    #[should_panic(expected = "Index 5 out of bounds")]
    fn test_value_access_out_of_bounds() {
        let array = NullArray::new(3);
        array.value(5);
    }

    #[test]
    #[should_panic(expected = "Index 10 out of bounds")]
    fn test_is_null_out_of_bounds() {
        let array = NullArray::new(5);
        array.is_null(10);
    }

    #[test]
    fn test_slice() {
        let array = NullArray::new(10);
        let sliced = array.slice(3, 4);
        let sliced_array = sliced.as_any().downcast_ref::<NullArray>().unwrap();

        assert_eq!(sliced_array.len(), 4);
        assert_eq!(sliced_array.null_count(), 4);
        assert_eq!(sliced_array.offset, 3);

        for i in 0..4 {
            assert_eq!(sliced_array.value(i), None);
            assert!(sliced_array.is_null(i));
        }
    }

    #[test]
    fn test_slice_boundary_conditions() {
        let array = NullArray::new(5);

        // Empty slice
        let empty_slice = array.slice(2, 0);
        let empty_array = empty_slice.as_any().downcast_ref::<NullArray>().unwrap();
        assert_eq!(empty_array.len(), 0);
        assert_eq!(empty_array.null_count(), 0);

        // Single element slice
        let single_slice = array.slice(1, 1);
        let single_array = single_slice.as_any().downcast_ref::<NullArray>().unwrap();
        assert_eq!(single_array.len(), 1);
        assert_eq!(single_array.null_count(), 1);
        assert_eq!(single_array.value(0), None);

        // Full array slice
        let full_slice = array.slice(0, 5);
        let full_array = full_slice.as_any().downcast_ref::<NullArray>().unwrap();
        assert_eq!(full_array.len(), 5);
        assert_eq!(full_array.null_count(), 5);
    }

    #[test]
    #[should_panic(expected = "Slice out of bounds")]
    fn test_slice_out_of_bounds() {
        let array = NullArray::new(3);
        array.slice(2, 5);
    }

    #[test]
    fn test_iterator() {
        let array = NullArray::new(3);
        let collected: Vec<_> = array.iter().collect();
        assert_eq!(collected, vec![None, None, None]);
    }

    #[test]
    fn test_iterator_empty() {
        let array = NullArray::new(0);
        let collected: Vec<_> = array.iter().collect();
        assert_eq!(collected, Vec::<Option<()>>::new());
    }

    #[test]
    fn test_iterator_size_hint() {
        let array = NullArray::new(5);
        let mut iter = array.iter();

        assert_eq!(iter.size_hint(), (5, Some(5)));

        iter.next();
        assert_eq!(iter.size_hint(), (4, Some(4)));

        iter.next();
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
    }

    #[test]
    fn test_builder() {
        let mut builder = NullArrayBuilder::new();
        assert_eq!(builder.len(), 0);
        assert!(builder.is_empty());

        builder.append_null();
        builder.append_null();
        builder.append_null();

        assert_eq!(builder.len(), 3);
        assert!(!builder.is_empty());

        let array = builder.finish();
        assert_eq!(array.len(), 3);
        assert_eq!(array.null_count(), 3);

        for i in 0..3 {
            assert_eq!(array.value(i), None);
        }
    }

    #[test]
    fn test_builder_with_capacity() {
        let mut builder = NullArrayBuilder::with_capacity(1000);
        assert_eq!(builder.len(), 0);

        for _ in 0..50 {
            builder.append_null();
        }

        let array = builder.finish();
        assert_eq!(array.len(), 50);
        assert_eq!(array.null_count(), 50);
    }

    #[test]
    fn test_builder_default() {
        let builder = NullArrayBuilder::default();
        let array = builder.finish();

        assert_eq!(array.len(), 0);
        assert_eq!(array.null_count(), 0);
    }

    #[test]
    fn test_memory_efficiency() {
        let small_array = NullArray::new(10);
        let large_array = NullArray::new(1_000_000);

        // Memory usage should be constant regardless of length
        assert_eq!(small_array.memory_size(), large_array.memory_size());
        assert_eq!(small_array.memory_size(), std::mem::size_of::<NullArray>());
    }

    #[test]
    fn test_data_type() {
        let array = NullArray::new(10);
        assert_eq!(array.data_type(), &DataType::Null);
    }

    #[test]
    fn test_as_any_downcast() {
        let array: ArrayRef = Arc::new(NullArray::new(5));
        let null_array = array.as_any().downcast_ref::<NullArray>();
        assert!(null_array.is_some());

        let downcasted = null_array.unwrap();
        assert_eq!(downcasted.len(), 5);
        assert_eq!(downcasted.null_count(), 5);
    }

    #[test]
    fn test_slice_preserves_properties() {
        let array = NullArray::new(20);
        let sliced = array.slice(5, 10);
        let sliced_array = sliced.as_any().downcast_ref::<NullArray>().unwrap();

        // All values should still be null
        for i in 0..sliced_array.len() {
            assert_eq!(sliced_array.value(i), None);
            assert!(sliced_array.is_null(i));
            assert!(!sliced_array.is_valid(i));
        }

        // Properties should be maintained
        assert_eq!(sliced_array.len(), sliced_array.null_count());
        assert_eq!(sliced_array.data_type(), &DataType::Null);
    }

    #[test]
    fn test_chained_slicing() {
        let array = NullArray::new(50);
        let slice1 = array.slice(10, 30);
        let slice1_array = slice1.as_any().downcast_ref::<NullArray>().unwrap();

        let slice2 = slice1_array.slice(5, 15);
        let slice2_array = slice2.as_any().downcast_ref::<NullArray>().unwrap();

        assert_eq!(slice2_array.len(), 15);
        assert_eq!(slice2_array.null_count(), 15);
        assert_eq!(slice2_array.offset, 15); // 10 + 5

        for i in 0..15 {
            assert_eq!(slice2_array.value(i), None);
        }
    }

    #[test]
    fn test_large_null_array_performance() {
        let size = 1_000_000;

        let start = std::time::Instant::now();
        let array = NullArray::new(size);
        let construction_time = start.elapsed();

        println!(
            "Construction time for {} nulls: {:?}",
            size, construction_time
        );

        // Construction should be near-instant regardless of size
        assert!(construction_time.as_millis() < 10);

        let start = std::time::Instant::now();
        let null_count = array.null_count();
        let null_count_time = start.elapsed();

        println!("Null count time: {:?}", null_count_time);

        assert_eq!(null_count, size);
        // Null count should be O(1)
        assert!(null_count_time.as_millis() < 1);
    }

    #[test]
    fn test_iterator_exact_size() {
        let array = NullArray::new(100);
        let iter = array.iter();

        // ExactSizeIterator should provide exact length
        assert_eq!(iter.len(), 100);

        let collected: Vec<_> = array.iter().collect();
        assert_eq!(collected.len(), 100);

        // All items should be None
        for item in collected {
            assert_eq!(item, None);
        }
    }

    #[test]
    fn test_zero_length_edge_cases() {
        let array = NullArray::new(0);

        // Iterator should handle empty arrays
        let mut iter = array.iter();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));

        // Slice should handle empty arrays
        let sliced = array.slice(0, 0);
        let sliced_array = sliced.as_any().downcast_ref::<NullArray>().unwrap();
        assert_eq!(sliced_array.len(), 0);
    }

    #[test]
    fn test_consistency_with_array_trait() {
        let array = NullArray::new(42);

        // Array trait methods should be consistent
        assert_eq!(array.len(), 42);
        assert_eq!(array.null_count(), array.len());
        assert_eq!(array.data_type(), &DataType::Null);

        // Slice should maintain consistency
        let sliced = array.slice(10, 20);
        assert_eq!(sliced.len(), 20);
        assert_eq!(sliced.null_count(), 20);
        assert_eq!(sliced.data_type(), &DataType::Null);
    }
}
