use super::bitmap::BitMap;
use super::{Array, ArrayRef, DataType};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub trait PrimitiveType: Copy + Send + Sync + std::fmt::Debug + 'static {
    const DATA_TYPE: DataType;
}

impl PrimitiveType for i64 {
    const DATA_TYPE: DataType = DataType::Int64;
}

impl PrimitiveType for f64 {
    const DATA_TYPE: DataType = DataType::Float64;
}

const UNINITIALIZED_NULL_COUNT: usize = usize::MAX;

#[derive(Debug)]
pub struct PrimitiveArray<T> {
    values: Arc<[T]>,
    null_bitmap: Option<Arc<BitMap>>,
    data_type: DataType,
    offset: usize,
    length: usize,
    cached_null_count: AtomicUsize,
}

impl<T: PrimitiveType> PrimitiveArray<T> {
    pub fn new(values: Vec<T>, validity: Option<Vec<bool>>) -> Self {
        let length = values.len();
        let null_bitmap = validity.map(|v| Arc::new(BitMap::from_bool_slice(&v)));
        Self {
            values: Arc::from(values),
            null_bitmap,
            data_type: T::DATA_TYPE,
            offset: 0,
            length,
            cached_null_count: AtomicUsize::new(UNINITIALIZED_NULL_COUNT),
        }
    }

    pub fn from_values(values: Vec<T>) -> Self {
        Self::new(values, None)
    }

    pub fn value(&self, index: usize) -> Option<T> {
        assert!(index < self.length, "Index {} out of bounds", index);

        let logical_index = self.offset + index;

        if let Some(bitmap) = &self.null_bitmap {
            if !bitmap.get_bit(logical_index) {
                return None;
            }
        }

        Some(self.values[logical_index])
    }

    pub fn values(&self) -> &[T] {
        &self.values[self.offset..self.offset + self.length]
    }

    pub fn null_bitmap(&self) -> Option<&BitMap> {
        self.null_bitmap.as_deref()
    }

    pub fn iter(&self) -> PrimitiveArrayIter<T> {
        PrimitiveArrayIter {
            array: self,
            index: 0,
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.values.len() * std::mem::size_of::<T>()
    }
}

impl<T: PrimitiveType> Array for PrimitiveArray<T> {
    fn len(&self) -> usize {
        self.length
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn null_count(&self) -> usize {
        let cached = self.cached_null_count.load(Ordering::Relaxed);
        if cached != UNINITIALIZED_NULL_COUNT {
            return cached;
        }

        let count = if let Some(bitmap) = &self.null_bitmap {
            bitmap.count_zeros_range(self.offset, self.length)
        } else {
            0
        };

        self.cached_null_count.store(count, Ordering::Relaxed);
        count
    }

    fn slice(&self, offset: usize, length: usize) -> ArrayRef {
        assert!(offset + length <= self.length);
        Arc::new(PrimitiveArray {
            values: self.values.clone(),
            null_bitmap: self.null_bitmap.clone(),
            data_type: self.data_type.clone(),
            offset: self.offset + offset,
            length,
            cached_null_count: AtomicUsize::new(UNINITIALIZED_NULL_COUNT),
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct PrimitiveArrayIter<'a, T> {
    array: &'a PrimitiveArray<T>,
    index: usize,
}

impl<'a, T: PrimitiveType> Iterator for PrimitiveArrayIter<'a, T> {
    type Item = Option<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.array.length {
            None
        } else {
            let value = self.array.value(self.index);
            self.index += 1;
            Some(value)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.length - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: PrimitiveType> ExactSizeIterator for PrimitiveArrayIter<'a, T> {}

pub struct PrimitiveArrayBuilder<T> {
    values: Vec<T>,
    null_builder: super::bitmap::BitmapBuilder,
}

impl<T: PrimitiveType> PrimitiveArrayBuilder<T> {
    pub fn new() -> Self {
        PrimitiveArrayBuilder {
            values: Vec::new(),
            null_builder: super::bitmap::BitmapBuilder::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        PrimitiveArrayBuilder {
            values: Vec::with_capacity(capacity),
            null_builder: super::bitmap::BitmapBuilder::with_capacity(capacity),
        }
    }

    pub fn append_value(&mut self, value: T) {
        self.null_builder.append(true);
        self.values.push(value);
    }

    pub fn append_null(&mut self, placeholder: T) {
        self.null_builder.append(false);
        self.values.push(placeholder);
    }

    pub fn finish(self) -> PrimitiveArray<T> {
        let null_bitmap = if self.null_builder.has_nulls() {
            Some(Arc::new(self.null_builder.finish()))
        } else {
            None
        };

        let length = self.values.len();

        PrimitiveArray {
            values: Arc::from(self.values),
            null_bitmap,
            data_type: T::DATA_TYPE,
            offset: 0,
            length,
            cached_null_count: AtomicUsize::new(UNINITIALIZED_NULL_COUNT),
        }
    }
}

impl<T: PrimitiveType> Default for PrimitiveArrayBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

//generated tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_new_with_no_nulls() {
        let values = vec![1i64, 2, 3, 4, 5];
        let array = PrimitiveArray::new(values.clone(), None);

        assert_eq!(array.len(), 5);
        assert_eq!(array.data_type(), &DataType::Int64);
        assert_eq!(array.null_count(), 0);
        assert!(array.null_bitmap.is_none());

        // Verify all values accessible
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(array.value(i), Some(expected));
        }
    }

    #[test]
    fn test_new_with_some_nulls() {
        let values = vec![1i64, 2, 3, 4, 5];
        let validity = vec![true, false, true, false, true];
        let array = PrimitiveArray::new(values, Some(validity));

        assert_eq!(array.len(), 5);
        assert_eq!(array.null_count(), 2); // Two false bits
        assert!(array.null_bitmap.is_some());

        // Check specific values
        assert_eq!(array.value(0), Some(1)); // valid
        assert_eq!(array.value(1), None); // null
        assert_eq!(array.value(2), Some(3)); // valid
        assert_eq!(array.value(3), None); // null
        assert_eq!(array.value(4), Some(5)); // valid
    }

    #[test]
    fn test_all_nulls() {
        let values = vec![0i64, 0, 0]; // Placeholder values
        let validity = vec![false, false, false];
        let array = PrimitiveArray::new(values, Some(validity));

        assert_eq!(array.len(), 3);
        assert_eq!(array.null_count(), 3);

        for i in 0..3 {
            assert_eq!(array.value(i), None);
        }
    }

    #[test]
    fn test_empty_array() {
        let array = PrimitiveArray::<i64>::new(vec![], None);

        assert_eq!(array.len(), 0);
        assert_eq!(array.null_count(), 0);
        assert!(array.null_bitmap.is_none());
    }

    #[test]
    fn test_from_values_constructor() {
        let values = vec![10i64, 20, 30];
        let array = PrimitiveArray::from_values(values.clone());

        assert_eq!(array.len(), 3);
        assert_eq!(array.null_count(), 0);
        assert!(array.null_bitmap.is_none());

        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(array.value(i), Some(expected));
        }
    }

    #[test]
    fn test_slice_operation() {
        let values = vec![1i64, 2, 3, 4, 5, 6];
        let validity = vec![true, false, true, false, true, false];
        let array = PrimitiveArray::new(values, Some(validity));

        // Slice middle portion: elements 2, 3, 4 (indices 2, 3, 4)
        let sliced = array.slice(2, 3);
        let sliced_array = sliced
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();

        assert_eq!(sliced_array.len(), 3);
        assert_eq!(sliced_array.offset, 2);

        // Check values in slice
        assert_eq!(sliced_array.value(0), Some(3)); // Original index 2
        assert_eq!(sliced_array.value(1), None); // Original index 3 (null)
        assert_eq!(sliced_array.value(2), Some(5)); // Original index 4

        // Verify null count in slice
        assert_eq!(sliced_array.null_count(), 1);
    }

    #[test]
    fn test_slice_boundary_conditions() {
        let values = vec![1i64, 2, 3];
        let array = PrimitiveArray::new(values, None);

        // Empty slice
        let empty_slice = array.slice(1, 0);
        let empty_array = empty_slice
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        assert_eq!(empty_array.len(), 0);

        // Single element slice
        let single_slice = array.slice(1, 1);
        let single_array = single_slice
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        assert_eq!(single_array.len(), 1);
        assert_eq!(single_array.value(0), Some(2));

        // Full array slice
        let full_slice = array.slice(0, 3);
        let full_array = full_slice
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        assert_eq!(full_array.len(), 3);
    }

    #[test]
    #[should_panic]
    fn test_slice_out_of_bounds() {
        let array = PrimitiveArray::<i64>::new(vec![1, 2, 3], None);
        array.slice(2, 5); // Should panic
    }

    #[test]
    #[should_panic(expected = "Index 5 out of bounds")]
    fn test_value_access_out_of_bounds() {
        let array = PrimitiveArray::<i64>::new(vec![1, 2, 3], None);
        array.value(5); // Should panic
    }

    #[test]
    fn test_cached_null_count_consistency() {
        let values = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
        let validity = vec![true, false, true, false, true, false, true, false];
        let array = PrimitiveArray::new(values, Some(validity));

        // First call should calculate and cache
        let null_count1 = array.null_count();
        let cached_value = array.cached_null_count.load(Ordering::Relaxed);
        assert_ne!(cached_value, UNINITIALIZED_NULL_COUNT);
        assert_eq!(cached_value, null_count1);

        // Subsequent calls should use cache
        let null_count2 = array.null_count();
        assert_eq!(null_count1, null_count2);
        assert_eq!(null_count1, 4); // Four false bits
    }

    #[test]
    fn test_cache_invalidation_on_slice() {
        let values = vec![1i64, 2, 3, 4, 5, 6];
        let validity = vec![true, false, true, false, true, false];
        let array = PrimitiveArray::new(values, Some(validity));

        // Cache original array null count
        let original_null_count = array.null_count();
        assert_eq!(original_null_count, 3);

        // Create slice
        let sliced = array.slice(1, 4); // Indices 1,2,3,4
        let sliced_array = sliced
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();

        // Slice should have uninitialized cache
        let slice_cached = sliced_array.cached_null_count.load(Ordering::Relaxed);
        assert_eq!(slice_cached, UNINITIALIZED_NULL_COUNT);

        // Calculate slice null count
        let slice_null_count = sliced_array.null_count();
        assert_eq!(slice_null_count, 2); // Two nulls in slice

        // Verify cache is now set for slice
        let slice_cached_after = sliced_array.cached_null_count.load(Ordering::Relaxed);
        assert_eq!(slice_cached_after, slice_null_count);
    }

    #[test]
    fn test_consistency_values_bitmap_nullcount() {
        let values = vec![10i64, 20, 30, 40, 50];
        let validity = vec![true, false, true, false, true];
        let array = PrimitiveArray::new(values.clone(), Some(validity.clone()));

        // Manual verification of consistency
        let bitmap = array.null_bitmap.as_ref().unwrap();

        // Check that bitmap length matches array length
        assert_eq!(bitmap.bit_count(), array.len());

        // Count nulls manually from validity vector
        let expected_null_count = validity.iter().filter(|&&b| !b).count();
        assert_eq!(array.null_count(), expected_null_count);

        // Verify each position consistency
        for (i, &is_valid) in validity.iter().enumerate() {
            let bitmap_bit = bitmap.get_bit(i);
            assert_eq!(bitmap_bit, is_valid, "Bitmap inconsistency at index {}", i);

            if is_valid {
                assert_eq!(array.value(i), Some(values[i]));
            } else {
                assert_eq!(array.value(i), None);
            }
        }
    }

    #[test]
    fn test_multiple_slices_consistency() {
        let values: Vec<i64> = (0..20).collect();
        let validity: Vec<bool> = (0..20).map(|i| i % 3 != 0).collect(); // Every 3rd is null
        let array = PrimitiveArray::new(values, Some(validity));

        // Create multiple overlapping slices
        let slice1 = array.slice(0, 10);
        let slice2 = array.slice(5, 10);
        let slice3 = array.slice(10, 10);

        let s1 = slice1
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        let s2 = slice2
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        let s3 = slice3
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();

        // Verify consistency across slices
        assert_eq!(s1.null_count() + s3.null_count(), array.null_count());

        // Test overlapping region consistency (indices 5-9 in original)
        for i in 0..5 {
            assert_eq!(s1.value(i + 5), s2.value(i));
        }
    }

    #[test]
    fn test_float64_primitive_array() {
        let values = vec![1.5f64, 2.7, 3.14, 4.0];
        let validity = vec![true, false, true, true];
        let array = PrimitiveArray::new(values, Some(validity));

        assert_eq!(array.data_type(), &DataType::Float64);
        assert_eq!(array.len(), 4);
        assert_eq!(array.null_count(), 1);

        assert_eq!(array.value(0), Some(1.5));
        assert_eq!(array.value(1), None);
        assert_eq!(array.value(2), Some(3.14));
        assert_eq!(array.value(3), Some(4.0));
    }

    #[test]
    fn test_as_any_downcast() {
        let array: ArrayRef = Arc::new(PrimitiveArray::new(vec![1i64, 2, 3], None));

        // Successful downcast
        let int_array = array.as_any().downcast_ref::<PrimitiveArray<i64>>();
        assert!(int_array.is_some());

        // Failed downcast
        let float_array = array.as_any().downcast_ref::<PrimitiveArray<f64>>();
        assert!(float_array.is_none());
    }

    #[test]
    fn test_large_array_performance() {
        // Test with larger array to verify null counting performance
        let size = 10_000;
        let values: Vec<i64> = (0..size).collect();
        let validity: Vec<bool> = (0..size).map(|i| i % 7 != 0).collect();

        let array = PrimitiveArray::new(values, Some(validity.clone()));

        let expected_nulls = validity.iter().filter(|&&b| !b).count();

        // First call (should calculate)
        let start = std::time::Instant::now();
        let null_count = array.null_count();
        let first_duration = start.elapsed();

        // Second call (should use cache)
        let start = std::time::Instant::now();
        let null_count2 = array.null_count();
        let second_duration = start.elapsed();

        assert_eq!(null_count, expected_nulls);
        assert_eq!(null_count2, expected_nulls);

        // Cache should be significantly faster
        assert!(second_duration < first_duration / 2);
    }

    #[test]
    fn test_iterator() {
        let values = vec![1i64, 2, 3, 4, 5];
        let validity = vec![true, false, true, false, true];
        let array = PrimitiveArray::new(values, Some(validity));

        let collected: Vec<Option<i64>> = array.iter().collect();
        assert_eq!(collected, vec![Some(1), None, Some(3), None, Some(5)]);
    }

    #[test]
    fn test_iterator_no_nulls() {
        let values = vec![10i64, 20, 30];
        let array = PrimitiveArray::from_values(values);

        let collected: Vec<Option<i64>> = array.iter().collect();
        assert_eq!(collected, vec![Some(10), Some(20), Some(30)]);
    }

    #[test]
    fn test_iterator_empty() {
        let array = PrimitiveArray::<i64>::new(vec![], None);
        let collected: Vec<Option<i64>> = array.iter().collect();
        assert_eq!(collected, vec![]);
    }

    #[test]
    fn test_primitive_array_builder() {
        let mut builder = PrimitiveArrayBuilder::<i64>::new();

        builder.append_value(10);
        builder.append_null(0); // Placeholder
        builder.append_value(20);
        builder.append_value(30);
        builder.append_null(0); // Placeholder

        let array = builder.finish();

        assert_eq!(array.len(), 5);
        assert_eq!(array.null_count(), 2);

        assert_eq!(array.value(0), Some(10));
        assert_eq!(array.value(1), None);
        assert_eq!(array.value(2), Some(20));
        assert_eq!(array.value(3), Some(30));
        assert_eq!(array.value(4), None);
    }

    #[test]
    fn test_primitive_array_builder_with_capacity() {
        let mut builder = PrimitiveArrayBuilder::<f64>::with_capacity(100);

        for i in 0..50 {
            if i % 3 == 0 {
                builder.append_null(0.0);
            } else {
                builder.append_value(i as f64 * 1.5);
            }
        }

        let array = builder.finish();
        assert_eq!(array.len(), 50);
        assert_eq!(array.data_type(), &DataType::Float64);

        let expected_nulls = (0..50).filter(|i| i % 3 == 0).count();
        assert_eq!(array.null_count(), expected_nulls);
    }

    #[test]
    fn test_primitive_array_builder_no_nulls() {
        let mut builder = PrimitiveArrayBuilder::<i64>::new();

        for i in 1..=5 {
            builder.append_value(i * 10);
        }

        let array = builder.finish();
        assert_eq!(array.len(), 5);
        assert_eq!(array.null_count(), 0);
        assert!(array.null_bitmap.is_none());

        for i in 0..5 {
            assert_eq!(array.value(i), Some((i as i64 + 1) * 10));
        }
    }

    #[test]
    fn test_total_bytes() {
        let values = vec![1i64, 2, 3, 4, 5];
        let array = PrimitiveArray::new(values, None);

        // i64 = 8 bytes, 5 elements = 40 bytes
        assert_eq!(array.total_bytes(), 40);

        let float_values = vec![1.0f64, 2.0, 3.0];
        let float_array = PrimitiveArray::new(float_values, None);

        // f64 = 8 bytes, 3 elements = 24 bytes
        assert_eq!(float_array.total_bytes(), 24);
    }

    #[test]
    fn test_iterator_size_hint() {
        let values = vec![1i64, 2, 3, 4, 5];
        let array = PrimitiveArray::new(values, None);

        let mut iter = array.iter();
        assert_eq!(iter.size_hint(), (5, Some(5)));

        iter.next();
        assert_eq!(iter.size_hint(), (4, Some(4)));

        iter.next();
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
    }

    #[test]
    fn test_builder_default() {
        let builder = PrimitiveArrayBuilder::<i64>::default();
        let array = builder.finish();

        assert_eq!(array.len(), 0);
        assert_eq!(array.null_count(), 0);
    }
}
