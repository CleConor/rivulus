use super::super::schema::DataType;
use super::bitmap::{BitMap, BitmapBuilder};
use super::{Array, ArrayRef};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const UNINITIALIZED_NULL_COUNT: usize = usize::MAX;

#[derive(Debug)]
pub struct BooleanArray {
    values: Arc<BitMap>,
    null_bitmap: Option<Arc<BitMap>>,
    offset: usize,
    length: usize,
    cached_null_count: AtomicUsize,
}

impl BooleanArray {
    pub fn new(booleans: Vec<Option<bool>>) -> Self {
        let mut values_builder = BitmapBuilder::new();
        let mut null_builder = BitmapBuilder::new();

        for bool_opt in &booleans {
            match bool_opt {
                Some(value) => {
                    values_builder.append(*value);
                    null_builder.append(true);
                }
                None => {
                    values_builder.append(false);
                    null_builder.append(false);
                }
            }
        }

        let values = Arc::new(values_builder.finish());
        let null_bitmap = if null_builder.has_nulls() {
            Some(Arc::new(null_builder.finish()))
        } else {
            None
        };

        BooleanArray {
            values,
            null_bitmap,
            offset: 0,
            length: booleans.len(),
            cached_null_count: AtomicUsize::new(UNINITIALIZED_NULL_COUNT),
        }
    }

    pub fn from_bools(booleans: Vec<bool>) -> Self {
        let bool_opts: Vec<Option<bool>> = booleans.into_iter().map(Some).collect();
        Self::new(bool_opts)
    }

    pub fn new_null(length: usize) -> Self {
        let values = Arc::new(BitMap::new(length));
        let null_bitmap = Arc::new(BitMap::all_false(length));
        BooleanArray {
            values,
            null_bitmap: Some(null_bitmap),
            offset: 0,
            length,
            cached_null_count: AtomicUsize::new(length),
        }
    }

    pub fn all_true(length: usize) -> Self {
        let values = Arc::new(BitMap::all_true(length));
        BooleanArray {
            values,
            null_bitmap: None,
            offset: 0,
            length,
            cached_null_count: AtomicUsize::new(UNINITIALIZED_NULL_COUNT),
        }
    }

    pub fn all_false(length: usize) -> Self {
        let values = Arc::new(BitMap::new(length));
        BooleanArray {
            values,
            null_bitmap: None,
            offset: 0,
            length,
            cached_null_count: AtomicUsize::new(UNINITIALIZED_NULL_COUNT),
        }
    }

    pub fn value(&self, index: usize) -> Option<bool> {
        assert!(index < self.length, "Index {} out of bounds", index);

        let logical_index = self.offset + index;

        if let Some(bitmap) = &self.null_bitmap {
            if !bitmap.get_bit(logical_index) {
                return None;
            }
        }

        Some(self.values.get_bit(logical_index))
    }

    pub fn iter(&self) -> BooleanArrayIter {
        BooleanArrayIter {
            array: self,
            index: 0,
        }
    }

    pub fn total_bits(&self) -> usize {
        self.values.bit_count()
    }

    pub fn total_bytes(&self) -> usize {
        (self.total_bits() + 7) / 8
    }

    pub fn and(&self, other: &BooleanArray) -> Result<BooleanArray, &'static str> {
        if self.len() != other.len() {
            return Err("Array lengths must match for logical operations");
        }

        let mut builder = BooleanArrayBuilder::with_capacity(self.len());

        for i in 0..self.len() {
            match (self.value(i), other.value(i)) {
                (Some(a), Some(b)) => builder.append_value(a && b),
                _ => builder.append_null(),
            }
        }

        Ok(builder.finish())
    }

    pub fn or(&self, other: &BooleanArray) -> Result<BooleanArray, &'static str> {
        if self.len() != other.len() {
            return Err("Array lengths must match for logical operations");
        }

        let mut builder = BooleanArrayBuilder::with_capacity(self.len());

        for i in 0..self.len() {
            match (self.value(i), other.value(i)) {
                (Some(a), Some(b)) => builder.append_value(a || b),
                _ => builder.append_null(),
            }
        }

        Ok(builder.finish())
    }

    pub fn not(&self) -> BooleanArray {
        let mut builder = BooleanArrayBuilder::with_capacity(self.len());

        for i in 0..self.len() {
            match self.value(i) {
                Some(value) => builder.append_value(!value),
                None => builder.append_null(),
            }
        }

        builder.finish()
    }

    pub fn count_true(&self) -> usize {
        (0..self.len())
            .filter_map(|i| self.value(i))
            .filter(|&b| b)
            .count()
    }

    pub fn count_false(&self) -> usize {
        (0..self.len())
            .filter_map(|i| self.value(i))
            .filter(|&b| !b)
            .count()
    }
}

impl Array for BooleanArray {
    fn len(&self) -> usize {
        self.length
    }

    fn data_type(&self) -> &DataType {
        &DataType::Boolean
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
        assert!(offset + length <= self.length, "Slice out of bounds");

        Arc::new(BooleanArray {
            values: self.values.clone(),
            null_bitmap: self.null_bitmap.clone(),
            offset: self.offset + offset,
            length,
            cached_null_count: AtomicUsize::new(UNINITIALIZED_NULL_COUNT),
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct BooleanArrayIter<'a> {
    array: &'a BooleanArray,
    index: usize,
}

impl<'a> Iterator for BooleanArrayIter<'a> {
    type Item = Option<bool>;

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

impl<'a> ExactSizeIterator for BooleanArrayIter<'a> {}

pub struct BooleanArrayBuilder {
    values_builder: BitmapBuilder,
    null_builder: BitmapBuilder,
}

impl BooleanArrayBuilder {
    pub fn new() -> Self {
        BooleanArrayBuilder {
            values_builder: BitmapBuilder::new(),
            null_builder: BitmapBuilder::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        BooleanArrayBuilder {
            values_builder: BitmapBuilder::with_capacity(capacity),
            null_builder: BitmapBuilder::with_capacity(capacity),
        }
    }

    pub fn append_value(&mut self, value: bool) {
        self.values_builder.append(value);
        self.null_builder.append(true);
    }

    pub fn append_null(&mut self) {
        self.values_builder.append(false);
        self.null_builder.append(false);
    }

    pub fn finish(self) -> BooleanArray {
        let values = Arc::new(self.values_builder.finish());
        let null_bitmap = if self.null_builder.has_nulls() {
            Some(Arc::new(self.null_builder.finish()))
        } else {
            None
        };

        let length = values.bit_count();

        BooleanArray {
            values,
            null_bitmap,
            offset: 0,
            length,
            cached_null_count: AtomicUsize::new(UNINITIALIZED_NULL_COUNT),
        }
    }
}

impl Default for BooleanArrayBuilder {
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
        let booleans = vec![Some(true), Some(false), Some(true), Some(false), Some(true)];
        let array = BooleanArray::new(booleans);

        assert_eq!(array.len(), 5);
        assert_eq!(array.data_type(), &DataType::Boolean);
        assert_eq!(array.null_count(), 0);
        assert!(array.null_bitmap.is_none());

        assert_eq!(array.value(0), Some(true));
        assert_eq!(array.value(1), Some(false));
        assert_eq!(array.value(2), Some(true));
        assert_eq!(array.value(3), Some(false));
        assert_eq!(array.value(4), Some(true));
    }

    #[test]
    fn test_new_with_some_nulls() {
        let booleans = vec![Some(true), None, Some(false), None, Some(true)];
        let array = BooleanArray::new(booleans);

        assert_eq!(array.len(), 5);
        assert_eq!(array.null_count(), 2);
        assert!(array.null_bitmap.is_some());

        assert_eq!(array.value(0), Some(true));
        assert_eq!(array.value(1), None);
        assert_eq!(array.value(2), Some(false));
        assert_eq!(array.value(3), None);
        assert_eq!(array.value(4), Some(true));
    }

    #[test]
    fn test_all_nulls() {
        let booleans = vec![None, None, None];
        let array = BooleanArray::new(booleans);

        assert_eq!(array.len(), 3);
        assert_eq!(array.null_count(), 3);

        for i in 0..3 {
            assert_eq!(array.value(i), None);
        }
    }

    #[test]
    fn test_empty_array() {
        let array = BooleanArray::new(vec![]);

        assert_eq!(array.len(), 0);
        assert_eq!(array.null_count(), 0);
        assert!(array.null_bitmap.is_none());
    }

    #[test]
    fn test_from_bools_constructor() {
        let booleans = vec![true, false, true, false];
        let array = BooleanArray::from_bools(booleans);

        assert_eq!(array.len(), 4);
        assert_eq!(array.null_count(), 0);
        assert!(array.null_bitmap.is_none());

        assert_eq!(array.value(0), Some(true));
        assert_eq!(array.value(1), Some(false));
        assert_eq!(array.value(2), Some(true));
        assert_eq!(array.value(3), Some(false));
    }

    #[test]
    fn test_factory_methods() {
        // All true
        let all_true = BooleanArray::all_true(3);
        assert_eq!(all_true.len(), 3);
        assert_eq!(all_true.null_count(), 0);
        for i in 0..3 {
            assert_eq!(all_true.value(i), Some(true));
        }

        // All false
        let all_false = BooleanArray::all_false(3);
        assert_eq!(all_false.len(), 3);
        assert_eq!(all_false.null_count(), 0);
        for i in 0..3 {
            assert_eq!(all_false.value(i), Some(false));
        }

        // All null
        let all_null = BooleanArray::new_null(3);
        assert_eq!(all_null.len(), 3);
        assert_eq!(all_null.null_count(), 3);
        for i in 0..3 {
            assert_eq!(all_null.value(i), None);
        }
    }

    #[test]
    fn test_slice_operation() {
        let booleans = vec![Some(true), None, Some(false), None, Some(true), Some(false)];
        let array = BooleanArray::new(booleans);

        // Slice middle portion: elements 2, 3, 4 (indices 2, 3, 4)
        let sliced = array.slice(2, 3);
        let sliced_array = sliced.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(sliced_array.len(), 3);
        assert_eq!(sliced_array.offset, 2);

        // Check values in slice
        assert_eq!(sliced_array.value(0), Some(false)); // Original index 2
        assert_eq!(sliced_array.value(1), None); // Original index 3 (null)
        assert_eq!(sliced_array.value(2), Some(true)); // Original index 4

        // Verify null count in slice
        assert_eq!(sliced_array.null_count(), 1);
    }

    #[test]
    fn test_slice_boundary_conditions() {
        let booleans = vec![Some(true), Some(false), Some(true)];
        let array = BooleanArray::new(booleans);

        // Empty slice
        let empty_slice = array.slice(1, 0);
        let empty_array = empty_slice.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(empty_array.len(), 0);

        // Single element slice
        let single_slice = array.slice(1, 1);
        let single_array = single_slice
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert_eq!(single_array.len(), 1);
        assert_eq!(single_array.value(0), Some(false));

        // Full array slice
        let full_slice = array.slice(0, 3);
        let full_array = full_slice.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(full_array.len(), 3);
    }

    #[test]
    #[should_panic]
    fn test_slice_out_of_bounds() {
        let booleans = vec![Some(true), Some(false)];
        let array = BooleanArray::new(booleans);
        array.slice(1, 5); // Should panic
    }

    #[test]
    #[should_panic(expected = "Index 5 out of bounds")]
    fn test_value_access_out_of_bounds() {
        let booleans = vec![Some(true)];
        let array = BooleanArray::new(booleans);
        array.value(5); // Should panic
    }

    #[test]
    fn test_cached_null_count_consistency() {
        let booleans = vec![Some(true), None, Some(false), None, Some(true), None];
        let array = BooleanArray::new(booleans);

        // First call should calculate and cache
        let null_count1 = array.null_count();
        let cached_value = array.cached_null_count.load(Ordering::Relaxed);
        assert_ne!(cached_value, UNINITIALIZED_NULL_COUNT);
        assert_eq!(cached_value, null_count1);

        // Subsequent calls should use cache
        let null_count2 = array.null_count();
        assert_eq!(null_count1, null_count2);
        assert_eq!(null_count1, 3); // Three nulls
    }

    #[test]
    fn test_cache_invalidation_on_slice() {
        let booleans = vec![Some(true), None, Some(false), None, Some(true), None];
        let array = BooleanArray::new(booleans);

        // Cache original array null count
        let original_null_count = array.null_count();
        assert_eq!(original_null_count, 3);

        // Create slice
        let sliced = array.slice(1, 4); // Indices 1,2,3,4
        let sliced_array = sliced.as_any().downcast_ref::<BooleanArray>().unwrap();

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
        let booleans = vec![Some(true), None, Some(false), None, Some(true)];
        let array = BooleanArray::new(booleans.clone());

        // Manual verification of consistency
        let null_bitmap = array.null_bitmap.as_ref().unwrap();

        // Check that bitmap length matches array length
        assert_eq!(null_bitmap.bit_count(), array.len());

        // Count nulls manually from input
        let expected_null_count = booleans.iter().filter(|b| b.is_none()).count();
        assert_eq!(array.null_count(), expected_null_count);

        // Verify each position consistency
        for (i, bool_opt) in booleans.iter().enumerate() {
            let bitmap_bit = null_bitmap.get_bit(i);
            assert_eq!(
                bitmap_bit,
                bool_opt.is_some(),
                "Null bitmap inconsistency at index {}",
                i
            );

            match bool_opt {
                Some(b) => assert_eq!(array.value(i), Some(*b)),
                None => assert_eq!(array.value(i), None),
            }
        }
    }

    #[test]
    fn test_iterator() {
        let booleans = vec![Some(true), None, Some(false)];
        let array = BooleanArray::new(booleans);

        let collected: Vec<Option<bool>> = array.iter().collect();
        assert_eq!(collected, vec![Some(true), None, Some(false)]);
    }

    #[test]
    fn test_boolean_array_builder() {
        let mut builder = BooleanArrayBuilder::new();

        builder.append_value(true);
        builder.append_null();
        builder.append_value(false);
        builder.append_value(true);
        builder.append_null();

        let array = builder.finish();

        assert_eq!(array.len(), 5);
        assert_eq!(array.null_count(), 2);

        assert_eq!(array.value(0), Some(true));
        assert_eq!(array.value(1), None);
        assert_eq!(array.value(2), Some(false));
        assert_eq!(array.value(3), Some(true));
        assert_eq!(array.value(4), None);
    }

    #[test]
    fn test_boolean_array_builder_with_capacity() {
        let mut builder = BooleanArrayBuilder::with_capacity(100);

        for i in 0..50 {
            if i % 3 == 0 {
                builder.append_null();
            } else {
                builder.append_value(i % 2 == 0);
            }
        }

        let array = builder.finish();
        assert_eq!(array.len(), 50);

        let expected_nulls = (0..50).filter(|i| i % 3 == 0).count();
        assert_eq!(array.null_count(), expected_nulls);
    }

    #[test]
    fn test_boolean_array_builder_no_nulls() {
        let mut builder = BooleanArrayBuilder::new();

        for i in 0..5 {
            builder.append_value(i % 2 == 0);
        }

        let array = builder.finish();
        assert_eq!(array.len(), 5);
        assert_eq!(array.null_count(), 0);
        assert!(array.null_bitmap.is_none());

        assert_eq!(array.value(0), Some(true)); // 0 % 2 == 0
        assert_eq!(array.value(1), Some(false)); // 1 % 2 != 0
        assert_eq!(array.value(2), Some(true)); // 2 % 2 == 0
        assert_eq!(array.value(3), Some(false)); // 3 % 2 != 0
        assert_eq!(array.value(4), Some(true)); // 4 % 2 == 0
    }

    #[test]
    fn test_total_bytes_and_bits() {
        let booleans = vec![Some(true); 17]; // 17 booleans = 3 bytes (17 bits)
        let array = BooleanArray::new(booleans);

        assert_eq!(array.total_bits(), 17);
        assert_eq!(array.total_bytes(), 3); // (17 + 7) / 8 = 3
    }

    #[test]
    fn test_logical_and_operation() {
        let array1 =
            BooleanArray::new(vec![Some(true), Some(false), Some(true), None, Some(false)]);
        let array2 = BooleanArray::new(vec![Some(true), Some(true), Some(false), Some(true), None]);

        let result = array1.and(&array2).unwrap();

        assert_eq!(result.value(0), Some(true)); // true && true = true
        assert_eq!(result.value(1), Some(false)); // false && true = false
        assert_eq!(result.value(2), Some(false)); // true && false = false
        assert_eq!(result.value(3), None); // null && true = null
        assert_eq!(result.value(4), None); // false && null = null
    }

    #[test]
    fn test_logical_or_operation() {
        let array1 =
            BooleanArray::new(vec![Some(true), Some(false), Some(true), None, Some(false)]);
        let array2 =
            BooleanArray::new(vec![Some(false), Some(true), Some(false), Some(true), None]);

        let result = array1.or(&array2).unwrap();

        assert_eq!(result.value(0), Some(true)); // true || false = true
        assert_eq!(result.value(1), Some(true)); // false || true = true
        assert_eq!(result.value(2), Some(true)); // true || false = true
        assert_eq!(result.value(3), None); // null || true = null
        assert_eq!(result.value(4), None); // false || null = null
    }

    #[test]
    fn test_logical_not_operation() {
        let array = BooleanArray::new(vec![Some(true), Some(false), None, Some(true)]);

        let result = array.not();

        assert_eq!(result.value(0), Some(false)); // !true = false
        assert_eq!(result.value(1), Some(true)); // !false = true
        assert_eq!(result.value(2), None); // !null = null
        assert_eq!(result.value(3), Some(false)); // !true = false
    }

    #[test]
    fn test_count_true_false() {
        let array = BooleanArray::new(vec![
            Some(true),
            Some(false),
            Some(true),
            None,
            Some(false),
            Some(true),
        ]);

        assert_eq!(array.count_true(), 3);
        assert_eq!(array.count_false(), 2);
        // Note: nulls are not counted in either
    }

    #[test]
    fn test_and_mismatched_lengths() {
        let array1 = BooleanArray::new(vec![Some(true), Some(false)]);
        let array2 = BooleanArray::new(vec![Some(true)]);

        assert!(array1.and(&array2).is_err());
    }

    #[test]
    fn test_iterator_size_hint() {
        let booleans = vec![Some(true), Some(false), None, Some(true)];
        let array = BooleanArray::new(booleans);

        let mut iter = array.iter();
        assert_eq!(iter.size_hint(), (4, Some(4)));

        iter.next();
        assert_eq!(iter.size_hint(), (3, Some(3)));

        iter.next();
        iter.next();
        assert_eq!(iter.size_hint(), (1, Some(1)));
    }

    #[test]
    fn test_as_any_downcast() {
        let array: ArrayRef = Arc::new(BooleanArray::new(vec![Some(true), Some(false)]));

        // Successful downcast
        let bool_array = array.as_any().downcast_ref::<BooleanArray>();
        assert!(bool_array.is_some());
    }

    #[test]
    fn test_large_boolean_array_performance() {
        let size = 10_000;
        let booleans: Vec<Option<bool>> = (0..size)
            .map(|i| if i % 7 == 0 { None } else { Some(i % 2 == 0) })
            .collect();

        let array = BooleanArray::new(booleans.clone());
        let expected_nulls = booleans.iter().filter(|b| b.is_none()).count();

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
    fn test_builder_default() {
        let builder = BooleanArrayBuilder::default();
        let array = builder.finish();

        assert_eq!(array.len(), 0);
        assert_eq!(array.null_count(), 0);
    }
}
