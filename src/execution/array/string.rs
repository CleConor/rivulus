use super::bitmap::{BitMap, BitmapBuilder};
use super::{Array, ArrayRef, DataType};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const UNINITIALIZED_NULL_COUNT: usize = usize::MAX;

#[derive(Debug)]
pub struct StringArray {
    data: Arc<[u8]>,
    offsets: Arc<[i32]>,
    null_bitmap: Option<Arc<BitMap>>,
    offset: usize,
    length: usize,
    cached_null_count: AtomicUsize,
}

impl StringArray {
    pub fn new(strings: Vec<Option<String>>) -> Self {
        let mut offsets = Vec::with_capacity(strings.len() + 1);
        let mut data = Vec::new();
        let mut null_builder = BitmapBuilder::new();

        offsets.push(0i32);

        for string_opt in &strings {
            match string_opt {
                Some(s) => {
                    null_builder.append(true);
                    data.extend_from_slice(s.as_bytes());
                    offsets.push(data.len() as i32);
                }
                None => {
                    null_builder.append(false);
                    offsets.push(data.len() as i32);
                }
            }
        }

        let null_bitmap = if null_builder.has_nulls() {
            Some(Arc::new(null_builder.finish()))
        } else {
            None
        };

        let array = StringArray {
            offsets: Arc::from(offsets),
            data: Arc::from(data),
            null_bitmap,
            offset: 0,
            length: strings.len(),
            cached_null_count: AtomicUsize::new(UNINITIALIZED_NULL_COUNT),
        };

        Self::validate_utf8(&array.data, &array.offsets).expect("Invalid UTF-8 in string data");

        array
    }

    pub fn from_strings(strings: Vec<String>) -> Self {
        let string_opts: Vec<Option<String>> = strings.into_iter().map(Some).collect();
        Self::new(string_opts)
    }

    pub fn new_null(length: usize) -> Self {
        let offsets = vec![0i32; length + 1];
        let data = Vec::new();
        let null_bitmap = Arc::new(BitMap::all_false(length));

        StringArray {
            offsets: Arc::from(offsets),
            data: Arc::from(data),
            null_bitmap: Some(null_bitmap),
            offset: 0,
            length,
            cached_null_count: AtomicUsize::new(length),
        }
    }

    pub fn value(&self, index: usize) -> Option<&str> {
        assert!(index < self.length, "Index {} out of bounds", index);

        let logical_index = self.offset + index;

        if let Some(bitmap) = &self.null_bitmap {
            if !bitmap.get_bit(logical_index) {
                return None;
            }
        }

        let start_offset = self.offsets[logical_index] as usize;
        let end_offset = self.offsets[logical_index + 1] as usize;

        let string_bytes = &self.data[start_offset..end_offset];

        Some(unsafe { std::str::from_utf8_unchecked(string_bytes) })
    }

    pub fn byte_len(&self, index: usize) -> Option<usize> {
        assert!(index < self.length, "Index {} out of bounds", index);

        let logical_index = self.offset + index;

        if let Some(bitmap) = &self.null_bitmap {
            if !bitmap.get_bit(logical_index) {
                return None;
            }
        }

        let start_offset = self.offsets[logical_index] as usize;
        let end_offset = self.offsets[logical_index + 1] as usize;

        Some(end_offset - start_offset)
    }

    pub fn char_len(&self, index: usize) -> Option<usize> {
        self.value(index).map(|s| s.chars().count())
    }

    pub fn iter(&self) -> StringArrayIter {
        StringArrayIter {
            array: self,
            index: 0,
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.data.len()
    }

    fn validate_utf8(data: &[u8], offsets: &[i32]) -> Result<(), &'static str> {
        for window in offsets.windows(2) {
            let start = window[0] as usize;
            let end = window[1] as usize;

            if end > data.len() {
                return Err("Offset out of bounds");
            }

            let string_slice = &data[start..end];
            if std::str::from_utf8(string_slice).is_err() {
                return Err("Invalid UTF-8 sequence");
            }
        }
        Ok(())
    }
}

impl Array for StringArray {
    fn len(&self) -> usize {
        self.length
    }

    fn data_type(&self) -> &DataType {
        &DataType::String
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

        Arc::new(StringArray {
            offsets: self.offsets.clone(),
            data: self.data.clone(),
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

pub struct StringArrayIter<'a> {
    array: &'a StringArray,
    index: usize,
}

impl<'a> Iterator for StringArrayIter<'a> {
    type Item = Option<&'a str>;

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

impl<'a> ExactSizeIterator for StringArrayIter<'a> {}

pub struct StringBuilder {
    offsets: Vec<i32>,
    data: Vec<u8>,
    null_builder: BitmapBuilder,
    current_offset: i32,
}

impl StringBuilder {
    pub fn new() -> Self {
        let mut offsets = Vec::new();
        offsets.push(0);

        StringBuilder {
            offsets,
            data: Vec::new(),
            null_builder: BitmapBuilder::new(),
            current_offset: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(capacity + 1);
        offsets.push(0);

        StringBuilder {
            offsets,
            data: Vec::new(),
            null_builder: BitmapBuilder::with_capacity(capacity),
            current_offset: 0,
        }
    }

    pub fn append_value(&mut self, value: &str) {
        self.null_builder.append(true);
        self.data.extend_from_slice(value.as_bytes());
        self.current_offset += value.len() as i32;
        self.offsets.push(self.current_offset);
    }

    pub fn append_null(&mut self) {
        self.null_builder.append(false);
        self.offsets.push(self.current_offset);
    }

    pub fn finish(self) -> StringArray {
        let null_bitmap = if self.null_builder.has_nulls() {
            Some(Arc::new(self.null_builder.finish()))
        } else {
            None
        };

        let length = self.offsets.len() - 1;

        let array = StringArray {
            offsets: Arc::from(self.offsets),
            data: Arc::from(self.data),
            null_bitmap,
            offset: 0,
            length,
            cached_null_count: AtomicUsize::new(UNINITIALIZED_NULL_COUNT),
        };

        StringArray::validate_utf8(&array.data, &array.offsets)
            .expect("Invalid UTF-8 in string data");

        array
    }
}

impl Default for StringBuilder {
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
        let strings = vec![
            Some("hello".to_string()),
            Some("world".to_string()),
            Some("test".to_string()),
        ];
        let array = StringArray::new(strings);

        assert_eq!(array.len(), 3);
        assert_eq!(array.data_type(), &DataType::String);
        assert_eq!(array.null_count(), 0);
        assert!(array.null_bitmap.is_none());

        assert_eq!(array.value(0), Some("hello"));
        assert_eq!(array.value(1), Some("world"));
        assert_eq!(array.value(2), Some("test"));
    }

    #[test]
    fn test_new_with_some_nulls() {
        let strings = vec![
            Some("hello".to_string()),
            None,
            Some("world".to_string()),
            None,
            Some("test".to_string()),
        ];
        let array = StringArray::new(strings);

        assert_eq!(array.len(), 5);
        assert_eq!(array.null_count(), 2);
        assert!(array.null_bitmap.is_some());

        assert_eq!(array.value(0), Some("hello"));
        assert_eq!(array.value(1), None);
        assert_eq!(array.value(2), Some("world"));
        assert_eq!(array.value(3), None);
        assert_eq!(array.value(4), Some("test"));
    }

    #[test]
    fn test_all_nulls() {
        let strings = vec![None, None, None];
        let array = StringArray::new(strings);

        assert_eq!(array.len(), 3);
        assert_eq!(array.null_count(), 3);

        for i in 0..3 {
            assert_eq!(array.value(i), None);
        }
    }

    #[test]
    fn test_empty_array() {
        let array = StringArray::new(vec![]);

        assert_eq!(array.len(), 0);
        assert_eq!(array.null_count(), 0);
        assert!(array.null_bitmap.is_none());
    }

    #[test]
    fn test_empty_strings() {
        let strings = vec![
            Some("hello".to_string()),
            Some("".to_string()), // Empty string
            Some("world".to_string()),
        ];
        let array = StringArray::new(strings);

        assert_eq!(array.len(), 3);
        assert_eq!(array.null_count(), 0);

        assert_eq!(array.value(0), Some("hello"));
        assert_eq!(array.value(1), Some(""));
        assert_eq!(array.value(2), Some("world"));

        assert_eq!(array.byte_len(0), Some(5));
        assert_eq!(array.byte_len(1), Some(0));
        assert_eq!(array.byte_len(2), Some(5));
    }

    #[test]
    fn test_from_strings_constructor() {
        let strings = vec!["hello".to_string(), "world".to_string()];
        let array = StringArray::from_strings(strings);

        assert_eq!(array.len(), 2);
        assert_eq!(array.null_count(), 0);
        assert!(array.null_bitmap.is_none());

        assert_eq!(array.value(0), Some("hello"));
        assert_eq!(array.value(1), Some("world"));
    }

    #[test]
    fn test_new_null_constructor() {
        let array = StringArray::new_null(3);

        assert_eq!(array.len(), 3);
        assert_eq!(array.null_count(), 3);
        assert!(array.null_bitmap.is_some());

        for i in 0..3 {
            assert_eq!(array.value(i), None);
        }
    }

    #[test]
    fn test_slice_operation() {
        let strings = vec![
            Some("a".to_string()),
            None,
            Some("b".to_string()),
            None,
            Some("c".to_string()),
            Some("d".to_string()),
        ];
        let array = StringArray::new(strings);

        // Slice middle portion: elements 2, 3, 4 (indices 2, 3, 4)
        let sliced = array.slice(2, 3);
        let sliced_array = sliced.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(sliced_array.len(), 3);
        assert_eq!(sliced_array.offset, 2);

        // Check values in slice
        assert_eq!(sliced_array.value(0), Some("b")); // Original index 2
        assert_eq!(sliced_array.value(1), None); // Original index 3 (null)
        assert_eq!(sliced_array.value(2), Some("c")); // Original index 4

        // Verify null count in slice
        assert_eq!(sliced_array.null_count(), 1);
    }

    #[test]
    fn test_slice_boundary_conditions() {
        let strings = vec![
            Some("hello".to_string()),
            Some("world".to_string()),
            Some("test".to_string()),
        ];
        let array = StringArray::new(strings);

        // Empty slice
        let empty_slice = array.slice(1, 0);
        let empty_array = empty_slice.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(empty_array.len(), 0);

        // Single element slice
        let single_slice = array.slice(1, 1);
        let single_array = single_slice.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(single_array.len(), 1);
        assert_eq!(single_array.value(0), Some("world"));

        // Full array slice
        let full_slice = array.slice(0, 3);
        let full_array = full_slice.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(full_array.len(), 3);
    }

    #[test]
    #[should_panic]
    fn test_slice_out_of_bounds() {
        let strings = vec![Some("a".to_string()), Some("b".to_string())];
        let array = StringArray::new(strings);
        array.slice(1, 5); // Should panic
    }

    #[test]
    #[should_panic(expected = "Index 5 out of bounds")]
    fn test_value_access_out_of_bounds() {
        let strings = vec![Some("a".to_string())];
        let array = StringArray::new(strings);
        array.value(5); // Should panic
    }

    #[test]
    fn test_cached_null_count_consistency() {
        let strings = vec![
            Some("a".to_string()),
            None,
            Some("b".to_string()),
            None,
            Some("c".to_string()),
            None,
        ];
        let array = StringArray::new(strings);

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
        let strings = vec![
            Some("a".to_string()),
            None,
            Some("b".to_string()),
            None,
            Some("c".to_string()),
            None,
        ];
        let array = StringArray::new(strings);

        // Cache original array null count
        let original_null_count = array.null_count();
        assert_eq!(original_null_count, 3);

        // Create slice
        let sliced = array.slice(1, 4); // Indices 1,2,3,4
        let sliced_array = sliced.as_any().downcast_ref::<StringArray>().unwrap();

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
        let strings = vec![
            Some("hello".to_string()),
            None,
            Some("world".to_string()),
            None,
            Some("test".to_string()),
        ];
        let array = StringArray::new(strings.clone());

        // Manual verification of consistency
        let bitmap = array.null_bitmap.as_ref().unwrap();

        // Check that bitmap length matches array length
        assert_eq!(bitmap.bit_count(), array.len());

        // Count nulls manually from input
        let expected_null_count = strings.iter().filter(|s| s.is_none()).count();
        assert_eq!(array.null_count(), expected_null_count);

        // Verify each position consistency
        for (i, string_opt) in strings.iter().enumerate() {
            let bitmap_bit = bitmap.get_bit(i);
            assert_eq!(
                bitmap_bit,
                string_opt.is_some(),
                "Bitmap inconsistency at index {}",
                i
            );

            match string_opt {
                Some(s) => assert_eq!(array.value(i), Some(s.as_str())),
                None => assert_eq!(array.value(i), None),
            }
        }
    }

    #[test]
    fn test_byte_len_method() {
        let strings = vec![
            Some("hello".to_string()),
            None,
            Some("world!!!".to_string()),
            Some("".to_string()),
        ];
        let array = StringArray::new(strings);

        assert_eq!(array.byte_len(0), Some(5));
        assert_eq!(array.byte_len(1), None);
        assert_eq!(array.byte_len(2), Some(8));
        assert_eq!(array.byte_len(3), Some(0));
    }

    #[test]
    fn test_char_len_method() {
        let strings = vec![
            Some("hello".to_string()),
            None,
            Some("world!!!".to_string()),
            Some("".to_string()),
        ];
        let array = StringArray::new(strings);

        assert_eq!(array.char_len(0), Some(5));
        assert_eq!(array.char_len(1), None);
        assert_eq!(array.char_len(2), Some(8));
        assert_eq!(array.char_len(3), Some(0));
    }

    #[test]
    fn test_byte_vs_char_length_utf8() {
        let strings = vec![
            Some("ascii".to_string()), // 5 bytes, 5 chars
            Some("café".to_string()),  // 5 bytes, 4 chars (é = 2 bytes)
            Some("🦀".to_string()),    // 4 bytes, 1 char (emoji = 4 bytes)
            Some("🦀🔥".to_string()),  // 8 bytes, 2 chars
            Some("".to_string()),      // 0 bytes, 0 chars
            None,                      // null
        ];
        let array = StringArray::new(strings);

        // Byte lengths
        assert_eq!(array.byte_len(0), Some(5));
        assert_eq!(array.byte_len(1), Some(5));
        assert_eq!(array.byte_len(2), Some(4));
        assert_eq!(array.byte_len(3), Some(8));
        assert_eq!(array.byte_len(4), Some(0));
        assert_eq!(array.byte_len(5), None);

        // Character lengths
        assert_eq!(array.char_len(0), Some(5));
        assert_eq!(array.char_len(1), Some(4)); // Different from byte_len!
        assert_eq!(array.char_len(2), Some(1)); // Different from byte_len!
        assert_eq!(array.char_len(3), Some(2)); // Different from byte_len!
        assert_eq!(array.char_len(4), Some(0));
        assert_eq!(array.char_len(5), None);
    }

    #[test]
    fn test_iterator() {
        let strings = vec![Some("a".to_string()), None, Some("b".to_string())];
        let array = StringArray::new(strings);

        let collected: Vec<Option<&str>> = array.iter().collect();
        assert_eq!(collected, vec![Some("a"), None, Some("b")]);
    }

    #[test]
    fn test_string_builder() {
        let mut builder = StringBuilder::new();

        builder.append_value("hello");
        builder.append_null();
        builder.append_value("world");
        builder.append_value(""); // Empty string

        let array = builder.finish();

        assert_eq!(array.len(), 4);
        assert_eq!(array.null_count(), 1);

        assert_eq!(array.value(0), Some("hello"));
        assert_eq!(array.value(1), None);
        assert_eq!(array.value(2), Some("world"));
        assert_eq!(array.value(3), Some(""));
    }

    #[test]
    fn test_string_builder_with_capacity() {
        let mut builder = StringBuilder::with_capacity(100);

        for i in 0..50 {
            if i % 3 == 0 {
                builder.append_null();
            } else {
                builder.append_value(&format!("string_{}", i));
            }
        }

        let array = builder.finish();
        assert_eq!(array.len(), 50);

        let expected_nulls = (0..50).filter(|i| i % 3 == 0).count();
        assert_eq!(array.null_count(), expected_nulls);
    }

    #[test]
    fn test_total_bytes() {
        let strings = vec![
            Some("hello".to_string()), // 5 bytes
            Some("world".to_string()), // 5 bytes
            None,                      // 0 bytes
            Some("test".to_string()),  // 4 bytes
        ];
        let array = StringArray::new(strings);

        assert_eq!(array.total_bytes(), 14); // 5 + 5 + 0 + 4
    }

    #[test]
    fn test_utf8_validation() {
        // Valid UTF-8
        let data = "hello world".as_bytes();
        let offsets = vec![0, 5, 11];
        assert!(StringArray::validate_utf8(data, &offsets).is_ok());

        // Invalid offsets
        let invalid_offsets = vec![0, 20]; // Beyond data length
        assert!(StringArray::validate_utf8(data, &invalid_offsets).is_err());
    }

    #[test]
    fn test_as_any_downcast() {
        let array: ArrayRef = Arc::new(StringArray::new(vec![Some("test".to_string())]));

        // Successful downcast
        let string_array = array.as_any().downcast_ref::<StringArray>();
        assert!(string_array.is_some());

        // Failed downcast to different type would fail
        // (Cannot test easily without importing PrimitiveArray here)
    }

    #[test]
    fn test_large_string_array_performance() {
        let size = 1000;
        let strings: Vec<Option<String>> = (0..size)
            .map(|i| {
                if i % 7 == 0 {
                    None
                } else {
                    Some(format!("string_value_{}", i))
                }
            })
            .collect();

        let array = StringArray::new(strings.clone());
        let expected_nulls = strings.iter().filter(|s| s.is_none()).count();

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
}
