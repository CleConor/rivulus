use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct BitMap {
    buffer: Arc<[u8]>,
    bit_count: usize,
    offset: usize,
}

impl BitMap {
    pub fn new(bit_count: usize) -> Self {
        let byte_count = (bit_count + 7) / 8;
        let buffer = vec![0u8; byte_count];
        Self {
            buffer: Arc::from(buffer),
            bit_count,
            offset: 0,
        }
    }

    pub fn all_true(bit_count: usize) -> Self {
        let byte_count = (bit_count + 7) / 8;
        let mut buffer = vec![0xFFu8; byte_count];

        if bit_count % 8 != 0 {
            let last_byte_bits = bit_count % 8;
            let mask = (1u8 << last_byte_bits) - 1;
            if let Some(last_byte) = buffer.last_mut() {
                *last_byte = mask;
            }
        }

        Self {
            buffer: Arc::from(buffer),
            bit_count,
            offset: 0,
        }
    }

    pub fn all_false(bit_count: usize) -> Self {
        Self::new(bit_count)
    }

    pub fn from_bool_slice(values: &[bool]) -> Self {
        let byte_count = (values.len() + 7) / 8;
        let mut buffer = vec![0u8; byte_count];

        for i in 0..values.len() {
            let byte_pos = i / 8;
            let bit_pos = i % 8;
            buffer[byte_pos] |= (values[i] as u8) << bit_pos;
        }

        Self {
            buffer: Arc::from(buffer),
            bit_count: values.len(),
            offset: 0,
        }
    }

    pub fn get_bit(&self, index: usize) -> bool {
        assert!(index < self.bit_count);

        let i = (index + self.offset) / 8;
        let j = (index + self.offset) % 8;

        (self.buffer[i] >> j & 1) != 0
    }

    pub fn bit_count(&self) -> usize {
        self.bit_count
    }

    fn count(&self, value: u8, offset: usize, length: usize) -> usize {
        let mut count: usize = 0;

        for i in offset..offset + length {
            let byte_pos = i / 8;
            let bit_pos = i % 8;
            if ((self.buffer[byte_pos] >> bit_pos) & 1) == value {
                count += 1;
            }
        }

        count
    }

    pub fn count_ones(&self) -> usize {
        self.count(1, self.offset, self.bit_count)
    }

    pub fn count_zeros(&self) -> usize {
        self.count(0, self.offset, self.bit_count)
    }

    pub fn count_ones_range(&self, offset: usize, length: usize) -> usize {
        self.count(1, offset, length)
    }

    pub fn count_zeros_range(&self, offset: usize, length: usize) -> usize {
        self.count(0, offset, length)
    }

    pub fn slice(&self, offset: usize, length: usize) -> Self {
        assert!(offset + length <= self.bit_count);

        Self {
            buffer: self.buffer.clone(),
            bit_count: length,
            offset: offset + self.offset,
        }
    }
}

pub struct BitmapBuilder {
    buffer: Vec<u8>,
    bit_count: usize,
    current_byte: u8,
    current_bit_pos: usize,
}

impl BitmapBuilder {
    pub fn new() -> Self {
        BitmapBuilder {
            buffer: Vec::new(),
            bit_count: 0,
            current_byte: 0,
            current_bit_pos: 0,
        }
    }

    pub fn with_capacity(capacity_bits: usize) -> Self {
        let byte_capacity = (capacity_bits + 7) / 8;
        BitmapBuilder {
            buffer: Vec::with_capacity(byte_capacity),
            bit_count: 0,
            current_byte: 0,
            current_bit_pos: 0,
        }
    }

    pub fn append(&mut self, value: bool) {
        if value {
            self.current_byte |= 1 << self.current_bit_pos;
        }

        self.current_bit_pos += 1;
        self.bit_count += 1;

        if self.current_bit_pos == 8 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.current_bit_pos = 0;
        }
    }

    pub fn has_nulls(&self) -> bool {
        if self.bit_count == 0 {
            return false;
        }

        for &byte in &self.buffer {
            if byte != 0xFF {
                return true;
            }
        }

        if self.current_bit_pos > 0 {
            let expected_mask = (1u8 << self.current_bit_pos) - 1;
            if self.current_byte != expected_mask {
                return true;
            }
        }

        false
    }

    pub fn finish(mut self) -> BitMap {
        if self.current_bit_pos > 0 {
            self.buffer.push(self.current_byte);
        }

        BitMap {
            buffer: Arc::from(self.buffer),
            bit_count: self.bit_count,
            offset: 0,
        }
    }
}

impl Default for BitmapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

//generated tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn ones_in(values: &[bool]) -> usize {
        values.iter().copied().filter(|&b| b).count()
    }

    fn pattern(len: usize) -> Vec<bool> {
        // pattern non banale, non periodico su 8
        (0..len).map(|i| (i % 3 == 0) ^ (i % 5 == 0)).collect()
    }

    #[test]
    fn new_is_zeroed_and_sizes_ok() {
        let n = 100;
        let bm = BitMap::new(n);
        // proprietà di base
        assert_eq!(bm.bit_count, n);
        assert_eq!(bm.offset, 0);
        // buffer in byte deve essere ceil(n/8)
        let expected_bytes = (n + 7) / 8;
        assert_eq!(bm.buffer.len(), expected_bytes);
        // tutti i bit a 0
        for i in 0..n {
            assert!(!bm.get_bit(i), "bit {} dovrebbe essere 0", i);
        }
        // count_ones deve essere O(1) e restituire 0
        assert_eq!(bm.count_ones(), 0);
    }

    #[test]
    fn from_bool_slice_roundtrip_unaligned() {
        let vals = pattern(37); // non multiplo di 8
        let bm = BitMap::from_bool_slice(&vals);
        assert_eq!(bm.bit_count, vals.len());
        assert_eq!(bm.offset, 0);

        for i in 0..vals.len() {
            assert_eq!(bm.get_bit(i), vals[i], "mismatch al bit {}", i);
        }
        assert_eq!(bm.count_ones(), ones_in(&vals));
    }

    #[test]
    fn from_bool_slice_all_true() {
        let n = 65; // attraversa due byte boundary
        let vals = vec![true; n];
        let bm = BitMap::from_bool_slice(&vals);
        for i in 0..n {
            assert!(bm.get_bit(i));
        }
        assert_eq!(bm.count_ones(), n);
    }

    #[test]
    fn slice_is_view_and_counts_match() {
        let vals = pattern(64);
        let bm = BitMap::from_bool_slice(&vals);

        let off = 7;
        let len = 25;
        let s = bm.slice(off, len);

        // la slice è una vista: condivide il buffer e aggiorna offset/len
        assert!(
            Arc::ptr_eq(&bm.buffer, &s.buffer),
            "la slice dovrebbe condividere il buffer"
        );
        assert_eq!(s.offset, bm.offset + off);
        assert_eq!(s.bit_count, len);

        // get_bit nella slice è relativo alla vista
        for i in 0..len {
            assert_eq!(
                s.get_bit(i),
                vals[off + i],
                "mismatch nella slice al bit {}",
                i
            );
        }
        assert_eq!(s.count_ones(), ones_in(&vals[off..off + len]));
    }

    #[test]
    fn chained_slice_equivalence() {
        let vals = pattern(91);
        let bm = BitMap::from_bool_slice(&vals);

        let a = 10;
        let b = 50;
        let c = 7;
        let d = 20;

        let s1 = bm.slice(a, b).slice(c, d);
        let s2 = bm.slice(a + c, d);

        assert_eq!(s1.bit_count, s2.bit_count);
        assert_eq!(s1.offset, s2.offset);
        assert!(Arc::ptr_eq(&s1.buffer, &s2.buffer));

        for i in 0..d {
            assert_eq!(
                s1.get_bit(i),
                s2.get_bit(i),
                "chained slice mismatch al bit {}",
                i
            );
        }
        assert_eq!(s1.count_ones(), s2.count_ones());
    }

    #[test]
    fn zero_length_slice() {
        let vals = pattern(13);
        let bm = BitMap::from_bool_slice(&vals);
        let s = bm.slice(5, 0);
        assert_eq!(s.bit_count, 0);
        assert_eq!(s.count_ones(), 0);
        // get_bit(0) dovrebbe panicare perché out-of-bounds
    }

    #[test]
    #[should_panic]
    fn slice_out_of_bounds_should_panic() {
        let bm = BitMap::new(16);
        // offset + length supera bit_count
        let _ = bm.slice(9, 8);
    }

    #[test]
    #[should_panic]
    fn get_bit_out_of_bounds_should_panic() {
        let bm = BitMap::new(10);
        let _ = bm.get_bit(10); // valido: 0..=9
    }

    #[test]
    fn boundaries_around_bytes() {
        // pattern che colpisce boundary 7->8 e 15->16
        let vals = (0..17)
            .map(|i| i == 7 || i == 8 || i == 16)
            .collect::<Vec<_>>();
        let bm = BitMap::from_bool_slice(&vals);

        for i in 0..vals.len() {
            assert_eq!(bm.get_bit(i), vals[i], "boundary mismatch al bit {}", i);
        }
        assert_eq!(bm.count_ones(), 3);

        // slice che attraversa due boundary
        let s = bm.slice(6, 6); // copre 6..11 (include 7 e 8)
        let expect = &vals[6..12];
        for i in 0..expect.len() {
            assert_eq!(
                s.get_bit(i),
                expect[i],
                "boundary slice mismatch al bit {}",
                i
            );
        }
        assert_eq!(s.count_ones(), ones_in(expect));
    }
}
