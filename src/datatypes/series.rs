use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Index;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AnyValue {
    Null,
    Int64(i64),
    Float64(f64),
    String(String),
    Boolean(bool),
}

impl AnyValue {
    pub fn is_null(&self) -> bool {
        self.data_type() == DataType::Null
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Self::Null => DataType::Null,
            Self::Int64(_) => DataType::Int64,
            Self::Float64(_) => DataType::Float64,
            Self::String(_) => DataType::String,
            Self::Boolean(_) => DataType::Boolean,
        }
    }
}

impl From<i64> for AnyValue {
    fn from(item: i64) -> Self {
        AnyValue::Int64(item)
    }
}

impl From<f64> for AnyValue {
    fn from(item: f64) -> Self {
        AnyValue::Float64(item)
    }
}

impl From<String> for AnyValue {
    fn from(item: String) -> Self {
        AnyValue::String(item)
    }
}

impl From<&str> for AnyValue {
    fn from(item: &str) -> Self {
        AnyValue::String(item.to_string())
    }
}

impl From<bool> for AnyValue {
    fn from(item: bool) -> Self {
        AnyValue::Boolean(item)
    }
}

impl fmt::Display for AnyValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Null => write!(f, "null"),
            Self::Int64(v) => write!(f, "{}", v),
            Self::Float64(v) => write!(f, "{}", v),
            Self::String(v) => write!(f, "{}", v),
            Self::Boolean(v) => write!(f, "{}", v),
        }
    }
}

impl Hash for AnyValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            AnyValue::Null => 0.hash(state),
            AnyValue::Int64(v) => v.hash(state),
            AnyValue::Float64(v) => v.to_bits().hash(state),
            AnyValue::String(v) => v.hash(state),
            AnyValue::Boolean(v) => v.hash(state),
        }
    }
}

impl Eq for AnyValue {}

impl PartialEq for AnyValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AnyValue::Null, AnyValue::Null) => true,
            (AnyValue::Int64(a), AnyValue::Int64(b)) => a == b,
            (AnyValue::Float64(a), AnyValue::Float64(b)) => a == b,
            (AnyValue::String(a), AnyValue::String(b)) => a == b,
            (AnyValue::Boolean(a), AnyValue::Boolean(b)) => a == b,
            _ => false,
        }
    }
}

impl PartialOrd for AnyValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        match (self, other) {
            (AnyValue::Null, AnyValue::Null) => Some(Ordering::Equal),
            (AnyValue::Null, _) => Some(Ordering::Less),
            (_, AnyValue::Null) => Some(Ordering::Greater),

            (AnyValue::Int64(a), AnyValue::Int64(b)) => a.partial_cmp(b),
            (AnyValue::Float64(a), AnyValue::Float64(b)) => a.partial_cmp(b),
            (AnyValue::String(a), AnyValue::String(b)) => a.partial_cmp(b),
            (AnyValue::Boolean(a), AnyValue::Boolean(b)) => a.partial_cmp(b),

            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Series {
    name: String,
    data: Vec<AnyValue>,
    dtype: DataType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Int64,
    Float64,
    String,
    Boolean,
    Null,
}

impl DataType {
    pub fn is_numeric(&self) -> bool {
        match self {
            Self::String => false,
            Self::Boolean => false,
            Self::Null => false,
            _ => true,
        }
    }
    pub fn is_comparable_with(&self, other: &DataType) -> bool {
        if self == other {
            return true;
        }

        match (self, other) {
            (DataType::Int64, DataType::Float64)
            | (DataType::Float64, DataType::Int64)
            | (DataType::Null, _) => true,
            (DataType::Int64, DataType::Null) => true,
            (DataType::Float64, DataType::Null) => true,
            (DataType::String, DataType::Null) => true,
            (DataType::Boolean, DataType::Null) => true,
            _ => false,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Null => write!(f, "Null"),
            Self::Int64 => write!(f, "Int64"),
            Self::Float64 => write!(f, "Float64"),
            Self::String => write!(f, "String"),
            Self::Boolean => write!(f, "Boolean"),
        }
    }
}

#[derive(Debug, Error)]
pub enum SeriesError {
    #[error("Mixed types in series: expected {expected:?}, found {found:?}")]
    MixedTypes { expected: DataType, found: DataType },
    #[error("Empty series not allowed")]
    EmptyData,
    #[error("Index {index} out of bounds for series of length {length}")]
    OutOfBounds { index: usize, length: usize },
}

impl Series {
    pub fn new(name: &str, data: Vec<AnyValue>) -> Result<Self, SeriesError> {
        if data.is_empty() {
            return Err(SeriesError::EmptyData);
        }

        let mut dtype = None;
        for value in &data {
            if !value.is_null() {
                dtype = Some(value.data_type());
                break;
            }
        }

        let mut dtype = dtype.unwrap_or(DataType::Null);

        for value in &data {
            if !value.is_null() {
                let current_type = value.data_type();
                if !Self::are_types_compatible(&dtype, &current_type) {
                    return Err(SeriesError::MixedTypes {
                        expected: dtype,
                        found: current_type,
                    });
                }

                if dtype == DataType::Int64 && current_type == DataType::Float64 {
                    dtype = DataType::Float64;
                }
            }
        }

        Ok(Series {
            name: name.to_string(),
            data,
            dtype,
        })
    }

    pub fn empty(name: &str, dtype: DataType) -> Self {
        Series {
            name: name.to_string(),
            data: vec![],
            dtype,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    pub fn get(&self, index: usize) -> Option<&AnyValue> {
        self.data.get(index)
    }

    pub fn iter(&self) -> std::slice::Iter<AnyValue> {
        self.data.iter()
    }

    fn are_types_compatible(expected: &DataType, found: &DataType) -> bool {
        if expected == found {
            return true;
        }

        match (expected, found) {
            (DataType::Int64, DataType::Float64) | (DataType::Float64, DataType::Int64) => true,
            _ => false,
        }
    }
}

impl fmt::Display for Series {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Series: numbers [{}; {}]", self.dtype(), self.len())
    }
}

impl Index<usize> for Series {
    type Output = AnyValue;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len() {
            panic!(
                "{}",
                SeriesError::OutOfBounds {
                    index,
                    length: self.len()
                }
            );
        }
        &self.data[index]
    }
}

// Generated tests
#[cfg(test)]
mod tests {
    use super::*;

    // ============ AnyValue Tests ============

    #[test]
    fn test_anyvalue_creation_and_types() {
        assert_eq!(AnyValue::Null.data_type(), DataType::Null);
        assert_eq!(AnyValue::Int64(42).data_type(), DataType::Int64);
        assert_eq!(AnyValue::Float64(3.14).data_type(), DataType::Float64);
        assert_eq!(
            AnyValue::String("hello".to_string()).data_type(),
            DataType::String
        );
        assert_eq!(AnyValue::Boolean(true).data_type(), DataType::Boolean);
    }

    #[test]
    fn test_anyvalue_is_null() {
        assert!(AnyValue::Null.is_null());
        assert!(!AnyValue::Int64(42).is_null());
        assert!(!AnyValue::Float64(3.14).is_null());
        assert!(!AnyValue::String("hello".to_string()).is_null());
        assert!(!AnyValue::Boolean(false).is_null());
    }

    #[test]
    fn test_anyvalue_from_implementations() {
        let v1: AnyValue = 42i64.into();
        assert_eq!(v1, AnyValue::Int64(42));

        let v2: AnyValue = 3.14f64.into();
        assert_eq!(v2, AnyValue::Float64(3.14));

        let v3: AnyValue = "hello".into();
        assert_eq!(v3, AnyValue::String("hello".to_string()));

        let v4: AnyValue = String::from("world").into();
        assert_eq!(v4, AnyValue::String("world".to_string()));

        let v5: AnyValue = true.into();
        assert_eq!(v5, AnyValue::Boolean(true));
    }

    #[test]
    fn test_anyvalue_display() {
        assert_eq!(format!("{}", AnyValue::Null), "null");
        assert_eq!(format!("{}", AnyValue::Int64(42)), "42");
        assert_eq!(format!("{}", AnyValue::Float64(3.14)), "3.14");
        assert_eq!(
            format!("{}", AnyValue::String("hello".to_string())),
            "hello"
        );
        assert_eq!(format!("{}", AnyValue::Boolean(true)), "true");
        assert_eq!(format!("{}", AnyValue::Boolean(false)), "false");
    }

    #[test]
    fn test_anyvalue_partial_ord() {
        // Same types should compare correctly
        assert!(AnyValue::Int64(1) < AnyValue::Int64(2));
        assert!(AnyValue::Float64(1.0) < AnyValue::Float64(2.0));
        assert!(AnyValue::String("a".to_string()) < AnyValue::String("b".to_string()));
        assert!(AnyValue::Boolean(false) < AnyValue::Boolean(true));

        // Null should be less than everything else
        assert!(AnyValue::Null < AnyValue::Int64(0));
        assert!(AnyValue::Null < AnyValue::Boolean(false));

        // Different types should not be comparable (return None)
        assert_eq!(
            AnyValue::Int64(1).partial_cmp(&AnyValue::String("1".to_string())),
            None
        );
    }

    // ============ DataType Tests ============

    #[test]
    fn test_datatype_is_numeric() {
        assert!(!DataType::Null.is_numeric());
        assert!(!DataType::Boolean.is_numeric());
        assert!(DataType::Int64.is_numeric());
        assert!(DataType::Float64.is_numeric());
        assert!(!DataType::String.is_numeric());
    }

    #[test]
    fn test_datatype_is_comparable_with() {
        // Same types are always comparable
        assert!(DataType::Int64.is_comparable_with(&DataType::Int64));
        assert!(DataType::String.is_comparable_with(&DataType::String));

        // Numeric types are comparable with each other
        assert!(DataType::Int64.is_comparable_with(&DataType::Float64));
        assert!(DataType::Float64.is_comparable_with(&DataType::Int64));

        // Null is comparable with everything
        assert!(DataType::Null.is_comparable_with(&DataType::Int64));
        assert!(DataType::String.is_comparable_with(&DataType::Null));

        // Non-numeric different types are not comparable
        assert!(!DataType::String.is_comparable_with(&DataType::Boolean));
        assert!(!DataType::Boolean.is_comparable_with(&DataType::Int64));
    }

    // ============ Series Tests ============

    #[test]
    fn test_series_creation_homogeneous_int() {
        let data = vec![AnyValue::Int64(1), AnyValue::Int64(2), AnyValue::Int64(3)];
        let series = Series::new("numbers", data).unwrap();

        assert_eq!(series.name(), "numbers");
        assert_eq!(series.len(), 3);
        assert_eq!(series.dtype(), &DataType::Int64);
        assert!(!series.is_empty());
    }

    #[test]
    fn test_series_creation_homogeneous_string() {
        let data = vec![
            AnyValue::String("a".to_string()),
            AnyValue::String("b".to_string()),
            AnyValue::String("c".to_string()),
        ];
        let series = Series::new("letters", data).unwrap();

        assert_eq!(series.name(), "letters");
        assert_eq!(series.len(), 3);
        assert_eq!(series.dtype(), &DataType::String);
    }

    #[test]
    fn test_series_creation_with_nulls() {
        let data = vec![AnyValue::Int64(1), AnyValue::Null, AnyValue::Int64(3)];
        let series = Series::new("with_nulls", data).unwrap();

        // Dtype should be determined by non-null values
        assert_eq!(series.dtype(), &DataType::Int64);
        assert_eq!(series.len(), 3);
    }

    #[test]
    fn test_series_creation_all_nulls() {
        let data = vec![AnyValue::Null, AnyValue::Null];
        let series = Series::new("nulls", data).unwrap();

        assert_eq!(series.dtype(), &DataType::Null);
        assert_eq!(series.len(), 2);
    }

    #[test]
    fn test_series_creation_empty_fails() {
        let data = vec![];
        let result = Series::new("empty", data);

        assert!(result.is_err());
        match result.unwrap_err() {
            SeriesError::EmptyData => {}
            _ => panic!("Expected EmptyData error"),
        }
    }

    #[test]
    fn test_series_creation_mixed_types_fails() {
        let data = vec![AnyValue::Int64(1), AnyValue::String("hello".to_string())];
        let result = Series::new("mixed", data);

        assert!(result.is_err());
        match result.unwrap_err() {
            SeriesError::MixedTypes { expected, found } => {
                assert_eq!(expected, DataType::Int64);
                assert_eq!(found, DataType::String);
            }
            _ => panic!("Expected MixedTypes error"),
        }
    }

    #[test]
    fn test_series_get_access() {
        let data = vec![AnyValue::Int64(1), AnyValue::Int64(2), AnyValue::Int64(3)];
        let series = Series::new("numbers", data).unwrap();

        assert_eq!(series.get(0), Some(&AnyValue::Int64(1)));
        assert_eq!(series.get(1), Some(&AnyValue::Int64(2)));
        assert_eq!(series.get(2), Some(&AnyValue::Int64(3)));
        assert_eq!(series.get(3), None);
    }

    #[test]
    fn test_series_index_trait() {
        let data = vec![AnyValue::Int64(1), AnyValue::Int64(2), AnyValue::Int64(3)];
        let series = Series::new("numbers", data).unwrap();

        assert_eq!(series[0], AnyValue::Int64(1));
        assert_eq!(series[1], AnyValue::Int64(2));
        assert_eq!(series[2], AnyValue::Int64(3));
    }

    #[test]
    #[should_panic(expected = "Index 3 out of bounds for series of length 3")]
    fn test_series_index_out_of_bounds_panics() {
        let data = vec![AnyValue::Int64(1), AnyValue::Int64(2), AnyValue::Int64(3)];
        let series = Series::new("numbers", data).unwrap();
        let _ = series[3]; // Should panic
    }

    #[test]
    fn test_series_iter() {
        let data = vec![AnyValue::Int64(1), AnyValue::Int64(2), AnyValue::Int64(3)];
        let series = Series::new("numbers", data.clone()).unwrap();

        let collected: Vec<&AnyValue> = series.iter().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], &AnyValue::Int64(1));
        assert_eq!(collected[1], &AnyValue::Int64(2));
        assert_eq!(collected[2], &AnyValue::Int64(3));
    }

    #[test]
    fn test_series_display() {
        let data = vec![AnyValue::Int64(1), AnyValue::Int64(2), AnyValue::Int64(3)];
        let series = Series::new("numbers", data).unwrap();

        let display_str = format!("{}", series);
        assert_eq!(display_str, "Series: numbers [Int64; 3]");
    }

    #[test]
    fn test_series_numeric_compatibility() {
        // Int64 and Float64 should be compatible (both numeric)
        let data = vec![
            AnyValue::Int64(1),
            AnyValue::Float64(2.5),
            AnyValue::Int64(3),
        ];
        let series = Series::new("mixed_numeric", data).unwrap();

        // Should default to Float64 when mixed numeric types are present
        assert_eq!(series.dtype(), &DataType::Float64);
        assert_eq!(series.len(), 3);
    }
}
