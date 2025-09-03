use crate::datatypes::series::{Series, SeriesError};
use std::collections::HashSet;
use std::fmt;
use std::ops::Index;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct DataFrame {
    columns: Vec<Series>,
}

#[derive(Debug, Error)]
pub enum DataFrameError {
    #[error("Column lengths mismatch: expected {expected}, found {found} for column '{column}'")]
    LengthMismatch {
        expected: usize,
        found: usize,
        column: String,
    },
    #[error("Duplicate column name: '{name}'")]
    DuplicateColumn { name: String },
    #[error("Column not found: '{name}'")]
    ColumnNotFound { name: String },
    #[error("Series error: {0}")]
    SeriesError(#[from] SeriesError),
}

impl DataFrame {
    pub fn new(columns: Vec<Series>) -> Result<Self, DataFrameError> {
        if columns.is_empty() {
            return Ok(DataFrame {
                columns: Vec::new(),
            });
        }

        let mut seen = HashSet::new();
        if let Some(dup) = columns.iter().find(|c| !seen.insert(c.name())) {
            return Err(DataFrameError::DuplicateColumn {
                name: dup.name().to_string(),
            });
        }

        if let Some(first) = columns.first() {
            let expected = first.len();

            if let Some(col) = columns.iter().find(|c| c.len() != expected) {
                return Err(DataFrameError::LengthMismatch {
                    expected,
                    found: col.len(),
                    column: col.name().to_string(),
                });
            }
        }

        Ok(DataFrame { columns })
    }

    pub fn empty() -> Self {
        DataFrame {
            columns: Vec::new(),
        }
    }

    pub fn height(&self) -> usize {
        if self.columns.is_empty() {
            return 0;
        }

        self.columns[0].len()
    }

    pub fn width(&self) -> usize {
        self.columns.len()
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.height(), self.width())
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    pub fn column(&self, name: &str) -> Option<&Series> {
        self.columns.iter().find(|s| s.name() == name)
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|s| s.name()).collect()
    }

    pub fn columns(&self) -> &[Series] {
        self.columns.as_slice()
    }

    pub fn select(&self, column_names: &[&str]) -> Result<DataFrame, DataFrameError> {
        let mut columns = Vec::with_capacity(column_names.len());
        for name in column_names {
            if let Some(s) = self.columns.iter().find(|s| s.name() == *name) {
                columns.push(s.clone())
            } else {
                return Err(DataFrameError::ColumnNotFound {
                    name: name.to_string(),
                });
            }
        }

        Ok(DataFrame { columns })
    }
}

impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            write!(f, "empty DataFrame")?;
        }

        let df_to_print = self.columns.iter().take(10).collect::<Vec<&Series>>();

        for s in df_to_print {
            write!(f, "[{}]", s.name())?;
            for col in s.iter() {
                write!(f, "[{}]", col)?;
            }
            write!(f, "\n")?;
        }

        if self.height() > 10 {
            write!(f, "...")?;
        }

        Ok(())
    }
}

impl Index<&str> for DataFrame {
    type Output = Series;

    fn index(&self, column_name: &str) -> &Self::Output {
        self.column(column_name).unwrap_or_else(|| {
            panic!(
                "{}",
                DataFrameError::ColumnNotFound {
                    name: column_name.to_string()
                }
            )
        })
    }
}

// Generated tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::series::{AnyValue, Series};

    // Helper function to create test series
    fn create_test_series() -> (Series, Series, Series) {
        let names = Series::new(
            "name",
            vec![
                AnyValue::String("Alice".to_string()),
                AnyValue::String("Bob".to_string()),
                AnyValue::String("Charlie".to_string()),
            ],
        )
        .unwrap();

        let ages = Series::new(
            "age",
            vec![
                AnyValue::Int64(25),
                AnyValue::Int64(30),
                AnyValue::Int64(35),
            ],
        )
        .unwrap();

        let scores = Series::new(
            "score",
            vec![
                AnyValue::Float64(85.5),
                AnyValue::Float64(92.0),
                AnyValue::Float64(78.5),
            ],
        )
        .unwrap();

        (names, ages, scores)
    }

    // ============ DataFrame Creation Tests ============

    #[test]
    fn test_dataframe_creation_success() {
        let (names, ages, scores) = create_test_series();
        let df = DataFrame::new(vec![names, ages, scores]).unwrap();

        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 3);
        assert_eq!(df.shape(), (3, 3));
        assert!(!df.is_empty());
    }

    #[test]
    fn test_dataframe_empty_creation() {
        let df = DataFrame::empty();

        assert_eq!(df.height(), 0);
        assert_eq!(df.width(), 0);
        assert_eq!(df.shape(), (0, 0));
        assert!(df.is_empty());
    }

    #[test]
    fn test_dataframe_creation_empty_columns_list() {
        let df = DataFrame::new(vec![]).unwrap();

        assert_eq!(df.height(), 0);
        assert_eq!(df.width(), 0);
        assert!(df.is_empty());
    }

    #[test]
    fn test_dataframe_creation_single_column() {
        let names = Series::new(
            "name",
            vec![
                AnyValue::String("Alice".to_string()),
                AnyValue::String("Bob".to_string()),
            ],
        )
        .unwrap();

        let df = DataFrame::new(vec![names]).unwrap();

        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 1);
        assert_eq!(df.shape(), (2, 1));
    }

    #[test]
    fn test_dataframe_creation_length_mismatch_fails() {
        let names = Series::new(
            "name",
            vec![
                AnyValue::String("Alice".to_string()),
                AnyValue::String("Bob".to_string()),
            ],
        )
        .unwrap();

        let ages = Series::new(
            "age",
            vec![
                AnyValue::Int64(25),
                AnyValue::Int64(30),
                AnyValue::Int64(35), // Extra element!
            ],
        )
        .unwrap();

        let result = DataFrame::new(vec![names, ages]);

        assert!(result.is_err());
        match result.unwrap_err() {
            DataFrameError::LengthMismatch {
                expected,
                found,
                column,
            } => {
                assert_eq!(expected, 2);
                assert_eq!(found, 3);
                assert_eq!(column, "age");
            }
            _ => panic!("Expected LengthMismatch error"),
        }
    }

    #[test]
    fn test_dataframe_creation_duplicate_column_names_fails() {
        let names1 = Series::new(
            "name",
            vec![
                AnyValue::String("Alice".to_string()),
                AnyValue::String("Bob".to_string()),
            ],
        )
        .unwrap();

        let names2 = Series::new(
            "name",
            vec![
                AnyValue::String("Charlie".to_string()),
                AnyValue::String("David".to_string()),
            ],
        )
        .unwrap();

        let result = DataFrame::new(vec![names1, names2]);

        assert!(result.is_err());
        match result.unwrap_err() {
            DataFrameError::DuplicateColumn { name } => {
                assert_eq!(name, "name");
            }
            _ => panic!("Expected DuplicateColumn error"),
        }
    }

    // ============ Column Access Tests ============

    #[test]
    fn test_dataframe_column_access() {
        let (names, ages, scores) = create_test_series();
        let df = DataFrame::new(vec![names, ages, scores]).unwrap();

        let name_col = df.column("name").unwrap();
        assert_eq!(name_col.name(), "name");
        assert_eq!(name_col.len(), 3);

        let age_col = df.column("age").unwrap();
        assert_eq!(age_col.name(), "age");
        assert_eq!(age_col.len(), 3);

        assert!(df.column("nonexistent").is_none());
    }

    #[test]
    fn test_dataframe_column_names() {
        let (names, ages, scores) = create_test_series();
        let df = DataFrame::new(vec![names, ages, scores]).unwrap();

        let col_names = df.column_names();
        assert_eq!(col_names, vec!["name", "age", "score"]);
    }

    #[test]
    fn test_dataframe_columns_access() {
        let (names, ages, scores) = create_test_series();
        let df = DataFrame::new(vec![names, ages, scores]).unwrap();

        let columns = df.columns();
        assert_eq!(columns.len(), 3);
        assert_eq!(columns[0].name(), "name");
        assert_eq!(columns[1].name(), "age");
        assert_eq!(columns[2].name(), "score");
    }

    #[test]
    fn test_dataframe_index_trait() {
        let (names, ages, scores) = create_test_series();
        let df = DataFrame::new(vec![names, ages, scores]).unwrap();

        let name_col = &df["name"];
        assert_eq!(name_col.name(), "name");
        assert_eq!(name_col.len(), 3);

        let age_col = &df["age"];
        assert_eq!(age_col.name(), "age");
    }

    #[test]
    #[should_panic(expected = "Column not found: 'nonexistent'")]
    fn test_dataframe_index_nonexistent_panics() {
        let (names, ages, scores) = create_test_series();
        let df = DataFrame::new(vec![names, ages, scores]).unwrap();

        let _ = &df["nonexistent"]; // Should panic
    }

    // ============ Select Operation Tests ============

    #[test]
    fn test_dataframe_select_success() {
        let (names, ages, scores) = create_test_series();
        let df = DataFrame::new(vec![names, ages, scores]).unwrap();

        let selected = df.select(&["name", "score"]).unwrap();

        assert_eq!(selected.width(), 2);
        assert_eq!(selected.height(), 3);

        let col_names = selected.column_names();
        assert_eq!(col_names, vec!["name", "score"]);

        // Verify order is preserved
        assert_eq!(selected.columns()[0].name(), "name");
        assert_eq!(selected.columns()[1].name(), "score");
    }

    #[test]
    fn test_dataframe_select_single_column() {
        let (names, ages, scores) = create_test_series();
        let df = DataFrame::new(vec![names, ages, scores]).unwrap();

        let selected = df.select(&["age"]).unwrap();

        assert_eq!(selected.width(), 1);
        assert_eq!(selected.height(), 3);
        assert_eq!(selected.column_names(), vec!["age"]);
    }

    #[test]
    fn test_dataframe_select_reorder_columns() {
        let (names, ages, scores) = create_test_series();
        let df = DataFrame::new(vec![names, ages, scores]).unwrap();

        // Select in different order
        let selected = df.select(&["score", "name", "age"]).unwrap();

        assert_eq!(selected.width(), 3);
        let col_names = selected.column_names();
        assert_eq!(col_names, vec!["score", "name", "age"]);
    }

    #[test]
    fn test_dataframe_select_nonexistent_column_fails() {
        let (names, ages, scores) = create_test_series();
        let df = DataFrame::new(vec![names, ages, scores]).unwrap();

        let result = df.select(&["name", "nonexistent"]);

        assert!(result.is_err());
        match result.unwrap_err() {
            DataFrameError::ColumnNotFound { name } => {
                assert_eq!(name, "nonexistent");
            }
            _ => panic!("Expected ColumnNotFound error"),
        }
    }

    #[test]
    fn test_dataframe_select_empty_list() {
        let (names, ages, scores) = create_test_series();
        let df = DataFrame::new(vec![names, ages, scores]).unwrap();

        let selected = df.select(&[]).unwrap();

        assert_eq!(selected.width(), 0);
        assert_eq!(selected.height(), 0);
        assert!(selected.column_names().is_empty());
    }

    #[test]
    fn test_dataframe_select_duplicate_column_names() {
        let (names, ages, scores) = create_test_series();
        let df = DataFrame::new(vec![names, ages, scores]).unwrap();

        // Selecting same column twice - should appear twice in result
        let selected = df.select(&["name", "age", "name"]).unwrap();

        assert_eq!(selected.width(), 3);
        let col_names = selected.column_names();
        assert_eq!(col_names, vec!["name", "age", "name"]);
    }

    // ============ Display Tests ============

    #[test]
    fn test_dataframe_display_small() {
        let names = Series::new(
            "name",
            vec![
                AnyValue::String("Alice".to_string()),
                AnyValue::String("Bob".to_string()),
            ],
        )
        .unwrap();

        let ages = Series::new("age", vec![AnyValue::Int64(25), AnyValue::Int64(30)]).unwrap();

        let df = DataFrame::new(vec![names, ages]).unwrap();
        let display_str = format!("{}", df);

        // Should contain column headers and data
        assert!(display_str.contains("name"));
        assert!(display_str.contains("age"));
        assert!(display_str.contains("Alice"));
        assert!(display_str.contains("Bob"));
        assert!(display_str.contains("25"));
        assert!(display_str.contains("30"));
    }

    #[test]
    fn test_dataframe_display_empty() {
        let df = DataFrame::empty();
        let display_str = format!("{}", df);

        assert!(display_str.contains("empty DataFrame"));
    }

    #[test]
    fn test_dataframe_display_truncation() {
        // Create DataFrame with more than 10 rows to test truncation
        let mut values = Vec::new();
        for i in 0..15 {
            values.push(AnyValue::Int64(i));
        }

        let numbers = Series::new("number", values).unwrap();
        let df = DataFrame::new(vec![numbers]).unwrap();
        let display_str = format!("{}", df);

        // Should show first few rows and indicate truncation
        assert!(display_str.contains("...") || display_str.contains("truncated"));
    }

    // ============ Edge Cases Tests ============

    #[test]
    fn test_dataframe_with_all_null_column() {
        let nulls = Series::new(
            "nulls",
            vec![AnyValue::Null, AnyValue::Null, AnyValue::Null],
        )
        .unwrap();

        let numbers = Series::new(
            "numbers",
            vec![AnyValue::Int64(1), AnyValue::Int64(2), AnyValue::Int64(3)],
        )
        .unwrap();

        let df = DataFrame::new(vec![nulls, numbers]).unwrap();

        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 2);

        let null_col = df.column("nulls").unwrap();
        assert_eq!(null_col.dtype(), &crate::datatypes::series::DataType::Null);
    }

    #[test]
    fn test_dataframe_with_mixed_numeric_column() {
        let mixed = Series::new(
            "mixed",
            vec![
                AnyValue::Int64(1),
                AnyValue::Float64(2.5),
                AnyValue::Int64(3),
            ],
        )
        .unwrap();

        let df = DataFrame::new(vec![mixed]).unwrap();

        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 1);

        let mixed_col = df.column("mixed").unwrap();
        assert_eq!(
            mixed_col.dtype(),
            &crate::datatypes::series::DataType::Float64
        );
    }
}
