use crate::datatypes::{AnyValue, DataFrame, DataType, Series};
use crate::expressions::BinaryOperator;
use crate::logical_plan::JoinType;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug)]
pub enum PhysicalPlan {
    DataFrameSource {
        df: DataFrame,
    },
    Select {
        input: Box<PhysicalPlan>,
        columns: Vec<String>,
    },
    Filter {
        input: Box<PhysicalPlan>,
        column: String,
        value: AnyValue,
        op: BinaryOperator,
    },
    Limit {
        input: Box<PhysicalPlan>,
        n: usize,
    },
    HashJoin {
        build_side: Box<PhysicalPlan>,
        probe_side: Box<PhysicalPlan>,
        build_key: String,
        probe_key: String,
        join_type: JoinType,
    },
}

#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("Column not found: '{name}'")]
    ColumnNotFound { name: String },

    #[error("Series length mismatch: expected {expected}, found {found}")]
    SeriesLengthMismatch { expected: usize, found: usize },

    #[error("Type mismatch: cannot compare {left:?} with {right:?}")]
    TypeMismatch { left: DataType, right: DataType },

    #[error("Invalid operation: {op:?} not supported for types {left:?} and {right:?}")]
    InvalidOperation {
        op: BinaryOperator,
        left: DataType,
        right: DataType,
    },

    #[error("DataFrame error: {0}")]
    DataFrameError(#[from] crate::datatypes::DataFrameError),

    #[error("Series error: {0}")]
    SeriesError(#[from] crate::datatypes::SeriesError),

    #[error("Execution failed: {message}")]
    General { message: String },
}

impl PhysicalPlan {
    pub fn execute(self) -> Result<DataFrame, ExecutionError> {
        match self {
            Self::DataFrameSource { df } => Ok(df),
            Self::Select { input, columns } => {
                let input_df = input.execute()?;

                for col_name in &columns {
                    if input_df.column(col_name).is_none() {
                        return Err(ExecutionError::ColumnNotFound {
                            name: col_name.clone(),
                        });
                    }
                }

                input_df
                    .select(&columns.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                    .map_err(ExecutionError::from)
            }
            Self::Filter {
                input,
                column,
                value,
                op,
            } => {
                let input_df = input.execute()?;

                let filter_series =
                    input_df
                        .column(&column)
                        .ok_or_else(|| ExecutionError::ColumnNotFound {
                            name: column.clone(),
                        })?;

                let mut mask = Vec::with_capacity(input_df.height());
                for row_value in filter_series.iter() {
                    let keep_row = match op {
                        BinaryOperator::Eq => row_value == &value,
                        BinaryOperator::NotEq => row_value != &value,
                        BinaryOperator::Lt => row_value < &value,
                        BinaryOperator::Gt => row_value > &value,
                        BinaryOperator::LtEq => row_value <= &value,
                        BinaryOperator::GtEq => row_value >= &value,
                        _ => {
                            return Err(ExecutionError::InvalidOperation {
                                op,
                                left: filter_series.dtype().clone(),
                                right: value.data_type(),
                            });
                        }
                    };
                    mask.push(keep_row);
                }

                let mut filtered_columns = Vec::with_capacity(input_df.width());
                for series in input_df.columns() {
                    let filtered_data: Vec<AnyValue> = series
                        .iter()
                        .zip(&mask)
                        .filter_map(|(value, &keep)| if keep { Some(value.clone()) } else { None })
                        .collect();

                    let filtered_series = if filtered_data.is_empty() {
                        Series::empty(series.name(), series.dtype().clone())
                    } else {
                        Series::new(series.name(), filtered_data)?
                    };

                    filtered_columns.push(filtered_series);
                }

                Ok(DataFrame::new(filtered_columns)?)
            }
            Self::Limit { input, n } => {
                let input_df = input.execute()?;

                if n == 0 || input_df.is_empty() {
                    let empty_columns: Vec<Series> = input_df
                        .columns()
                        .iter()
                        .map(|s| Series::empty(s.name(), s.dtype().clone()))
                        .collect();
                    return Ok(DataFrame::new(empty_columns)?);
                }

                let limit = n.min(input_df.height());
                let mut limited_columns = Vec::with_capacity(input_df.width());

                for series in input_df.columns() {
                    let limited_data: Vec<AnyValue> = series.iter().take(limit).cloned().collect();

                    limited_columns.push(Series::new(series.name(), limited_data)?);
                }

                Ok(DataFrame::new(limited_columns)?)
            }
            Self::HashJoin {
                build_side,
                probe_side,
                build_key,
                ref probe_key,
                ref join_type,
            } => match join_type {
                JoinType::Inner => {
                    let build_df = build_side.execute()?;
                    let build_key_series = build_df.column(&build_key).unwrap();

                    let mut hash_table: HashMap<AnyValue, Vec<usize>> = HashMap::new();

                    for (row_idx, key_value) in build_key_series.iter().enumerate() {
                        hash_table
                            .entry(key_value.clone())
                            .or_insert_with(Vec::new)
                            .push(row_idx);
                    }

                    let probe_df = probe_side.execute()?;
                    let probe_key_series = probe_df.column(&probe_key).unwrap();

                    let mut result_pairs = Vec::new();
                    for (probe_idx, probe_key_value) in probe_key_series.iter().enumerate() {
                        if let Some(build_indices) = hash_table.get(probe_key_value) {
                            for &build_idx in build_indices {
                                result_pairs.push((probe_idx, build_idx));
                            }
                        }
                    }

                    Self::materialize_join_result(probe_df, build_df, result_pairs, &build_key)
                }
            },
        }
    }

    fn materialize_join_result(
        probe_df: DataFrame,
        build_df: DataFrame,
        pairs: Vec<(usize, usize)>,
        build_key: &str,
    ) -> Result<DataFrame, ExecutionError> {
        if pairs.is_empty() {
            return Ok(Self::create_empty_join_result(
                &probe_df, &build_df, build_key,
            )?);
        }

        let mut result_columns = Vec::new();

        for probe_series in probe_df.columns() {
            let selected_data: Vec<AnyValue> = pairs
                .iter()
                .map(|(probe_idx, _)| probe_series[*probe_idx].clone())
                .collect();
            result_columns.push(Series::new(probe_series.name(), selected_data)?);
        }

        for build_series in build_df.columns() {
            if build_series.name() == build_key {
                continue;
            }

            let selected_data: Vec<AnyValue> = pairs
                .iter()
                .map(|(_, build_idx)| build_series[*build_idx].clone())
                .collect();

            let final_name = if probe_df.column(build_series.name()).is_some() {
                format!("{}_right", build_series.name())
            } else {
                build_series.name().to_string()
            };

            result_columns.push(Series::new(&final_name, selected_data)?);
        }

        DataFrame::new(result_columns).map_err(ExecutionError::from)
    }

    fn create_empty_join_result(
        probe_df: &DataFrame,
        build_df: &DataFrame,
        build_key: &str,
    ) -> Result<DataFrame, ExecutionError> {
        let mut empty_columns = Vec::new();

        for probe_series in probe_df.columns() {
            let empty_series = Series::empty(probe_series.name(), probe_series.dtype().clone());
            empty_columns.push(empty_series);
        }

        for build_series in build_df.columns() {
            if build_series.name() == build_key {
                continue;
            }

            let final_name = if probe_df.column(build_series.name()).is_some() {
                format!("{}_right", build_series.name())
            } else {
                build_series.name().to_string()
            };

            let empty_series = Series::empty(&final_name, build_series.dtype().clone());
            empty_columns.push(empty_series);
        }

        DataFrame::new(empty_columns).map_err(ExecutionError::from)
    }
}

//Generated tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::{AnyValue, DataFrame, Series};
    use crate::expressions::BinaryOperator;

    // Helper function to create test DataFrame
    fn create_test_dataframe() -> DataFrame {
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

        DataFrame::new(vec![names, ages, scores]).unwrap()
    }

    // ============ PhysicalPlan Creation Tests ============

    #[test]
    fn test_dataframe_source_creation() {
        let df = create_test_dataframe();
        let plan = PhysicalPlan::DataFrameSource { df: df.clone() };

        match plan {
            PhysicalPlan::DataFrameSource { df: plan_df } => {
                assert_eq!(plan_df.width(), df.width());
                assert_eq!(plan_df.height(), df.height());
            }
            _ => panic!("Expected DataFrameSource variant"),
        }
    }

    #[test]
    fn test_select_creation() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let columns = vec!["name".to_string(), "age".to_string()];
        let plan = PhysicalPlan::Select {
            input: Box::new(source),
            columns,
        };

        match plan {
            PhysicalPlan::Select { columns, .. } => {
                assert_eq!(columns.len(), 2);
                assert_eq!(columns[0], "name");
                assert_eq!(columns[1], "age");
            }
            _ => panic!("Expected Select variant"),
        }
    }

    #[test]
    fn test_filter_creation() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let plan = PhysicalPlan::Filter {
            input: Box::new(source),
            column: "age".to_string(),
            value: AnyValue::Int64(30),
            op: BinaryOperator::Gt,
        };

        match plan {
            PhysicalPlan::Filter {
                column, value, op, ..
            } => {
                assert_eq!(column, "age");
                assert_eq!(value, AnyValue::Int64(30));
                assert_eq!(op, BinaryOperator::Gt);
            }
            _ => panic!("Expected Filter variant"),
        }
    }

    #[test]
    fn test_limit_creation() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let plan = PhysicalPlan::Limit {
            input: Box::new(source),
            n: 5,
        };

        match plan {
            PhysicalPlan::Limit { n, .. } => {
                assert_eq!(n, 5);
            }
            _ => panic!("Expected Limit variant"),
        }
    }

    // ============ PhysicalPlan Execution Tests ============

    #[test]
    fn test_execute_dataframe_source() {
        let df = create_test_dataframe();
        let plan = PhysicalPlan::DataFrameSource { df: df.clone() };

        let result = plan.execute().expect("Execution should succeed");

        assert_eq!(result.width(), df.width());
        assert_eq!(result.height(), df.height());
        assert_eq!(result.column_names(), df.column_names());
    }

    #[test]
    fn test_execute_select_single_column() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let plan = PhysicalPlan::Select {
            input: Box::new(source),
            columns: vec!["name".to_string()],
        };

        let result = plan.execute().expect("Execution should succeed");

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 3);
        assert_eq!(result.column_names(), vec!["name"]);

        let name_col = result.column("name").unwrap();
        assert_eq!(name_col[0], AnyValue::String("Alice".to_string()));
        assert_eq!(name_col[1], AnyValue::String("Bob".to_string()));
        assert_eq!(name_col[2], AnyValue::String("Charlie".to_string()));
    }

    #[test]
    fn test_execute_select_multiple_columns() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let plan = PhysicalPlan::Select {
            input: Box::new(source),
            columns: vec!["name".to_string(), "age".to_string()],
        };

        let result = plan.execute().expect("Execution should succeed");

        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 3);
        assert_eq!(result.column_names(), vec!["name", "age"]);
    }

    #[test]
    fn test_execute_select_reorder_columns() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        // Select in different order: score, name, age
        let plan = PhysicalPlan::Select {
            input: Box::new(source),
            columns: vec!["score".to_string(), "name".to_string(), "age".to_string()],
        };

        let result = plan.execute().expect("Execution should succeed");

        assert_eq!(result.width(), 3);
        assert_eq!(result.column_names(), vec!["score", "name", "age"]);
    }

    #[test]
    fn test_execute_select_nonexistent_column() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let plan = PhysicalPlan::Select {
            input: Box::new(source),
            columns: vec!["nonexistent".to_string()],
        };

        let result = plan.execute();
        assert!(result.is_err());

        match result.unwrap_err() {
            ExecutionError::ColumnNotFound { name } => {
                assert_eq!(name, "nonexistent");
            }
            _ => panic!("Expected ColumnNotFound error"),
        }
    }

    #[test]
    fn test_execute_filter_gt() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let plan = PhysicalPlan::Filter {
            input: Box::new(source),
            column: "age".to_string(),
            value: AnyValue::Int64(25),
            op: BinaryOperator::Gt,
        };

        let result = plan.execute().expect("Execution should succeed");

        // Should keep Bob (30) and Charlie (35), filter out Alice (25)
        assert_eq!(result.height(), 2);
        assert_eq!(result.width(), 3);

        let ages = result.column("age").unwrap();
        assert_eq!(ages[0], AnyValue::Int64(30)); // Bob
        assert_eq!(ages[1], AnyValue::Int64(35)); // Charlie
    }

    #[test]
    fn test_execute_filter_eq() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let plan = PhysicalPlan::Filter {
            input: Box::new(source),
            column: "name".to_string(),
            value: AnyValue::String("Bob".to_string()),
            op: BinaryOperator::Eq,
        };

        let result = plan.execute().expect("Execution should succeed");

        // Should keep only Bob
        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        let names = result.column("name").unwrap();
        assert_eq!(names[0], AnyValue::String("Bob".to_string()));
    }

    #[test]
    fn test_execute_filter_lt() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let plan = PhysicalPlan::Filter {
            input: Box::new(source),
            column: "score".to_string(),
            value: AnyValue::Float64(90.0),
            op: BinaryOperator::Lt,
        };

        let result = plan.execute().expect("Execution should succeed");

        // Should keep Alice (85.5) and Charlie (78.5), filter out Bob (92.0)
        assert_eq!(result.height(), 2);

        let names = result.column("name").unwrap();
        assert_eq!(names[0], AnyValue::String("Alice".to_string()));
        assert_eq!(names[1], AnyValue::String("Charlie".to_string()));
    }

    #[test]
    fn test_execute_filter_no_matches() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let plan = PhysicalPlan::Filter {
            input: Box::new(source),
            column: "age".to_string(),
            value: AnyValue::Int64(100),
            op: BinaryOperator::Gt,
        };

        let result = plan.execute().expect("Execution should succeed");

        // Should return empty DataFrame with correct columns
        assert_eq!(result.height(), 0);
        assert_eq!(result.width(), 3);
        assert_eq!(result.column_names(), vec!["name", "age", "score"]);
    }

    #[test]
    fn test_execute_filter_nonexistent_column() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let plan = PhysicalPlan::Filter {
            input: Box::new(source),
            column: "nonexistent".to_string(),
            value: AnyValue::Int64(0),
            op: BinaryOperator::Eq,
        };

        let result = plan.execute();
        assert!(result.is_err());

        match result.unwrap_err() {
            ExecutionError::ColumnNotFound { name } => {
                assert_eq!(name, "nonexistent");
            }
            _ => panic!("Expected ColumnNotFound error"),
        }
    }

    #[test]
    fn test_execute_limit() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let plan = PhysicalPlan::Limit {
            input: Box::new(source),
            n: 2,
        };

        let result = plan.execute().expect("Execution should succeed");

        // Should keep first 2 rows
        assert_eq!(result.height(), 2);
        assert_eq!(result.width(), 3);

        let names = result.column("name").unwrap();
        assert_eq!(names[0], AnyValue::String("Alice".to_string()));
        assert_eq!(names[1], AnyValue::String("Bob".to_string()));
    }

    #[test]
    fn test_execute_limit_larger_than_data() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let plan = PhysicalPlan::Limit {
            input: Box::new(source),
            n: 10,
        };

        let result = plan.execute().expect("Execution should succeed");

        // Should return all rows since limit > data size
        assert_eq!(result.height(), 3);
        assert_eq!(result.width(), 3);
    }

    #[test]
    fn test_execute_limit_zero() {
        let df = create_test_dataframe();
        let source = PhysicalPlan::DataFrameSource { df };

        let plan = PhysicalPlan::Limit {
            input: Box::new(source),
            n: 0,
        };

        let result = plan.execute().expect("Execution should succeed");

        // Should return empty DataFrame
        assert_eq!(result.height(), 0);
        assert_eq!(result.width(), 3);
    }

    // ============ Complex PhysicalPlan Tests ============

    #[test]
    fn test_execute_chained_operations() {
        let df = create_test_dataframe();

        // Build: source -> select -> filter -> limit
        let source = PhysicalPlan::DataFrameSource { df };

        let select = PhysicalPlan::Select {
            input: Box::new(source),
            columns: vec!["name".to_string(), "age".to_string(), "score".to_string()],
        };

        let filter = PhysicalPlan::Filter {
            input: Box::new(select),
            column: "age".to_string(),
            value: AnyValue::Int64(25),
            op: BinaryOperator::Gt,
        };

        let limit = PhysicalPlan::Limit {
            input: Box::new(filter),
            n: 1,
        };

        let result = limit.execute().expect("Execution should succeed");

        // Should filter out Alice (age <= 25), then limit to 1 row (Bob)
        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        let names = result.column("name").unwrap();
        assert_eq!(names[0], AnyValue::String("Bob".to_string()));
    }

    #[test]
    fn test_execute_filter_then_select() {
        let df = create_test_dataframe();

        let source = PhysicalPlan::DataFrameSource { df };

        let filter = PhysicalPlan::Filter {
            input: Box::new(source),
            column: "age".to_string(),
            value: AnyValue::Int64(30),
            op: BinaryOperator::GtEq,
        };

        let select = PhysicalPlan::Select {
            input: Box::new(filter),
            columns: vec!["name".to_string(), "score".to_string()],
        };

        let result = select.execute().expect("Execution should succeed");

        // Should keep Bob (30) and Charlie (35), then select name + score
        assert_eq!(result.height(), 2);
        assert_eq!(result.width(), 2);
        assert_eq!(result.column_names(), vec!["name", "score"]);

        let names = result.column("name").unwrap();
        assert_eq!(names[0], AnyValue::String("Bob".to_string()));
        assert_eq!(names[1], AnyValue::String("Charlie".to_string()));
    }

    // ============ Error Handling Tests ============

    #[test]
    fn test_execute_nested_error_propagation() {
        let df = create_test_dataframe();

        let source = PhysicalPlan::DataFrameSource { df };

        // Create a filter that will fail due to nonexistent column
        let bad_filter = PhysicalPlan::Filter {
            input: Box::new(source),
            column: "nonexistent".to_string(),
            value: AnyValue::Int64(0),
            op: BinaryOperator::Eq,
        };

        // Wrap it in a select - error should propagate up
        let select = PhysicalPlan::Select {
            input: Box::new(bad_filter),
            columns: vec!["name".to_string()],
        };

        let result = select.execute();
        assert!(result.is_err());

        match result.unwrap_err() {
            ExecutionError::ColumnNotFound { name } => {
                assert_eq!(name, "nonexistent");
            }
            _ => panic!("Expected ColumnNotFound error"),
        }
    }

    // ============ Debug Tests ============

    #[test]
    fn test_physical_plan_debug() {
        let df = create_test_dataframe();
        let plan = PhysicalPlan::DataFrameSource { df };

        let debug_str = format!("{:?}", plan);
        assert!(debug_str.contains("DataFrameSource"));
    }

    #[test]
    fn test_execution_error_debug() {
        let error = ExecutionError::ColumnNotFound {
            name: "test".to_string(),
        };

        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("ColumnNotFound"));
        assert!(debug_str.contains("test"));
    }
}
