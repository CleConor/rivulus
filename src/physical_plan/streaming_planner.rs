use crate::logical_plan::plan::LogicalPlan;
#[allow(unused_imports)]
use crate::logical_plan::plan::JoinType;
use crate::physical_plan::streaming::{StreamingPhysicalPlan, StreamingExecutionError};
use crate::expressions::expr::Expr;

pub type Result<T> = std::result::Result<T, StreamingPlannerError>;

#[derive(Debug, thiserror::Error)]
pub enum StreamingPlannerError {
    #[error("Unsupported logical operation: {operation}")]
    UnsupportedOperation { operation: String },

    #[error("Expression conversion error: {message}")]
    ExpressionError { message: String },

    #[error("Type conversion error: {message}")]
    TypeConversionError { message: String },

    #[error("Column extraction error: {message}")]
    ColumnExtractionError { message: String },

    #[error("Streaming execution error: {0}")]
    StreamingExecution(#[from] StreamingExecutionError),
}


pub fn logical_to_streaming(plan: LogicalPlan) -> Result<StreamingPhysicalPlan> {
    match plan {
        LogicalPlan::DataFrameSource { df, .. } => {
            
            Ok(StreamingPhysicalPlan::dataframe_source(df, 1024))
        }


        LogicalPlan::Select { input, expressions } => {
            let streaming_input = logical_to_streaming(*input)?;
            let column_names = extract_column_names_from_expressions(&expressions)?;
            Ok(streaming_input.select(column_names))
        }

        LogicalPlan::Filter { input, predicate } => {
            let streaming_input = logical_to_streaming(*input)?;
            let predicate_column = extract_boolean_predicate_column(&predicate)?;
            Ok(streaming_input.filter(predicate_column))
        }

        LogicalPlan::Limit { input, n } => {
            let streaming_input = logical_to_streaming(*input)?;
            Ok(streaming_input.limit(n))
        }

        LogicalPlan::Join {
            left,
            right,
            left_key,
            right_key,
            join_type
        } => {
            let streaming_left = logical_to_streaming(*left)?;
            let streaming_right = logical_to_streaming(*right)?;

            Ok(StreamingPhysicalPlan::HashJoin {
                build_side: Box::new(streaming_left),
                probe_side: Box::new(streaming_right),
                build_key: left_key,
                probe_key: right_key,
                join_type,
            })
        }
    }
}

// Extract column names from select expressions
// For now, only supports simple column references
fn extract_column_names_from_expressions(expressions: &[Expr]) -> Result<Vec<String>> {
    let mut column_names = Vec::new();

    for expr in expressions {
        match expr {
            Expr::Column(name) => {
                column_names.push(name.clone());
            }
            Expr::Alias(inner_expr, _alias_name) => {
                // For aliases, we use the original column name for data extraction
                match inner_expr.as_ref() {
                    Expr::Column(original_name) => {
                        column_names.push(original_name.clone());
                    }
                    _ => {
                        return Err(StreamingPlannerError::ExpressionError {
                            message: format!("Complex expressions with aliases not yet supported: {:?}", expr),
                        });
                    }
                }
            }
            _ => {
                return Err(StreamingPlannerError::ExpressionError {
                    message: format!("Complex expressions not yet supported in streaming mode: {:?}", expr),
                });
            }
        }
    }

    Ok(column_names)
}

// Extract boolean predicate column from filter expression
// For now, only supports simple boolean column predicates
fn extract_boolean_predicate_column(predicate: &Expr) -> Result<String> {
    match predicate {
        // Simple boolean column reference (e.g., "active")
        Expr::Column(name) => Ok(name.clone()),

        // Binary expression that should evaluate to boolean
        Expr::BinaryExpr { left, op: _, right: _ } => {
            // For now, we need the predicate to be a simple boolean column
            // In a full implementation, we'd evaluate the binary expression and create a boolean array
            match left.as_ref() {
                Expr::Column(name) => {
                    return Err(StreamingPlannerError::ExpressionError {
                        message: format!(
                            "Binary expressions not yet supported in streaming mode. \
                            Found expression on column '{}'. \
                            Currently only simple boolean column references are supported (e.g., .filter(col('is_active')))",
                            name
                        ),
                    });
                }
                _ => {
                    return Err(StreamingPlannerError::ExpressionError {
                        message: "Complex binary expressions not supported in streaming mode".to_string(),
                    });
                }
            }
        }

        _ => {
            Err(StreamingPlannerError::ExpressionError {
                message: format!("Unsupported filter expression type: {:?}", predicate),
            })
        }
    }
}

//generated tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::{DataFrame, Series, AnyValue, DataType};
    use crate::expressions::expr::Expr;

    fn create_test_dataframe() -> DataFrame {
        let names = Series::new(
            "name",
            vec![
                AnyValue::String("Alice".to_string()),
                AnyValue::String("Bob".to_string()),
                AnyValue::String("Charlie".to_string()),
            ],
        ).unwrap();

        let ages = Series::new(
            "age",
            vec![
                AnyValue::Int64(25),
                AnyValue::Int64(30),
                AnyValue::Int64(35),
            ],
        ).unwrap();

        let active = Series::new(
            "active",
            vec![
                AnyValue::Boolean(true),
                AnyValue::Boolean(false),
                AnyValue::Boolean(true),
            ],
        ).unwrap();

        DataFrame::new(vec![names, ages, active]).unwrap()
    }

    #[test]
    fn test_dataframe_source_conversion() {
        let df = create_test_dataframe();
        let logical_plan = LogicalPlan::DataFrameSource {
            df: df.clone(),
            schema: vec![
                ("name".to_string(), DataType::String),
                ("age".to_string(), DataType::Int64),
                ("active".to_string(), DataType::Boolean),
            ],
        };

        let streaming_plan = logical_to_streaming(logical_plan).unwrap();

        // Should be able to execute and get results
        let result = streaming_plan.collect().unwrap();
        assert_eq!(result.num_rows(), 3);
        assert_eq!(result.num_columns(), 3);
    }

    #[test]
    fn test_simple_select_conversion() {
        let df = create_test_dataframe();
        let logical_plan = LogicalPlan::Select {
            input: Box::new(LogicalPlan::DataFrameSource {
                df,
                schema: vec![
                    ("name".to_string(), DataType::String),
                    ("age".to_string(), DataType::Int64),
                    ("active".to_string(), DataType::Boolean),
                ],
            }),
            expressions: vec![
                Expr::Column("name".to_string()),
                Expr::Column("age".to_string()),
            ],
        };

        let streaming_plan = logical_to_streaming(logical_plan).unwrap();
        let result = streaming_plan.collect().unwrap();

        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.schema().field(0).name(), "name");
        assert_eq!(result.schema().field(1).name(), "age");
    }

    #[test]
    fn test_simple_filter_conversion() {
        let df = create_test_dataframe();
        let logical_plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::DataFrameSource {
                df,
                schema: vec![
                    ("name".to_string(), DataType::String),
                    ("age".to_string(), DataType::Int64),
                    ("active".to_string(), DataType::Boolean),
                ],
            }),
            predicate: Expr::Column("active".to_string()),
        };

        let streaming_plan = logical_to_streaming(logical_plan).unwrap();
        let result = streaming_plan.collect().unwrap();

        // Should filter to only active=true rows (Alice and Charlie)
        assert_eq!(result.num_rows(), 2);
    }

    #[test]
    fn test_limit_conversion() {
        let df = create_test_dataframe();
        let logical_plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::DataFrameSource {
                df,
                schema: vec![
                    ("name".to_string(), DataType::String),
                    ("age".to_string(), DataType::Int64),
                    ("active".to_string(), DataType::Boolean),
                ],
            }),
            n: 2,
        };

        let streaming_plan = logical_to_streaming(logical_plan).unwrap();
        let result = streaming_plan.collect().unwrap();

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 3);
    }

    #[test]
    fn test_chained_operations_conversion() {
        let df = create_test_dataframe();
        let logical_plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Select {
                input: Box::new(LogicalPlan::Filter {
                    input: Box::new(LogicalPlan::DataFrameSource {
                        df,
                        schema: vec![
                            ("name".to_string(), DataType::String),
                            ("age".to_string(), DataType::Int64),
                            ("active".to_string(), DataType::Boolean),
                        ],
                    }),
                    predicate: Expr::Column("active".to_string()),
                }),
                expressions: vec![Expr::Column("name".to_string())],
            }),
            n: 1,
        };

        let streaming_plan = logical_to_streaming(logical_plan).unwrap();
        let result = streaming_plan.collect().unwrap();

        // Should filter active=true, select name column, limit to 1 row
        assert_eq!(result.num_rows(), 1);
        assert_eq!(result.num_columns(), 1);
        assert_eq!(result.schema().field(0).name(), "name");
    }


    #[test]
    fn test_complex_expression_not_supported() {
        let df = create_test_dataframe();
        let logical_plan = LogicalPlan::Select {
            input: Box::new(LogicalPlan::DataFrameSource {
                df,
                schema: vec![("age".to_string(), DataType::Int64)],
            }),
            expressions: vec![
                Expr::BinaryExpr {
                    left: Box::new(Expr::Column("age".to_string())),
                    op: crate::expressions::expr::BinaryOperator::Plus,
                    right: Box::new(Expr::Literal(AnyValue::Int64(10))),
                },
            ],
        };

        let result = logical_to_streaming(logical_plan);
        assert!(result.is_err());

        match result.unwrap_err() {
            StreamingPlannerError::ExpressionError { .. } => {
                // Expected - complex expressions not yet supported
            }
            _ => panic!("Expected ExpressionError"),
        }
    }

    #[test]
    fn test_binary_filter_not_supported() {
        let df = create_test_dataframe();
        let logical_plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::DataFrameSource {
                df,
                schema: vec![("age".to_string(), DataType::Int64)],
            }),
            predicate: Expr::BinaryExpr {
                left: Box::new(Expr::Column("age".to_string())),
                op: crate::expressions::expr::BinaryOperator::Gt,
                right: Box::new(Expr::Literal(AnyValue::Int64(30))),
            },
        };

        let result = logical_to_streaming(logical_plan);
        assert!(result.is_err());

        match result.unwrap_err() {
            StreamingPlannerError::ExpressionError { message } => {
                assert!(message.contains("Binary expressions not yet supported"));
            }
            _ => panic!("Expected ExpressionError for binary expressions"),
        }
    }
}