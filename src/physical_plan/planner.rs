use super::plan::PhysicalPlan;
use crate::logical_plan::LogicalPlan;

use crate::datatypes::AnyValue;
use crate::expressions::{BinaryOperator, Expr};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConversionError {
    #[error("Unsupported expression: {expr:?}")]
    UnsupportedExpression { expr: String },

    #[error("Unsupported filter: only simple column comparisons supported, found: {predicate:?}")]
    UnsupportedFilter { predicate: String },

    #[error("Filter must be a binary comparison, found: {expr_type}")]
    InvalidFilterStructure { expr_type: String },

    #[error("Filter left side must be a column reference, found: {expr:?}")]
    FilterLeftNotColumn { expr: String },

    #[error("Filter right side must be a literal value, found: {expr:?}")]
    FilterRightNotLiteral { expr: String },

    #[error("Unsupported binary operator in filter: {op:?}")]
    UnsupportedFilterOperator { op: BinaryOperator },

    #[error("Select expression must be a column or alias, found: {expr:?}")]
    InvalidSelectExpression { expr: String },

    #[error("Nested conversion failed: {source}")]
    NestedConversion {
        #[from]
        source: Box<ConversionError>,
    },

    #[error("Conversion failed: {message}")]
    General { message: String },
}

pub fn logical_to_physical(logical: LogicalPlan) -> Result<PhysicalPlan, ConversionError> {
    match logical {
        LogicalPlan::DataFrameSource { df, .. } => Ok(PhysicalPlan::DataFrameSource { df }),

        LogicalPlan::Select { input, expressions } => {
            let physical_input = Box::new(logical_to_physical(*input)?);

            let mut columns = Vec::new();
            for expr in expressions {
                let column_name = convert_select_expr(&expr)?;
                columns.push(column_name);
            }

            Ok(PhysicalPlan::Select {
                input: physical_input,
                columns,
            })
        }

        LogicalPlan::Filter { input, predicate } => {
            let physical_input = Box::new(logical_to_physical(*input)?);

            let (column, value, op) = convert_filter_predicate(&predicate)?;

            Ok(PhysicalPlan::Filter {
                input: physical_input,
                column,
                value,
                op,
            })
        }

        LogicalPlan::Limit { input, n } => {
            let physical_input = Box::new(logical_to_physical(*input)?);

            Ok(PhysicalPlan::Limit {
                input: physical_input,
                n,
            })
        }
        LogicalPlan::Join {
            left,
            right,
            left_key,
            right_key,
            join_type,
        } => {
            let physical_left = Box::new(logical_to_physical(*left)?);
            let physical_right = Box::new(logical_to_physical(*right)?);

            //smallest in build side, because in theory build side must stay in memory
            //not now
            Ok(PhysicalPlan::HashJoin {
                build_side: physical_left,
                probe_side: physical_right,
                build_key: left_key,
                probe_key: right_key,
                join_type,
            })
        }
    }
}

fn convert_select_expr(expr: &Expr) -> Result<String, ConversionError> {
    match expr {
        Expr::Column(name) => Ok(name.clone()),

        Expr::Alias(inner_expr, _alias) => match inner_expr.as_ref() {
            Expr::Column(original_name) => Ok(original_name.clone()),
            _ => Err(ConversionError::UnsupportedExpression {
                expr: format!("{:?}", expr),
            }),
        },

        Expr::BinaryExpr { .. } => Err(ConversionError::UnsupportedExpression {
            expr: format!("{:?}", expr),
        }),

        Expr::Literal(_) => Err(ConversionError::InvalidSelectExpression {
            expr: format!("{:?}", expr),
        }),
    }
}

fn convert_filter_predicate(
    predicate: &Expr,
) -> Result<(String, AnyValue, BinaryOperator), ConversionError> {
    match predicate {
        Expr::BinaryExpr { left, op, right } => {
            match op {
                BinaryOperator::Eq
                | BinaryOperator::NotEq
                | BinaryOperator::Lt
                | BinaryOperator::Gt
                | BinaryOperator::LtEq
                | BinaryOperator::GtEq => {}
                BinaryOperator::And | BinaryOperator::Or => {
                    return Err(ConversionError::UnsupportedFilter {
                        predicate: format!("{:?}", predicate),
                    });
                }
                BinaryOperator::Plus
                | BinaryOperator::Minus
                | BinaryOperator::Multiply
                | BinaryOperator::Divide => {
                    return Err(ConversionError::UnsupportedFilterOperator { op: op.clone() });
                }
            }

            let column = match left.as_ref() {
                Expr::Column(name) => name.clone(),
                _ => {
                    return Err(ConversionError::FilterLeftNotColumn {
                        expr: format!("{:?}", left),
                    });
                }
            };

            let value = match right.as_ref() {
                Expr::Literal(val) => val.clone(),
                _ => {
                    return Err(ConversionError::FilterRightNotLiteral {
                        expr: format!("{:?}", right),
                    });
                }
            };

            Ok((column, value, op.clone()))
        }

        _ => Err(ConversionError::InvalidFilterStructure {
            expr_type: match predicate {
                Expr::Column(_) => "Column".to_string(),
                Expr::Literal(_) => "Literal".to_string(),
                Expr::Alias(_, _) => "Alias".to_string(),
                Expr::BinaryExpr { .. } => unreachable!(),
            },
        }),
    }
}

//Generated tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::{AnyValue, DataFrame, Series};
    use crate::expressions::BinaryOperator;
    use crate::physical_plan::ExecutionError;

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
