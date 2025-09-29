use crate::datatypes::dataframe::DataFrame;
use crate::datatypes::series::DataType;
use crate::expressions::expr::{BinaryOperator, Expr};
use std::collections::HashSet;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum LogicalPlan {
    DataFrameSource {
        df: DataFrame,
        schema: Vec<(String, DataType)>,
    },
    CsvFileSource {
        path: PathBuf,
        schema: Vec<(String, DataType)>,
        batch_size: Option<usize>,
        delimiter: Option<char>,
    },
    Select {
        input: Box<LogicalPlan>,
        expressions: Vec<Expr>,
    },
    Filter {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },
    Limit {
        input: Box<LogicalPlan>,
        n: usize,
    },
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        left_key: String,
        right_key: String,
        join_type: JoinType,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum JoinType {
    Inner,
    //Left,
    //Right,
    //Outer,
}

#[derive(Debug, Error)]
pub enum LogicalPlanError {
    #[error("Column not found: '{name}'")]
    ColumnNotFound { name: String },
    #[error("Invalid limit: '{limit}'")]
    InvalidLimit { limit: usize },
    #[error("Incompatible join key types: '{left_type:?}' and '{right_type:?}'")]
    IncompatibleJoinKeys {
        left_type: DataType,
        right_type: DataType,
    },
}

impl LogicalPlan {
    pub fn schema(&self) -> Vec<(String, DataType)> {
        match self {
            Self::DataFrameSource { schema, .. } => schema.clone(),
            Self::CsvFileSource { schema, .. } => schema.clone(),
            Self::Select { input, expressions } => {
                let input_schema = input.schema();

                let mut result_schema = Vec::new();
                for expr in expressions {
                    let (name, dtype) = self.resolve_expr_schema(expr, &input_schema);
                    result_schema.push((name, dtype));
                }
                result_schema
            }
            Self::Filter { input, .. } => input.schema(),
            Self::Limit { input, .. } => input.schema(),
            Self::Join {
                left,
                right,
                left_key: _,
                right_key,
                join_type,
            } => {
                let left_schema = left.schema();
                let right_schema = right.schema();

                let mut result_schema = left_schema.clone();

                match join_type {
                    JoinType::Inner => {
                        for (right_col, right_type) in right_schema {
                            if right_col == *right_key {
                                continue;
                            }

                            let final_name =
                                if left_schema.iter().any(|(name, _)| name == &right_col) {
                                    format!("{}_right", right_col)
                                } else {
                                    right_col
                                };

                            result_schema.push((final_name, right_type));
                        }

                        result_schema
                    }
                }
            }
        }
    }

    pub fn validate(&self) -> Result<(), LogicalPlanError> {
        match self {
            Self::DataFrameSource { df, schema } => {
                let df_names: HashSet<&str> = df.column_names().into_iter().collect();
                for (schema_name, _) in schema {
                    if !df_names.contains(schema_name.as_str()) {
                        return Err(LogicalPlanError::ColumnNotFound {
                            name: schema_name.clone(),
                        });
                    }
                }
                Ok(())
            }
            Self::CsvFileSource { .. } => Ok(()),
            Self::Select { input, expressions } => {
                input.validate()?;

                let input_schema = input.schema();
                for expr in expressions {
                    self.validate_expr_columns(expr, &input_schema)?;
                }

                Ok(())
            }
            Self::Filter { input, predicate } => {
                input.validate()?;

                let input_schema = input.schema();
                self.validate_expr_columns(predicate, &input_schema)?;

                Ok(())
            }
            Self::Limit { input, .. } => {
                input.validate()?;

                /*if *n == 0 {
                    return Err(LogicalPlanError::InvalidLimit { limit: *n });
                }*/

                Ok(())
            }
            Self::Join {
                right,
                left,
                right_key,
                left_key,
                ..
            } => {
                right.validate()?;
                left.validate()?;

                let left_schema = left.schema();
                if !left_schema.iter().any(|(col, _)| col == left_key) {
                    return Err(LogicalPlanError::ColumnNotFound {
                        name: left_key.clone(),
                    });
                }

                let right_schema = right.schema();
                if !right_schema.iter().any(|(col, _)| col == right_key) {
                    return Err(LogicalPlanError::ColumnNotFound {
                        name: right_key.clone(),
                    });
                }

                let left_key_type = left_schema
                    .iter()
                    .find(|(col, _)| col == left_key)
                    .map(|(_, dtype)| dtype)
                    .unwrap();

                let right_key_type = right_schema
                    .iter()
                    .find(|(col, _)| col == right_key)
                    .map(|(_, dtype)| dtype)
                    .unwrap();

                if !left_key_type.is_comparable_with(right_key_type) {
                    return Err(LogicalPlanError::IncompatibleJoinKeys {
                        left_type: left_key_type.clone(),
                        right_type: right_key_type.clone(),
                    });
                }

                Ok(())
            }
        }
    }

    fn resolve_expr_schema(
        &self,
        expr: &Expr,
        input_schema: &[(String, DataType)],
    ) -> (String, DataType) {
        match expr {
            Expr::Column(name) => {
                let dtype = input_schema
                    .iter()
                    .find(|(col_name, _)| col_name == name)
                    .map(|(_, dtype)| dtype.clone())
                    .unwrap_or(DataType::Null);
                (name.clone(), dtype)
            }

            Expr::Alias(inner_expr, alias) => {
                let (_, dtype) = self.resolve_expr_schema(inner_expr, input_schema);
                (alias.clone(), dtype)
            }

            Expr::BinaryExpr { left, op, right } => {
                let (left_name, left_type) = self.resolve_expr_schema(left, input_schema);
                let (_, right_type) = self.resolve_expr_schema(right, input_schema);
                let result_type = Self::infer_binary_result_type(&left_type, op, &right_type);
                (left_name, result_type)
            }

            Expr::Literal(value) => ("literal".to_string(), value.data_type()),
        }
    }

    fn infer_binary_result_type(
        left: &DataType,
        op: &BinaryOperator,
        right: &DataType,
    ) -> DataType {
        match op {
            BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Lt
            | BinaryOperator::Gt
            | BinaryOperator::LtEq
            | BinaryOperator::GtEq => DataType::Boolean,

            BinaryOperator::And | BinaryOperator::Or => DataType::Boolean,

            BinaryOperator::Plus
            | BinaryOperator::Minus
            | BinaryOperator::Multiply
            | BinaryOperator::Divide => match (left, right) {
                (DataType::Float64, _) | (_, DataType::Float64) => DataType::Float64,
                (DataType::Int64, DataType::Int64) => DataType::Int64,

                (DataType::Null, other) | (other, DataType::Null) => other.clone(),

                _ => DataType::Null,
            },
        }
    }

    fn validate_expr_columns(
        &self,
        expr: &Expr,
        schema: &[(String, DataType)],
    ) -> Result<(), LogicalPlanError> {
        match expr {
            Expr::Column(name) => {
                if !schema.iter().any(|(col, _)| col == name) {
                    return Err(LogicalPlanError::ColumnNotFound { name: name.clone() });
                }
            }
            Expr::BinaryExpr { left, right, .. } => {
                self.validate_expr_columns(left, schema)?;
                self.validate_expr_columns(right, schema)?;
            }
            Expr::Alias(inner, _) => {
                self.validate_expr_columns(inner, schema)?;
            }
            Expr::Literal(_) => {}
        }
        Ok(())
    }
}

//Generated tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::dataframe::DataFrame;
    use crate::datatypes::series::{AnyValue, DataType, Series};
    use crate::expressions::expr::{BinaryOperator, Expr};

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

    // ============ LogicalPlan Creation Tests ============

    #[test]
    fn test_dataframe_source_creation() {
        let df = create_test_dataframe();
        let schema = vec![
            ("name".to_string(), DataType::String),
            ("age".to_string(), DataType::Int64),
            ("score".to_string(), DataType::Float64),
        ];

        let plan = LogicalPlan::DataFrameSource {
            df: df.clone(),
            schema: schema.clone(),
        };

        match plan {
            LogicalPlan::DataFrameSource {
                df: plan_df,
                schema: plan_schema,
            } => {
                assert_eq!(plan_df.width(), df.width());
                assert_eq!(plan_df.height(), df.height());
                assert_eq!(plan_schema, schema);
            }
            _ => panic!("Expected DataFrameSource variant"),
        }
    }

    #[test]
    fn test_select_creation() {
        let df = create_test_dataframe();
        let source = LogicalPlan::DataFrameSource {
            df,
            schema: vec![
                ("name".to_string(), DataType::String),
                ("age".to_string(), DataType::Int64),
            ],
        };

        let expressions = vec![
            Expr::col("name"),
            Expr::col("age").mul(Expr::lit(2)).alias("double_age"),
        ];

        let plan = LogicalPlan::Select {
            input: Box::new(source),
            expressions,
        };

        match plan {
            LogicalPlan::Select { expressions, .. } => {
                assert_eq!(expressions.len(), 2);
            }
            _ => panic!("Expected Select variant"),
        }
    }

    #[test]
    fn test_filter_creation() {
        let df = create_test_dataframe();
        let source = LogicalPlan::DataFrameSource {
            df,
            schema: vec![("age".to_string(), DataType::Int64)],
        };

        let predicate = Expr::col("age").gt(Expr::lit(25));

        let plan = LogicalPlan::Filter {
            input: Box::new(source),
            predicate: predicate.clone(),
        };

        match plan {
            LogicalPlan::Filter {
                predicate: plan_predicate,
                ..
            } => {
                // Compare the structure of predicates
                match plan_predicate {
                    Expr::BinaryExpr {
                        op: BinaryOperator::Gt,
                        ..
                    } => {}
                    _ => panic!("Expected Gt comparison in predicate"),
                }
            }
            _ => panic!("Expected Filter variant"),
        }
    }

    #[test]
    fn test_limit_creation() {
        let df = create_test_dataframe();
        let source = LogicalPlan::DataFrameSource {
            df,
            schema: vec![("name".to_string(), DataType::String)],
        };

        let plan = LogicalPlan::Limit {
            input: Box::new(source),
            n: 10,
        };

        match plan {
            LogicalPlan::Limit { n, .. } => {
                assert_eq!(n, 10);
            }
            _ => panic!("Expected Limit variant"),
        }
    }

    // ============ Schema Computation Tests ============

    #[test]
    fn test_dataframe_source_schema() {
        let df = create_test_dataframe();
        let expected_schema = vec![
            ("name".to_string(), DataType::String),
            ("age".to_string(), DataType::Int64),
            ("score".to_string(), DataType::Float64),
        ];

        let plan = LogicalPlan::DataFrameSource {
            df,
            schema: expected_schema.clone(),
        };

        let computed_schema = plan.schema();
        assert_eq!(computed_schema, expected_schema);
    }

    #[test]
    fn test_select_schema_simple_columns() {
        let df = create_test_dataframe();
        let source = LogicalPlan::DataFrameSource {
            df,
            schema: vec![
                ("name".to_string(), DataType::String),
                ("age".to_string(), DataType::Int64),
                ("score".to_string(), DataType::Float64),
            ],
        };

        let expressions = vec![Expr::col("name"), Expr::col("age")];

        let plan = LogicalPlan::Select {
            input: Box::new(source),
            expressions,
        };

        let schema = plan.schema();
        let expected = vec![
            ("name".to_string(), DataType::String),
            ("age".to_string(), DataType::Int64),
        ];

        assert_eq!(schema, expected);
    }

    #[test]
    fn test_select_schema_with_alias() {
        let df = create_test_dataframe();
        let source = LogicalPlan::DataFrameSource {
            df,
            schema: vec![
                ("name".to_string(), DataType::String),
                ("age".to_string(), DataType::Int64),
            ],
        };

        let expressions = vec![Expr::col("name"), Expr::col("age").alias("user_age")];

        let plan = LogicalPlan::Select {
            input: Box::new(source),
            expressions,
        };

        let schema = plan.schema();
        let expected = vec![
            ("name".to_string(), DataType::String),
            ("user_age".to_string(), DataType::Int64),
        ];

        assert_eq!(schema, expected);
    }

    #[test]
    fn test_select_schema_with_arithmetic() {
        let df = create_test_dataframe();
        let source = LogicalPlan::DataFrameSource {
            df,
            schema: vec![
                ("age".to_string(), DataType::Int64),
                ("score".to_string(), DataType::Float64),
            ],
        };

        let expressions = vec![
            Expr::col("age").mul(Expr::lit(2)),       // Int64 * Int64 = Int64
            Expr::col("score").add(Expr::lit(10.0)),  // Float64 + Float64 = Float64
            Expr::col("age").add(Expr::col("score")), // Int64 + Float64 = Float64
        ];

        let plan = LogicalPlan::Select {
            input: Box::new(source),
            expressions,
        };

        let schema = plan.schema();
        let expected = vec![
            ("age".to_string(), DataType::Int64),     // age * 2
            ("score".to_string(), DataType::Float64), // score + 10.0
            ("age".to_string(), DataType::Float64),   // age + score (promoted)
        ];

        assert_eq!(schema, expected);
    }

    #[test]
    fn test_filter_schema_unchanged() {
        let df = create_test_dataframe();
        let source_schema = vec![
            ("name".to_string(), DataType::String),
            ("age".to_string(), DataType::Int64),
        ];

        let source = LogicalPlan::DataFrameSource {
            df,
            schema: source_schema.clone(),
        };

        let predicate = Expr::col("age").gt(Expr::lit(30));

        let plan = LogicalPlan::Filter {
            input: Box::new(source),
            predicate,
        };

        let schema = plan.schema();
        assert_eq!(schema, source_schema);
    }

    #[test]
    fn test_limit_schema_unchanged() {
        let df = create_test_dataframe();
        let source_schema = vec![
            ("name".to_string(), DataType::String),
            ("age".to_string(), DataType::Int64),
        ];

        let source = LogicalPlan::DataFrameSource {
            df,
            schema: source_schema.clone(),
        };

        let plan = LogicalPlan::Limit {
            input: Box::new(source),
            n: 5,
        };

        let schema = plan.schema();
        assert_eq!(schema, source_schema);
    }

    // ============ Validation Tests ============

    #[test]
    fn test_validate_simple_plan() {
        let df = create_test_dataframe();
        let plan = LogicalPlan::DataFrameSource {
            df,
            schema: vec![
                ("name".to_string(), DataType::String),
                ("age".to_string(), DataType::Int64),
            ],
        };

        assert!(plan.validate().is_ok());
    }

    #[test]
    fn test_validate_select_valid_columns() {
        let df = create_test_dataframe();
        let source = LogicalPlan::DataFrameSource {
            df,
            schema: vec![
                ("name".to_string(), DataType::String),
                ("age".to_string(), DataType::Int64),
            ],
        };

        let expressions = vec![
            Expr::col("name"), // Valid column
            Expr::col("age"),  // Valid column
        ];

        let plan = LogicalPlan::Select {
            input: Box::new(source),
            expressions,
        };

        assert!(plan.validate().is_ok());
    }

    #[test]
    fn test_validate_select_invalid_column() {
        let df = create_test_dataframe();
        let source = LogicalPlan::DataFrameSource {
            df,
            schema: vec![
                ("name".to_string(), DataType::String),
                ("age".to_string(), DataType::Int64),
            ],
        };

        let expressions = vec![
            Expr::col("name"),
            Expr::col("invalid_column"), // This column doesn't exist
        ];

        let plan = LogicalPlan::Select {
            input: Box::new(source),
            expressions,
        };

        let result = plan.validate();
        assert!(result.is_err());

        match result.unwrap_err() {
            LogicalPlanError::ColumnNotFound { name } => {
                assert_eq!(name, "invalid_column");
            }
            _ => panic!("Expected ColumnNotFound error"),
        }
    }

    #[test]
    fn test_validate_filter_valid_column() {
        let df = create_test_dataframe();
        let source = LogicalPlan::DataFrameSource {
            df,
            schema: vec![("age".to_string(), DataType::Int64)],
        };

        let predicate = Expr::col("age").gt(Expr::lit(25)); // age exists

        let plan = LogicalPlan::Filter {
            input: Box::new(source),
            predicate,
        };

        assert!(plan.validate().is_ok());
    }

    #[test]
    fn test_validate_filter_invalid_column() {
        let df = create_test_dataframe();
        let source = LogicalPlan::DataFrameSource {
            df,
            schema: vec![("age".to_string(), DataType::Int64)],
        };

        let predicate = Expr::col("invalid").gt(Expr::lit(25)); // invalid doesn't exist

        let plan = LogicalPlan::Filter {
            input: Box::new(source),
            predicate,
        };

        let result = plan.validate();
        assert!(result.is_err());

        match result.unwrap_err() {
            LogicalPlanError::ColumnNotFound { name } => {
                assert_eq!(name, "invalid");
            }
            _ => panic!("Expected ColumnNotFound error"),
        }
    }

    // ============ Complex Plan Tests ============

    #[test]
    fn test_chained_operations() {
        let df = create_test_dataframe();

        // Build: source -> filter -> select -> limit
        let source = LogicalPlan::DataFrameSource {
            df,
            schema: vec![
                ("name".to_string(), DataType::String),
                ("age".to_string(), DataType::Int64),
                ("score".to_string(), DataType::Float64),
            ],
        };

        let filtered = LogicalPlan::Filter {
            input: Box::new(source),
            predicate: Expr::col("age").gt(Expr::lit(25)),
        };

        let selected = LogicalPlan::Select {
            input: Box::new(filtered),
            expressions: vec![Expr::col("name"), Expr::col("score").alias("final_score")],
        };

        let limited = LogicalPlan::Limit {
            input: Box::new(selected),
            n: 10,
        };

        // Should validate successfully
        assert!(limited.validate().is_ok());

        // Final schema should be from select
        let schema = limited.schema();
        let expected = vec![
            ("name".to_string(), DataType::String),
            ("final_score".to_string(), DataType::Float64),
        ];
        assert_eq!(schema, expected);
    }

    #[test]
    fn test_nested_select_operations() {
        let df = create_test_dataframe();

        let source = LogicalPlan::DataFrameSource {
            df,
            schema: vec![
                ("name".to_string(), DataType::String),
                ("age".to_string(), DataType::Int64),
                ("score".to_string(), DataType::Float64),
            ],
        };

        // First select: create derived columns
        let first_select = LogicalPlan::Select {
            input: Box::new(source),
            expressions: vec![
                Expr::col("name"),
                Expr::col("age").mul(Expr::lit(2)).alias("double_age"),
                Expr::col("score"),
            ],
        };

        // Second select: use derived column
        let second_select = LogicalPlan::Select {
            input: Box::new(first_select),
            expressions: vec![
                Expr::col("name"),
                Expr::col("double_age")
                    .add(Expr::lit(10))
                    .alias("adjusted_age"),
            ],
        };

        assert!(second_select.validate().is_ok());

        let schema = second_select.schema();
        let expected = vec![
            ("name".to_string(), DataType::String),
            ("adjusted_age".to_string(), DataType::Int64),
        ];
        assert_eq!(schema, expected);
    }

    // ============ Debug and Clone Tests ============

    #[test]
    fn test_logical_plan_debug() {
        let df = create_test_dataframe();
        let plan = LogicalPlan::DataFrameSource {
            df,
            schema: vec![("test".to_string(), DataType::String)],
        };

        let debug_str = format!("{:?}", plan);
        assert!(debug_str.contains("DataFrameSource"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_logical_plan_clone() {
        let df = create_test_dataframe();
        let plan1 = LogicalPlan::DataFrameSource {
            df,
            schema: vec![("test".to_string(), DataType::String)],
        };

        let plan2 = plan1.clone();

        // Both plans should have the same structure
        match (plan1, plan2) {
            (
                LogicalPlan::DataFrameSource { schema: s1, .. },
                LogicalPlan::DataFrameSource { schema: s2, .. },
            ) => {
                assert_eq!(s1, s2);
            }
            _ => panic!("Clone should preserve plan structure"),
        }
    }
}
