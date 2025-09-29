use crate::datatypes::dataframe::DataFrame;
use crate::datatypes::series::DataType;
use crate::execution::RecordBatch;
use crate::expressions::expr::Expr;
use crate::logical_plan::QueryOptimizer;
use crate::logical_plan::plan::{JoinType, LogicalPlan, LogicalPlanError};
use crate::physical_plan::{StreamingPlannerError, logical_to_physical, logical_to_streaming};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct LazyFrame {
    logical_plan: LogicalPlan,
}

#[derive(Debug, Error)]
pub enum QueryError {
    #[error("Logical plan error: {0}")]
    LogicalPlanError(#[from] LogicalPlanError),
    #[error("Execution error: {0}")]
    ExecutionError(String),
    #[error("Streaming planner error: {0}")]
    StreamingPlannerError(#[from] StreamingPlannerError),
}

impl LazyFrame {
    pub fn from_dataframe(df: DataFrame) -> Self {
        Self {
            logical_plan: LogicalPlan::DataFrameSource {
                df: df.clone(),
                schema: df
                    .columns()
                    .iter()
                    .map(|s| s.name().to_string())
                    .zip(df.columns().iter().map(|s| s.dtype().clone()))
                    .collect(),
            },
        }
    }

    pub fn from_csv<P: Into<PathBuf>>(
        path: P,
        schema: Vec<(String, DataType)>,
        batch_size: Option<usize>,
        delimiter: Option<char>,
    ) -> Self {
        Self {
            logical_plan: LogicalPlan::CsvFileSource {
                path: path.into(),
                schema,
                batch_size,
                delimiter,
            },
        }
    }

    pub fn select(self, exprs: Vec<Expr>) -> Self {
        Self {
            logical_plan: LogicalPlan::Select {
                input: Box::new(self.logical_plan),
                expressions: exprs,
            },
        }
    }

    pub fn filter(self, predicate: Expr) -> Self {
        Self {
            logical_plan: LogicalPlan::Filter {
                input: Box::new(self.logical_plan),
                predicate,
            },
        }
    }

    pub fn limit(self, n: usize) -> Self {
        Self {
            logical_plan: LogicalPlan::Limit {
                input: Box::new(self.logical_plan),
                n,
            },
        }
    }

    pub fn inner_join(self, right: LazyFrame, left_key: String, right_key: String) -> Self {
        Self {
            logical_plan: LogicalPlan::Join {
                left: Box::new(self.logical_plan),
                right: Box::new(right.logical_plan),
                left_key,
                right_key,
                join_type: JoinType::Inner,
            },
        }
    }

    pub fn collect(self) -> Result<DataFrame, QueryError> {
        let optimized_plan = QueryOptimizer::optimize(self.logical_plan);
        optimized_plan.validate()?;
        let physical_plan = logical_to_physical(optimized_plan)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;
        physical_plan
            .execute()
            .map_err(|e| QueryError::ExecutionError(e.to_string()))
    }

    pub fn collect_streaming(self) -> Result<RecordBatch, QueryError> {
        let optimized_plan = QueryOptimizer::optimize(self.logical_plan);
        optimized_plan.validate()?;
        let streaming_plan = logical_to_streaming(optimized_plan)?;
        streaming_plan
            .collect()
            .map_err(|e| QueryError::ExecutionError(e.to_string()))
    }
}

//Generated tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::dataframe::DataFrame;
    use crate::datatypes::series::{AnyValue, DataType, Series};

    use crate::expressions::expr::{BinaryOperator, Expr};

    use crate::logical_plan::plan::LogicalPlanError;

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

    // ============ LazyFrame Creation Tests ============

    #[test]
    fn test_from_dataframe_creation() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df.clone());

        // LazyFrame should contain a DataFrameSource plan
        match &lazy.logical_plan {
            LogicalPlan::DataFrameSource {
                df: plan_df,
                schema,
            } => {
                assert_eq!(plan_df.width(), df.width());
                assert_eq!(plan_df.height(), df.height());
                assert_eq!(schema.len(), 3);
                assert_eq!(schema[0].0, "name");
                assert_eq!(schema[1].0, "age");
                assert_eq!(schema[2].0, "score");
            }
            _ => panic!("Expected DataFrameSource plan"),
        }
    }

    // ============ Select Operation Tests ============

    #[test]
    fn test_select_single_column() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df).select(vec![Expr::col("name")]);

        match &lazy.logical_plan {
            LogicalPlan::Select { expressions, .. } => {
                assert_eq!(expressions.len(), 1);
                match &expressions[0] {
                    Expr::Column(name) => assert_eq!(name, "name"),
                    _ => panic!("Expected Column expression"),
                }
            }
            _ => panic!("Expected Select plan"),
        }
    }

    #[test]
    fn test_select_multiple_columns() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df).select(vec![
            Expr::col("name"),
            Expr::col("age"),
            Expr::col("score"),
        ]);

        match &lazy.logical_plan {
            LogicalPlan::Select { expressions, .. } => {
                assert_eq!(expressions.len(), 3);
            }
            _ => panic!("Expected Select plan"),
        }
    }

    #[test]
    fn test_select_with_alias() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df)
            .select(vec![Expr::col("name"), Expr::col("age").alias("user_age")]);

        match &lazy.logical_plan {
            LogicalPlan::Select { expressions, .. } => {
                assert_eq!(expressions.len(), 2);
                match &expressions[1] {
                    Expr::Alias(_, alias) => assert_eq!(alias, "user_age"),
                    _ => panic!("Expected Alias expression"),
                }
            }
            _ => panic!("Expected Select plan"),
        }
    }

    #[test]
    fn test_select_with_arithmetic() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df).select(vec![
            Expr::col("name"),
            Expr::col("age").mul(Expr::lit(2)).alias("double_age"),
        ]);

        match &lazy.logical_plan {
            LogicalPlan::Select { expressions, .. } => {
                assert_eq!(expressions.len(), 2);
                // Second expression should be an alias containing binary expr
                match &expressions[1] {
                    Expr::Alias(inner, alias) => {
                        assert_eq!(alias, "double_age");
                        match inner.as_ref() {
                            Expr::BinaryExpr { .. } => {}
                            _ => panic!("Expected BinaryExpr inside alias"),
                        }
                    }
                    _ => panic!("Expected Alias expression"),
                }
            }
            _ => panic!("Expected Select plan"),
        }
    }

    // ============ Filter Operation Tests ============

    #[test]
    fn test_filter_simple_condition() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df).filter(Expr::col("age").gt(Expr::lit(30)));

        match &lazy.logical_plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryExpr { op, .. } => {
                    assert_eq!(*op, BinaryOperator::Gt);
                }
                _ => panic!("Expected BinaryExpr predicate"),
            },
            _ => panic!("Expected Filter plan"),
        }
    }

    #[test]
    fn test_filter_complex_condition() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df).filter(
            Expr::col("age")
                .gt(Expr::lit(25))
                .and(Expr::col("score").lt(Expr::lit(90.0))),
        );

        match &lazy.logical_plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryExpr { op, .. } => {
                    assert_eq!(*op, BinaryOperator::And);
                }
                _ => panic!("Expected And BinaryExpr predicate"),
            },
            _ => panic!("Expected Filter plan"),
        }
    }

    #[test]
    fn test_filter_string_condition() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df).filter(Expr::col("name").eq(Expr::lit("Alice")));

        match &lazy.logical_plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::BinaryExpr { left, op, right } => {
                    assert_eq!(*op, BinaryOperator::Eq);
                    match (left.as_ref(), right.as_ref()) {
                        (Expr::Column(name), Expr::Literal(AnyValue::String(val))) => {
                            assert_eq!(name, "name");
                            assert_eq!(val, "Alice");
                        }
                        _ => panic!("Expected name == 'Alice' condition"),
                    }
                }
                _ => panic!("Expected BinaryExpr predicate"),
            },
            _ => panic!("Expected Filter plan"),
        }
    }

    // ============ Limit Operation Tests ============

    #[test]
    fn test_limit_operation() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df).limit(5);

        match &lazy.logical_plan {
            LogicalPlan::Limit { n, .. } => {
                assert_eq!(*n, 5);
            }
            _ => panic!("Expected Limit plan"),
        }
    }

    #[test]
    fn test_limit_zero() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df).limit(0);

        match &lazy.logical_plan {
            LogicalPlan::Limit { n, .. } => {
                assert_eq!(*n, 0);
            }
            _ => panic!("Expected Limit plan"),
        }
    }

    // ============ Chaining Operations Tests ============

    #[test]
    fn test_select_then_filter() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df)
            .select(vec![Expr::col("name"), Expr::col("age")])
            .filter(Expr::col("age").gt(Expr::lit(25)));

        // Should result in nested plans: Filter -> Select -> DataFrameSource
        match &lazy.logical_plan {
            LogicalPlan::Filter { input, .. } => match input.as_ref() {
                LogicalPlan::Select {
                    input: inner_input, ..
                } => match inner_input.as_ref() {
                    LogicalPlan::DataFrameSource { .. } => {}
                    _ => panic!("Expected DataFrameSource at bottom"),
                },
                _ => panic!("Expected Select as Filter input"),
            },
            _ => panic!("Expected Filter plan at top"),
        }
    }

    #[test]
    fn test_filter_then_select() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df)
            .filter(Expr::col("age").gt(Expr::lit(25)))
            .select(vec![Expr::col("name"), Expr::col("score")]);

        // Should result in nested plans: Select -> Filter -> DataFrameSource
        match &lazy.logical_plan {
            LogicalPlan::Select { input, .. } => match input.as_ref() {
                LogicalPlan::Filter {
                    input: inner_input, ..
                } => match inner_input.as_ref() {
                    LogicalPlan::DataFrameSource { .. } => {}
                    _ => panic!("Expected DataFrameSource at bottom"),
                },
                _ => panic!("Expected Filter as Select input"),
            },
            _ => panic!("Expected Select plan at top"),
        }
    }

    #[test]
    fn test_complex_chaining() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df)
            .select(vec![
                Expr::col("name"),
                Expr::col("age"),
                Expr::col("score"),
            ])
            .filter(Expr::col("age").gt(Expr::lit(25)))
            .select(vec![
                Expr::col("name"),
                Expr::col("score")
                    .mul(Expr::lit(1.1))
                    .alias("boosted_score"),
            ])
            .limit(10);

        // Should be: Limit -> Select -> Filter -> Select -> DataFrameSource
        match &lazy.logical_plan {
            LogicalPlan::Limit { input, n } => {
                assert_eq!(*n, 10);
                match input.as_ref() {
                    LogicalPlan::Select { .. } => {} // Second select
                    _ => panic!("Expected Select under Limit"),
                }
            }
            _ => panic!("Expected Limit at top"),
        }
    }

    // ============ Collect (Execution) Tests ============

    #[test]
    fn test_collect_simple_select() {
        let df = create_test_dataframe();
        let result = LazyFrame::from_dataframe(df)
            .select(vec![Expr::col("name"), Expr::col("age")])
            .collect()
            .expect("Collection should succeed");

        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 3);

        let names = result.column_names();
        assert_eq!(names, vec!["name", "age"]);
    }

    #[test]
    fn test_collect_with_filter() {
        let df = create_test_dataframe();
        let result = LazyFrame::from_dataframe(df)
            .filter(Expr::col("age").gt(Expr::lit(25)))
            .collect()
            .expect("Collection should succeed");

        // Should filter out Alice (age 25), leaving Bob (30) and Charlie (35)
        assert_eq!(result.height(), 2);
        assert_eq!(result.width(), 3);
    }

    #[test]
    fn test_collect_with_limit() {
        let df = create_test_dataframe();
        let result = LazyFrame::from_dataframe(df)
            .limit(2)
            .collect()
            .expect("Collection should succeed");

        assert_eq!(result.height(), 2);
        assert_eq!(result.width(), 3);
    }

    /*#[test]
    fn test_collect_complex_query() {
        let df = create_test_dataframe();
        let result = LazyFrame::from_dataframe(df)
            .select(vec![
                Expr::col("name"),
                Expr::col("age").mul(Expr::lit(2)).alias("double_age"),
                Expr::col("score"),
            ])
            .filter(Expr::col("score").gt(Expr::lit(80.0)))
            .limit(5)
            .collect()
            .expect("Collection should succeed");

        assert_eq!(result.width(), 3);
        let names = result.column_names();
        assert_eq!(names, vec!["name", "double_age", "score"]);
    }*/

    // ============ Error Handling Tests ============

    #[test]
    fn test_collect_invalid_column_in_select() {
        let df = create_test_dataframe();
        let result = LazyFrame::from_dataframe(df)
            .select(vec![Expr::col("nonexistent")])
            .collect();

        assert!(result.is_err());
        match result.unwrap_err() {
            QueryError::LogicalPlanError(err) => match err {
                LogicalPlanError::ColumnNotFound { name } => {
                    assert_eq!(name, "nonexistent");
                }
                _ => panic!("Expected ColumnNotFound error"),
            },
            _ => panic!("Expected LogicalPlanError"),
        }
    }

    #[test]
    fn test_collect_invalid_column_in_filter() {
        let df = create_test_dataframe();
        let result = LazyFrame::from_dataframe(df)
            .filter(Expr::col("nonexistent").gt(Expr::lit(0)))
            .collect();

        assert!(result.is_err());
        match result.unwrap_err() {
            QueryError::LogicalPlanError(err) => match err {
                LogicalPlanError::ColumnNotFound { name } => {
                    assert_eq!(name, "nonexistent");
                }
                _ => panic!("Expected ColumnNotFound error"),
            },
            _ => panic!("Expected LogicalPlanError"),
        }
    }

    // ============ Schema Inference Tests ============

    #[test]
    fn test_schema_inference_select() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df)
            .select(vec![Expr::col("name"), Expr::col("age").alias("user_age")]);

        // Test that we can get schema without collecting
        let schema = lazy.logical_plan.schema();
        assert_eq!(schema.len(), 2);
        assert_eq!(schema[0].0, "name");
        assert_eq!(schema[1].0, "user_age");
        assert_eq!(schema[0].1, DataType::String);
        assert_eq!(schema[1].1, DataType::Int64);
    }

    #[test]
    fn test_schema_inference_arithmetic() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df).select(vec![
            Expr::col("age")
                .add(Expr::col("score"))
                .alias("age_plus_score"),
        ]);

        let schema = lazy.logical_plan.schema();
        assert_eq!(schema.len(), 1);
        assert_eq!(schema[0].0, "age_plus_score");
        // Int64 + Float64 should promote to Float64
        assert_eq!(schema[0].1, DataType::Float64);
    }

    // ============ Streaming Collect Tests ============

    #[test]
    fn test_collect_streaming_simple() {
        let df = create_test_dataframe();
        let result = LazyFrame::from_dataframe(df)
            .select(vec![Expr::col("name"), Expr::col("age")])
            .collect_streaming()
            .expect("Streaming collection should succeed");

        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 3);
        assert_eq!(result.schema().field(0).name(), "name");
        assert_eq!(result.schema().field(1).name(), "age");
    }

    #[test]
    fn test_collect_streaming_with_filter() {
        let df = create_test_dataframe();
        let result = LazyFrame::from_dataframe(df)
            .filter(Expr::col("active")) // Simple boolean column filter
            .collect_streaming();

        // This should fail because we need to create a DataFrame with a boolean column
        // For now, let's test that it fails gracefully
        assert!(result.is_err());
    }

    #[test]
    fn test_collect_streaming_vs_collect() {
        let df = create_test_dataframe();

        // Old system
        let old_result = LazyFrame::from_dataframe(df.clone())
            .select(vec![Expr::col("name")])
            .collect()
            .expect("Old collection should succeed");

        // New system
        let new_result = LazyFrame::from_dataframe(df)
            .select(vec![Expr::col("name")])
            .collect_streaming()
            .expect("New collection should succeed");

        // Should have same dimensions
        assert_eq!(old_result.width(), new_result.num_columns());
        assert_eq!(old_result.height(), new_result.num_rows());
    }

    // ============ Clone and Debug Tests ============

    #[test]
    fn test_lazyframe_clone() {
        let df = create_test_dataframe();
        let lazy1 = LazyFrame::from_dataframe(df)
            .select(vec![Expr::col("name")])
            .filter(Expr::col("name").eq(Expr::lit("Alice")));

        let lazy2 = lazy1.clone();

        // Both should be able to collect independently
        let result1 = lazy1.collect().expect("First collection should succeed");
        let result2 = lazy2.collect().expect("Second collection should succeed");

        assert_eq!(result1.height(), result2.height());
        assert_eq!(result1.width(), result2.width());
    }

    #[test]
    fn test_lazyframe_debug() {
        let df = create_test_dataframe();
        let lazy = LazyFrame::from_dataframe(df).select(vec![Expr::col("name")]);

        let debug_str = format!("{:?}", lazy);
        assert!(debug_str.contains("LazyFrame"));
        assert!(debug_str.contains("logical_plan"));
    }
}
