use crate::Expr;
use crate::logical_plan::LogicalPlan;

pub struct QueryOptimizer;

impl QueryOptimizer {
    pub fn optimize(plan: LogicalPlan) -> LogicalPlan {
        let mut optimized = plan;

        optimized = Self::push_predicates_down(optimized);

        optimized
    }

    fn push_predicates_down(plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Select { input, expressions } => match *input {
                LogicalPlan::Filter {
                    input: filter_input,
                    predicate,
                } => {
                    if Self::predicate_uses_only_selected_columns(&predicate, &expressions) {
                        LogicalPlan::Filter {
                            input: Box::new(LogicalPlan::Select {
                                input: filter_input,
                                expressions,
                            }),
                            predicate,
                        }
                    } else {
                        LogicalPlan::Select {
                            input: Box::new(LogicalPlan::Filter {
                                input: filter_input,
                                predicate,
                            }),
                            expressions,
                        }
                    }
                }
                other => LogicalPlan::Select {
                    input: Box::new(Self::push_predicates_down(other)),
                    expressions,
                },
            },
            LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
                input: Box::new(Self::push_predicates_down(*input)),
                predicate,
            },
            LogicalPlan::Join {
                left,
                right,
                left_key,
                right_key,
                join_type,
            } => LogicalPlan::Join {
                left: Box::new(Self::push_predicates_down(*left)),
                right: Box::new(Self::push_predicates_down(*right)),
                left_key,
                right_key,
                join_type,
            },
            other => other,
        }
    }

    fn predicate_uses_only_selected_columns(predicate: &Expr, expressions: &[Expr]) -> bool {
        let predicate_columns = Self::extract_column_names(predicate);
        let selected_columns = Self::extract_selected_columns(expressions);

        predicate_columns
            .iter()
            .all(|col| selected_columns.contains(col))
    }

    fn extract_column_names(expr: &Expr) -> Vec<String> {
        match expr {
            Expr::Column(name) => vec![name.clone()],
            Expr::BinaryExpr { left, right, .. } => {
                let mut cols = Self::extract_column_names(left);
                cols.extend(Self::extract_column_names(right));
                cols
            }
            Expr::Alias(inner, _) => Self::extract_column_names(inner),
            Expr::Literal(_) => vec![],
        }
    }

    fn extract_selected_columns(expressions: &[Expr]) -> Vec<String> {
        expressions
            .iter()
            .flat_map(|expr| match expr {
                Expr::Column(name) => vec![name.clone()],
                Expr::Alias(inner_expr, _) => match inner_expr.as_ref() {
                    Expr::Column(name) => vec![name.clone()],
                    _ => vec![],
                },
                _ => vec![],
            })
            .collect()
    }
}
