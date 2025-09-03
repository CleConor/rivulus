use crate::datatypes::series::AnyValue;

#[derive(Debug, Clone)]
pub enum Expr {
    Column(String),
    Literal(AnyValue),
    BinaryExpr {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },
    Alias(Box<Expr>, String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Plus,
    Minus,
    Multiply,
    Divide,
    Eq,    // ==
    NotEq, // !=
    Lt,    // <
    Gt,    // >
    LtEq,  // <=
    GtEq,  // >=
    And,
    Or,
}

impl Expr {
    pub fn col(name: &str) -> Self {
        Expr::Column(name.to_string())
    }

    pub fn lit<T: Into<AnyValue>>(value: T) -> Self {
        Expr::Literal(value.into())
    }

    pub fn alias(self, name: &str) -> Self {
        Expr::Alias(Box::new(self), name.to_string())
    }

    pub fn add(self, other: Expr) -> Self {
        Expr::BinaryExpr {
            left: Box::new(self),
            op: BinaryOperator::Plus,
            right: Box::new(other),
        }
    }

    pub fn sub(self, other: Expr) -> Self {
        Expr::BinaryExpr {
            left: Box::new(self),
            op: BinaryOperator::Minus,
            right: Box::new(other),
        }
    }

    pub fn mul(self, other: Expr) -> Self {
        Expr::BinaryExpr {
            left: Box::new(self),
            op: BinaryOperator::Multiply,
            right: Box::new(other),
        }
    }

    pub fn div(self, other: Expr) -> Self {
        Expr::BinaryExpr {
            left: Box::new(self),
            op: BinaryOperator::Divide,
            right: Box::new(other),
        }
    }

    pub fn eq(self, other: Expr) -> Self {
        Expr::BinaryExpr {
            left: Box::new(self),
            op: BinaryOperator::Eq,
            right: Box::new(other),
        }
    }

    pub fn neq(self, other: Expr) -> Self {
        Expr::BinaryExpr {
            left: Box::new(self),
            op: BinaryOperator::NotEq,
            right: Box::new(other),
        }
    }

    pub fn lt(self, other: Expr) -> Self {
        Expr::BinaryExpr {
            left: Box::new(self),
            op: BinaryOperator::Lt,
            right: Box::new(other),
        }
    }

    pub fn gt(self, other: Expr) -> Self {
        Expr::BinaryExpr {
            left: Box::new(self),
            op: BinaryOperator::Gt,
            right: Box::new(other),
        }
    }

    pub fn lte(self, other: Expr) -> Self {
        Expr::BinaryExpr {
            left: Box::new(self),
            op: BinaryOperator::LtEq,
            right: Box::new(other),
        }
    }

    pub fn gte(self, other: Expr) -> Self {
        Expr::BinaryExpr {
            left: Box::new(self),
            op: BinaryOperator::GtEq,
            right: Box::new(other),
        }
    }

    pub fn and(self, other: Expr) -> Self {
        Expr::BinaryExpr {
            left: Box::new(self),
            op: BinaryOperator::And,
            right: Box::new(other),
        }
    }

    pub fn or(self, other: Expr) -> Self {
        Expr::BinaryExpr {
            left: Box::new(self),
            op: BinaryOperator::Or,
            right: Box::new(other),
        }
    }
}

//Generated tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::series::AnyValue;

    // ============ BinaryOperator Tests ============

    #[test]
    fn test_binary_operator_equality() {
        assert_eq!(BinaryOperator::Plus, BinaryOperator::Plus);
        assert_eq!(BinaryOperator::Eq, BinaryOperator::Eq);
        assert_ne!(BinaryOperator::Plus, BinaryOperator::Minus);
    }

    #[test]
    fn test_binary_operator_debug() {
        assert_eq!(format!("{:?}", BinaryOperator::Plus), "Plus");
        assert_eq!(format!("{:?}", BinaryOperator::Eq), "Eq");
        assert_eq!(format!("{:?}", BinaryOperator::And), "And");
    }

    // ============ Expr Creation Tests ============

    #[test]
    fn test_expr_column_creation() {
        let expr = Expr::Column("name".to_string());

        match expr {
            Expr::Column(name) => assert_eq!(name, "name"),
            _ => panic!("Expected Column variant"),
        }
    }

    #[test]
    fn test_expr_literal_creation() {
        let expr = Expr::Literal(AnyValue::Int64(42));

        match expr {
            Expr::Literal(AnyValue::Int64(42)) => {}
            _ => panic!("Expected Literal with Int64(42)"),
        }
    }

    #[test]
    fn test_expr_binary_creation() {
        let left = Box::new(Expr::Column("age".to_string()));
        let right = Box::new(Expr::Literal(AnyValue::Int64(25)));
        let expr = Expr::BinaryExpr {
            left,
            op: BinaryOperator::Gt,
            right,
        };

        match expr {
            Expr::BinaryExpr { op, .. } => assert_eq!(op, BinaryOperator::Gt),
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_expr_alias_creation() {
        let inner = Box::new(Expr::Column("age".to_string()));
        let expr = Expr::Alias(inner, "user_age".to_string());

        match expr {
            Expr::Alias(_, alias) => assert_eq!(alias, "user_age"),
            _ => panic!("Expected Alias variant"),
        }
    }

    // ============ Helper Functions Tests ============

    #[test]
    fn test_col_helper() {
        let expr = Expr::col("name");

        match expr {
            Expr::Column(name) => assert_eq!(name, "name"),
            _ => panic!("Expected Column variant"),
        }
    }

    #[test]
    fn test_lit_helper_int() {
        let expr = Expr::lit(42i64);

        match expr {
            Expr::Literal(AnyValue::Int64(42)) => {}
            _ => panic!("Expected Literal with Int64(42)"),
        }
    }

    #[test]
    fn test_lit_helper_float() {
        let expr = Expr::lit(3.14f64);

        match expr {
            Expr::Literal(AnyValue::Float64(v)) => assert_eq!(v, 3.14),
            _ => panic!("Expected Literal with Float64"),
        }
    }

    #[test]
    fn test_lit_helper_string() {
        let expr = Expr::lit("hello");

        match expr {
            Expr::Literal(AnyValue::String(s)) => assert_eq!(s, "hello"),
            _ => panic!("Expected Literal with String"),
        }
    }

    #[test]
    fn test_lit_helper_bool() {
        let expr = Expr::lit(true);

        match expr {
            Expr::Literal(AnyValue::Boolean(true)) => {}
            _ => panic!("Expected Literal with Boolean(true)"),
        }
    }

    #[test]
    fn test_alias_helper() {
        let expr = Expr::col("age").alias("user_age");

        match expr {
            Expr::Alias(inner, alias) => {
                assert_eq!(alias, "user_age");
                match *inner {
                    Expr::Column(name) => assert_eq!(name, "age"),
                    _ => panic!("Expected Column inside Alias"),
                }
            }
            _ => panic!("Expected Alias variant"),
        }
    }

    // ============ Binary Operations Tests ============

    #[test]
    fn test_eq_method() {
        let expr = Expr::col("age").eq(Expr::lit(25));

        match expr {
            Expr::BinaryExpr { left, op, right } => {
                assert_eq!(op, BinaryOperator::Eq);
                match (*left, *right) {
                    (Expr::Column(name), Expr::Literal(AnyValue::Int64(25))) => {
                        assert_eq!(name, "age");
                    }
                    _ => panic!("Unexpected left/right expressions"),
                }
            }
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_gt_method() {
        let expr = Expr::col("score").gt(Expr::lit(80.0));

        match expr {
            Expr::BinaryExpr { op, .. } => assert_eq!(op, BinaryOperator::Gt),
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_lt_method() {
        let expr = Expr::col("age").lt(Expr::lit(30));

        match expr {
            Expr::BinaryExpr { op, .. } => assert_eq!(op, BinaryOperator::Lt),
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_gte_method() {
        let expr = Expr::col("score").gte(Expr::lit(90));

        match expr {
            Expr::BinaryExpr { op, .. } => assert_eq!(op, BinaryOperator::GtEq),
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_lte_method() {
        let expr = Expr::col("age").lte(Expr::lit(65));

        match expr {
            Expr::BinaryExpr { op, .. } => assert_eq!(op, BinaryOperator::LtEq),
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_neq_method() {
        let expr = Expr::col("status").neq(Expr::lit("inactive"));

        match expr {
            Expr::BinaryExpr { op, .. } => assert_eq!(op, BinaryOperator::NotEq),
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_add_method() {
        let expr = Expr::col("a").add(Expr::col("b"));

        match expr {
            Expr::BinaryExpr { left, op, right } => {
                assert_eq!(op, BinaryOperator::Plus);
                match (*left, *right) {
                    (Expr::Column(a), Expr::Column(b)) => {
                        assert_eq!(a, "a");
                        assert_eq!(b, "b");
                    }
                    _ => panic!("Unexpected left/right expressions"),
                }
            }
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_sub_method() {
        let expr = Expr::col("total").sub(Expr::lit(10));

        match expr {
            Expr::BinaryExpr { op, .. } => assert_eq!(op, BinaryOperator::Minus),
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_mul_method() {
        let expr = Expr::col("price").mul(Expr::lit(2));

        match expr {
            Expr::BinaryExpr { op, .. } => assert_eq!(op, BinaryOperator::Multiply),
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_div_method() {
        let expr = Expr::col("total").div(Expr::lit(3));

        match expr {
            Expr::BinaryExpr { op, .. } => assert_eq!(op, BinaryOperator::Divide),
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_and_method() {
        let expr = Expr::col("active").and(Expr::col("verified"));

        match expr {
            Expr::BinaryExpr { op, .. } => assert_eq!(op, BinaryOperator::And),
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_or_method() {
        let expr = Expr::col("admin").or(Expr::col("moderator"));

        match expr {
            Expr::BinaryExpr { op, .. } => assert_eq!(op, BinaryOperator::Or),
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    // ============ Complex Expression Tests ============

    #[test]
    fn test_chained_operations() {
        // (age > 18) AND (score >= 80)
        let expr = Expr::col("age")
            .gt(Expr::lit(18))
            .and(Expr::col("score").gte(Expr::lit(80)));

        match expr {
            Expr::BinaryExpr { left, op, right } => {
                assert_eq!(op, BinaryOperator::And);

                // Left side: age > 18
                match *left {
                    Expr::BinaryExpr {
                        op: BinaryOperator::Gt,
                        ..
                    } => {}
                    _ => panic!("Expected Gt operation on left"),
                }

                // Right side: score >= 80
                match *right {
                    Expr::BinaryExpr {
                        op: BinaryOperator::GtEq,
                        ..
                    } => {}
                    _ => panic!("Expected GtEq operation on right"),
                }
            }
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_arithmetic_with_alias() {
        // (price * 2).alias("double_price")
        let expr = Expr::col("price").mul(Expr::lit(2)).alias("double_price");

        match expr {
            Expr::Alias(inner, alias) => {
                assert_eq!(alias, "double_price");
                match *inner {
                    Expr::BinaryExpr {
                        op: BinaryOperator::Multiply,
                        ..
                    } => {}
                    _ => panic!("Expected Multiply operation inside alias"),
                }
            }
            _ => panic!("Expected Alias variant"),
        }
    }

    #[test]
    fn test_nested_expressions() {
        // ((a + b) * c)
        let expr = Expr::col("a").add(Expr::col("b")).mul(Expr::col("c"));

        match expr {
            Expr::BinaryExpr { left, op, right } => {
                assert_eq!(op, BinaryOperator::Multiply);

                // Left side should be (a + b)
                match *left {
                    Expr::BinaryExpr {
                        op: BinaryOperator::Plus,
                        ..
                    } => {}
                    _ => panic!("Expected Plus operation on left"),
                }

                // Right side should be column c
                match *right {
                    Expr::Column(name) => assert_eq!(name, "c"),
                    _ => panic!("Expected Column on right"),
                }
            }
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    // ============ Clone and Debug Tests ============

    #[test]
    fn test_expr_clone() {
        let expr1 = Expr::col("test").gt(Expr::lit(42));
        let expr2 = expr1.clone();

        // Both expressions should be equal after clone
        match (expr1, expr2) {
            (Expr::BinaryExpr { op: op1, .. }, Expr::BinaryExpr { op: op2, .. }) => {
                assert_eq!(op1, op2);
                assert_eq!(op1, BinaryOperator::Gt);
            }
            _ => panic!("Clone should preserve expression structure"),
        }
    }

    #[test]
    fn test_expr_debug() {
        let expr = Expr::col("test");
        let debug_str = format!("{:?}", expr);

        assert!(debug_str.contains("Column"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_binary_expr_debug() {
        let expr = Expr::col("a").eq(Expr::lit(1));
        let debug_str = format!("{:?}", expr);

        assert!(debug_str.contains("BinaryExpr"));
        assert!(debug_str.contains("Eq"));
    }

    // ============ Edge Cases Tests ============

    #[test]
    fn test_multiple_aliases() {
        // col("x").alias("y").alias("z") - last alias should win
        let expr = Expr::col("x").alias("y").alias("z");

        match expr {
            Expr::Alias(inner, alias) => {
                assert_eq!(alias, "z");
                match *inner {
                    Expr::Alias(inner2, alias2) => {
                        assert_eq!(alias2, "y");
                        match *inner2 {
                            Expr::Column(name) => assert_eq!(name, "x"),
                            _ => panic!("Expected Column at core"),
                        }
                    }
                    _ => panic!("Expected nested Alias"),
                }
            }
            _ => panic!("Expected Alias variant"),
        }
    }

    #[test]
    fn test_literal_operations() {
        // lit(5).add(lit(3))
        let expr = Expr::lit(5).add(Expr::lit(3));

        match expr {
            Expr::BinaryExpr { left, op, right } => {
                assert_eq!(op, BinaryOperator::Plus);
                match (*left, *right) {
                    (Expr::Literal(AnyValue::Int64(5)), Expr::Literal(AnyValue::Int64(3))) => {}
                    _ => panic!("Expected Int64 literals"),
                }
            }
            _ => panic!("Expected BinaryExpr variant"),
        }
    }

    #[test]
    fn test_empty_string_operations() {
        let expr = Expr::col("").eq(Expr::lit(""));

        match expr {
            Expr::BinaryExpr { left, op, right } => {
                assert_eq!(op, BinaryOperator::Eq);
                match (*left, *right) {
                    (Expr::Column(name), Expr::Literal(AnyValue::String(s))) => {
                        assert_eq!(name, "");
                        assert_eq!(s, "");
                    }
                    _ => panic!("Expected empty strings"),
                }
            }
            _ => panic!("Expected BinaryExpr variant"),
        }
    }
}
