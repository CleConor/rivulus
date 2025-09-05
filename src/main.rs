use rivulus::datatypes::{AnyValue, DataFrame, Series};
use rivulus::expressions::Expr;
use rivulus::logical_plan::LazyFrame;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple Query Engine Demo ===\n");

    // 1. Crea dati di test
    let names = Series::new(
        "name",
        vec![
            AnyValue::String("Alice".to_string()),
            AnyValue::String("Bob".to_string()),
            AnyValue::String("Charlie".to_string()),
            AnyValue::String("Diana".to_string()),
            AnyValue::String("Eve".to_string()),
        ],
    )?;

    let ages = Series::new(
        "age",
        vec![
            AnyValue::Int64(25),
            AnyValue::Int64(30),
            AnyValue::Int64(35),
            AnyValue::Int64(28),
            AnyValue::Int64(42),
        ],
    )?;

    let scores = Series::new(
        "score",
        vec![
            AnyValue::Float64(85.5),
            AnyValue::Float64(92.0),
            AnyValue::Float64(78.5),
            AnyValue::Float64(94.5),
            AnyValue::Float64(88.0),
        ],
    )?;

    let df = DataFrame::new(vec![names, ages, scores])?;

    println!("Original DataFrame:");
    println!("{}\n", df);

    // 2. Query semplice: SELECT name, age FROM df WHERE age > 30
    println!("Query 1: SELECT name, age WHERE age > 30");
    let result1 = LazyFrame::from_dataframe(df.clone())
        .select(vec![Expr::col("name"), Expr::col("age")])
        .filter(Expr::col("age").gt(Expr::lit(30)))
        .collect()?;

    println!("Result:");
    println!("{}\n", result1);

    // 3. Query con alias: SELECT name, age as user_age WHERE score >= 90
    println!("Query 2: SELECT name, age as user_age WHERE score >= 90");
    let result2 = LazyFrame::from_dataframe(df.clone())
        .filter(Expr::col("score").gte(Expr::lit(90.0)))
        .select(vec![
            Expr::col("name"),
            Expr::col("age"), /*.alias("user_age")*/
        ]) //alias bug
        .collect()?;

    println!("Result:");
    println!("{}\n", result2);

    // 4. Query con LIMIT: SELECT * WHERE age < 40 LIMIT 2
    println!("Query 3: SELECT * WHERE age < 40 LIMIT 2");
    let result3 = LazyFrame::from_dataframe(df.clone())
        .filter(Expr::col("age").lt(Expr::lit(40)))
        .limit(2)
        .collect()?;

    println!("Result:");
    println!("{}\n", result3);

    // 5. Query che non trova risultati
    println!("Query 4: SELECT * WHERE age > 100 (no matches)");
    let result4 = LazyFrame::from_dataframe(df.clone())
        .filter(Expr::col("age").gt(Expr::lit(100)))
        .collect()?;

    println!("Result:");
    println!("{}\n", result4);

    // 6. Query con LIMIT 0
    println!("Query 5: SELECT name LIMIT 0");
    let result5 = LazyFrame::from_dataframe(df)
        .select(vec![Expr::col("name")])
        .limit(0)
        .collect()?;

    println!("Result:");
    println!("{}\n", result5);

    println!("Query 6: Inner Join - Users with their Orders");
    // Crea tabella users
    let user_ids = Series::new(
        "user_id",
        vec![
            AnyValue::Int64(1),
            AnyValue::Int64(2),
            AnyValue::Int64(3),
            AnyValue::Int64(4),
        ],
    )?;

    let user_names = Series::new(
        "name",
        vec![
            AnyValue::String("Alice".to_string()),
            AnyValue::String("Bob".to_string()),
            AnyValue::String("Charlie".to_string()),
            AnyValue::String("Diana".to_string()),
        ],
    )?;

    let user_cities = Series::new(
        "city",
        vec![
            AnyValue::String("NYC".to_string()),
            AnyValue::String("LA".to_string()),
            AnyValue::String("Chicago".to_string()),
            AnyValue::String("Boston".to_string()),
        ],
    )?;

    let users_df = DataFrame::new(vec![user_ids, user_names, user_cities])?;

    // Crea tabella orders
    let order_ids = Series::new(
        "order_id",
        vec![
            AnyValue::Int64(101),
            AnyValue::Int64(102),
            AnyValue::Int64(103),
            AnyValue::Int64(104),
            AnyValue::Int64(105),
        ],
    )?;

    let order_user_ids = Series::new(
        "user_id",
        vec![
            AnyValue::Int64(1),  // Alice
            AnyValue::Int64(2),  // Bob
            AnyValue::Int64(1),  // Alice again
            AnyValue::Int64(3),  // Charlie
            AnyValue::Int64(99), // User che non esiste
        ],
    )?;

    let amounts = Series::new(
        "amount",
        vec![
            AnyValue::Float64(25.99),
            AnyValue::Float64(15.50),
            AnyValue::Float64(99.99),
            AnyValue::Float64(45.00),
            AnyValue::Float64(12.99),
        ],
    )?;

    let orders_df = DataFrame::new(vec![order_ids, order_user_ids, amounts])?;

    println!("Users table:");
    println!("{}\n", users_df);
    println!("Orders table:");
    println!("{}\n", orders_df);

    // JOIN: SELECT * FROM users u INNER JOIN orders o ON u.user_id = o.user_id
    let join_result = LazyFrame::from_dataframe(users_df.clone())
        .inner_join(
            LazyFrame::from_dataframe(orders_df.clone()),
            "user_id".to_string(),
            "user_id".to_string(),
        )
        .collect()?;

    println!("Join Result (users with orders):");
    println!("{}\n", join_result);

    // 7. Join con Select: mostra solo alcune colonne
    println!("Query 7: Join + Select - Show only name and amount");
    let join_select_result = LazyFrame::from_dataframe(users_df.clone())
        .inner_join(
            LazyFrame::from_dataframe(orders_df.clone()),
            "user_id".to_string(),
            "user_id".to_string(),
        )
        .select(vec![
            Expr::col("name"),
            Expr::col("amount"),
            Expr::col("city"),
        ])
        .collect()?;

    println!("Result:");
    println!("{}\n", join_select_result);

    println!("=== Demo completed successfully! ===");

    Ok(())
}
