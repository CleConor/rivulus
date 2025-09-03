# Rivulus

A simple query engine implementation in Rust, inspired by [Polars](https://pola.rs/posts/polars_birds_eye_view) architecture and concepts from [How Query Engines Work](https://howqueryengineswork.com/).

## Status

**⚠️ This is an incomplete and unoptimized version.**

Currently implements basic query operations (select, filter, limit) with a simplified execution model. Before adding more complex operations (joins, aggregations, window functions), the focus is on optimizing and parallelizing the existing codebase.

## Features

- Basic DataFrame operations
- Lazy evaluation with fluent API
- Simple expression system
- Logical and physical query planning
- Support for basic data types (Int64, Float64, String, Boolean)

## Example

```rust
use rivulus::{DataFrame, Series, AnyValue, Expr, LazyFrame};

// Create data
let names = Series::new("name", vec![
   AnyValue::String("Alice".to_string()),
   AnyValue::String("Bob".to_string()),
])?;

let ages = Series::new("age", vec![
   AnyValue::Int64(25),
   AnyValue::Int64(30),
])?;

let df = DataFrame::new(vec![names, ages])?;

// Query: SELECT name WHERE age > 25
let result = LazyFrame::from_dataframe(df)
   .select(vec![Expr::col("name")])
   .filter(Expr::col("age").gt(Expr::lit(25)))
   .collect()?;
```
