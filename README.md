# Rivulus

A simple query engine implementation in Rust, inspired by [Polars](https://pola.rs/posts/polars_birds_eye_view) architecture and concepts from [How Query Engines Work](https://howqueryengineswork.com/).

## Status

**⚠️ This is an incomplete and unoptimized version.**

Currently implements basic query operations (select, filter, limit) with both traditional eager execution and a new streaming execution engine. Recent additions include:

- ✅ **Streaming execution system** with lazy evaluation and early termination
- ✅ **Memory-efficient array implementations** (BitMap-backed BooleanArray, NullArray, zero-copy slicing)
- ✅ **RecordBatch-based processing** with Arrow-compatible memory layout
- ✅ **FileStream implementation**: with adaptive batch sizing
- ⏳ **Planned**: Memory optimizations, parallelization, and advanced operations

**Note**: The streaming system currently supports basic operations only. Complex operations (joins, aggregations, binary expressions in filters) still need implementation.

## Features

### Core Query Engine
- Basic DataFrame operations
- Fluent API with lazy evaluation
- Expression system
- Logical and physical query planning
- Support for basic data types (Int64, Float64, String, Boolean)

### Streaming Execution Engine
- **Lazy evaluation** with early termination (LIMIT operations)
- **Memory-bounded processing** using RecordBatch chunking
- **Zero-copy operations** where possible
- **Dual execution modes**: `collect()` for traditional, `collect_streaming()` for streaming

### Memory-Efficient Arrays
- **BooleanArray**: BitMap-backed storage with 8x memory savings
- **PrimitiveArray<T>**: Generic arrays for numeric types with null support
- **StringArray**: Variable-length strings with offset buffer design
- **NullArray**: Ultra-compact storage for completely null columns
- **Zero-copy slicing**: All arrays support efficient slicing via `Arc<[T]>`

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
