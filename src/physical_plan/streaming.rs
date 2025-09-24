use crate::execution::{DataStream, DataStreamRef, RecordBatch, Schema};
use crate::execution::stream::{MemoryStream, FilterStream, SelectStream, StreamError};
use crate::datatypes::DataFrame;
use crate::logical_plan::JoinType;
use std::sync::Arc;

pub type Result<T> = std::result::Result<T, StreamingExecutionError>;

#[derive(Debug, thiserror::Error)]
pub enum StreamingExecutionError {
    #[error("Stream error: {0}")]
    Stream(#[from] StreamError),

    #[error("Conversion error: {message}")]
    Conversion { message: String },

    #[error("Column not found: '{name}'")]
    ColumnNotFound { name: String },

    #[error("Type mismatch: {message}")]
    TypeMismatch { message: String },

    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },
}


#[derive(Debug)]
pub enum StreamingPhysicalPlan {
    
    MemorySource {
        batches: Vec<RecordBatch>,
    },
    
    DataFrameSource {
        df: DataFrame,
        batch_size: usize,
    },
    
    Filter {
        input: Box<StreamingPhysicalPlan>,
        predicate_column: String,
    },
    
    Select {
        input: Box<StreamingPhysicalPlan>,
        columns: Vec<String>,
    },
    
    Limit {
        input: Box<StreamingPhysicalPlan>,
        n: usize,
    },
    
    HashJoin {
        build_side: Box<StreamingPhysicalPlan>,
        probe_side: Box<StreamingPhysicalPlan>,
        build_key: String,
        probe_key: String,
        join_type: JoinType,
    },
}

impl StreamingPhysicalPlan {
    
    pub fn execute(self) -> Result<DataStreamRef> {
        match self {
            Self::MemorySource { batches } => {
                if batches.is_empty() {
                    return Err(StreamingExecutionError::InvalidOperation {
                        message: "Cannot create stream from empty batch list".to_string(),
                    });
                }

                let schema = batches[0].schema().clone();
                let stream = MemoryStream::new(schema, batches)?;
                Ok(Box::new(stream))
            }

            Self::DataFrameSource { df, batch_size } => {
                
                let batches = Self::dataframe_to_batches(df, batch_size)?;
                let schema = if batches.is_empty() {
                    
                    Arc::new(Schema::empty())
                } else {
                    batches[0].schema().clone()
                };
                let stream = MemoryStream::new(schema, batches)?;
                Ok(Box::new(stream))
            }

            Self::Filter { input, predicate_column } => {
                let input_stream = input.execute()?;
                let filter_stream = FilterStream::new(input_stream, predicate_column);
                Ok(Box::new(filter_stream))
            }

            Self::Select { input, columns } => {
                let input_stream = input.execute()?;
                let select_stream = SelectStream::new(input_stream, columns)?;
                Ok(Box::new(select_stream))
            }

            Self::Limit { input, n } => {
                let input_stream = input.execute()?;
                let limit_stream = LimitStream::new(input_stream, n);
                Ok(Box::new(limit_stream))
            }

            Self::HashJoin { .. } => {
                // TODO: Implement streaming hash join
                todo!("Streaming hash join not yet implemented")
            }
        }
    }

    
    fn dataframe_to_batches(df: DataFrame, batch_size: usize) -> Result<Vec<RecordBatch>> {
        use crate::execution::array::{PrimitiveArray, StringArray, BooleanArray, NullArray};
        use crate::execution::schema::{Schema, Field, DataType as ExecutionDataType};
        use crate::execution::array::ArrayRef;

        if df.is_empty() {
            return Ok(Vec::new());
        }

        let num_rows = df.height();
        let num_batches = (num_rows + batch_size - 1) / batch_size; 
        let mut batches = Vec::with_capacity(num_batches);

        
        let fields: Vec<_> = df.columns()
            .iter()
            .map(|series| {
                let execution_dtype = match series.dtype() {
                    crate::datatypes::series::DataType::Int64 => ExecutionDataType::Int64,
                    crate::datatypes::series::DataType::Float64 => ExecutionDataType::Float64,
                    crate::datatypes::series::DataType::String => ExecutionDataType::String,
                    crate::datatypes::series::DataType::Boolean => ExecutionDataType::Boolean,
                    crate::datatypes::series::DataType::Null => ExecutionDataType::Null,
                };
                Field::new(series.name(), execution_dtype, true) 
            })
            .collect();
        let schema = Arc::new(Schema::new(fields));

        
        for batch_idx in 0..num_batches {
            let start_row = batch_idx * batch_size;
            let end_row = ((batch_idx + 1) * batch_size).min(num_rows);
            let batch_rows = end_row - start_row;

            let mut arrays = Vec::with_capacity(df.width());

            
            for series in df.columns() {
                let array: ArrayRef = match series.dtype() {
                    crate::datatypes::series::DataType::Int64 => {
                        let values: Vec<i64> = (start_row..end_row)
                            .map(|i| match &series[i] {
                                crate::datatypes::series::AnyValue::Int64(val) => *val,
                                crate::datatypes::series::AnyValue::Null => 0, 
                                _ => panic!("Type mismatch in Int64 series"),
                            })
                            .collect();
                        Arc::new(PrimitiveArray::<i64>::from_values(values))
                    }

                    crate::datatypes::series::DataType::Float64 => {
                        let values: Vec<f64> = (start_row..end_row)
                            .map(|i| match &series[i] {
                                crate::datatypes::series::AnyValue::Float64(val) => *val,
                                crate::datatypes::series::AnyValue::Null => 0.0, 
                                _ => panic!("Type mismatch in Float64 series"),
                            })
                            .collect();
                        Arc::new(PrimitiveArray::<f64>::from_values(values))
                    }

                    crate::datatypes::series::DataType::String => {
                        let values: Vec<Option<String>> = (start_row..end_row)
                            .map(|i| match &series[i] {
                                crate::datatypes::series::AnyValue::String(val) => Some(val.clone()),
                                crate::datatypes::series::AnyValue::Null => None,
                                _ => panic!("Type mismatch in String series"),
                            })
                            .collect();
                        Arc::new(StringArray::new(values))
                    }

                    crate::datatypes::series::DataType::Boolean => {
                        let values: Vec<bool> = (start_row..end_row)
                            .map(|i| match &series[i] {
                                crate::datatypes::series::AnyValue::Boolean(val) => *val,
                                crate::datatypes::series::AnyValue::Null => false, 
                                _ => panic!("Type mismatch in Boolean series"),
                            })
                            .collect();
                        Arc::new(BooleanArray::from_bools(values))
                    }

                    crate::datatypes::series::DataType::Null => {
                        Arc::new(NullArray::new(batch_rows))
                    }
                };
                arrays.push(array);
            }

            let batch = RecordBatch::try_new(schema.clone(), arrays)
                .map_err(|e| StreamingExecutionError::Conversion { message: e })?;

            batches.push(batch);
        }

        Ok(batches)
    }

    
    pub fn collect(self) -> Result<RecordBatch> {
        let mut stream = self.execute()?;
        collect_stream_batches(&mut *stream)
    }

    
    pub fn collect_batches(self) -> Result<Vec<RecordBatch>> {
        let mut stream = self.execute()?;
        collect_all_batches(&mut *stream)
    }
}


#[derive(Debug)]
pub struct LimitStream {
    input: DataStreamRef,
    limit: usize,
    rows_returned: usize,
}

impl LimitStream {
    pub fn new(input: DataStreamRef, limit: usize) -> Self {
        Self {
            input,
            limit,
            rows_returned: 0,
        }
    }
}

impl DataStream for LimitStream {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema()
    }

    fn next_batch(&mut self) -> crate::execution::stream::Result<Option<RecordBatch>> {
        if self.rows_returned >= self.limit {
            return Ok(None);
        }

        if let Some(batch) = self.input.next_batch()? {
            let remaining = self.limit - self.rows_returned;

            if batch.num_rows() <= remaining {
                
                self.rows_returned += batch.num_rows();
                Ok(Some(batch))
            } else {
                
                let limited_batch = batch.slice(0, remaining);
                self.rows_returned += remaining;
                Ok(Some(limited_batch))
            }
        } else {
            Ok(None)
        }
    }
}


impl StreamingPhysicalPlan {
    pub fn memory_source(batches: Vec<RecordBatch>) -> Self {
        Self::MemorySource { batches }
    }

    pub fn dataframe_source(df: DataFrame, batch_size: usize) -> Self {
        Self::DataFrameSource { df, batch_size }
    }

    pub fn filter(self, predicate_column: String) -> Self {
        Self::Filter {
            input: Box::new(self),
            predicate_column,
        }
    }

    pub fn select(self, columns: Vec<String>) -> Self {
        Self::Select {
            input: Box::new(self),
            columns,
        }
    }

    pub fn limit(self, n: usize) -> Self {
        Self::Limit {
            input: Box::new(self),
            n,
        }
    }
}


fn collect_all_batches(stream: &mut dyn DataStream) -> Result<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    while let Some(batch) = stream.next_batch()? {
        batches.push(batch);
    }
    Ok(batches)
}

fn collect_stream_batches(stream: &mut dyn DataStream) -> Result<RecordBatch> {
    let schema = stream.schema();
    let batches = collect_all_batches(stream)?;

    if batches.is_empty() {
        return Ok(RecordBatch::empty(schema));
    }

    RecordBatch::concat(&batches)
        .map_err(|e| StreamingExecutionError::Conversion { message: e })
}

//generated tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::array::{PrimitiveArray, StringArray, BooleanArray};
    use crate::execution::schema::{DataType, Field};

    fn to_array_ref<T: crate::execution::array::Array + 'static>(
        array: T,
    ) -> crate::execution::array::ArrayRef {
        Arc::new(array)
    }

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::String, true),
            Field::new("active", DataType::Boolean, false),
        ]))
    }

    fn create_test_batch(id_start: i64) -> RecordBatch {
        let schema = create_test_schema();
        let columns = vec![
            to_array_ref(PrimitiveArray::<i64>::from_values(vec![id_start, id_start + 1])),
            to_array_ref(StringArray::new(vec![
                Some(format!("name_{}", id_start)),
                Some(format!("name_{}", id_start + 1)),
            ])),
            to_array_ref(BooleanArray::from_bools(vec![true, false])),
        ];
        RecordBatch::try_new(schema, columns).unwrap()
    }

    #[test]
    fn test_streaming_plan_memory_source() {
        let batch1 = create_test_batch(1);
        let batch2 = create_test_batch(3);

        let plan = StreamingPhysicalPlan::memory_source(vec![batch1, batch2]);
        let result = plan.collect().unwrap();

        assert_eq!(result.num_rows(), 4);
        assert_eq!(result.num_columns(), 3);
    }

    #[test]
    fn test_streaming_plan_filter() {
        let batch1 = create_test_batch(1);
        let batch2 = create_test_batch(3);

        let plan = StreamingPhysicalPlan::memory_source(vec![batch1, batch2])
            .filter("active".to_string());

        let result = plan.collect().unwrap();

        // Only rows with active=true should remain (1 per batch)
        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 3);
    }

    #[test]
    fn test_streaming_plan_select() {
        let batch1 = create_test_batch(1);
        let batch2 = create_test_batch(3);

        let plan = StreamingPhysicalPlan::memory_source(vec![batch1, batch2])
            .select(vec!["id".to_string(), "name".to_string()]);

        let result = plan.collect().unwrap();

        assert_eq!(result.num_rows(), 4);
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.schema().field(0).name(), "id");
        assert_eq!(result.schema().field(1).name(), "name");
    }

    #[test]
    fn test_streaming_plan_limit() {
        let batch1 = create_test_batch(1);
        let batch2 = create_test_batch(3);

        let plan = StreamingPhysicalPlan::memory_source(vec![batch1, batch2])
            .limit(3);

        let result = plan.collect().unwrap();

        assert_eq!(result.num_rows(), 3);
        assert_eq!(result.num_columns(), 3);
    }

    #[test]
    fn test_streaming_plan_chained_operations() {
        let batch1 = create_test_batch(1);
        let batch2 = create_test_batch(3);

        let plan = StreamingPhysicalPlan::memory_source(vec![batch1, batch2])
            .filter("active".to_string())
            .select(vec!["name".to_string()])
            .limit(1);

        let result = plan.collect().unwrap();

        assert_eq!(result.num_rows(), 1);
        assert_eq!(result.num_columns(), 1);
        assert_eq!(result.schema().field(0).name(), "name");
    }

    #[test]
    fn test_limit_stream_exact_batch_size() {
        let batch1 = create_test_batch(1); // 2 rows
        let batch2 = create_test_batch(3); // 2 rows

        let stream = MemoryStream::new(create_test_schema(), vec![batch1, batch2]).unwrap();
        let mut limit_stream = LimitStream::new(Box::new(stream), 2);

        let first = limit_stream.next_batch().unwrap();
        assert!(first.is_some());
        assert_eq!(first.unwrap().num_rows(), 2);

        let second = limit_stream.next_batch().unwrap();
        assert!(second.is_none());
    }

    #[test]
    fn test_limit_stream_partial_batch() {
        let batch1 = create_test_batch(1); // 2 rows
        let batch2 = create_test_batch(3); // 2 rows

        let stream = MemoryStream::new(create_test_schema(), vec![batch1, batch2]).unwrap();
        let mut limit_stream = LimitStream::new(Box::new(stream), 3);

        let first = limit_stream.next_batch().unwrap();
        assert!(first.is_some());
        assert_eq!(first.unwrap().num_rows(), 2);

        let second = limit_stream.next_batch().unwrap();
        assert!(second.is_some());
        assert_eq!(second.unwrap().num_rows(), 1); // Limited to 1 row

        let third = limit_stream.next_batch().unwrap();
        assert!(third.is_none());
    }

    #[test]
    fn test_collect_vs_collect_batches() {
        let batch1 = create_test_batch(1);
        let batch2 = create_test_batch(3);

        // Test collect (concatenates)
        let plan1 = StreamingPhysicalPlan::memory_source(vec![batch1.clone(), batch2.clone()]);
        let single_batch = plan1.collect().unwrap();
        assert_eq!(single_batch.num_rows(), 4);

        // Test collect_batches (keeps separate)
        let plan2 = StreamingPhysicalPlan::memory_source(vec![batch1, batch2]);
        let batch_vec = plan2.collect_batches().unwrap();
        assert_eq!(batch_vec.len(), 2);
        assert_eq!(batch_vec[0].num_rows(), 2);
        assert_eq!(batch_vec[1].num_rows(), 2);
    }
}