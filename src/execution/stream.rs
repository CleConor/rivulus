use super::record_batch::RecordBatch;
use super::schema::Schema;
use std::sync::Arc;

pub type Result<T> = std::result::Result<T, StreamError>;

#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("Stream execution error: {message}")]
    Execution { message: String },

    #[error("Schema mismatch: expected {expected:?}, found {actual:?}")]
    SchemaMismatch {
        expected: Arc<Schema>,
        actual: Arc<Schema>,
    },

    #[error("Stream exhausted")]
    Exhausted,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub trait DataStream: Send + Sync + std::fmt::Debug {
    fn schema(&self) -> Arc<Schema>;

    fn next_batch(&mut self) -> Result<Option<RecordBatch>>;

    fn collect(mut self) -> Result<Vec<RecordBatch>>
    where
        Self: Sized,
    {
        let mut batches = Vec::new();
        while let Some(batch) = self.next_batch()? {
            batches.push(batch);
        }
        Ok(batches)
    }

    fn concatenate(self) -> Result<RecordBatch>
    where
        Self: Sized,
    {
        let schema = self.schema();
        let batches = self.collect()?;

        if batches.is_empty() {
            return Ok(RecordBatch::empty(schema));
        }

        RecordBatch::concat(&batches).map_err(|e| StreamError::Execution { message: e })
    }
}

pub type DataStreamRef = Box<dyn DataStream>;

#[derive(Debug)]
pub struct MemoryStream {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
    current_index: usize,
}

impl MemoryStream {
    pub fn new(schema: Arc<Schema>, batches: Vec<RecordBatch>) -> Result<Self> {
        for batch in &batches {
            if batch.schema().as_ref() != schema.as_ref() {
                return Err(StreamError::SchemaMismatch {
                    expected: schema.clone(),
                    actual: batch.schema().clone(),
                });
            }
        }

        Ok(Self {
            schema,
            batches,
            current_index: 0,
        })
    }

    pub fn from_single_batch(batch: RecordBatch) -> Self {
        Self {
            schema: batch.schema().clone(),
            batches: vec![batch],
            current_index: 0,
        }
    }

    pub fn empty(schema: Arc<Schema>) -> Self {
        Self {
            schema,
            batches: Vec::new(),
            current_index: 0,
        }
    }
}

impl DataStream for MemoryStream {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.current_index < self.batches.len() {
            let batch = self.batches[self.current_index].clone();
            self.current_index += 1;
            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug)]
pub struct FilterStream {
    input: DataStreamRef,
    predicate_column: String,
}

impl FilterStream {
    pub fn new(input: DataStreamRef, predicate_column: String) -> Self {
        Self {
            input,
            predicate_column,
        }
    }
}

impl DataStream for FilterStream {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema().clone()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if let Some(batch) = self.input.next_batch()? {
            let schema = batch.schema();
            let col_index =
                schema
                    .index_of(&self.predicate_column)
                    .ok_or_else(|| StreamError::Execution {
                        message: format!("Column '{}' not found in schema", self.predicate_column),
                    })?;

            let predicate_array = batch.column(col_index);
            if predicate_array.data_type() != &super::schema::DataType::Boolean {
                return Err(StreamError::Execution {
                    message: format!(
                        "Predicate column '{}' is not of boolean type",
                        self.predicate_column
                    ),
                });
            }

            let filtered_batch = RecordBatch::filter(&batch, predicate_array)
                .map_err(|e| StreamError::Execution { message: e })?;
            Ok(Some(filtered_batch))
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug)]
pub struct SelectStream {
    input: DataStreamRef,
    column_names: Vec<String>,
    output_schema: Arc<Schema>,
}

impl SelectStream {
    pub fn new(input: DataStreamRef, column_names: Vec<String>) -> Result<Self> {
        let input_schema = input.schema();
        let mut fields = Vec::new();

        for col_name in &column_names {
            let field =
                input_schema
                    .field_by_name(col_name)
                    .ok_or_else(|| StreamError::Execution {
                        message: format!("Column '{}' not found in schema", col_name),
                    })?;
            fields.push(field.clone());
        }

        let output_schema = Arc::new(Schema::new(fields));

        Ok(Self {
            input,
            column_names,
            output_schema,
        })
    }
}

impl DataStream for SelectStream {
    fn schema(&self) -> Arc<Schema> {
        self.output_schema.clone()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if let Some(batch) = self.input.next_batch()? {
            let column_names_str: Vec<&str> =
                self.column_names.iter().map(|s| s.as_str()).collect();
            let selected_batch = RecordBatch::select_columns_by_name(&batch, &column_names_str)
                .map_err(|e| StreamError::Execution { message: e })?;
            Ok(Some(selected_batch))
        } else {
            Ok(None)
        }
    }
}

//generated tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::array::{BooleanArray, PrimitiveArray, StringArray};
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
            to_array_ref(PrimitiveArray::<i64>::from_values(vec![
                id_start,
                id_start + 1,
            ])),
            to_array_ref(StringArray::new(vec![
                Some(format!("name_{}", id_start)),
                Some(format!("name_{}", id_start + 1)),
            ])),
            to_array_ref(BooleanArray::from_bools(vec![true, false])),
        ];
        RecordBatch::try_new(schema, columns).unwrap()
    }

    // ============ MemoryStream Tests ============

    #[test]
    fn test_memory_stream_creation() {
        let schema = create_test_schema();
        let batch = create_test_batch(1);
        let stream = MemoryStream::new(schema.clone(), vec![batch]).unwrap();

        assert_eq!(stream.schema().as_ref(), schema.as_ref());
    }

    #[test]
    fn test_memory_stream_schema_mismatch() {
        let schema1 = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let schema2 = Arc::new(Schema::new(vec![Field::new(
            "name",
            DataType::String,
            false,
        )]));

        let batch = RecordBatch::empty(schema2);
        let result = MemoryStream::new(schema1, vec![batch]);

        assert!(result.is_err());
        match result.unwrap_err() {
            StreamError::SchemaMismatch { .. } => {}
            _ => panic!("Expected SchemaMismatch error"),
        }
    }

    #[test]
    fn test_memory_stream_from_single_batch() {
        let batch = create_test_batch(1);
        let expected_rows = batch.num_rows();
        let mut stream = MemoryStream::from_single_batch(batch.clone());

        let first = stream.next_batch().unwrap();
        assert!(first.is_some());
        assert_eq!(first.unwrap().num_rows(), expected_rows);

        let second = stream.next_batch().unwrap();
        assert!(second.is_none());
    }

    #[test]
    fn test_memory_stream_empty() {
        let schema = create_test_schema();
        let mut stream = MemoryStream::empty(schema.clone());

        assert_eq!(stream.schema().as_ref(), schema.as_ref());

        let result = stream.next_batch().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_memory_stream_multiple_batches() {
        let schema = create_test_schema();
        let batch1 = create_test_batch(1);
        let batch2 = create_test_batch(3);

        let mut stream = MemoryStream::new(schema, vec![batch1, batch2]).unwrap();

        let first = stream.next_batch().unwrap();
        assert!(first.is_some());
        assert_eq!(first.unwrap().num_rows(), 2);

        let second = stream.next_batch().unwrap();
        assert!(second.is_some());
        assert_eq!(second.unwrap().num_rows(), 2);

        let third = stream.next_batch().unwrap();
        assert!(third.is_none());
    }

    #[test]
    fn test_stream_collect() {
        let schema = create_test_schema();
        let batch1 = create_test_batch(1);
        let batch2 = create_test_batch(3);

        let stream = MemoryStream::new(schema, vec![batch1, batch2]).unwrap();
        let collected = stream.collect().unwrap();

        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].num_rows(), 2);
        assert_eq!(collected[1].num_rows(), 2);
    }

    #[test]
    fn test_stream_concatenate() {
        let schema = create_test_schema();
        let batch1 = create_test_batch(1);
        let batch2 = create_test_batch(3);

        let stream = MemoryStream::new(schema, vec![batch1, batch2]).unwrap();
        let concatenated = stream.concatenate().unwrap();

        assert_eq!(concatenated.num_rows(), 4);
        assert_eq!(concatenated.num_columns(), 3);
    }

    #[test]
    fn test_stream_concatenate_empty() {
        let schema = create_test_schema();
        let stream = MemoryStream::empty(schema.clone());
        let concatenated = stream.concatenate().unwrap();

        assert_eq!(concatenated.num_rows(), 0);
        assert_eq!(concatenated.schema().as_ref(), schema.as_ref());
    }

    // ============ FilterStream Tests ============

    #[test]
    fn test_filter_stream_creation() {
        let batch = create_test_batch(1);
        let input = Box::new(MemoryStream::from_single_batch(batch));
        let filter = FilterStream::new(input, "active".to_string());

        assert_eq!(filter.schema().num_fields(), 3);
    }

    #[test]
    fn test_filter_stream_boolean_predicate() {
        let batch = create_test_batch(1);
        let input = Box::new(MemoryStream::from_single_batch(batch));
        let mut filter = FilterStream::new(input, "active".to_string());

        let result = filter.next_batch().unwrap();
        assert!(result.is_some());

        let filtered_batch = result.unwrap();
        // Il batch di test ha [true, false] per active, quindi dovremmo avere 1 riga
        assert_eq!(filtered_batch.num_rows(), 1);
    }

    #[test]
    fn test_filter_stream_no_matches() {
        // Creiamo un batch dove tutti i valori di 'active' sono false
        let schema = create_test_schema();
        let columns = vec![
            to_array_ref(PrimitiveArray::<i64>::from_values(vec![1, 2])),
            to_array_ref(StringArray::new(vec![
                Some("a".to_string()),
                Some("b".to_string()),
            ])),
            to_array_ref(BooleanArray::from_bools(vec![false, false])),
        ];
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        let input = Box::new(MemoryStream::from_single_batch(batch));
        let mut filter = FilterStream::new(input, "active".to_string());

        let result = filter.next_batch().unwrap();
        assert!(result.is_some());

        let filtered_batch = result.unwrap();
        assert_eq!(filtered_batch.num_rows(), 0);
    }

    #[test]
    fn test_filter_stream_multiple_batches() {
        let schema = create_test_schema();
        let batch1 = create_test_batch(1);
        let batch2 = create_test_batch(3);

        let input = Box::new(MemoryStream::new(schema, vec![batch1, batch2]).unwrap());
        let mut filter = FilterStream::new(input, "active".to_string());

        let first = filter.next_batch().unwrap();
        assert!(first.is_some());
        assert_eq!(first.unwrap().num_rows(), 1); // Solo true values

        let second = filter.next_batch().unwrap();
        assert!(second.is_some());
        assert_eq!(second.unwrap().num_rows(), 1); // Solo true values

        let third = filter.next_batch().unwrap();
        assert!(third.is_none());
    }

    // ============ SelectStream Tests ============

    #[test]
    fn test_select_stream_creation() {
        let batch = create_test_batch(1);
        let input = Box::new(MemoryStream::from_single_batch(batch));
        let select = SelectStream::new(input, vec!["id".to_string(), "name".to_string()]).unwrap();

        assert_eq!(select.schema().num_fields(), 2);
        assert_eq!(select.schema().field(0).name(), "id");
        assert_eq!(select.schema().field(1).name(), "name");
    }

    #[test]
    fn test_select_stream_single_column() {
        let batch = create_test_batch(1);
        let input = Box::new(MemoryStream::from_single_batch(batch));
        let mut select = SelectStream::new(input, vec!["name".to_string()]).unwrap();

        let result = select.next_batch().unwrap();
        assert!(result.is_some());

        let selected_batch = result.unwrap();
        assert_eq!(selected_batch.num_columns(), 1);
        assert_eq!(selected_batch.num_rows(), 2);
        assert_eq!(selected_batch.schema().field(0).name(), "name");
    }

    #[test]
    fn test_select_stream_reorder_columns() {
        let batch = create_test_batch(1);
        let input = Box::new(MemoryStream::from_single_batch(batch));
        let mut select =
            SelectStream::new(input, vec!["active".to_string(), "id".to_string()]).unwrap();

        let result = select.next_batch().unwrap();
        assert!(result.is_some());

        let selected_batch = result.unwrap();
        assert_eq!(selected_batch.num_columns(), 2);
        assert_eq!(selected_batch.schema().field(0).name(), "active");
        assert_eq!(selected_batch.schema().field(1).name(), "id");
    }

    #[test]
    fn test_select_stream_invalid_column() {
        let batch = create_test_batch(1);
        let input = Box::new(MemoryStream::from_single_batch(batch));
        let result = SelectStream::new(input, vec!["nonexistent".to_string()]);

        assert!(result.is_err());
        match result.unwrap_err() {
            StreamError::Execution { message } => {
                assert!(message.contains("nonexistent"));
            }
            _ => panic!("Expected Execution error"),
        }
    }

    #[test]
    fn test_select_stream_multiple_batches() {
        let schema = create_test_schema();
        let batch1 = create_test_batch(1);
        let batch2 = create_test_batch(3);

        let input = Box::new(MemoryStream::new(schema, vec![batch1, batch2]).unwrap());
        let mut select = SelectStream::new(input, vec!["id".to_string()]).unwrap();

        let first = select.next_batch().unwrap();
        assert!(first.is_some());
        assert_eq!(first.unwrap().num_columns(), 1);

        let second = select.next_batch().unwrap();
        assert!(second.is_some());
        assert_eq!(second.unwrap().num_columns(), 1);

        let third = select.next_batch().unwrap();
        assert!(third.is_none());
    }

    // ============ Integration Tests ============

    #[test]
    fn test_chained_operations() {
        let batch = create_test_batch(1);
        let input = Box::new(MemoryStream::from_single_batch(batch));

        // Prima filtriamo per active=true, poi selezioniamo solo name
        let filter = FilterStream::new(input, "active".to_string());
        let mut select = SelectStream::new(Box::new(filter), vec!["name".to_string()]).unwrap();

        let result = select.next_batch().unwrap();
        assert!(result.is_some());

        let final_batch = result.unwrap();
        assert_eq!(final_batch.num_columns(), 1);
        assert_eq!(final_batch.num_rows(), 1); // Solo true values
        assert_eq!(final_batch.schema().field(0).name(), "name");
    }

    #[test]
    fn test_collect_vs_concatenate() {
        let schema = create_test_schema();
        let batch1 = create_test_batch(1);
        let batch2 = create_test_batch(3);

        // Test collect
        let stream1 =
            MemoryStream::new(schema.clone(), vec![batch1.clone(), batch2.clone()]).unwrap();
        let collected = stream1.collect().unwrap();
        assert_eq!(collected.len(), 2);

        // Test concatenate
        let stream2 = MemoryStream::new(schema, vec![batch1, batch2]).unwrap();
        let concatenated = stream2.concatenate().unwrap();
        assert_eq!(concatenated.num_rows(), 4);
    }
}
