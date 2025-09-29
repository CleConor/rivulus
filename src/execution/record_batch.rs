use super::array::{
    Array, ArrayRef, BooleanArray, NullArray, PrimitiveArray, PrimitiveArrayBuilder, StringArray,
};
use super::schema::{DataType, Field, Schema};
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct RecordBatch {
    schema: Arc<Schema>,
    columns: Vec<ArrayRef>,
    num_rows: usize,
}

impl RecordBatch {
    pub fn try_new(schema: Arc<Schema>, columns: Vec<ArrayRef>) -> Result<Self, String> {
        if schema.num_fields() != columns.len() {
            return Err(format!(
                "Schema has {} fields but {} columns provided",
                schema.num_fields(),
                columns.len()
            ));
        }

        let num_rows = if columns.is_empty() {
            0
        } else {
            columns[0].len()
        };

        for (i, column) in columns.iter().enumerate() {
            if column.len() != num_rows {
                return Err(format!(
                    "Column {} has length {} but expected {}",
                    i,
                    column.len(),
                    num_rows
                ));
            }
        }

        for (i, (field, column)) in schema.fields().iter().zip(columns.iter()).enumerate() {
            if field.data_type() != column.data_type() {
                return Err(format!(
                    "Column {} has type {:?} but schema expects {:?}",
                    i,
                    column.data_type(),
                    field.data_type()
                ));
            }
        }

        Ok(Self {
            schema,
            columns,
            num_rows,
        })
    }

    pub fn new_unchecked(schema: Arc<Schema>, columns: Vec<ArrayRef>, num_rows: usize) -> Self {
        Self {
            schema,
            columns,
            num_rows,
        }
    }

    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn column(&self, index: usize) -> &ArrayRef {
        &self.columns[index]
    }

    pub fn column_by_name(&self, name: &str) -> Option<&ArrayRef> {
        self.schema.index_of(name).map(|index| &self.columns[index])
    }

    pub fn columns(&self) -> &[ArrayRef] {
        &self.columns
    }

    pub fn slice(&self, offset: usize, length: usize) -> Self {
        assert!(offset + length <= self.num_rows, "Slice out of bounds");

        let sliced_columns: Vec<ArrayRef> = self
            .columns
            .iter()
            .map(|col| col.slice(offset, length))
            .collect();

        Self {
            schema: self.schema.clone(),
            columns: sliced_columns,
            num_rows: length,
        }
    }

    pub fn take(&self, indices: &[usize]) -> Result<Self, String> {
        for &index in indices {
            if index >= self.num_rows {
                return Err(format!(
                    "Index {} out of bounds for {} rows",
                    index, self.num_rows
                ));
            }
        }

        let taken_columns: Result<Vec<ArrayRef>, String> = self
            .columns
            .iter()
            .map(|col| self.take_array(col, indices))
            .collect();

        Ok(Self {
            schema: self.schema.clone(),
            columns: taken_columns?,
            num_rows: indices.len(),
        })
    }

    fn take_array(&self, array: &ArrayRef, indices: &[usize]) -> Result<ArrayRef, String> {
        use super::array::{BooleanArray, NullArray, PrimitiveArray, StringArray};

        match array.data_type() {
            DataType::Int64 => {
                let src = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i64>>()
                    .unwrap();
                let mut builder = PrimitiveArrayBuilder::<i64>::with_capacity(indices.len());
                for &index in indices {
                    match src.value(index) {
                        Some(val) => builder.append_value(val),
                        None => builder.append_null(0),
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::Float64 => {
                let src = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f64>>()
                    .unwrap();
                let mut builder = PrimitiveArrayBuilder::<f64>::with_capacity(indices.len());
                for &index in indices {
                    match src.value(index) {
                        Some(val) => builder.append_value(val),
                        None => builder.append_null(0.0),
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::String => {
                let src = array.as_any().downcast_ref::<StringArray>().unwrap();
                let values: Vec<Option<String>> = indices
                    .iter()
                    .map(|&i| src.value(i).map(|s| s.to_string()))
                    .collect();
                Ok(Arc::new(StringArray::new(values)))
            }
            DataType::Boolean => {
                let src = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                let values: Vec<Option<bool>> = indices.iter().map(|&i| src.value(i)).collect();
                Ok(Arc::new(BooleanArray::new(values)))
            }
            DataType::Null => Ok(Arc::new(NullArray::new(indices.len()))),
        }
    }

    pub fn select_columns(&self, indices: &[usize]) -> Result<Self, String> {
        for &index in indices {
            if index >= self.num_columns() {
                return Err(format!(
                    "Column index {} out of bounds for {} columns",
                    index,
                    self.num_columns()
                ));
            }
        }

        let selected_fields: Vec<Field> = indices
            .iter()
            .map(|&i| self.schema.field(i).clone())
            .collect();

        let selected_columns: Vec<ArrayRef> =
            indices.iter().map(|&i| self.columns[i].clone()).collect();

        let new_schema = Arc::new(Schema::new(selected_fields));

        Ok(Self {
            schema: new_schema,
            columns: selected_columns,
            num_rows: self.num_rows,
        })
    }

    pub fn select_columns_by_name(&self, names: &[&str]) -> Result<Self, String> {
        let indices: Result<Vec<usize>, String> = names
            .iter()
            .map(|name| {
                self.schema
                    .index_of(name)
                    .ok_or_else(|| format!("Column '{}' not found", name))
            })
            .collect();

        self.select_columns(&indices?)
    }

    pub fn filter(&self, predicate: &ArrayRef) -> Result<Self, String> {
        if predicate.len() != self.num_rows {
            return Err(format!(
                "Predicate length {} doesn't match batch length {}",
                predicate.len(),
                self.num_rows
            ));
        }

        let bool_array = predicate
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or("Predicate must be a BooleanArray")?;

        let mut selected_indices = Vec::new();
        for i in 0..bool_array.len() {
            if let Some(true) = bool_array.value(i) {
                selected_indices.push(i);
            }
        }

        self.take(&selected_indices)
    }

    pub fn concat(batches: &[Self]) -> Result<Self, String> {
        if batches.is_empty() {
            return Err("Cannot concatenate empty batch list".to_string());
        }

        let first_schema = &batches[0].schema;
        for batch in batches.iter().skip(1) {
            if batch.schema.as_ref() != first_schema.as_ref() {
                return Err("All batches must have the same schema".to_string());
            }
        }

        let total_rows: usize = batches.iter().map(|b| b.num_rows).sum();

        let mut concatenated_columns = Vec::new();
        for col_idx in 0..first_schema.num_fields() {
            let column_arrays: Vec<&ArrayRef> = batches
                .iter()
                .map(|batch| &batch.columns[col_idx])
                .collect();

            let concatenated_column = Self::concat_arrays(&column_arrays)?;
            concatenated_columns.push(concatenated_column);
        }

        Ok(Self {
            schema: first_schema.clone(),
            columns: concatenated_columns,
            num_rows: total_rows,
        })
    }

    fn concat_arrays(arrays: &[&ArrayRef]) -> Result<ArrayRef, String> {
        use super::array::{BooleanArray, NullArray, PrimitiveArray, StringArray};

        if arrays.is_empty() {
            return Err("Cannot concatenate empty array list".to_string());
        }

        let data_type = arrays[0].data_type();
        let total_len: usize = arrays.iter().map(|a| a.len()).sum();

        match data_type {
            DataType::Int64 => {
                let mut builder = PrimitiveArrayBuilder::<i64>::with_capacity(total_len);
                for array in arrays {
                    let primitive_array = array
                        .as_any()
                        .downcast_ref::<PrimitiveArray<i64>>()
                        .unwrap();
                    for i in 0..primitive_array.len() {
                        match primitive_array.value(i) {
                            Some(val) => builder.append_value(val),
                            None => builder.append_null(0),
                        }
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::Float64 => {
                let mut builder = PrimitiveArrayBuilder::<f64>::with_capacity(total_len);
                for array in arrays {
                    let primitive_array = array
                        .as_any()
                        .downcast_ref::<PrimitiveArray<f64>>()
                        .unwrap();
                    for i in 0..primitive_array.len() {
                        match primitive_array.value(i) {
                            Some(val) => builder.append_value(val),
                            None => builder.append_null(0.0),
                        }
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::String => {
                let mut all_values = Vec::with_capacity(total_len);
                for array in arrays {
                    let string_array = array.as_any().downcast_ref::<StringArray>().unwrap();
                    for i in 0..string_array.len() {
                        all_values.push(string_array.value(i).map(|s| s.to_string()));
                    }
                }
                Ok(Arc::new(StringArray::new(all_values)))
            }
            DataType::Boolean => {
                let mut all_values = Vec::with_capacity(total_len);
                for array in arrays {
                    let bool_array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                    for i in 0..bool_array.len() {
                        all_values.push(bool_array.value(i));
                    }
                }
                Ok(Arc::new(BooleanArray::new(all_values)))
            }
            DataType::Null => Ok(Arc::new(NullArray::new(total_len))),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema.num_fields() != self.columns.len() {
            return Err(format!(
                "Schema has {} fields but {} columns present",
                self.schema.num_fields(),
                self.columns.len()
            ));
        }

        for (i, column) in self.columns.iter().enumerate() {
            if column.len() != self.num_rows {
                return Err(format!(
                    "Column {} has length {} but expected {}",
                    i,
                    column.len(),
                    self.num_rows
                ));
            }

            let expected_type = self.schema.field(i).data_type();
            let actual_type = column.data_type();
            if expected_type != actual_type {
                return Err(format!(
                    "Column {} has type {:?} but schema expects {:?}",
                    i, actual_type, expected_type
                ));
            }
        }

        Ok(())
    }

    pub fn memory_size(&self) -> usize {
        let schema_size = std::mem::size_of_val(self.schema.as_ref());
        let columns_overhead = std::mem::size_of::<Vec<ArrayRef>>();
        let array_refs_size = self.columns.len() * std::mem::size_of::<ArrayRef>();

        // Estimate array data sizes
        let arrays_data_size: usize = self
            .columns
            .iter()
            .map(|col| {
                match col.data_type() {
                    DataType::Int64 | DataType::Float64 => col.len() * 8,
                    DataType::Boolean => (col.len() + 7) / 8, // Packed bits
                    DataType::String => col.len() * 20,       // Rough estimate
                    DataType::Null => 16,                     // Just metadata
                }
            })
            .sum();

        schema_size + columns_overhead + array_refs_size + arrays_data_size
    }

    pub fn empty(schema: Arc<Schema>) -> Self {
        let cols = schema
            .fields()
            .iter()
            .map(|f| match f.data_type() {
                DataType::Int64 => Arc::new(PrimitiveArray::<i64>::from_values(vec![])) as ArrayRef,
                DataType::Float64 => {
                    Arc::new(PrimitiveArray::<f64>::from_values(vec![])) as ArrayRef
                }
                DataType::String => Arc::new(StringArray::new(vec![])) as ArrayRef,
                DataType::Boolean => Arc::new(BooleanArray::from_bools(vec![])) as ArrayRef,
                DataType::Null => Arc::new(NullArray::new(0)) as ArrayRef,
            })
            .collect();
        Self {
            schema,
            columns: cols,
            num_rows: 0,
        }
    }
}

impl fmt::Display for RecordBatch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.num_rows == 0 {
            return write!(f, "empty RecordBatch");
        }

        for field in self.schema.fields().iter() {
            write!(f, "[{}]", field.name())?;
        }
        writeln!(f)?;

        let rows_to_show = self.num_rows.min(10);
        for row_idx in 0..rows_to_show {
            for column in self.columns.iter() {
                let value_str = match column.data_type() {
                    DataType::Int64 => {
                        if let Some(array) = column.as_any().downcast_ref::<PrimitiveArray<i64>>() {
                            match array.value(row_idx) {
                                Some(val) => val.to_string(),
                                None => "null".to_string(),
                            }
                        } else {
                            "?".to_string()
                        }
                    }
                    DataType::Float64 => {
                        if let Some(array) = column.as_any().downcast_ref::<PrimitiveArray<f64>>() {
                            match array.value(row_idx) {
                                Some(val) => val.to_string(),
                                None => "null".to_string(),
                            }
                        } else {
                            "?".to_string()
                        }
                    }
                    DataType::String => {
                        if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
                            match array.value(row_idx) {
                                Some(val) => val.to_string(),
                                None => "null".to_string(),
                            }
                        } else {
                            "?".to_string()
                        }
                    }
                    DataType::Boolean => {
                        if let Some(array) = column.as_any().downcast_ref::<BooleanArray>() {
                            match array.value(row_idx) {
                                Some(val) => val.to_string(),
                                None => "null".to_string(),
                            }
                        } else {
                            "?".to_string()
                        }
                    }
                    DataType::Null => "null".to_string(),
                };

                write!(f, "[{}]", value_str)?;
            }
            writeln!(f)?;
        }

        if self.num_rows > 10 {
            writeln!(f, "... ({} more rows)", self.num_rows - 10)?;
        }

        Ok(())
    }
}

pub struct RecordBatchBuilder {
    schema: Arc<Schema>,
    columns: Vec<ArrayRef>,
    _capacity: usize,
}

impl RecordBatchBuilder {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self {
            schema,
            columns: Vec::new(),
            _capacity: 0,
        }
    }

    pub fn with_capacity(schema: Arc<Schema>, capacity: usize) -> Self {
        let num_fields = schema.num_fields();
        Self {
            schema,
            columns: Vec::with_capacity(num_fields),
            _capacity: capacity,
        }
    }

    pub fn add_column(&mut self, column: ArrayRef) -> Result<(), String> {
        if self.columns.len() >= self.schema.num_fields() {
            return Err("Cannot add more columns than schema defines".to_string());
        }

        let expected_field = self.schema.field(self.columns.len());
        if column.data_type() != expected_field.data_type() {
            return Err(format!(
                "Column type {:?} doesn't match expected type {:?}",
                column.data_type(),
                expected_field.data_type()
            ));
        }

        if !self.columns.is_empty() {
            let expected_len = self.columns[0].len();
            if column.len() != expected_len {
                return Err(format!(
                    "Column length {} doesn't match expected length {}",
                    column.len(),
                    expected_len
                ));
            }
        }

        self.columns.push(column);
        Ok(())
    }

    pub fn finish(self) -> Result<RecordBatch, String> {
        if self.columns.len() != self.schema.num_fields() {
            return Err(format!(
                "Expected {} columns but only {} provided",
                self.schema.num_fields(),
                self.columns.len()
            ));
        }

        RecordBatch::try_new(self.schema, self.columns)
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn is_complete(&self) -> bool {
        self.columns.len() == self.schema.num_fields()
    }
}

impl Default for RecordBatchBuilder {
    fn default() -> Self {
        Self::new(Arc::new(Schema::empty()))
    }
}

//generated tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::array::{BooleanArray, NullArray, PrimitiveArray, StringArray};
    use crate::execution::schema::{Field, Schema};

    fn to_array_ref<T: Array + 'static>(array: T) -> ArrayRef {
        Arc::new(array)
    }

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::String, true),
            Field::new("active", DataType::Boolean, false),
        ]))
    }

    fn create_test_columns() -> Vec<ArrayRef> {
        vec![
            to_array_ref(PrimitiveArray::<i64>::from_values(vec![1, 2, 3])),
            to_array_ref(StringArray::new(vec![
                Some("Alice".to_string()),
                None,
                Some("Charlie".to_string()),
            ])),
            to_array_ref(BooleanArray::from_bools(vec![true, false, true])),
        ]
    }

    #[test]
    fn test_new_record_batch_valid() {
        let schema = create_test_schema();
        let columns = create_test_columns();

        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.schema().as_ref(), schema.as_ref());
    }

    #[test]
    fn test_new_record_batch_schema_mismatch() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Float64, true), // Wrong type
        ]));
        let columns = vec![
            to_array_ref(PrimitiveArray::<i64>::from_values(vec![1, 2, 3])),
            to_array_ref(StringArray::new(vec![Some("Alice".to_string())])),
        ];

        let result = RecordBatch::try_new(schema, columns);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_record_batch_length_mismatch() {
        let schema = create_test_schema();
        let columns = vec![
            to_array_ref(PrimitiveArray::<i64>::from_values(vec![1, 2, 3])),
            to_array_ref(StringArray::new(vec![Some("Alice".to_string())])), // Wrong length
            to_array_ref(BooleanArray::from_bools(vec![true, false, true])),
        ];

        let result = RecordBatch::try_new(schema, columns);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_record_batch() {
        let schema = Arc::new(Schema::empty());
        let batch = RecordBatch::empty(schema.clone());

        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 0);
        assert!(batch.is_empty());
        assert_eq!(batch.schema().as_ref(), schema.as_ref());
    }

    #[test]
    fn test_column_access_by_index() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        let col0 = batch.column(0);
        let col1 = batch.column(1);
        let col2 = batch.column(2);

        assert_eq!(col0.data_type(), &DataType::Int64);
        assert_eq!(col1.data_type(), &DataType::String);
        assert_eq!(col2.data_type(), &DataType::Boolean);
    }

    #[test]
    fn test_column_access_by_name() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        let id_col = batch.column_by_name("id").unwrap();
        let name_col = batch.column_by_name("name").unwrap();
        let active_col = batch.column_by_name("active").unwrap();

        assert_eq!(id_col.data_type(), &DataType::Int64);
        assert_eq!(name_col.data_type(), &DataType::String);
        assert_eq!(active_col.data_type(), &DataType::Boolean);

        assert!(batch.column_by_name("nonexistent").is_none());
    }

    #[test]
    #[should_panic]
    fn test_column_access_out_of_bounds() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        batch.column(5); // Should panic
    }

    #[test]
    fn test_slice_operation() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();

        let sliced = batch.slice(1, 2);

        assert_eq!(sliced.num_rows(), 2);
        assert_eq!(sliced.num_columns(), 3);
        assert_eq!(sliced.schema().as_ref(), schema.as_ref());

        // Verify slicing preserved data
        let id_col = sliced
            .column(0)
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        assert_eq!(id_col.value(0), Some(2));
        assert_eq!(id_col.value(1), Some(3));
    }

    #[test]
    fn test_slice_boundary_conditions() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();

        // Empty slice
        let empty_slice = batch.slice(1, 0);
        assert_eq!(empty_slice.num_rows(), 0);
        assert!(empty_slice.is_empty());

        // Full slice
        let full_slice = batch.slice(0, 3);
        assert_eq!(full_slice.num_rows(), 3);

        // Single element slice
        let single_slice = batch.slice(2, 1);
        assert_eq!(single_slice.num_rows(), 1);
    }

    #[test]
    #[should_panic]
    fn test_slice_out_of_bounds() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        batch.slice(2, 5); // Should panic
    }

    #[test]
    fn test_take_operation() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();

        let taken = batch.take(&[2, 0, 1]).unwrap();

        assert_eq!(taken.num_rows(), 3);
        assert_eq!(taken.num_columns(), 3);

        let id_col = taken
            .column(0)
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        assert_eq!(id_col.value(0), Some(3)); // Index 2
        assert_eq!(id_col.value(1), Some(1)); // Index 0
        assert_eq!(id_col.value(2), Some(2)); // Index 1
    }

    #[test]
    fn test_take_empty_indices() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        let taken = batch.take(&[]).unwrap();
        assert_eq!(taken.num_rows(), 0);
        assert!(taken.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_take_invalid_indices() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        batch.take(&[0, 5, 1]).unwrap(); // Index 5 out of bounds
    }

    #[test]
    fn test_select_columns_by_index() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        let selected = batch.select_columns(&[0, 2]).unwrap();

        assert_eq!(selected.num_rows(), 3);
        assert_eq!(selected.num_columns(), 2);
        assert_eq!(selected.schema().field(0).name(), "id");
        assert_eq!(selected.schema().field(1).name(), "active");
    }

    #[test]
    fn test_select_columns_by_name() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        let selected = batch.select_columns_by_name(&["name", "id"]).unwrap();

        assert_eq!(selected.num_rows(), 3);
        assert_eq!(selected.num_columns(), 2);
        assert_eq!(selected.schema().field(0).name(), "name");
        assert_eq!(selected.schema().field(1).name(), "id");
    }

    #[test]
    fn test_filter_operation() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        let predicate = to_array_ref(BooleanArray::from_bools(vec![true, false, true]));
        let filtered = batch.filter(&predicate).unwrap();

        assert_eq!(filtered.num_rows(), 2);
        assert_eq!(filtered.num_columns(), 3);

        let id_col = filtered
            .column(0)
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        assert_eq!(id_col.value(0), Some(1));
        assert_eq!(id_col.value(1), Some(3));
    }

    #[test]
    fn test_filter_all_true() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        let predicate = to_array_ref(BooleanArray::all_true(3));
        let filtered = batch.filter(&predicate).unwrap();

        assert_eq!(filtered.num_rows(), 3);
        assert_eq!(filtered.num_columns(), 3);
    }

    #[test]
    fn test_filter_all_false() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        let predicate = to_array_ref(BooleanArray::all_false(3));
        let filtered = batch.filter(&predicate).unwrap();

        assert_eq!(filtered.num_rows(), 0);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_with_nulls() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        let predicate = to_array_ref(BooleanArray::new(vec![Some(true), None, Some(false)]));
        let filtered = batch.filter(&predicate).unwrap();

        // Nulls should be treated as false
        assert_eq!(filtered.num_rows(), 1);
    }

    #[test]
    fn test_concat_batches() {
        let schema = create_test_schema();

        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                to_array_ref(PrimitiveArray::<i64>::from_values(vec![1, 2])),
                to_array_ref(StringArray::new(vec![Some("A".to_string()), None])),
                to_array_ref(BooleanArray::from_bools(vec![true, false])),
            ],
        )
        .unwrap();

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                to_array_ref(PrimitiveArray::<i64>::from_values(vec![3, 4])),
                to_array_ref(StringArray::new(vec![
                    Some("B".to_string()),
                    Some("C".to_string()),
                ])),
                to_array_ref(BooleanArray::from_bools(vec![true, true])),
            ],
        )
        .unwrap();

        let concatenated = RecordBatch::concat(&[batch1, batch2]).unwrap();

        assert_eq!(concatenated.num_rows(), 4);
        assert_eq!(concatenated.num_columns(), 3);

        let id_col = concatenated
            .column(0)
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        assert_eq!(id_col.value(0), Some(1));
        assert_eq!(id_col.value(1), Some(2));
        assert_eq!(id_col.value(2), Some(3));
        assert_eq!(id_col.value(3), Some(4));
    }

    #[test]
    fn test_concat_empty_batches() {
        let schema = create_test_schema();
        let empty1 = RecordBatch::empty(schema.clone());
        let empty2 = RecordBatch::empty(schema.clone());

        let concatenated = RecordBatch::concat(&[empty1, empty2]).unwrap();
        assert_eq!(concatenated.num_rows(), 0);
        assert!(concatenated.is_empty());
    }

    #[test]
    fn test_concat_schema_mismatch() {
        let schema1 = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let schema2 = Arc::new(Schema::new(vec![Field::new(
            "name",
            DataType::String,
            false,
        )]));

        let batch1 = RecordBatch::empty(schema1);
        let batch2 = RecordBatch::empty(schema2);

        let result = RecordBatch::concat(&[batch1, batch2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_size_calculation() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        let memory_size = batch.memory_size();
        assert!(memory_size > 0);
    }

    #[test]
    fn test_validation_success() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        assert!(batch.validate().is_ok());
    }

    #[test]
    fn test_validation_failures() {
        let schema = create_test_schema();
        let mismatched_columns = vec![
            Arc::new(PrimitiveArray::<i64>::from_values(vec![1, 2, 3])),
            to_array_ref(StringArray::new(vec![Some("Alice".to_string())])), // Wrong length
            to_array_ref(BooleanArray::from_bools(vec![true, false, true])),
        ];

        let batch = RecordBatch::new_unchecked(schema, mismatched_columns, 3);
        assert!(batch.validate().is_err());
    }

    #[test]
    fn test_builder_basic_usage() {
        let schema = create_test_schema();
        let mut builder = RecordBatchBuilder::new(schema.clone());

        assert_eq!(builder.num_columns(), 0);
        assert!(!builder.is_complete());

        builder
            .add_column(Arc::new(PrimitiveArray::<i64>::from_values(vec![1, 2, 3])))
            .unwrap();
        builder
            .add_column(to_array_ref(StringArray::new(vec![
                Some("Alice".to_string()),
                None,
                Some("Charlie".to_string()),
            ])))
            .unwrap();
        builder
            .add_column(to_array_ref(BooleanArray::from_bools(vec![
                true, false, true,
            ])))
            .unwrap();

        assert_eq!(builder.num_columns(), 3);
        assert!(builder.is_complete());

        let batch = builder.finish().unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 3);
    }

    #[test]
    fn test_builder_with_capacity() {
        let schema = create_test_schema();
        let builder = RecordBatchBuilder::with_capacity(schema, 1000);

        assert_eq!(builder.num_columns(), 0);
    }

    #[test]
    fn test_builder_column_validation() {
        let schema = create_test_schema();
        let mut builder = RecordBatchBuilder::new(schema);

        // Try to add wrong type
        let wrong_type = to_array_ref(StringArray::new(vec![Some("test".to_string())]));
        let result = builder.add_column(wrong_type);
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_incomplete_error() {
        let schema = create_test_schema();
        let mut builder = RecordBatchBuilder::new(schema);

        builder
            .add_column(Arc::new(PrimitiveArray::<i64>::from_values(vec![1, 2, 3])))
            .unwrap();
        // Only added 1 of 3 expected columns

        let result = builder.finish();
        assert!(result.is_err());
    }

    #[test]
    fn test_mixed_column_types() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("int_col", DataType::Int64, false),
            Field::new("float_col", DataType::Float64, true),
            Field::new("string_col", DataType::String, true),
            Field::new("bool_col", DataType::Boolean, false),
            Field::new("null_col", DataType::Null, true),
        ]));

        let columns = vec![
            to_array_ref(PrimitiveArray::<i64>::from_values(vec![1, 2, 3])),
            to_array_ref(PrimitiveArray::<f64>::from_values(vec![1.1, 2.2, 3.3])),
            to_array_ref(StringArray::new(vec![
                Some("A".to_string()),
                None,
                Some("C".to_string()),
            ])),
            to_array_ref(BooleanArray::from_bools(vec![true, false, true])),
            to_array_ref(NullArray::new(3)),
        ];

        let batch = RecordBatch::try_new(schema, columns).unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 5);
    }

    #[test]
    fn test_large_batch_performance() {
        let size = 10_000;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ]));

        let columns = vec![
            to_array_ref(PrimitiveArray::<i64>::from_values((0..size).collect())),
            to_array_ref(PrimitiveArray::<f64>::from_values(
                (0..size).map(|i| i as f64 * 1.5).collect(),
            )),
        ];

        let start = std::time::Instant::now();
        let batch = RecordBatch::try_new(schema, columns).unwrap();
        let creation_time = start.elapsed();

        println!("Large batch creation time: {:?}", creation_time);
        assert_eq!(batch.num_rows(), size as usize);

        let start = std::time::Instant::now();
        let sliced = batch.slice(1000, 5000);
        let slice_time = start.elapsed();

        println!("Slice time: {:?}", slice_time);
        assert_eq!(sliced.num_rows(), 5000);
    }

    #[test]
    fn test_slice_preserves_schema() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();

        let sliced = batch.slice(1, 1);
        assert_eq!(sliced.schema().as_ref(), schema.as_ref());
        assert_eq!(sliced.schema().num_fields(), 3);
    }

    #[test]
    fn test_chained_operations() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        // Chain slice -> select -> filter
        let sliced = batch.slice(0, 3);
        let selected = sliced.select_columns(&[0, 2]).unwrap();
        let predicate = to_array_ref(BooleanArray::from_bools(vec![true, false, true]));
        let filtered = selected.filter(&predicate).unwrap();

        assert_eq!(filtered.num_rows(), 2);
        assert_eq!(filtered.num_columns(), 2);
    }

    #[test]
    fn test_null_column_handling() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("null_col", DataType::Null, true),
        ]));

        let columns = vec![
            to_array_ref(PrimitiveArray::<i64>::from_values(vec![1, 2, 3])),
            to_array_ref(NullArray::new(3)),
        ];

        let batch = RecordBatch::try_new(schema, columns).unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);

        let null_col = batch.column(1);
        assert_eq!(null_col.data_type(), &DataType::Null);
        assert_eq!(null_col.null_count(), 3);
    }

    #[test]
    fn test_as_any_downcast() {
        let schema = create_test_schema();
        let columns = create_test_columns();
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        let id_col = batch.column(0);
        let int_array = id_col.as_any().downcast_ref::<PrimitiveArray<i64>>();
        assert!(int_array.is_some());

        let name_col = batch.column(1);
        let string_array = name_col.as_any().downcast_ref::<StringArray>();
        assert!(string_array.is_some());

        let active_col = batch.column(2);
        let bool_array = active_col.as_any().downcast_ref::<BooleanArray>();
        assert!(bool_array.is_some());
    }
}
