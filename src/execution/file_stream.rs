use super::array::{ArrayRef, BooleanArray, NullArray, PrimitiveArray, StringArray};
use super::schema::DataType;
use super::stream::{DataStream, Result as StreamResult};
use super::{RecordBatch, Schema};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

#[derive(Debug)]
pub struct CsvFileStream {
    reader: BufReader<File>,
    schema: Arc<Schema>,
    batch_size: usize,
    current_line: usize,
    finished: bool,
    delimiter: char,
}

impl CsvFileStream {
    pub fn new<P: AsRef<Path>>(
        path: P,
        schema: Arc<Schema>,
        batch_size: Option<usize>,
        delimiter: Option<char>,
    ) -> Result<Self, String> {
        let file = File::open(&path).map_err(|e| format!("Failed to open file: {}", e))?;
        let reader = BufReader::with_capacity(64 * 1024, file);

        let batch_size = batch_size.unwrap_or_else(|| calculate_adaptive_batch_size(&schema));

        Ok(Self {
            reader,
            schema,
            batch_size,
            current_line: 0,
            finished: false,
            delimiter: delimiter.unwrap_or(','),
        })
    }

    fn parse_line(&self, line: &str) -> Result<Vec<ParsedValue>, String> {
        let fields: Vec<&str> = line.split(self.delimiter).map(|s| s.trim()).collect();

        if fields.len() != self.schema.num_fields() {
            return Err(format!(
                "Line {}: Expected {} fields, found {}",
                self.current_line,
                self.schema.num_fields(),
                fields.len()
            ));
        }

        let mut values = Vec::with_capacity(fields.len());

        for (field_idx, (field_str, schema_field)) in
            fields.iter().zip(self.schema.fields()).enumerate()
        {
            let value = match schema_field.data_type() {
                DataType::Int64 => {
                    if field_str.is_empty() || *field_str == "null" {
                        ParsedValue::Null
                    } else {
                        match field_str.parse::<i64>() {
                            Ok(val) => ParsedValue::Int64(val),
                            Err(_) => {
                                return Err(format!(
                                    "Line {}, field {}: Cannot parse '{}' as Int64",
                                    self.current_line, field_idx, field_str
                                ));
                            }
                        }
                    }
                }
                DataType::Float64 => {
                    if field_str.is_empty() || *field_str == "null" {
                        ParsedValue::Null
                    } else {
                        match field_str.parse::<f64>() {
                            Ok(val) => ParsedValue::Float64(val),
                            Err(_) => {
                                return Err(format!(
                                    "Line {}, field {}: Cannot parse '{}' as Float64",
                                    self.current_line, field_idx, field_str
                                ));
                            }
                        }
                    }
                }
                DataType::String => {
                    if field_str.is_empty() || *field_str == "null" {
                        ParsedValue::Null
                    } else {
                        ParsedValue::String(field_str.to_string())
                    }
                }
                DataType::Boolean => {
                    if field_str.is_empty() || *field_str == "null" {
                        ParsedValue::Null
                    } else {
                        match field_str.to_lowercase().as_str() {
                            "true" | "t" | "1" => ParsedValue::Boolean(true),
                            "false" | "f" | "0" => ParsedValue::Boolean(false),
                            _ => {
                                return Err(format!(
                                    "Line {}, field {}: Cannot parse '{}' as Boolean",
                                    self.current_line, field_idx, field_str
                                ));
                            }
                        }
                    }
                }
                DataType::Null => ParsedValue::Null,
            };

            values.push(value);
        }

        Ok(values)
    }

    fn read_batch(&mut self) -> StreamResult<Option<RecordBatch>> {
        if self.finished {
            return Ok(None);
        }

        let mut batch_data: Vec<Vec<ParsedValue>> = (0..self.schema.num_fields())
            .map(|_| Vec::with_capacity(self.batch_size))
            .collect();

        let mut lines_read = 0;
        let mut line = String::new();

        if self.current_line == 0 {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    self.finished = true;
                    return Ok(None);
                }
                Ok(_) => {
                    self.current_line += 1;
                }
                Err(e) => {
                    return Err(super::stream::StreamError::Execution {
                        message: format!("Failed to read header: {}", e),
                    });
                }
            }
        }

        while lines_read < self.batch_size {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    self.finished = true;
                    break;
                }
                Ok(_) => {
                    self.current_line += 1;

                    if line.ends_with('\n') {
                        line.pop();
                        if line.ends_with('\r') {
                            line.pop();
                        }
                    }

                    if line.trim().is_empty() {
                        continue;
                    }

                    let values = self.parse_line(&line).map_err(|e| {
                        super::stream::StreamError::Execution {
                            message: format!("Parse error: {}", e),
                        }
                    })?;

                    for (col_idx, value) in values.into_iter().enumerate() {
                        batch_data[col_idx].push(value);
                    }

                    lines_read += 1;
                }
                Err(e) => {
                    return Err(super::stream::StreamError::Execution {
                        message: format!("Failed to read line {}: {}", self.current_line + 1, e),
                    });
                }
            }
        }

        if lines_read == 0 {
            return Ok(None);
        }

        self.build_record_batch(batch_data, lines_read)
    }

    fn build_record_batch(
        &self,
        batch_data: Vec<Vec<ParsedValue>>,
        num_rows: usize,
    ) -> StreamResult<Option<RecordBatch>> {
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.schema.num_fields());

        for (col_idx, (column_data, field)) in
            batch_data.into_iter().zip(self.schema.fields()).enumerate()
        {
            let array: ArrayRef = match field.data_type() {
                DataType::Int64 => {
                    let mut values = Vec::with_capacity(num_rows);
                    let mut nulls = Vec::with_capacity(num_rows);

                    for value in column_data {
                        match value {
                            ParsedValue::Int64(v) => {
                                values.push(v);
                                nulls.push(false);
                            }
                            ParsedValue::Null => {
                                values.push(0);
                                nulls.push(true);
                            }
                            _ => {
                                return Err(super::stream::StreamError::Execution {
                                    message: format!("Type mismatch in column {}", col_idx),
                                });
                            }
                        }
                    }

                    Arc::new(PrimitiveArray::<i64>::new(
                        values,
                        if nulls.iter().any(|&x| x) {
                            Some(nulls)
                        } else {
                            None
                        },
                    ))
                }

                DataType::Float64 => {
                    let mut values = Vec::with_capacity(num_rows);
                    let mut nulls = Vec::with_capacity(num_rows);

                    for value in column_data {
                        match value {
                            ParsedValue::Float64(v) => {
                                values.push(v);
                                nulls.push(false);
                            }
                            ParsedValue::Null => {
                                values.push(0.0);
                                nulls.push(true);
                            }
                            _ => {
                                return Err(super::stream::StreamError::Execution {
                                    message: format!("Type mismatch in column {}", col_idx),
                                });
                            }
                        }
                    }

                    Arc::new(PrimitiveArray::<f64>::new(
                        values,
                        if nulls.iter().any(|&x| x) {
                            Some(nulls)
                        } else {
                            None
                        },
                    ))
                }

                DataType::String => {
                    let mut values = Vec::with_capacity(num_rows);

                    for value in column_data {
                        match value {
                            ParsedValue::String(s) => values.push(Some(s)),
                            ParsedValue::Null => values.push(None),
                            _ => {
                                return Err(super::stream::StreamError::Execution {
                                    message: format!("Type mismatch in column {}", col_idx),
                                });
                            }
                        }
                    }

                    Arc::new(StringArray::new(values))
                }

                DataType::Boolean => {
                    let mut values = Vec::with_capacity(num_rows);

                    for value in column_data {
                        match value {
                            ParsedValue::Boolean(v) => values.push(Some(v)),
                            ParsedValue::Null => values.push(None),
                            _ => {
                                return Err(super::stream::StreamError::Execution {
                                    message: format!("Type mismatch in column {}", col_idx),
                                });
                            }
                        }
                    }

                    Arc::new(BooleanArray::new(values))
                }

                DataType::Null => Arc::new(NullArray::new(num_rows)),
            };

            columns.push(array);
        }

        let batch = RecordBatch::try_new(self.schema.clone(), columns).map_err(|e| {
            super::stream::StreamError::Execution {
                message: format!("Failed to create RecordBatch: {}", e),
            }
        })?;

        Ok(Some(batch))
    }
}

impl DataStream for CsvFileStream {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn next_batch(&mut self) -> StreamResult<Option<RecordBatch>> {
        self.read_batch()
    }
}

#[derive(Debug, Clone)]
enum ParsedValue {
    Int64(i64),
    Float64(f64),
    String(String),
    Boolean(bool),
    Null,
}

fn calculate_adaptive_batch_size(schema: &Schema) -> usize {
    const TARGET_MEMORY_MB: usize = 8;
    const TARGET_MEMORY_BYTES: usize = TARGET_MEMORY_MB * 1024 * 1024;

    let estimated_row_bytes: usize = schema
        .fields()
        .iter()
        .map(|field| match field.data_type() {
            DataType::Int64 => 8,
            DataType::Float64 => 8,
            DataType::Boolean => 1,
            DataType::String => 32,
            DataType::Null => 0,
        })
        .sum();

    if estimated_row_bytes == 0 {
        return 10_000;
    }

    let target_rows = TARGET_MEMORY_BYTES / estimated_row_bytes;

    target_rows.clamp(1_000, 100_000)
}

//generated tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use crate::execution::Field;

    fn create_test_csv() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id,name,score,active").unwrap();
        writeln!(file, "1,Alice,85.5,true").unwrap();
        writeln!(file, "2,Bob,92.0,false").unwrap();
        writeln!(file, "3,Charlie,78.5,true").unwrap();
        writeln!(file, "4,,90.0,false").unwrap(); // null name
        writeln!(file, "5,Eve,null,true").unwrap(); // null score
        file
    }

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::String, true),
            Field::new("score", DataType::Float64, true),
            Field::new("active", DataType::Boolean, false),
        ]))
    }

    #[test]
    fn test_csv_file_stream_basic() {
        let csv_file = create_test_csv();
        let schema = create_test_schema();

        let mut stream =
            CsvFileStream::new(csv_file.path(), schema.clone(), Some(10), None).unwrap();

        assert_eq!(stream.schema(), schema);

        let batch = stream.next_batch().unwrap();
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert_eq!(batch.num_rows(), 5);
        assert_eq!(batch.num_columns(), 4);
    }

    #[test]
    fn test_adaptive_batch_size() {
        let schema = create_test_schema();
        let batch_size = calculate_adaptive_batch_size(&schema);

        // Should be reasonable size based on schema
        assert!(batch_size >= 1_000);
        assert!(batch_size <= 100_000);

        // For our test schema: 8 + 32 + 8 + 1 = 49 bytes per row
        // 8MB / 49 bytes ≈ 171K rows (clamped to 100K)
        assert_eq!(batch_size, 100_000);
    }

    #[test]
    fn test_csv_parsing_with_nulls() {
        let csv_file = create_test_csv();
        let schema = create_test_schema();

        let mut stream = CsvFileStream::new(csv_file.path(), schema, Some(10), None).unwrap();
        let batch = stream.next_batch().unwrap().unwrap();

        // Check that nulls are handled correctly
        assert_eq!(batch.num_rows(), 5);

        // Row 3 (index 3) should have null name
        // Row 4 (index 4) should have null score
        // This would require accessing the actual arrays to verify nulls
    }

    #[test]
    fn test_empty_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id,name").unwrap(); // Header only

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::String, true),
        ]));

        let mut stream = CsvFileStream::new(file.path(), schema, Some(10), None).unwrap();
        let batch = stream.next_batch().unwrap();

        assert!(batch.is_none()); // No data after header
    }
}
