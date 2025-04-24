import os
import pandas as pd
import json
import sqlite3
from typing import List, Dict, Any, Union
import numpy as np
from pandas.api.types import is_numeric_dtype
import uuid
import shutil

def flatten_json(nested_json: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """
    Flatten a nested JSON structure into a flat dictionary.
    
    Args:
        nested_json: Nested JSON object
        prefix: Prefix for flattened keys
        
    Returns:
        Flattened dictionary
    """
    flattened = {}
    
    for key, value in nested_json.items():
        # Create new key with prefix
        new_key = f"{prefix}_{key}" if prefix else key
        
        # If value is a dictionary, recursively flatten it
        if isinstance(value, dict):
            flattened.update(flatten_json(value, new_key))
        # If value is a list, check if it contains dictionaries
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # For simplicity, just take the first item in lists of dictionaries
            flattened.update(flatten_json(value[0], new_key))
        else:
            flattened[new_key] = value
            
    return flattened

def parse_table(path: str) -> pd.DataFrame:
    """
    Load a structured data file and return it as a pandas DataFrame.
    
    Supports CSV, Excel, and JSON formats.
    - CSV/Excel are loaded directly
    - JSON is parsed based on structure (flat or nested)
    
    Args:
        path: Path to the data file
        
    Returns:
        pandas DataFrame with inferred column types
    """
    file_extension = os.path.splitext(path)[1].lower()
    
    # Handle CSV files
    if file_extension == '.csv':
        # Try to infer encoding and delimiter
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            # Try with different encoding if default fails
            df = pd.read_csv(path, encoding='latin1')
    
    # Handle Excel files
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    
    # Handle JSON files
    elif file_extension == '.json':
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # If data is a list of dictionaries, convert directly
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
        # If it's a single dictionary, flatten nested structure
        elif isinstance(data, dict):
            flattened_data = flatten_json(data)
            df = pd.DataFrame([flattened_data])
        # If it's a dictionary with arrays/lists as values (common API response format)
        elif isinstance(data, dict) and any(isinstance(value, list) for value in data.values()):
            # Find the first list in the dictionary
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    df = pd.DataFrame(value)
                    break
            else:
                # If no suitable list found, flatten and create single row
                flattened_data = flatten_json(data)
                df = pd.DataFrame([flattened_data])
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # Infer better column types
    df = infer_column_types(df)
    
    return df

def infer_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Improve type inference for DataFrame columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with better type inference
    """
    for col in df.columns:
        # Skip columns that are already numeric
        if is_numeric_dtype(df[col]):
            continue
            
        # Try to convert string columns to numeric if appropriate
        if df[col].dtype == 'object':
            # Check if column might be numeric but stored as strings
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                # If most values could be converted (not resulting in NaN), use numeric
                if numeric_series.notnull().mean() > 0.8:  # If >80% are valid numbers
                    df[col] = numeric_series
            except:
                pass
                
            # Check for boolean-like columns
            if df[col].isin(['True', 'False', 'true', 'false', '0', '1']).all():
                df[col] = df[col].map({'True': True, 'true': True, '1': True, 
                                        'False': False, 'false': False, '0': False})
                
            # Try to convert to datetime
            try:
                datetime_series = pd.to_datetime(df[col], errors='coerce')
                # If most values converted successfully, use datetime
                if datetime_series.notnull().mean() > 0.8:
                    df[col] = datetime_series
            except:
                pass
                
    return df

class StructuredDataIndex:
    """Class to manage indexed structured data tables using a persistent SQLite database."""
    
    def __init__(self, db_path=None):
        """
        Initialize the structured data index with a file-based SQLite database.
        
        Args:
            db_path: Path to the SQLite database file. If None, a default path is used.
        """
        # Create database directory if it doesn't exist
        self.db_dir = os.path.join(os.getcwd(), 'structured_db')
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Create a unique database file if path not provided
        if db_path is None:
            db_id = str(uuid.uuid4())[:8]
            self.db_path = os.path.join(self.db_dir, f'structured_data_{db_id}.db')
        else:
            self.db_path = db_path
            
        # Create connection to the file-based SQLite database
        self.conn = sqlite3.connect(self.db_path)
        
        # Create tables to store metadata
        self._create_metadata_tables()
        
        # Dictionary to store table metadata (loaded from database)
        self.table_schemas = self._load_table_schemas()
        
        # List of table names
        self.table_names = list(self.table_schemas.keys())
        
        print(f"Using SQLite database at: {self.db_path}")
    
    def _create_metadata_tables(self):
        """Create metadata tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Table to store schemas
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_schemas (
            table_name TEXT PRIMARY KEY,
            original_name TEXT,
            row_count INTEGER,
            schema_json TEXT
        )
        ''')
        
        # Table to store column information
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_columns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT,
            column_name TEXT,
            column_type TEXT,
            FOREIGN KEY (table_name) REFERENCES table_schemas (table_name)
        )
        ''')
        
        self.conn.commit()
    
    def _load_table_schemas(self):
        """Load table schemas from the metadata tables."""
        schemas = {}
        cursor = self.conn.cursor()
        
        # Query table schemas
        cursor.execute("SELECT table_name, original_name, row_count, schema_json FROM table_schemas")
        for table_name, original_name, row_count, schema_json in cursor.fetchall():
            schema_data = json.loads(schema_json)
            schemas[table_name] = {
                'columns': schema_data['columns'],
                'types': schema_data['types'],
                'original_name': original_name,
                'row_count': row_count
            }
            
        return schemas
        
    def _sanitize_column_name(self, name: str) -> str:
        """Sanitize column names for SQLite."""
        # Replace spaces and special characters with underscores
        return name.replace(' ', '_').replace('-', '_').replace('.', '_')
    
    def _sanitize_table_name(self, name: str) -> str:
        """Sanitize table names for SQLite."""
        # Replace spaces and special characters with underscores
        return name.replace(' ', '_').replace('-', '_').replace('.', '_')
    
    def index_table(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Index a single DataFrame into SQLite.
        
        Args:
            df: DataFrame to index
            table_name: Name for the table
        """
        # Sanitize column names for SQLite
        df_copy = df.copy()
        df_copy.columns = [self._sanitize_column_name(col) for col in df_copy.columns]
        
        # Sanitize table name
        safe_table_name = self._sanitize_table_name(table_name)
        
        # Store DataFrame in SQLite
        # Use chunks for large dataframes to avoid memory issues
        chunk_size = 100000  # Adjust based on your memory constraints
        if len(df_copy) > chunk_size:
            # For first chunk, replace existing table
            first_chunk = True
            for chunk_start in range(0, len(df_copy), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(df_copy))
                chunk = df_copy.iloc[chunk_start:chunk_end]
                
                if first_chunk:
                    chunk.to_sql(safe_table_name, self.conn, if_exists='replace', index=False)
                    first_chunk = False
                else:
                    chunk.to_sql(safe_table_name, self.conn, if_exists='append', index=False)
                
                print(f"Indexed chunk {chunk_start//chunk_size + 1} of table {table_name} ({chunk_end}/{len(df_copy)} rows)")
        else:
            # For smaller dataframes, do it in one go
            df_copy.to_sql(safe_table_name, self.conn, if_exists='replace', index=False)
        
        # Store schema information in memory
        column_types = {col: str(dtype) for col, dtype in df_copy.dtypes.items()}
        self.table_schemas[safe_table_name] = {
            'columns': list(df_copy.columns),
            'types': column_types,
            'original_name': table_name,
            'row_count': len(df_copy)
        }
        
        # Store schema information in the database
        cursor = self.conn.cursor()
        
        # Save to table_schemas
        schema_json = json.dumps({
            'columns': list(df_copy.columns),
            'types': {col: str(dtype) for col, dtype in df_copy.dtypes.items()}
        })
        
        cursor.execute(
            "INSERT OR REPLACE INTO table_schemas (table_name, original_name, row_count, schema_json) VALUES (?, ?, ?, ?)",
            (safe_table_name, table_name, len(df_copy), schema_json)
        )
        
        # Save column information
        cursor.execute("DELETE FROM table_columns WHERE table_name = ?", (safe_table_name,))
        for col, dtype in df_copy.dtypes.items():
            cursor.execute(
                "INSERT INTO table_columns (table_name, column_name, column_type) VALUES (?, ?, ?)",
                (safe_table_name, col, str(dtype))
            )
        
        self.conn.commit()
        
        # Add to table names list if not already there
        if safe_table_name not in self.table_names:
            self.table_names.append(safe_table_name)
    
    def query(self, sql_query: str) -> pd.DataFrame:
        """
        Execute SQL query against the indexed tables.
        
        Args:
            sql_query: SQL query string
            
        Returns:
            Query results as DataFrame
        """
        try:
            # For potentially large result sets, use chunksize
            chunks = []
            for chunk in pd.read_sql_query(sql_query, self.conn, chunksize=10000):
                chunks.append(chunk)
                
            if chunks:
                return pd.concat(chunks, ignore_index=True)
            return pd.DataFrame()
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def get_schema_info(self) -> Dict:
        """
        Get schema information for all tables.
        
        Returns:
            Dictionary with table schema information
        """
        return self.table_schemas
    
    def get_table_names(self) -> List[str]:
        """
        Get list of table names.
        
        Returns:
            List of table names
        """
        return self.table_names
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print(f"Closed connection to database: {self.db_path}")
    
    def __del__(self):
        """Destructor to ensure connection is closed when object is deleted."""
        self.close()


def index_structured(tables: List[pd.DataFrame], table_names: List[str] = None, db_path: str = None) -> StructuredDataIndex:
    """
    Load DataFrames into a persistent SQLite database and record schemas for query routing.
    
    Args:
        tables: List of pandas DataFrames to index
        table_names: Optional list of table names (default: table_0, table_1, etc.)
        db_path: Optional path to the SQLite database file
        
    Returns:
        StructuredDataIndex object for querying the tables
    """
    # Create index object with specified database path
    index = StructuredDataIndex(db_path)
    
    # Generate default table names if not provided
    if table_names is None:
        table_names = [f"table_{i}" for i in range(len(tables))]
    
    # Ensure we have the right number of names
    if len(table_names) != len(tables):
        raise ValueError("Number of table names must match number of tables")
    
    # Index each table
    for df, name in zip(tables, table_names):
        print(f"Indexing table: {name} ({len(df)} rows, {len(df.columns)} columns)")
        index.index_table(df, name)
    
    return index

def load_existing_index(db_path: str) -> StructuredDataIndex:
    """
    Load an existing structured data index from a SQLite database file.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        StructuredDataIndex object for querying the tables
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
        
    return StructuredDataIndex(db_path)

# Example usage
if __name__ == "__main__":
    # Example 1: Parse a CSV file
    # df_csv = parse_table("example.csv")
    # print(f"CSV DataFrame shape: {df_csv.shape}")
    
    # Example 2: Parse multiple files and index them to a persistent database
    # dfs = [
    #     parse_table("example1.csv"),
    #     parse_table("example2.xlsx"),
    #     parse_table("example3.json")
    # ]
    # 
    # # Index the parsed tables
    # db_path = os.path.join(os.getcwd(), 'structured_db', 'my_database.db')
    # index = index_structured(dfs, ["sales", "customers", "products"], db_path)
    # 
    # # Query the indexed tables
    # result = index.query("SELECT * FROM sales JOIN customers ON sales.customer_id = customers.id LIMIT 10")
    # print(result)
    # 
    # # Close the connection
    # index.close()
    
    print("Structured data module loaded.") 