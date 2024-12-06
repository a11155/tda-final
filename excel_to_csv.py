import pandas as pd
import argparse
from pathlib import Path

def convert_excel_to_simple_csv(input_file: str, output_file: str, value_column: str = 'product') -> None:
    """
    Convert Excel file to a simple CSV containing only the numeric values.
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path to output CSV file
        value_column (str): Name of the column containing numeric values
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If value_column is not found in the Excel file
        pd.errors.EmptyDataError: If Excel file is empty
    """
    try:
        # Read Excel file
        df = pd.read_excel(input_file)
        
        # Verify the column exists
        if value_column not in df.columns:
            raise ValueError(f"Column '{value_column}' not found in Excel file. Available columns: {', '.join(df.columns)}")
        
        # Extract only the numeric values
        values = df[value_column]
        
        # Create a new dataframe with just the values
        pd.DataFrame(values).to_csv(output_file, index=False, header=False)
        
        print(f"Successfully converted '{input_file}' to '{output_file}'")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty")
        raise
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Excel file to simple CSV with only numeric values')
    parser.add_argument('input_file', help='Path to input Excel file')
    parser.add_argument('output_file', help='Path to output CSV file')
    parser.add_argument('--value-column', default='product', help='Name of the column containing numeric values')
    
    args = parser.parse_args()
    
    convert_excel_to_simple_csv(args.input_file, args.output_file, args.value_column)