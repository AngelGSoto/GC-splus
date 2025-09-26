#!/usr/bin/env python3
'''
Process all SPLUS fields to zero points and output in SPLUS main survey format
CORRECTED VERSION - Handles both corrected and calibrated column names
'''
import glob
import subprocess
import pandas as pd
import os
import numpy as np
import logging
import sys
from pathlib import Path
import argparse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('zero_points_batch_processing_splus_method.log')
    ]
)

def extract_field_name(csv_filename):
    """
    Safely extract field name from CSV filename with multiple patterns
    """
    filename = Path(csv_filename).name
    
    patterns = [
        '_gaia_xp_matches_splus_method.csv',
        '_gaia_xp_matches_corrected.csv', 
        '_gaia_xp_matches.csv',
        '.csv'
    ]
    
    for pattern in patterns:
        if pattern in filename:
            return filename.split(pattern)[0]
    
    return Path(csv_filename).stem

def is_valid_zero_points_file(zp_file):
    """
    Check if a zero points file is valid (not empty and contains actual data)
    """
    if not os.path.exists(zp_file):
        return False
    
    try:
        # Check file size
        if os.path.getsize(zp_file) == 0:
            logging.warning(f"Zero points file is empty: {zp_file}")
            return False
        
        # Try to read the CSV
        df = pd.read_csv(zp_file)
        
        # Check if it has the expected columns and data
        if df.empty:
            logging.warning(f"Zero points file has no data: {zp_file}")
            return False
        
        if 'Filter' not in df.columns or 'Median_ZP' not in df.columns:
            logging.warning(f"Zero points file has unexpected format: {zp_file}")
            return False
        
        # Check if there are any valid (non-NaN) zero points
        valid_zps = df['Median_ZP'].dropna()
        if len(valid_zps) == 0:
            logging.warning(f"Zero points file has no valid values: {zp_file}")
            return False
        
        logging.info(f"Zero points file is valid: {zp_file} ({len(valid_zps)} valid measurements)")
        return True
        
    except Exception as e:
        logging.warning(f"Error reading zero points file {zp_file}: {e}")
        return False

def extract_filter_name(filter_column):
    """
    Extract filter name from column name, handling both corrected and calibrated prefixes
    """
    # List of possible prefixes
    prefixes = ['mag_inst_corrected_', 'mag_inst_calibrated_', 'mag_inst_']
    
    for prefix in prefixes:
        if filter_column.startswith(prefix):
            return filter_column.split(prefix)[1]
    
    # If no prefix matches, return the original name
    return filter_column

def process_field_results(field_name):
    """
    Process the results for a single field and extract the zero points
    """
    # Try multiple possible output file patterns
    output_patterns = [
        f'{field_name}_zero_points_splus_method.csv',
        f'{field_name}_zero_points.csv',
        f'zero_points_{field_name}.csv'
    ]
    
    detailed_file = None
    for pattern in output_patterns:
        if os.path.exists(pattern) and is_valid_zero_points_file(pattern):
            detailed_file = pattern
            break
    
    if not detailed_file:
        # Check if any file exists but is invalid
        for pattern in output_patterns:
            if os.path.exists(pattern):
                logging.warning(f"File exists but invalid: {pattern}")
        logging.warning(f"No valid results file found for {field_name}")
        return None, None
    
    try:
        # Read detailed results
        df_detailed = pd.read_csv(detailed_file)
        
        # Additional validation
        if df_detailed.empty:
            logging.warning(f"Results file is empty: {detailed_file}")
            return None, None
        
        if 'Filter' not in df_detailed.columns or 'Median_ZP' not in df_detailed.columns:
            logging.error(f"Unexpected format in {detailed_file}. Columns: {df_detailed.columns.tolist()}")
            return None, None
        
        df_detailed['Field'] = field_name
        
        # Create SPLUS format row
        splus_row = {'field': field_name}
        
        # Add zero points for each filter
        valid_filters = 0
        for _, row in df_detailed.iterrows():
            filter_name = row['Filter']
            median_zp = row['Median_ZP']
            
            # Skip invalid or missing zero points
            if pd.isna(median_zp):
                continue
            
            # Extract filter name using the new function that handles both prefixes
            splus_col = extract_filter_name(filter_name)
            
            # Only add if it's one of the expected SPLUS filters
            expected_filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
            if splus_col in expected_filters:
                splus_row[splus_col] = float(median_zp)
                valid_filters += 1
                logging.debug(f"Added filter {splus_col} = {median_zp:.3f}")
            else:
                logging.debug(f"Skipping unexpected filter name: {filter_name} -> {splus_col}")
        
        if valid_filters == 0:
            logging.warning(f"No valid zero points found in {detailed_file}")
            # Show what's actually in the file for debugging
            logging.info(f"Content of {detailed_file}:")
            logging.info(df_detailed.to_string())
            return None, None
            
        logging.info(f"Extracted zero points for {valid_filters} filters in {field_name}")
        return df_detailed, splus_row
        
    except Exception as e:
        logging.error(f"Error processing results for {field_name}: {str(e)}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Batch process SPLUS fields for zero points')
    parser.add_argument('--pattern', default='*_gaia_xp_matches_splus_method.csv', 
                       help='File pattern to match CSV files')
    
    args = parser.parse_args()
    
    # Find CSV files
    csv_files = glob.glob(args.pattern)
    
    if not csv_files:
        logging.error(f"No CSV files found matching pattern '{args.pattern}'")
        logging.info("Available CSV files:")
        for f in glob.glob("*.csv"):
            logging.info(f"  {f}")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to process")
    
    # Process each field
    all_detailed_results = []
    all_splus_results = []
    processed_fields = []
    failed_fields = []
    
    for i, csv_file in enumerate(sorted(csv_files), 1):
        field_name = extract_field_name(csv_file)
        logging.info(f"\n=== Processing file {i}/{len(csv_files)}: {csv_file} (field: {field_name}) ===")
        
        # Check if zero points file already exists and is valid
        zp_file = f'{field_name}_zero_points_splus_method.csv'
        
        if os.path.exists(zp_file) and is_valid_zero_points_file(zp_file):
            logging.info(f"Valid zero points file exists: {zp_file}")
            detailed_data, splus_data = process_field_results(field_name)
            
            if detailed_data is not None and splus_data is not None:
                all_detailed_results.append(detailed_data)
                all_splus_results.append(splus_data)
                processed_fields.append(field_name)
                logging.info(f"✓ Successfully processed {field_name}")
            else:
                failed_fields.append(field_name)
                logging.error(f"✗ Failed to process results for {field_name}")
        else:
            failed_fields.append(field_name)
            logging.error(f"✗ Zero points file missing or invalid: {zp_file}")
    
    # Create Results directory
    os.makedirs('Results', exist_ok=True)
    
    # Save detailed results (original format)
    if all_detailed_results:
        try:
            combined_detailed = pd.concat(all_detailed_results, ignore_index=True)
            
            # Save detailed results
            detailed_output = 'Results/all_fields_zero_points_detailed_splus_method.csv'
            combined_detailed.to_csv(detailed_output, index=False)
            logging.info(f"✓ Detailed results saved to {detailed_output}")
            
            # Calculate average zero points across all fields
            avg_zp = combined_detailed.groupby('Filter').agg({
                'Median_ZP': ['mean', 'std', 'count'],
                'STD_MAD': 'mean'
            }).reset_index()
            
            # Flatten column names
            avg_zp.columns = ['Filter', 'Average_Median_ZP', 'Std_Median_ZP', 'N_Fields', 'Average_STD_MAD']
            
            avg_output = 'Results/average_zero_points_detailed_splus_method.csv'
            avg_zp.to_csv(avg_output, index=False)
            logging.info(f"✓ Average zero points saved to {avg_output}")
            
            # Print detailed summary
            logging.info("\n=== DETAILED SUMMARY ===")
            for _, row in avg_zp.iterrows():
                if pd.notna(row['Average_Median_ZP']):
                    logging.info(f"{row['Filter']}: {row['Average_Median_ZP']:.3f} ± {row['Std_Median_ZP']:.3f} (from {int(row['N_Fields'])} fields)")
        
        except Exception as e:
            logging.error(f"Error saving detailed results: {str(e)}")
    
    # Save SPLUS format results (only field and filters)
    if all_splus_results:
        try:
            splus_df = pd.DataFrame(all_splus_results)
            
            # Define expected filter columns in correct order
            filter_cols = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
            column_order = ['field'] + filter_cols
            
            # Ensure all expected columns exist (fill missing with NaN)
            for col in column_order:
                if col not in splus_df.columns:
                    splus_df[col] = np.nan
            
            # Reorder columns and sort by field name
            splus_df = splus_df[column_order].sort_values('field')
            
            # Save with appropriate precision
            splus_output = 'Results/all_fields_zero_points_splus_format_splus_method.csv'
            splus_df.to_csv(splus_output, index=False, float_format='%.6f')
            logging.info(f"✓ SPLUS format results saved to {splus_output}")
            
            # Calculate averages for SPLUS format
            numeric_cols = splus_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                avg_splus = splus_df[numeric_cols].mean()
                std_splus = splus_df[numeric_cols].std()
                count_splus = splus_df[numeric_cols].count()
                
                avg_splus_df = pd.DataFrame({
                    'Filter': numeric_cols,
                    'Average_ZP': avg_splus.values,
                    'STD_ZP': std_splus.values,
                    'N_Fields': count_splus.values
                })
                
                avg_splus_output = 'Results/average_zero_points_splus_format_splus_method.csv'
                avg_splus_df.to_csv(avg_splus_output, index=False, float_format='%.6f')
                logging.info(f"✓ Average SPLUS format zero points saved to {avg_splus_output}")
                
                # Print SPLUS format summary
                logging.info("\n=== SPLUS FORMAT SUMMARY ===")
                logging.info(f"Successfully processed {len(splus_df)} fields")
                for _, row in avg_splus_df.iterrows():
                    if pd.notna(row['Average_ZP']):
                        logging.info(f"{row['Filter']}: {row['Average_ZP']:.3f} ± {row['STD_ZP']:.3f} (from {int(row['N_Fields'])} fields)")
        
        except Exception as e:
            logging.error(f"Error saving SPLUS format results: {str(e)}")
    
    # Summary statistics
    logging.info("\n" + "="*50)
    logging.info("PROCESSING SUMMARY")
    logging.info("="*50)
    logging.info(f"Total CSV files found: {len(csv_files)}")
    logging.info(f"Successfully processed: {len(processed_fields)}")
    logging.info(f"Failed: {len(failed_fields)}")
    
    if processed_fields:
        logging.info("\n✓ Successfully processed fields:")
        for field in sorted(processed_fields):
            logging.info(f"  - {field}")
    
    if failed_fields:
        logging.info("\n✗ Failed fields:")
        for field in sorted(failed_fields):
            logging.info(f"  - {field}")
    
    # Final output files summary
    logging.info("\n" + "="*50)
    logging.info("OUTPUT FILES")
    logging.info("="*50)
    logging.info("Detailed results: Results/all_fields_zero_points_detailed_splus_method.csv")
    logging.info("SPLUS format: Results/all_fields_zero_points_splus_format_splus_method.csv")
    logging.info("Format: field, F378, F395, F410, F430, F515, F660, F861")

if __name__ == "__main__":
    main()
