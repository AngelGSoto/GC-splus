#!/usr/bin/env python3
'''
Process all SPLUS fields to calculate zero points and output in SPLUS main survey format
Updated version for splus_method CSV files
'''
import glob
import subprocess
import pandas as pd
import os
import numpy as np
import logging
import sys
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('zero_points_batch_processing_splus_method.log')
    ]
)

def run_zero_point_calculation(csv_file, script_path='../Programs/ZeroPoints_calculations.py'):
    """
    Run zero point calculation for a single field with proper error handling
    """
    # Extract field name from new file naming convention
    if '_gaia_xp_matches_splus_method.csv' in csv_file:
        field_name = Path(csv_file).stem.replace('_gaia_xp_matches_splus_method', '')
    else:
        field_name = Path(csv_file).stem.replace('_gaia_xp_matches_corrected', '')
    
    # Check for new output file naming
    output_file = f'{field_name}_zero_points_splus_method.csv'
    if os.path.exists(output_file):
        logging.info(f"Zero points file already exists for {field_name}, skipping calculation")
        return True, field_name
    
    # Check if the calculation script exists
    if not os.path.exists(script_path):
        logging.error(f"Calculation script not found: {script_path}")
        return False, field_name
    
    logging.info(f"Processing {csv_file}...")
    
    try:
        # Run the zero point calculation with timeout
        result = subprocess.run([
            'python', script_path, 
            csv_file, 
            '--json-dir', '.',
            '--plot'
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            logging.info(f"Successfully processed {field_name}")
            
            # Verify the output file was created
            if os.path.exists(output_file):
                return True, field_name
            else:
                logging.error(f"Output file not created: {output_file}")
                return False, field_name
        else:
            logging.error(f"Error processing {field_name}: {result.stderr}")
            # Log the full error for debugging
            with open(f'{field_name}_zero_points_error.log', 'w') as f:
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")
            return False, field_name
            
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout processing {field_name}")
        return False, field_name
    except Exception as e:
        logging.error(f"Unexpected error processing {field_name}: {e}")
        return False, field_name

def process_field_results(field_name):
    """
    Process the results for a single field and extract the zero points
    """
    detailed_file = f'{field_name}_zero_points_splus_method.csv'
    
    if not os.path.exists(detailed_file):
        logging.warning(f"Detailed results file not found: {detailed_file}")
        return None, None
    
    try:
        # Read detailed results
        df_detailed = pd.read_csv(detailed_file)
        df_detailed['Field'] = field_name
        
        # Create SPLUS format row - only field name and zero points
        splus_row = {'field': field_name}
        
        # Add zero points for each filter
        valid_filters = 0
        for _, row in df_detailed.iterrows():
            filter_name = row['Filter']
            median_zp = row['Median_ZP']
            
            # Skip invalid or missing zero points
            if pd.isna(median_zp) or (isinstance(median_zp, str) and median_zp.lower() == 'nan'):
                continue
            
            # Extract filter name (e.g., 'F378' from 'mag_inst_corrected_F378')
            if filter_name.startswith('mag_inst_corrected_'):
                splus_col = filter_name.split('mag_inst_corrected_')[1]
                splus_row[splus_col] = float(median_zp)
                valid_filters += 1
            else:
                logging.debug(f"Skipping unexpected filter name format: {filter_name}")
        
        if valid_filters == 0:
            logging.warning(f"No valid zero points found for {field_name}")
            return None, None
            
        logging.info(f"Extracted zero points for {valid_filters} filters in {field_name}")
        return df_detailed, splus_row
        
    except Exception as e:
        logging.error(f"Error processing results for {field_name}: {e}")
        return None, None

def main():
    # Find all splus_method CSV files
    csv_files = glob.glob("CenA*_gaia_xp_matches_splus_method.csv")
    
    if not csv_files:
        logging.error("No splus_method CSV files found matching pattern 'CenA*_gaia_xp_matches_splus_method.csv'")
        logging.info("Available CSV files:")
        for f in glob.glob("*.csv"):
            logging.info(f"  {f}")
        return
    
    logging.info(f"Found {len(csv_files)} splus_method CSV files to process")
    
    # Process each field
    all_detailed_results = []
    all_splus_results = []
    processed_fields = []
    failed_fields = []
    
    for csv_file in sorted(csv_files):
        success, field_name = run_zero_point_calculation(csv_file)
        
        if success:
            detailed_data, splus_data = process_field_results(field_name)
            
            if detailed_data is not None and splus_data is not None:
                all_detailed_results.append(detailed_data)
                all_splus_results.append(splus_data)
                processed_fields.append(field_name)
            else:
                failed_fields.append(field_name)
        else:
            failed_fields.append(field_name)
    
    # Save detailed results (original format)
    if all_detailed_results:
        combined_detailed = pd.concat(all_detailed_results, ignore_index=True)
        
        # Ensure the output directory exists
        os.makedirs('Results', exist_ok=True)
        
        # Save detailed results
        detailed_output = 'Results/all_fields_zero_points_detailed_splus_method.csv'
        combined_detailed.to_csv(detailed_output, index=False)
        logging.info(f"Detailed results saved to {detailed_output}")
        
        # Calculate average zero points across all fields
        avg_zp = combined_detailed.groupby('Filter').agg({
            'Median_ZP': ['mean', 'std', 'count'],
            'STD_MAD': 'mean'
        }).reset_index()
        
        # Flatten column names
        avg_zp.columns = ['Filter', 'Average_Median_ZP', 'Std_Median_ZP', 'N_Fields', 'Average_STD_MAD']
        
        avg_output = 'Results/average_zero_points_detailed_splus_method.csv'
        avg_zp.to_csv(avg_output, index=False)
        logging.info(f"Average zero points saved to {avg_output}")
        
        # Print detailed summary
        logging.info("\n=== DETAILED SUMMARY ===")
        for _, row in avg_zp.iterrows():
            if pd.notna(row['Average_Median_ZP']):
                logging.info(f"{row['Filter']}: {row['Average_Median_ZP']:.6f} \u00b1 {row['Std_Median_ZP']:.6f} (from {int(row['N_Fields'])} fields)")

    # Save SPLUS format results (only field and filters)
    if all_splus_results:
        splus_df = pd.DataFrame(all_splus_results)
        
        # Define expected filter columns
        filter_cols = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        column_order = ['field'] + filter_cols
        
        # Ensure all expected columns exist (fill missing with NaN)
        for col in column_order:
            if col not in splus_df.columns:
                splus_df[col] = np.nan
        
        # Reorder columns
        splus_df = splus_df[column_order]
        
        # Save with appropriate precision
        splus_output = 'Results/all_fields_zero_points_splus_format_splus_method.csv'
        splus_df.to_csv(splus_output, index=False, float_format='%.6f')
        logging.info(f"SPLUS format results saved to {splus_output}")
        
        # Calculate averages for SPLUS format
        avg_splus = splus_df[filter_cols].mean()
        std_splus = splus_df[filter_cols].std()
        count_splus = splus_df[filter_cols].count()
        
        avg_splus_df = pd.DataFrame({
            'Filter': avg_splus.index,
            'Average_ZP': avg_splus.values,
            'STD_ZP': std_splus.values,
            'N_Fields': count_splus.values
        })
        
        avg_splus_output = 'Results/average_zero_points_splus_format_splus_method.csv'
        avg_splus_df.to_csv(avg_splus_output, index=False, float_format='%.6f')
        logging.info(f"Average SPLUS format zero points saved to {avg_splus_output}")
        
        # Print SPLUS format summary
        logging.info("\n=== SPLUS FORMAT SUMMARY ===")
        logging.info(f"Successfully processed {len(splus_df)} fields")
        for _, row in avg_splus_df.iterrows():
            if pd.notna(row['Average_ZP']):
                logging.info(f"{row['Filter']}: {row['Average_ZP']:.6f} \u00b1 {row['STD_ZP']:.6f} (from {int(row['N_Fields'])} fields)")
    
    # Summary statistics
    logging.info("\n=== PROCESSING SUMMARY ===")
    logging.info(f"Total fields found: {len(csv_files)}")
    logging.info(f"Successfully processed: {len(processed_fields)}")
    logging.info(f"Failed: {len(failed_fields)}")
    
    if processed_fields:
        logging.info("Successfully processed fields: " + ", ".join(sorted(processed_fields)))
    
    if failed_fields:
        logging.warning("Failed fields: " + ", ".join(sorted(failed_fields)))
    
    # Final output format reminder
    logging.info("\n=== OUTPUT FILES ===")
    logging.info("Detailed results: Results/all_fields_zero_points_detailed_splus_method.csv")
    logging.info("SPLUS format: Results/all_fields_zero_points_splus_format_splus_method.csv")
    logging.info("Final output format: field, F378, F395, F410, F430, F515, F660, F861")

if __name__ == "__main__":
    main()
