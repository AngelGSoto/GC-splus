'''
Process all SPLUS fields to calculate zero points and output in SPLUS main survey format
'''
import glob
import subprocess
import pandas as pd
import os
import numpy as np

# Find all CSV files
csv_files = glob.glob("CenA*_gaia_xp_matches.csv")

# Process each field
all_detailed_results = []
all_splus_results = []

for csv_file in csv_files:
    print(f"Processing {csv_file}...")
    
    # Run the zero point calculation
    result = subprocess.run([
        'python', '../Programs/calculate_zero_points.py', 
        csv_file, 
        '--json-dir', '.'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Successfully processed {csv_file}")
        
        # Read the detailed results file
        field_name = csv_file.split('_gaia_xp_matches.csv')[0]
        detailed_file = f'{field_name}_zero_points.csv'
        
        try:
            # Read detailed results
            df_detailed = pd.read_csv(detailed_file)
            df_detailed['Field'] = field_name
            all_detailed_results.append(df_detailed)
            
            # Create SPLUS format row - SOLO NOMBRE DEL CAMPO Y ZERO POINTS
            splus_row = {
                'field': field_name
            }
            
            # Add zero points for each filter
            for _, row in df_detailed.iterrows():
                filter_name = row['Filter']
                if filter_name.startswith('mag_'):
                    # Convert to SPLUS column name (e.g., mag_F378 -> F378)
                    splus_col = filter_name.replace('mag_', '')
                    splus_row[splus_col] = row['Median_ZP']
            
            all_splus_results.append(splus_row)
            
        except Exception as e:
            print(f"Could not process results for {field_name}: {e}")
    else:
        print(f"Error processing {csv_file}: {result.stderr}")

# Save detailed results (original format)
if all_detailed_results:
    combined_detailed = pd.concat(all_detailed_results, ignore_index=True)
    combined_detailed.to_csv('all_fields_zero_points_detailed.csv', index=False)
    print("Detailed results saved to all_fields_zero_points_detailed.csv")
    
    # Calculate average zero points across all fields
    avg_zp = combined_detailed.groupby('Filter').agg({
        'Median_ZP': 'mean',
        'STD_MAD': 'mean',
        'N_Stars': 'sum'
    }).reset_index()
    
    avg_zp.rename(columns={
        'Median_ZP': 'Average_Median_ZP',
        'STD_MAD': 'Average_STD_MAD'
    }, inplace=True)
    
    avg_zp.to_csv('average_zero_points_detailed.csv', index=False)
    print("Average zero points saved to average_zero_points_detailed.csv")

# Save SPLUS format results (sin RA y DEC)
if all_splus_results:
    splus_df = pd.DataFrame(all_splus_results)
    
    # Define column order - SOLO field y filtros
    column_order = ['field', 'F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
    
    # Ensure all columns are present (fill missing with NaN)
    for col in column_order:
        if col not in splus_df.columns:
            splus_df[col] = np.nan
    
    # Reorder columns
    splus_df = splus_df[column_order]
    
    # Save with appropriate precision
    splus_df.to_csv('all_fields_zero_points_splus_format.csv', index=False, float_format='%.6f')
    print("SPLUS format results saved to all_fields_zero_points_splus_format.csv")
    
    # Calculate averages for SPLUS format
    filter_cols = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
    avg_splus = splus_df[filter_cols].mean()
    std_splus = splus_df[filter_cols].std()
    
    avg_splus_df = pd.DataFrame({
        'Filter': avg_splus.index,
        'Average_ZP': avg_splus.values,
        'STD_ZP': std_splus.values
    })
    
    avg_splus_df.to_csv('average_zero_points_splus_format.csv', index=False, float_format='%.6f')
    print("Average SPLUS format zero points saved to average_zero_points_splus_format.csv")
    
    # Print summary
    print("\n=== SPLUS FORMAT SUMMARY ===")
    print(f"Processed {len(splus_df)} fields")
    for _, row in avg_splus_df.iterrows():
        print(f"{row['Filter']}: {row['Average_ZP']:.6f} ± {row['STD_ZP']:.6f}")

else:
    print("No results were processed!")

print("\n=== COMPLETED ===")
print("Final output format: field, F378, F395, F410, F430, F515, F660, F861")
