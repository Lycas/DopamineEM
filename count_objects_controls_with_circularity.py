import fnmatch
import math
import tifffile
import numpy as np
from scipy.ndimage import label, binary_erosion
import pandas as pd
import os
import gc

def calculate_perimeter_for_object(binary_mask):
    """Calculate perimeter by counting edge pixels - memory efficient"""
    # Simple perimeter calculation: count pixels that have at least one non-object neighbor
    eroded = binary_erosion(binary_mask)
    edge_pixels = binary_mask & (~eroded)
    return np.sum(edge_pixels)

# Focus only on control datasets
control_dirs = [
    "210706_Con_HR1_2_parts",
    "210706_Con_HR2_2_parts", 
    "210706_Con_TS6_parts",
    "210714_Con_HR1_parts",
    "210714_Con_HR2_parts",
    "210714_Con_HR6_parts"
]

print(f"Processing {len(control_dirs)} control datasets...")

for directory in control_dirs:
    print(f"\n{'='*60}")
    print(f"Processing: {directory}")
    print(f"{'='*60}")
    
    # Only processing mitochondria now - focusing on perimeter and circularity
    column_names = ["Volume", "Radius", "X", "Y", "Perimeter", "Circularity"]
    dfm = pd.DataFrame(columns=column_names)  # mitochondria only
    
    # First pass: determine the total image dimensions
    max_col, max_row = 0, 0
    tile_size_x, tile_size_y = 0, 0
    
    # Use raw data files to determine the complete grid dimensions
    raw_files = fnmatch.filter(os.listdir(directory),'*[0-9].tif')
    if not raw_files:
        print(f"No raw data files found in {directory}, skipping...")
        continue
        
    for f1 in raw_files:
        string = f1.split('_')
        col = int(string[-2])  # For raw files: col is at position -2
        row = int(string[-1].split('.')[0])  # For raw files: row is at position -1 (before .tif)
        max_col = max(max_col, col)
        max_row = max(max_row, row)
        
        # Get tile size from first file only
        if tile_size_x == 0:
            img = tifffile.imread(directory + '/' + f1)
            tile_size_x, tile_size_y = img.shape[0], img.shape[1]
    
    total_width = (max_col + 1) * tile_size_x
    total_height = (max_row + 1) * tile_size_y
    
    # Check if the total image would be too large (>8GB)
    estimated_memory_gb = (total_width * total_height * 8) / (1024**3)  # 8 bytes per float64
    print(f"Image dimensions: {max_col+1} × {max_row+1} tiles = {total_width} × {total_height} pixels")
    print(f"Estimated memory needed: {estimated_memory_gb:.1f} GB")
    
    if estimated_memory_gb > 8:
        print("Image too large, processing in quarters...")
        
        # Calculate quarter boundaries based on tile grid, not pixels
        mid_col = (max_col + 1) // 2
        mid_row = (max_row + 1) // 2
        
        quarters = [
            (0, 0, mid_col, mid_row),           # quarter 0,0: top-left
            (mid_col, 0, max_col + 1, mid_row), # quarter 1,0: top-right  
            (0, mid_row, mid_col, max_row + 1), # quarter 0,1: bottom-left
            (mid_col, mid_row, max_col + 1, max_row + 1) # quarter 1,1: bottom-right
        ]
        
        for quarter_idx, (start_col, start_row, end_col, end_row) in enumerate(quarters):
            print(f"Processing quarter {quarter_idx}: cols {start_col}-{end_col-1}, rows {start_row}-{end_row-1}")
            
            # Calculate actual pixel dimensions for this quarter
            quarter_width = (end_col - start_col) * tile_size_x
            quarter_height = (end_row - start_row) * tile_size_y
            quarter_mask = np.zeros((quarter_width, quarter_height))
            
            # Load only the labeled files that exist in this quarter
            labeled_files = fnmatch.filter(os.listdir(directory),'*L.tif')
            for f2 in labeled_files:
                string = f2.split('_')
                col = int(string[-3])  # For *L.tif files: col is at position -3
                row = int(string[-2])  # For *L.tif files: row is at position -2
                
                if start_col <= col < end_col and start_row <= row < end_row:
                    img = tifffile.imread(directory + '/' + f2)
                    local_col = col - start_col
                    local_row = row - start_row
                    
                    # Use actual image dimensions instead of assumed tile size
                    actual_height, actual_width = img.shape
                    start_x = local_col * tile_size_x
                    end_x = start_x + actual_height
                    start_y = local_row * tile_size_y  
                    end_y = start_y + actual_width
                    
                    # Ensure we don't exceed quarter_mask bounds
                    end_x = min(end_x, quarter_mask.shape[0])
                    end_y = min(end_y, quarter_mask.shape[1])
                    
                    # Only assign the portion that fits
                    quarter_mask[start_x:end_x, start_y:end_y] = img[:end_x-start_x, :end_y-start_y]
            
            # Process this quarter - only mitochondria
            imgM = np.where(quarter_mask != 1, 0, quarter_mask)
            
            # Process mitochondria with perimeter and circularity calculation
            labeled_arrayM, num_featuresM = label(imgM)
            for m in range(1, num_featuresM):
                print(f"Mitochondria {m} out of {num_featuresM} in quarter {quarter_idx}")
                
                # Get the binary mask for this mitochondria object
                mito_mask = (labeled_arrayM == m)
                vol = np.count_nonzero(mito_mask)
                rad = math.sqrt((vol * 1.25**2) / math.pi)
                
                # Calculate perimeter
                perimeter = calculate_perimeter_for_object(mito_mask)
                
                # Calculate circularity: Perimeter²/(4π × Area)
                if vol > 0 and perimeter > 0:
                    circularity = (perimeter**2) / (4 * math.pi * vol)
                else:
                    circularity = np.nan
                
                thing = np.where(mito_mask)
                # Adjust coordinates to global position
                cx = thing[0].mean() + start_col * tile_size_x
                cy = thing[1].mean() + start_row * tile_size_y
                
                df2m = pd.DataFrame([[vol, rad, cx, cy, perimeter, circularity]], columns=column_names)
                dfm = pd.concat([dfm, df2m])
            
            # Clear memory
            del quarter_mask, imgM, labeled_arrayM
            gc.collect()
    
    else:
        print("Processing full image...")
        # Original code for smaller images
        total_mask = np.zeros((total_width, total_height))
        
        # Load only the labeled files that exist
        labeled_files = fnmatch.filter(os.listdir(directory),'*L.tif')
        for f2 in labeled_files:
            print(f"Loading: {f2}")
            string = f2.split('_')
            col = int(string[-3])  # For *L.tif files: col is at position -3
            row = int(string[-2])  # For *L.tif files: row is at position -2
            img = tifffile.imread(directory + '/' + f2)
            
            # Use actual image dimensions instead of assumed tile size
            actual_height, actual_width = img.shape
            start_x = col * tile_size_x
            end_x = start_x + actual_height
            start_y = row * tile_size_y  
            end_y = start_y + actual_width
            
            # Ensure we don't exceed total_mask bounds
            end_x = min(end_x, total_mask.shape[0])
            end_y = min(end_y, total_mask.shape[1])
            
            # Only assign the portion that fits
            total_mask[start_x:end_x, start_y:end_y] = img[:end_x-start_x, :end_y-start_y]
        
        # Process only mitochondria
        imgM = np.where(total_mask!=1, 0, total_mask)
        
        # Process mitochondria with perimeter and circularity calculation
        labeled_arrayM, num_featuresM = label(imgM)
        for m in range(1,num_featuresM):
            print(f'Mitochondria {m} out of {num_featuresM}')
            
            # Get the binary mask for this mitochondria object
            mito_mask = (labeled_arrayM == m)
            vol = np.count_nonzero(mito_mask)
            rad = math.sqrt((vol * 1.25**2) / math.pi)
            
            # Calculate perimeter
            perimeter = calculate_perimeter_for_object(mito_mask)
            
            # Calculate circularity: Perimeter²/(4π × Area)
            if vol > 0 and perimeter > 0:
                circularity = (perimeter**2) / (4 * math.pi * vol)
            else:
                circularity = np.nan
            
            thing = np.where(mito_mask)
            cx = thing[0].mean()
            cy = thing[1].mean()
            
            df2 = pd.DataFrame([[vol, rad, cx, cy, perimeter, circularity]], columns=column_names)
            dfm = pd.concat([dfm, df2])
    
    # Save results with updated naming to indicate they include circularity
    dfm.to_csv('control_mito_' + directory + '.csv', index=False)
    
    # Print summary statistics for this dataset
    print(f"\nResults for {directory}:")
    print(f"  Mitochondria: {len(dfm)}")
    if len(dfm) > 0:
        valid_mito = dfm.dropna(subset=['Perimeter', 'Circularity'])
        print(f"  Mitochondria with valid circularity: {len(valid_mito)}")
        if len(valid_mito) > 0:
            print(f"  Mean volume: {valid_mito['Volume'].mean():.1f} pixels")
            print(f"  Mean perimeter: {valid_mito['Perimeter'].mean():.1f} pixels")
            print(f"  Mean circularity: {valid_mito['Circularity'].mean():.3f}")
    
    # Clean up variables that exist
    if 'total_mask' in locals():
        del total_mask
    if 'img' in locals():
        del img
    if 'imgM' in locals():
        del imgM
    
    del dfm
    gc.collect()

print(f"\n{'='*60}")
print("Processing complete!")
print("Output files saved with 'control_mito_' prefix")
print("Mitochondria files include Volume, Radius, X, Y, Perimeter, and Circularity columns")
print(f"{'='*60}") 