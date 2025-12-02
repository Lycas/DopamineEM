import numpy as np
import tifffile
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

def get_significance_stars(p_value):
    """Convert p-value to significance stars"""
    if p_value <= 0.0001:
        return '****'
    elif p_value <= 0.001:
        return '***'
    elif p_value <= 0.01:
        return '**'
    elif p_value <= 0.05:
        return '*'
    else:
        return 'ns'

def add_significance_annotation(ax, x1, x2, y, stars, y_offset_factor=0.12):
    """Add significance annotation between two groups"""
    y_max = ax.get_ylim()[1]
    y_pos = y_max * (1 - y_offset_factor)
    
    # Draw the bracket
    ax.plot([x1, x1, x2, x2], [y_pos, y_pos + y_max*0.02, y_pos + y_max*0.02, y_pos], 
            color='black', linewidth=2.0)
    
    # Add the significance text
    ax.text((x1 + x2) / 2, y_pos + y_max*0.03, stars, 
            ha='center', va='bottom', fontsize=16, fontweight='bold')

# Set matplotlib parameters for publication-quality figures
plt.rcParams.update({
    'font.size': 18,           # Base font size (increased)
    'font.weight': 'bold',     # Make all text bold
    'axes.titlesize': 20,      # Title font size
    'axes.titleweight': 'bold', # Bold titles
    'axes.labelsize': 20,      # Axis label font size (increased)
    'axes.labelweight': 'bold', # Bold axis labels
    'xtick.labelsize': 18,     # X-axis tick label size (increased)
    'ytick.labelsize': 18,     # Y-axis tick label size (increased)
    'legend.fontsize': 16,     # Legend font size
    'legend.frameon': True,    # Legend frame
    'axes.linewidth': 2.0,     # Axis line width
    'xtick.major.width': 2.0,  # Tick mark width
    'ytick.major.width': 2.0,  # Tick mark width
    'xtick.major.size': 6,     # Tick mark size
    'ytick.major.size': 6,     # Tick mark size
    'grid.linewidth': 1.0,     # Grid line width
    'lines.linewidth': 2.0     # Line width
})

# Functions from caphos_analysis_plots.py
def find_subobjects(probabilities_file):
    """Extract diameter measurements from probability images"""
    probabilities_image = tifffile.imread(probabilities_file)
    first_channel = probabilities_image[:,:,0]
    blurred = gaussian_filter(first_channel, sigma=3)

    # Thresholding to find objects
    thresholded = blurred > 0.3

    # Morphological closing to improve object detection
    closed = closing(thresholded, disk(2))

    # Label objects
    labeled_objects = label(closed)
    properties = regionprops(labeled_objects)

    # Extract diameter and other properties
    subobjects_info = []
    for i, prop in enumerate(properties, start=1):
        diameter = prop.equivalent_diameter  # Diameter of a circle with the same area as the region
        subobjects_info.append((i, diameter))

    return subobjects_info

def process_images_for_ratio(original_file, probabilities_file):
    """Process images to calculate ratio of detected objects"""
    # Load the original image and the probabilities image
    original_image = tifffile.imread(original_file)
    probabilities_image = tifffile.imread(probabilities_file)

    # Isolate the first channel of the probabilities image
    first_channel = probabilities_image[:,:,0]

    # Apply Gaussian blur
    blurred = gaussian_filter(first_channel, sigma=3)

    # Calculate the number of non-zero pixels in the original image
    non_zero_count_original = np.count_nonzero(original_image)

    # Calculate the number of pixels greater than 0.2 in the blurred image
    count_blurred_above_threshold = np.count_nonzero(blurred > 0.2)

    return non_zero_count_original, count_blurred_above_threshold

# Function from plot_treatment_violins.py
def load_and_classify_data(file_pattern, data_type):
    """Load CSV files and classify by treatment group"""
    files = glob.glob(file_pattern)
    all_data = []
    
    for file in files:
        try:
            df = pd.read_csv(file)
            if len(df) == 0:  # Skip empty files
                continue
                
            # Determine treatment group from filename
            if 'Con' in file:
                treatment = 'Control'
            elif 'DA' in file:
                treatment = 'Dopamine'
            elif 'Hal' in file:
                treatment = 'Haloperidol'
            else:
                continue  # Skip files that don't match our treatment groups
            
            df['Treatment'] = treatment
            df['Source_File'] = file
            all_data.append(df)
            print(f"Loaded {len(df)} {data_type} from {file} ({treatment})")
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal {data_type}: {len(combined_df)}")
        return combined_df
    else:
        print(f"No {data_type} data found!")
        return pd.DataFrame()

print("="*80)
print("PROCESSING DATA FOR ALL FOUR PLOTS")
print("="*80)

# Process data from caphos_analysis_plots.py
print("\nProcessing diameter data...")
data_folder = 'mitos'
prob_files = glob.glob(os.path.join(data_folder, '*_Probabilities.tif'))

# DataFrame for diameter results
diameter_df = pd.DataFrame(columns=['File', 'Subobject Number', 'Diameter', 'Condition'])

for prob_file in prob_files:
    subobjects_info = find_subobjects(prob_file)
    file_name = os.path.basename(prob_file)
    
    # Determine the condition
    if "DA" in file_name:
        condition = "DA"
    elif "Con" in file_name:
        condition = "Con"
    elif "Hal" in file_name:
        condition = "Hal"
    else:
        condition = "Unknown"

    for subobject_number, diameter in subobjects_info:
        new_row = pd.DataFrame({
            'File': [file_name],
            'Subobject Number': [subobject_number],
            'Diameter': [diameter],
            'Condition': [condition]
        })
        diameter_df = pd.concat([diameter_df, new_row], ignore_index=True)

print(f"Diameter data: {len(diameter_df)} measurements")

print("Processing ratio data...")
# Process ratio data
ratio_df = pd.DataFrame(columns=['Original File', 'Non Zero Count Original', 'Count Blurred Above Threshold', 'Ratio', 'Condition'])

# List all original TIF files
original_files = [f for f in glob.glob(os.path.join(data_folder, '*.tif')) if not f.endswith('_Probabilities.tif')]

for original_file in original_files:
    # Generate the corresponding probabilities file name
    probabilities_file = original_file[:-4] + '_Probabilities.tif'

    # Check if the probabilities file exists
    if not os.path.isfile(probabilities_file):
        continue

    non_zero_count_original, count_blurred_above_threshold = process_images_for_ratio(original_file, probabilities_file)

    # Compute the ratio
    ratio = count_blurred_above_threshold / non_zero_count_original if non_zero_count_original > 0 else 0

    # Determine the condition based on the file name
    file_name = os.path.basename(original_file)
    if "DA" in file_name:
        condition = "DA"
    elif "Con" in file_name:
        condition = "Con"
    elif "Hal" in file_name:
        condition = "Hal"
    else:
        condition = "Unknown"

    # Append the results to the DataFrame
    new_row = pd.DataFrame({
        'Original File': [file_name],
        'Condition': [condition],
        'Non Zero Count Original': [non_zero_count_original],
        'Count Blurred Above Threshold': [count_blurred_above_threshold],
        'Ratio': [ratio]
    })
    ratio_df = pd.concat([ratio_df, new_row], ignore_index=True)

print(f"Ratio data: {len(ratio_df)} measurements")

# Process data from plot_treatment_violins.py
print("\nLoading vesicle data...")
vesicle_df = load_and_classify_data("25_ves_*.csv", "vesicles")

print("\nLoading mitochondria data...")
mito_df = load_and_classify_data("25_mito_*.csv", "mitochondria")

# Data preprocessing for diameter data
print("\nPreprocessing diameter data...")
Q1 = diameter_df['Diameter'].quantile(0.25)
Q3 = diameter_df['Diameter'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

diameter_clean = diameter_df[(diameter_df['Diameter'] >= lower_bound) & (diameter_df['Diameter'] <= upper_bound)].copy()
diameter_clean['Scaled Diameter'] = diameter_clean['Diameter'] * 1.25
diameter_clean = diameter_clean[diameter_clean['Condition'].isin(['Con', 'DA', 'Hal'])]

# Map condition labels for both datasets
label_map = {'Con': 'Control', 'DA': 'Dopamine', 'Hal': 'Haloperidol'}
diameter_clean['Condition'] = diameter_clean['Condition'].map(label_map)

# Filter ratio data
ratio_clean = ratio_df[ratio_df['Condition'].isin(['Con', 'DA', 'Hal'])]
ratio_clean['Condition'] = ratio_clean['Condition'].map(label_map)

# Conversion factor for mitochondria volume (pixels to nm²)
pixel_to_nm2 = 1.25 * 1.25

print("="*80)
print("CREATING PUBLICATION-READY INDIVIDUAL PLOTS")
print("="*80)

# Plot 1: Vesicle Density
print("Creating vesicle density plot...")
if not vesicle_df.empty and 'friends_1000' in vesicle_df.columns:
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    
    vesicle_df_clean = vesicle_df[vesicle_df['friends_1000'] <= 250]
    
    control_vesicles = vesicle_df_clean[vesicle_df_clean['Treatment'] == 'Control']['friends_1000'].values
    da_vesicles = vesicle_df_clean[vesicle_df_clean['Treatment'] == 'Dopamine']['friends_1000'].values
    hal_vesicles = vesicle_df_clean[vesicle_df_clean['Treatment'] == 'Haloperidol']['friends_1000'].values
    
    parts1 = ax1.violinplot([control_vesicles, da_vesicles, hal_vesicles],
                           positions=[1, 2, 3], widths=0.8, showmeans=False, showmedians=True, showextrema=False)
    
    for pc in parts1['bodies']:
        pc.set_facecolor('green')
        pc.set_alpha(0.7)
    
    if 'cmedians' in parts1:
        parts1['cmedians'].set_color('black')
        parts1['cmedians'].set_linewidth(3)
    
    # Statistical testing
    vesicle_p_values = {}
    if len(control_vesicles) > 0 and len(da_vesicles) > 0:
        _, p_con_da = mannwhitneyu(control_vesicles, da_vesicles, alternative='two-sided')
        vesicle_p_values['Control vs DA'] = p_con_da
        
    if len(control_vesicles) > 0 and len(hal_vesicles) > 0:
        _, p_con_hal = mannwhitneyu(control_vesicles, hal_vesicles, alternative='two-sided')
        vesicle_p_values['Control vs Hal'] = p_con_hal
    
    ax1.set_ylabel('Neighbors within 1 μm')
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['Control', 'Dopamine', 'Haloperidol'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add significance annotations
    if 'Control vs DA' in vesicle_p_values:
        stars = get_significance_stars(vesicle_p_values['Control vs DA'])
        add_significance_annotation(ax1, 1, 2, ax1.get_ylim()[1], stars, y_offset_factor=0.20)
    
    if 'Control vs Hal' in vesicle_p_values:
        stars = get_significance_stars(vesicle_p_values['Control vs Hal'])
        add_significance_annotation(ax1, 1, 3, ax1.get_ylim()[1], stars, y_offset_factor=0.12)
    
    # Make tick labels bold
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig('vesicle_density_individual.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Vesicle density plot saved as: vesicle_density_individual.png")

# Plot 2: Mitochondria Size
print("Creating mitochondria size plot...")
if not mito_df.empty and 'Volume' in mito_df.columns:
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    
    mito_df['Volume_nm2'] = mito_df['Volume'] * pixel_to_nm2
    mito_df_clean = mito_df[mito_df['Volume_nm2'] > 100]
    
    control_data = mito_df_clean[mito_df_clean['Treatment'] == 'Control']['Volume_nm2'].values
    da_data = mito_df_clean[mito_df_clean['Treatment'] == 'Dopamine']['Volume_nm2'].values
    hal_data = mito_df_clean[mito_df_clean['Treatment'] == 'Haloperidol']['Volume_nm2'].values
    
    control_data_scaled = control_data / 1e6
    da_data_scaled = da_data / 1e6
    hal_data_scaled = hal_data / 1e6
    
    plot_data_scaled = []
    plot_labels = []
    plot_positions = []
    
    if len(control_data) > 0:
        plot_data_scaled.append(control_data_scaled)
        plot_labels.append('Control')
        plot_positions.append(1)
    
    if len(da_data) > 0:
        plot_data_scaled.append(da_data_scaled)
        plot_labels.append('Dopamine')
        plot_positions.append(2)
        
    if len(hal_data) > 0:
        plot_data_scaled.append(hal_data_scaled)
        plot_labels.append('Haloperidol')
        plot_positions.append(3)
    
    if plot_data_scaled:
        parts2 = ax2.violinplot(plot_data_scaled, positions=plot_positions, widths=0.8, 
                               showmeans=False, showmedians=True, showextrema=False)
        
        for pc in parts2['bodies']:
            pc.set_facecolor('cyan')
            pc.set_alpha(0.7)
        
        if 'cmedians' in parts2:
            parts2['cmedians'].set_color('black')
            parts2['cmedians'].set_linewidth(3)
        
        # Overlay individual points
        for i, (data, pos) in enumerate(zip(plot_data_scaled, plot_positions)):
            if len(data) > 0:
                x_pos = np.random.normal(pos, 0.04, len(data))
                ax2.scatter(x_pos, data, alpha=0.6, s=20, color='black')
    
        # Statistical testing
        mito_p_values = {}
        if len(control_data) > 0 and len(da_data) > 0:
            _, p_con_da = mannwhitneyu(control_data, da_data, alternative='two-sided')
            mito_p_values['Control vs DA'] = p_con_da
            
        if len(control_data) > 0 and len(hal_data) > 0:
            _, p_con_hal = mannwhitneyu(control_data, hal_data, alternative='two-sided')
            mito_p_values['Control vs Hal'] = p_con_hal
    
        ax2.set_ylabel('Area (nm²) ×10⁶')
        ax2.set_xticks(plot_positions)
        ax2.set_xticklabels(plot_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add significance annotations
        if len(plot_positions) >= 2 and 'Control vs DA' in mito_p_values:
            stars = get_significance_stars(mito_p_values['Control vs DA'])
            add_significance_annotation(ax2, plot_positions[0], plot_positions[1], ax2.get_ylim()[1], stars, y_offset_factor=0.20)
        
        if len(plot_positions) >= 3 and 'Control vs Hal' in mito_p_values:
            stars = get_significance_stars(mito_p_values['Control vs Hal'])
            add_significance_annotation(ax2, plot_positions[0], plot_positions[2], ax2.get_ylim()[1], stars, y_offset_factor=0.12)
        
        # Make tick labels bold
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig('mitochondria_size_individual.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Mitochondria size plot saved as: mitochondria_size_individual.png")

# Plot 3: Fractional Granule Content
print("Creating fractional granule content plot...")
fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))

ratio_groups = ['Control', 'Dopamine', 'Haloperidol']
ratio_data = []
ratio_positions = []

for i, group in enumerate(ratio_groups):
    group_data = ratio_clean[ratio_clean['Condition'] == group]['Ratio'].values
    if len(group_data) > 0:
        ratio_data.append(group_data)
        ratio_positions.append(i + 1)

if ratio_data:
    parts3 = ax3.violinplot(ratio_data, positions=ratio_positions, widths=0.8, 
                           showmeans=False, showmedians=True, showextrema=False)
    
    for pc in parts3['bodies']:
        pc.set_facecolor('lightgrey')
        pc.set_alpha(0.8)
    
    if 'cmedians' in parts3:
        parts3['cmedians'].set_color('black')
        parts3['cmedians'].set_linewidth(3)
    
    # Overlay individual points
    for i, (data, pos) in enumerate(zip(ratio_data, ratio_positions)):
        if len(data) > 0:
            x_pos = np.random.normal(pos, 0.04, len(data))
            ax3.scatter(x_pos, data, alpha=0.6, s=20, color='black')

    # Statistical testing
    ratio_p_values = {}
    control_ratio = ratio_clean[ratio_clean['Condition'] == 'Control']['Ratio'].values
    da_ratio = ratio_clean[ratio_clean['Condition'] == 'Dopamine']['Ratio'].values
    hal_ratio = ratio_clean[ratio_clean['Condition'] == 'Haloperidol']['Ratio'].values
    
    if len(control_ratio) > 0 and len(da_ratio) > 0:
        _, p_con_da = mannwhitneyu(control_ratio, da_ratio, alternative='two-sided')
        ratio_p_values['Control vs DA'] = p_con_da
        
    if len(control_ratio) > 0 and len(hal_ratio) > 0:
        _, p_con_hal = mannwhitneyu(control_ratio, hal_ratio, alternative='two-sided')
        ratio_p_values['Control vs Hal'] = p_con_hal

    ax3.set_ylabel('Fractional Granule\nContent')
    ax3.set_xticks(ratio_positions)
    ax3.set_xticklabels([ratio_groups[i] for i in range(len(ratio_positions))], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add significance annotations
    if 'Control vs DA' in ratio_p_values and len(ratio_positions) >= 2:
        stars = get_significance_stars(ratio_p_values['Control vs DA'])
        add_significance_annotation(ax3, ratio_positions[0], ratio_positions[1], ax3.get_ylim()[1], stars, y_offset_factor=0.20)
    
    if 'Control vs Hal' in ratio_p_values and len(ratio_positions) >= 3:
        stars = get_significance_stars(ratio_p_values['Control vs Hal'])
        add_significance_annotation(ax3, ratio_positions[0], ratio_positions[2], ax3.get_ylim()[1], stars, y_offset_factor=0.12)
    
    # Make tick labels bold
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig('fractional_granule_content_individual.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Fractional granule content plot saved as: fractional_granule_content_individual.png")

# Plot 4: Granule Diameter
print("Creating granule diameter plot...")
fig4, ax4 = plt.subplots(1, 1, figsize=(6, 6))

diameter_groups = ['Control', 'Dopamine', 'Haloperidol']
diameter_data = []
diameter_positions = []

for i, group in enumerate(diameter_groups):
    group_data = diameter_clean[diameter_clean['Condition'] == group]['Scaled Diameter'].values
    if len(group_data) > 0:
        diameter_data.append(group_data)
        diameter_positions.append(i + 1)

if diameter_data:
    parts4 = ax4.violinplot(diameter_data, positions=diameter_positions, widths=0.8, 
                           showmeans=False, showmedians=True, showextrema=False)
    
    for pc in parts4['bodies']:
        pc.set_facecolor('lightgrey')
        pc.set_alpha(0.8)
    
    if 'cmedians' in parts4:
        parts4['cmedians'].set_color('black')
        parts4['cmedians'].set_linewidth(3)

    # Statistical testing
    diameter_p_values = {}
    control_diam = diameter_clean[diameter_clean['Condition'] == 'Control']['Scaled Diameter'].values
    da_diam = diameter_clean[diameter_clean['Condition'] == 'Dopamine']['Scaled Diameter'].values
    hal_diam = diameter_clean[diameter_clean['Condition'] == 'Haloperidol']['Scaled Diameter'].values
    
    if len(control_diam) > 0 and len(da_diam) > 0:
        _, p_con_da = mannwhitneyu(control_diam, da_diam, alternative='two-sided')
        diameter_p_values['Control vs DA'] = p_con_da
        
    if len(control_diam) > 0 and len(hal_diam) > 0:
        _, p_con_hal = mannwhitneyu(control_diam, hal_diam, alternative='two-sided')
        diameter_p_values['Control vs Hal'] = p_con_hal

    ax4.set_ylabel('Diameter (nm)')
    ax4.set_xticks(diameter_positions)
    ax4.set_xticklabels([diameter_groups[i] for i in range(len(diameter_positions))], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Add significance annotations
    if 'Control vs DA' in diameter_p_values and len(diameter_positions) >= 2:
        stars = get_significance_stars(diameter_p_values['Control vs DA'])
        add_significance_annotation(ax4, diameter_positions[0], diameter_positions[1], ax4.get_ylim()[1], stars, y_offset_factor=0.20)
    
    if 'Control vs Hal' in diameter_p_values and len(diameter_positions) >= 3:
        stars = get_significance_stars(diameter_p_values['Control vs Hal'])
        add_significance_annotation(ax4, diameter_positions[0], diameter_positions[2], ax4.get_ylim()[1], stars, y_offset_factor=0.12)
    
    # Make tick labels bold
    for label in ax4.get_xticklabels() + ax4.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig('granule_diameter_individual.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Granule diameter plot saved as: granule_diameter_individual.png")

# All individual plots have been created and saved

print("="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Print comprehensive summary statistics
if not vesicle_df.empty:
    print("\nVESICLE DENSITY (neighbors within 1 μm):")
    for treatment in ['Control', 'Dopamine', 'Haloperidol']:
        data = vesicle_df_clean[vesicle_df_clean['Treatment'] == treatment]['friends_1000']
        if len(data) > 0:
            print(f"  {treatment}: n={len(data)}, mean={data.mean():.1f}, median={data.median():.1f}, std={data.std():.1f}")
    
    if 'vesicle_p_values' in locals():
        print("\n  Mann-Whitney U Test Results (Vesicle Density):")
        for comparison, p_val in vesicle_p_values.items():
            stars = get_significance_stars(p_val)
            if p_val == 0.0:
                print(f"    {comparison}: p < 1e-100 ({stars})")
            elif p_val < 1e-15:
                print(f"    {comparison}: p < 1e-15 ({stars})")
            elif p_val < 0.0001:
                print(f"    {comparison}: p = {p_val:.2e} ({stars})")
            else:
                print(f"    {comparison}: p = {p_val:.4f} ({stars})")

if not mito_df.empty:
    print("\nMITOCHONDRIA SIZE (nm²):")
    for treatment in ['Control', 'Dopamine', 'Haloperidol']:
        data = mito_df_clean[mito_df_clean['Treatment'] == treatment]['Volume_nm2']
        if len(data) > 0:
            print(f"  {treatment}: n={len(data)}, mean={data.mean():.0f}, median={data.median():.0f}, std={data.std():.0f}")
    
    if 'mito_p_values' in locals():
        print("\n  Mann-Whitney U Test Results (Mitochondria Size):")
        for comparison, p_val in mito_p_values.items():
            stars = get_significance_stars(p_val)
            print(f"    {comparison}: p = {p_val:.4f} ({stars})")

print("\nFRACTIONAL GRANULE CONTENT:")
for treatment in ['Control', 'Dopamine', 'Haloperidol']:
    data = ratio_clean[ratio_clean['Condition'] == treatment]['Ratio']
    if len(data) > 0:
        print(f"  {treatment}: n={len(data)}, mean={data.mean():.3f}, median={data.median():.3f}, std={data.std():.3f}")

if 'ratio_p_values' in locals():
    print("\n  Mann-Whitney U Test Results (Fractional Granule Content):")
    for comparison, p_val in ratio_p_values.items():
        stars = get_significance_stars(p_val)
        if p_val == 0.0:
            print(f"    {comparison}: p < 1e-100 ({stars})")
        elif p_val < 1e-15:
            print(f"    {comparison}: p < 1e-15 ({stars})")
        elif p_val < 0.0001:
            print(f"    {comparison}: p = {p_val:.2e} ({stars})")
        else:
            print(f"    {comparison}: p = {p_val:.4f} ({stars})")

print("\nGRANULE DIAMETER (nm):")
for treatment in ['Control', 'Dopamine', 'Haloperidol']:
    data = diameter_clean[diameter_clean['Condition'] == treatment]['Scaled Diameter']
    if len(data) > 0:
        print(f"  {treatment}: n={len(data)}, mean={data.mean():.1f}, median={data.median():.1f}, std={data.std():.1f}")

if 'diameter_p_values' in locals():
    print("\n  Mann-Whitney U Test Results (Granule Diameter):")
    for comparison, p_val in diameter_p_values.items():
        stars = get_significance_stars(p_val)
        if p_val == 0.0:
            print(f"    {comparison}: p < 1e-100 ({stars})")
        elif p_val < 1e-15:
            print(f"    {comparison}: p < 1e-15 ({stars})")
        elif p_val < 0.0001:
            print(f"    {comparison}: p = {p_val:.2e} ({stars})")
        else:
            print(f"    {comparison}: p = {p_val:.4f} ({stars})")

print(f"\n{'='*80}")
print("STATISTICAL TESTING")
print(f"{'='*80}")
print("Test: Mann-Whitney U (non-parametric, two-sided)")
print("Significance levels: **** p≤0.0001, *** p≤0.001, ** p≤0.01, * p≤0.05, ns p>0.05")

print(f"\nAll four individual publication-ready plots saved as:")
print("  - vesicle_density_individual.png")
print("  - mitochondria_size_individual.png") 
print("  - fractional_granule_content_individual.png")
print("  - granule_diameter_individual.png")
print("Each plot has consistent formatting, larger text, and no titles for publication.") 