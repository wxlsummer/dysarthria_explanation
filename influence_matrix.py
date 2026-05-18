'''
Docstring for dysarthria_explanation.influence_matrix


import os
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Directory containing the CSV files
csv_dir = "/srv/xw4e25/dysarthria_explanation/tracin_results/datacleaning_fold1_3518"

# Original speaker-to-severity mapping (for TEST samples)
original_mapping = {
    'F01': 2, 'M01': 2, 'M02': 2, 'M04': 2, 'M05': 2,
    'F03': 1,
    'F04': 0, 'M03': 0,
    'FC01': 3, 'FC02': 3, 'FC03': 3, 'MC01': 3, 'MC02': 3, 'MC04': 3, 'MC03': 3
}

# Remapping function
# 原来: 0=mild, 1=moderate, 2=severe, 3=control
# 现在: 0=normal, 1=mild, 2=moderate, 3=severe
def remap_severity(old_label):
    """Remap old severity labels to new labels"""
    mapping = {
        3: 0,  # control -> normal
        0: 1,  # mild -> mild
        1: 2,  # moderate -> moderate
        2: 3   # severe -> severe
    }
    return mapping[old_label]

# Create corrected TEST mapping
test_severity_mapping = {
    speaker: remap_severity(original_mapping[speaker]) 
    for speaker in original_mapping
}

print("=" * 70)
print("TEST SEVERITY MAPPING (corrected)")
print("=" * 70)
for speaker, level in sorted(test_severity_mapping.items()):
    level_names = {0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
    print(f"  {speaker}: {level} ({level_names[level]})")

# ============================================================================
# Helper functions
# ============================================================================

def extract_speaker_id(filename):
    """Extract speaker ID from filename (before first underscore)"""
    basename = os.path.basename(filename)
    basename = basename.replace('.csv', '').replace('.npy', '')
    speaker_id = basename.split('_')[0]
    return speaker_id

def get_test_severity(filename):
    """Get severity level for a test sample from filename"""
    speaker_id = extract_speaker_id(filename)
    return test_severity_mapping.get(speaker_id, None)

# ============================================================================
# Process CSV files and compute influence matrix
# ============================================================================

# Initialize the 4x4 matrix and counters
S = np.zeros((4, 4))
test_sample_count = np.zeros(4)

# Get all CSV files
csv_files = list(Path(csv_dir).glob("*.csv"))
print(f"\n{'=' * 70}")
print(f"Found {len(csv_files)} CSV files to process")
print("=" * 70)

if len(csv_files) == 0:
    print(f"ERROR: No CSV files found in {csv_dir}")
    exit(1)

# Process each test sample's influence file
processed = 0
skipped = 0

for csv_file in csv_files:
    # Get test sample severity from filename
    test_severity = get_test_severity(csv_file.name)
    
    if test_severity is None:
        print(f"WARNING: Unknown speaker in {csv_file.name}")
        skipped += 1
        continue
    
    # Count this test sample
    test_sample_count[test_severity] += 1
    
    # Read CSV file - try to auto-detect header
    try:
        # First try: assume has header
        df = pd.read_csv(csv_file)
        
        # Check if first row looks like header (contains 'label' or 'influence' etc)
        first_col_name = str(df.columns[1]).lower()
        if 'label' in first_col_name or 'train' in first_col_name:
            # Has header, columns are already named
            col1, col2, col3 = df.columns[0], df.columns[1], df.columns[2]
        else:
            # No header, first row is data
            df = pd.read_csv(csv_file, header=None)
            df.columns = ['col0', 'col1', 'col2'] + [f'col{i}' for i in range(3, df.shape[1])]
            col1, col2, col3 = 'col0', 'col1', 'col2'
        
        if df.shape[1] < 3:
            print(f"WARNING: {csv_file.name} has only {df.shape[1]} columns")
            skipped += 1
            continue
            
    except Exception as e:
        print(f"ERROR reading {csv_file.name}: {e}")
        skipped += 1
        continue
    
    # Process each row
    for idx, row in df.iterrows():
        try:
            # Column order: filename, label, influence
            train_label_old = int(row[col2])  # Column 2 (index 1): label
            influence = float(row[col3])      # Column 3 (index 2): influence
            
            # Remap training label
            train_severity = remap_severity(train_label_old)
            
            # Accumulate influence
            S[test_severity, train_severity] += influence
            
        except (ValueError, KeyError, IndexError) as e:
            # Skip invalid rows
            continue
    
    processed += 1
    if processed % 500 == 0:
        print(f"  Processed {processed}/{len(csv_files)} files...")

print(f"\n✓ Successfully processed: {processed} files")
print(f"✗ Skipped: {skipped} files")

# ============================================================================
# Compute averaged influence matrix
# ============================================================================

print(f"\n{'=' * 70}")
print("TEST SAMPLE DISTRIBUTION")
print("=" * 70)

level_names = ['Normal', 'Mild', 'Moderate', 'Severe']
for c in range(4):
    print(f"  Class-{c} ({level_names[c]:8s}): {int(test_sample_count[c]):4d} samples")

# Average by test sample count
S_final = np.zeros((4, 4))
for c in range(4):
    if test_sample_count[c] > 0:
        S_final[c, :] = S[c, :] / test_sample_count[c]

# ============================================================================
# Display results
# ============================================================================

print("\n" + "=" * 70)
print("CLASS-LEVEL INFLUENCE MATRIX S")
print("=" * 70)
print("\nS[c, c'] = Average influence from training level c' to test level c")
print("(Averaged over number of test samples)")
print("\n            Training Label")
print("            Normal    Mild      Moderate  Severe")
print("          ┌─────────┬─────────┬─────────┬─────────┐")

for c in range(4):
    row_str = f"Test {level_names[c]:8s} │"
    for cp in range(4):
        row_str += f" {S_final[c, cp]:7.1f} │"
    print(row_str)
    if c < 3:
        print("          ├─────────┼─────────┼─────────┼─────────┤")
    else:
        print("          └─────────┴─────────┴─────────┴─────────┘")

# ============================================================================
# Compute ordinal sensitivity S̄(d)
# ============================================================================

print("\n" + "=" * 70)
print("ORDINAL SENSITIVITY ANALYSIS")
print("=" * 70)

S_bar = np.zeros(4)
distance_counts = np.zeros(4)

for d in range(4):
    values = []
    for c in range(4):
        for cp in range(4):
            if abs(c - cp) == d:
                values.append(S_final[c, cp])
                distance_counts[d] += 1
    if len(values) > 0:
        S_bar[d] = np.mean(values)

print("\nS̄(d) = Average influence at ordinal distance d:")
distance_names = ['Same-level', 'Adjacent', 'Two-apart', 'Opposite']
for d in range(4):
    print(f"  d={d} ({distance_names[d]:10s}): {S_bar[d]:8.2f}  "
          f"({int(distance_counts[d])} entries)")

if S_bar[1] != 0:
    print(f"\nRatios:")
    print(f"  S̄(0)/S̄(1) = {S_bar[0]/S_bar[1]:.2f}×  (Same-level vs Adjacent)")
if S_bar[2] != 0:
    print(f"  S̄(0)/S̄(2) = {S_bar[0]/S_bar[2]:.2f}×  (Same-level vs Two-apart)")

# ============================================================================
# Save results
# ============================================================================

output_dir = "/srv/xw4e25/dysarthria_explanation/tracin_results/Figures"

np.savetxt(f"{output_dir}/influence_matrix_S.csv", S_final, 
           delimiter=",", fmt="%.3f",
           header="Normal,Mild,Moderate,Severe",
           comments="# S[test_c, train_c']: rows=test label, cols=train label\n# ")

s_bar_data = np.column_stack([np.arange(4), S_bar])
np.savetxt(f"{output_dir}/ordinal_sensitivity.csv", s_bar_data,
           delimiter=",", fmt="%.3f",
           header="distance,S_bar",
           comments="")

df_S = pd.DataFrame(S_final, 
                    index=[f"Test_{name}" for name in level_names],
                    columns=[f"Train_{name}" for name in level_names])
df_S.to_csv(f"{output_dir}/influence_matrix_S_labeled.csv")

print("\n" + "=" * 70)
print("✓ Results saved:")
print("=" * 70)
print(f"  - {output_dir}/influence_matrix_S.csv")
print(f"  - {output_dir}/influence_matrix_S_labeled.csv")
print(f"  - {output_dir}/ordinal_sensitivity.csv")
'''

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

csv_dir = "/srv/xw4e25/dysarthria_explanation/tracin_results/datacleaning_fold1_3518"
output_dir = "/srv/xw4e25/dysarthria_explanation/tracin_results/Figures"

original_mapping = {
    'F01': 2, 'M01': 2, 'M02': 2, 'M04': 2, 'M05': 2,
    'F03': 1,
    'F04': 0, 'M03': 0,
    'FC01': 3, 'FC02': 3, 'FC03': 3, 'MC01': 3, 'MC02': 3, 'MC04': 3, 'MC03': 3
}

print("=" * 70)
print("CLASS-LEVEL INFLUENCE MATRIX COMPUTATION")
print("=" * 70)

# ============================================================================
# Helper functions
# ============================================================================

def extract_speaker_id(filename):
    basename = os.path.basename(filename).replace('.csv', '').replace('.npy', '')
    return basename.split('_')[0]

def get_test_label_original(filename):
    speaker_id = extract_speaker_id(filename)
    return original_mapping.get(speaker_id, None)

# ============================================================================
# Step 1: Compute I^(c) for each test class c
# ============================================================================

print("\n" + "=" * 70)
print("STEP 1: Computing I^(c) for each test class")
print("=" * 70)
print("I^(c)_i = average influence of training sample i on test class c")
print("(Averaging over all test samples in class c)")

csv_files = list(Path(csv_dir).glob("*.csv"))
print(f"\nFound {len(csv_files)} CSV files")

# Initialize accumulators for each test class
# influence_sum[test_class][train_idx] = sum of influences
# We'll use dictionaries to store per training sample
influence_accumulators = {
    0: {},  # test_class 0 (mild in original)
    1: {},  # test_class 1 (moderate in original)
    2: {},  # test_class 2 (severe in original)
    3: {}   # test_class 3 (control in original)
}

test_counts = {0: 0, 1: 0, 2: 0, 3: 0}

# Process all CSVs
print("\nProcessing CSV files...")
processed = 0
skipped = 0

for csv_file in tqdm(csv_files, desc="Step 1", ncols=100):
    test_label = get_test_label_original(csv_file.name)
    
    if test_label is None:
        skipped += 1
        continue
    
    test_counts[test_label] += 1
    
    try:
        # Read CSV
        df = pd.read_csv(csv_file)
        if 'label' not in str(df.columns[1]).lower():
            df = pd.read_csv(csv_file, header=None)
        
        if df.shape[1] < 3:
            skipped += 1
            continue
        
        # Column 1: filename, Column 2: label, Column 3: influence
        train_filenames = df.iloc[:, 0].values
        train_labels = df.iloc[:, 1].values.astype(int)
        influences = df.iloc[:, 2].values.astype(float)
        
        # Accumulate influence for each training sample
        for i in range(len(train_filenames)):
            train_file = train_filenames[i]
            train_label = train_labels[i]
            influence = influences[i]
            
            # Use (filename, label) as key
            key = (train_file, train_label)
            
            if key not in influence_accumulators[test_label]:
                influence_accumulators[test_label][key] = 0.0
            
            influence_accumulators[test_label][key] += influence
        
        processed += 1
        
    except Exception as e:
        print(f"\nError in {csv_file.name}: {e}")
        skipped += 1
        continue

print(f"\n✓ Processed: {processed} files")
print(f"✗ Skipped: {skipped} files")

print("\nTest sample counts (original labels):")
orig_names = {0: 'Mild', 1: 'Moderate', 2: 'Severe', 3: 'Control'}
for c in range(4):
    print(f"  Class-{c} ({orig_names[c]:8s}): {test_counts[c]} samples")

# ============================================================================
# Step 2: Compute averages and save I^(c) CSVs
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Computing averages and saving I^(c) CSVs")
print("=" * 70)

I_dataframes = {}

for test_class in range(4):
    print(f"\nProcessing test class {test_class} ({orig_names[test_class]})...")
    
    if test_counts[test_class] == 0:
        print(f"  WARNING: No test samples for class {test_class}")
        continue
    
    # Convert accumulator to list
    data = []
    for (train_file, train_label), influence_sum in influence_accumulators[test_class].items():
        # Compute average: I^(c)_i = sum / |T_c|
        avg_influence = influence_sum / test_counts[test_class]
        data.append({
            'train_filename': train_file,
            'train_label': train_label,
            'I_c': avg_influence
        })
    
    # Create DataFrame
    df_I_c = pd.DataFrame(data)
    
    # Sort by influence (descending)
    df_I_c = df_I_c.sort_values('I_c', ascending=False).reset_index(drop=True)
    
    # Save CSV (original labels, NOT remapped)
    csv_filename = f"{output_dir}/I_class{test_class}_original.csv"
    df_I_c.to_csv(csv_filename, index=False)
    print(f"  ✓ Saved: {csv_filename}")
    print(f"    {len(df_I_c)} training samples")
    
    # Store for next step
    I_dataframes[test_class] = df_I_c

# ============================================================================
# Step 3: Compute S matrix (original labels)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: Computing S matrix from I^(c)")
print("=" * 70)
print("S[c, c'] = average of I^(c)_i for training samples with label c'")

S_original = np.zeros((4, 4))
train_counts = {0: 0, 1: 0, 2: 0, 3: 0}

# First, count training samples per class
for test_class in range(4):
    if test_class not in I_dataframes:
        continue
    df = I_dataframes[test_class]
    for label in range(4):
        count = (df['train_label'] == label).sum()
        train_counts[label] = max(train_counts[label], count)

print("\nTraining sample counts (original labels):")
for c in range(4):
    print(f"  Class-{c} ({orig_names[c]:8s}): {train_counts[c]} samples")

# Compute S[c, c'] for each (test_c, train_c') pair
for test_class in range(4):
    if test_class not in I_dataframes:
        continue
    
    df = I_dataframes[test_class]
    
    for train_class in range(4):
        # Filter training samples with label = train_class
        mask = df['train_label'] == train_class
        influences = df.loc[mask, 'I_c'].values
        
        if len(influences) > 0:
            # S_{c←c'} = mean(I^(c)_i for i with label c')
            S_original[test_class, train_class] = influences.mean()
        else:
            S_original[test_class, train_class] = 0.0

print("\nS matrix (original labels):")
print("          Train: 0(mild)  1(mod)   2(sev)   3(ctrl)")
for c in range(4):
    row_str = f"Test {c}:       "
    for cp in range(4):
        row_str += f"{S_original[c, cp]:8.4f} "
    print(row_str)

# ============================================================================
# Step 4: Remap to final labels
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Remapping to final labels")
print("=" * 70)
print("Remap: 0(mild)→1, 1(moderate)→2, 2(severe)→3, 3(control)→0")

remap = {0: 1, 1: 2, 2: 3, 3: 0}

S_final = np.zeros((4, 4))
test_counts_final = np.zeros(4)
train_counts_final = np.zeros(4)

# Remap matrix
for c_old in range(4):
    for cp_old in range(4):
        c_new = remap[c_old]
        cp_new = remap[cp_old]
        S_final[c_new, cp_new] = S_original[c_old, cp_old]

# Remap counts
for c_old in range(4):
    test_counts_final[remap[c_old]] = test_counts[c_old]
    train_counts_final[remap[c_old]] = train_counts[c_old]

# ============================================================================
# Display and save final results
# ============================================================================

print("\n" + "=" * 70)
print("FINAL RESULTS (remapped labels)")
print("=" * 70)

level_names = ['Normal', 'Mild', 'Moderate', 'Severe']

print("\nTest sample distribution:")
for c in range(4):
    print(f"  Class-{c} ({level_names[c]:8s}): {int(test_counts_final[c])} samples")

print("\nTraining sample distribution:")
for c in range(4):
    print(f"  Class-{c} ({level_names[c]:8s}): {int(train_counts_final[c])} samples")

print("\n" + "=" * 70)
print("CLASS-LEVEL INFLUENCE MATRIX S")
print("=" * 70)
print("\nS[c, c'] = Average influence from train class c' to test class c")
print("\n            Training Label")
print("            Normal    Mild      Moderate  Severe")
print("          ┌─────────┬─────────┬─────────┬─────────┐")

for c in range(4):
    row_str = f"Test {level_names[c]:8s} │"
    for cp in range(4):
        row_str += f" {S_final[c, cp]:7.4f} │"
    print(row_str)
    if c < 3:
        print("          ├─────────┼─────────┼─────────┼─────────┤")
    else:
        print("          └─────────┴─────────┴─────────┴─────────┘")

# Ordinal sensitivity
print("\n" + "=" * 70)
print("ORDINAL SENSITIVITY S̄(d)")
print("=" * 70)

S_bar = np.zeros(4)
for d in range(4):
    values = [S_final[c, cp] for c in range(4) for cp in range(4) if abs(c - cp) == d]
    S_bar[d] = np.mean(values) if values else 0

distance_names = ['Same-level', 'Adjacent', 'Two-apart', 'Opposite']
for d in range(4):
    print(f"  d={d} ({distance_names[d]:10s}): {S_bar[d]:8.4f}")

if S_bar[1] != 0:
    print(f"\nRatios:")
    print(f"  S̄(0)/S̄(1) = {S_bar[0]/S_bar[1]:.2f}×")
if S_bar[2] != 0:
    print(f"  S̄(0)/S̄(2) = {S_bar[0]/S_bar[2]:.2f}×")

# Save final results
print("\n" + "=" * 70)
print("Saving results...")
print("=" * 70)

# Save S matrix
np.savetxt(f"{output_dir}/influence_matrix_S.csv", S_final, 
           delimiter=",", fmt="%.6f", header="Normal,Mild,Moderate,Severe")
print(f"✓ Saved: {output_dir}/influence_matrix_S.csv")

# Save S̄(d)
s_bar_data = np.column_stack([np.arange(4), S_bar])
np.savetxt(f"{output_dir}/ordinal_sensitivity.csv", s_bar_data,
           delimiter=",", fmt="%.6f", header="distance,S_bar", comments="")
print(f"✓ Saved: {output_dir}/ordinal_sensitivity.csv")

# Save labeled version
df_S = pd.DataFrame(S_final, 
                    index=[f"Test_{name}" for name in level_names],
                    columns=[f"Train_{name}" for name in level_names])
df_S.to_csv(f"{output_dir}/influence_matrix_S_labeled.csv")
print(f"✓ Saved: {output_dir}/influence_matrix_S_labeled.csv")

print("\n" + "=" * 70)
print("✓ DONE!")
print("=" * 70)
print("\nGenerated files:")
print("  Intermediate (original labels 0,1,2,3):")
print("    - I_class0_original.csv (test=mild)")
print("    - I_class1_original.csv (test=moderate)")
print("    - I_class2_original.csv (test=severe)")
print("    - I_class3_original.csv (test=control)")
print("\n  Final (remapped labels 0=Normal, 1=Mild, 2=Moderate, 3=Severe):")
print("    - influence_matrix_S.csv")
print("    - influence_matrix_S_labeled.csv")
print("    - ordinal_sensitivity.csv")


