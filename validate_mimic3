import pandas as pd

# Read both datasets
df_mimic4 = df
df_mimic3 = pd.read_csv('/content/drive/MyDrive/bq-results-20250131-201215-1738354352708/bq-results-20250131-201215-1738354352708.csv')

# Compare categorical values
for df, name in [(df_mimic4, 'MIMIC-IV'), (df_mimic3, 'MIMIC-III')]:
    print(f"\n{name} unique values:")
    print("Gender:", df['gender'].unique())
    print("Race:", df['race'].unique())


# Extract all antipsychotics from the three columns
antipsychotic_columns = ['first_antipsychotic', 'second_antipsychotic', 'third_antipsychotic']

# Method 1: Get all unique values from all three columns combined
all_antipsychotics = []

for column in antipsychotic_columns:
    # Get non-null values from each column
    values = df_mimic3[column].dropna().astype(str)
    # Remove empty strings and 'nan' strings
    values = values[values.str.strip() != '']
    values = values[values.str.lower() != 'nan']
    all_antipsychotics.extend(values.tolist())

# Standardize capitalization (first letter cap, rest lowercase)
standardized_antipsychotics = [med.capitalize() for med in all_antipsychotics]

# Get unique antipsychotics after standardization
unique_antipsychotics = sorted(list(set(standardized_antipsychotics)))

print("All unique antipsychotics from MIMIC-III dataset (standardized capitalization):")
print("=" * 70)
for i, med in enumerate(unique_antipsychotics, 1):
    print(f"{i:2d}. {med}")

print(f"\nTotal number of unique antipsychotics: {len(unique_antipsychotics)}")

# Show original vs standardized comparison
print("\n" + "="*70)
print("ORIGINAL vs STANDARDIZED COMPARISON:")
print("="*70)
original_unique = sorted(list(set(all_antipsychotics)))
for orig in original_unique:
    standardized = orig.capitalize()
    if orig != standardized:
        print(f"'{orig}' -> '{standardized}'")
    else:
        print(f"'{orig}' (no change needed)")

# Method 2: Show breakdown by column
print("\n" + "="*60)
print("BREAKDOWN BY COLUMN:")
print("="*60)

for column in antipsychotic_columns:
    unique_in_column = df_mimic3[column].dropna().astype(str)
    unique_in_column = unique_in_column[unique_in_column.str.strip() != '']
    unique_in_column = unique_in_column[unique_in_column.str.lower() != 'nan']
    unique_in_column = sorted(unique_in_column.unique().tolist())

    print(f"\n{column.upper()}:")
    print(f"Number of unique medications: {len(unique_in_column)}")
    for med in unique_in_column:
        print(f"  - {med}")

# Method 3: Create a frequency count of standardized antipsychotics
print("\n" + "="*70)
print("FREQUENCY COUNT OF STANDARDIZED ANTIPSYCHOTICS:")
print("="*70)

from collections import Counter
standardized_counts = Counter(standardized_antipsychotics)
sorted_counts = sorted(standardized_counts.items(), key=lambda x: x[1], reverse=True)

for med, count in sorted_counts:
    print(f"{med:<30} : {count:>3} occurrences")

# Method 4: Check for any potential data quality issues
print("\n" + "="*60)
print("DATA QUALITY CHECK:")
print("="*60)

for column in antipsychotic_columns:
    total_rows = len(df_mimic3)
    non_null_count = df_mimic3[column].notna().sum()
    null_count = df_mimic3[column].isna().sum()

    print(f"\n{column}:")
    print(f"  Total rows: {total_rows}")
    print(f"  Non-null values: {non_null_count}")
    print(f"  Null values: {null_count}")
    print(f"  Percentage filled: {(non_null_count/total_rows)*100:.1f}%")

# Final clean list for easy copying
print("\n" + "="*70)
print("CLEAN STANDARDIZED LIST (for easy copying):")
print("="*70)
print("antipsychotic_list = [")
for med in unique_antipsychotics:
    print(f"    '{med}',")
print("]")
print(f"\n# Total: {len(unique_antipsychotics)} unique antipsychotics")

# Create mapping dictionaries for race and gender
race_mapping = {
    'WHITE': 'White',
    'HISPANIC OR LATINO': 'Hispanic/Latino',
    'BLACK/AFRICAN AMERICAN': 'Black/African American',
    'BLACK/CAPE VERDEAN': 'Black/African American',  # Mapping to closest equivalent
    'ASIAN': 'Asian',
    'MULTI RACE ETHNICITY': 'Other',  # No direct equivalent in MIMIC-IV
    'OTHER': 'Other',
    'UNKNOWN/NOT SPECIFIED': 'Unknown',
    'UNABLE TO OBTAIN': 'Unknown'
}

# Gender is already similar (M/F), but let's ensure consistency in case of other values
gender_mapping = {
    'M': 'M',
    'F': 'F'
    # Add other mappings if needed
}

# Apply the mappings to MIMIC-III
df_mimic3['race'] = df_mimic3['race'].map(race_mapping)
df_mimic3['gender'] = df_mimic3['gender'].map(gender_mapping)

# Display the updated unique values
print("\nUpdated MIMIC-III unique values:")
print("Gender:", df_mimic3['gender'].unique())
print("Race:", df_mimic3['race'].unique())

# Compare with MIMIC-IV
print("\nMIMIC-IV unique values:")
print("Gender:", df_mimic4['gender'].unique())
print("Race:", df_mimic4['race'].unique())

# prompt: mimic3 race frequencies

# Method 5: Frequency count of MIMIC-III Race categories
print("\n" + "="*60)
print("FREQUENCY COUNT OF MIMIC-III RACE:")
print("="*60)

# Use value_counts() to get frequencies, handle potential NaNs
race_counts = df_mimic3['race'].value_counts(dropna=False)
# Sort by count descending
sorted_race_counts = race_counts.sort_values(ascending=False)

# Print counts
total_count = sorted_race_counts.sum()
for race_cat, count in sorted_race_counts.items():
    # Handle NaN key gracefully for printing
    cat_name = 'NaN' if pd.isna(race_cat) else race_cat
    percentage = (count / total_count) * 100 if total_count > 0 else 0
    print(f"{cat_name:<25} : {count:>5} patients ({percentage:.1f}%)")

print(f"\nTotal patients (including NaN race): {total_count}")

# prompt: see cell above please delete if race is unkown or other

df_mimic3 = df_mimic3[~df_mimic3['race'].isin(['Unknown', 'Other'])].copy()

# Display the updated unique values and frequency count
print("\nUpdated MIMIC-III unique values after removing 'Unknown' and 'Other' races:")
print("Race:", df_mimic3['race'].unique())

print("\n" + "="*60)
print("FREQUENCY COUNT OF MIMIC-III RACE AFTER FILTERING:")
print("="*60)

race_counts_filtered = df_mimic3['race'].value_counts(dropna=False)
sorted_race_counts_filtered = race_counts_filtered.sort_values(ascending=False)

total_count_filtered = sorted_race_counts_filtered.sum()
for race_cat, count in sorted_race_counts_filtered.items():
    cat_name = 'NaN' if pd.isna(race_cat) else race_cat
    percentage = (count / total_count_filtered) * 100 if total_count_filtered > 0 else 0
    print(f"{cat_name:<25} : {count:>5} patients ({percentage:.1f}%)")

print(f"\nTotal patients remaining: {total_count_filtered}")

# Display the updated unique values
print("\nUpdated MIMIC-III unique values:")
print("Gender:", df_mimic3['gender'].unique())
print("Race:", df_mimic3['race'].unique())

# Compare with MIMIC-IV
print("\nMIMIC-IV unique values:")
print("Gender:", df_mimic4['gender'].unique())
print("Race:", df_mimic4['race'].unique())
# Print all columns for both dataframes
print("MIMIC-IV Columns:")
for col in df_mimic4.columns:
    print(f"- {col}")

print("\nMIMIC-III Columns:")
for col in df_mimic3.columns:
    print(f"- {col}")

# Optional: Compare column counts
print(f"\nTotal columns in MIMIC-IV: {len(df_mimic4.columns)}")
print(f"Total columns in MIMIC-III: {len(df_mimic3.columns)}")

# Optional: Find common columns
common_cols = set(df_mimic4.columns).intersection(set(df_mimic3.columns))
print(f"\nCommon columns between datasets: {len(common_cols)}")
print("Common columns:")
for col in sorted(common_cols):
    print(f"- {col}")

# Extract and print ICD codes from MIMIC-III
print("ICD Codes from MIMIC-III:")

# Check the data type and format of the icd_codes column
print("\nData type of icd_codes column:", type(df_mimic3['icd_codes'].iloc[0]))
print("Sample of first few ICD codes:")
print(df_mimic3['icd_codes'].head())

# If icd_codes is stored as a string (like a comma-separated list), extract all unique codes
if df_mimic3['icd_codes'].dtype == 'object':
    # Create a set to store all unique ICD codes
    all_icd_codes = set()

    # Function to handle different possible formats
    def extract_codes(code_string):
        if isinstance(code_string, str):
            # Try common delimiters
            if ',' in code_string:
                return code_string.split(',')
            elif ';' in code_string:
                return code_string.split(';')
            elif ' ' in code_string.strip():
                return code_string.strip().split()
            else:
                return [code_string]
        elif isinstance(code_string, list):
            return code_string
        else:
            return []

    # Extract all codes
    for codes in df_mimic3['icd_codes']:
        extracted_codes = extract_codes(codes)
        for code in extracted_codes:
            code = code.strip() if isinstance(code, str) else code
            if code:
                all_icd_codes.add(code)

    # Print the unique ICD codes
    print(f"\nTotal unique ICD codes: {len(all_icd_codes)}")
    print("\nUnique ICD codes (first 50):")
    for i, code in enumerate(sorted(all_icd_codes)):
        print(code, end=", ")
        if (i + 1) % 10 == 0:
            print()  # New line after every 10 codes
        if i >= 49:  # Only show first 50 codes
            print("...")
            break

else:
    # If icd_codes is stored in another format (like a list), adapt accordingly
    print("\nICD codes column is not in string format. Format:", df_mimic3['icd_codes'].dtype)
    print("Please provide more details about how ICD codes are stored for better extraction.")

# 1. Extract only the first ICD code from each row
def extract_first_code(code_string):
    if isinstance(code_string, str):
        # Split by common delimiters and take the first code
        if ',' in code_string:
            return code_string.split(',')[0].strip()
        elif ';' in code_string:
            return code_string.split(';')[0].strip()
        elif ' ' in code_string.strip():
            return code_string.strip().split()[0]
        else:
            return code_string.strip()
    return code_string

# Apply the extraction
df_mimic3['icd_codes'] = df_mimic3['icd_codes'].apply(extract_first_code)

print("First 10 rows after extracting first ICD code:")
print(df_mimic3['icd_codes'].head(10))

# 2. Apply ICD-9 to ICD-10 mapping
icd9_to_icd10_map = {
    # Schizoaffective Disorder (295.7x -> F25.x)
    '29570': 'F259', # Schizoaffective disorder, unspecified
    '29572': 'F259', # Schizoaffective disorder, unspecified (simplification)
    '29573': 'F259', # Schizoaffective disorder, unspecified (simplification)
    '29574': 'F259', # Schizoaffective disorder, unspecified (simplification)
    '29575': 'F259', # Schizoaffective disorder, unspecified (simplification)

    # Schizophrenia, Unspecified (295.9x -> F20.9)
    '29590': 'F209', # Schizophrenia, unspecified
    '29592': 'F209', # Schizophrenia, unspecified (simplification)
    '29594': 'F209', # Schizophrenia, unspecified (simplification)

    # Paranoid Schizophrenia (295.3x -> F20.0)
    '29530': 'F200', # Paranoid schizophrenia
    '29532': 'F200', # Paranoid schizophrenia (simplification)
    '29533': 'F200', # Paranoid schizophrenia (simplification)
    '29534': 'F200', # Paranoid schizophrenia (simplification)

    # Catatonic Schizophrenia (295.4x -> F20.2)
    '29540': 'F202', # Catatonic schizophrenia
    '29544': 'F202', # Catatonic schizophrenia (simplification)

    # Disorganized/Hebephrenic Schizophrenia (295.1x, 295.2x -> F20.1)
    '29510': 'F201', # Disorganized schizophrenia
    '29512': 'F201', # Disorganized schizophrenia (simplification)
    '29514': 'F201', # Disorganized schizophrenia (simplification)
    '29520': 'F201', # Hebephrenic -> Disorganized schizophrenia
    '29523': 'F201', # Hebephrenic -> Disorganized schizophrenia (simplification)
    '29524': 'F201', # Hebephrenic -> Disorganized schizophrenia (simplification)

    # Residual Schizophrenia (295.6x -> F20.5)
    '29560': 'F205', # Residual schizophrenia
    '29562': 'F205', # Residual schizophrenia (simplification)
    '29564': 'F205', # Residual schizophrenia (simplification)

    # Other Specified Schizophrenia (295.8x -> F20.89)
    '29580': 'F2089', # Other schizophrenia
    '29584': 'F2089', # Other schizophrenia (simplification)

    # Latent Schizophrenia (295.5x -> F21)
    '29550': 'F21',   # Latent schizophrenia -> Schizotypal disorder
}

# Store original codes before mapping for comparison
original_codes = df_mimic3['icd_codes'].copy()

# Apply the mapping to replace the original column
df_mimic3['icd_codes'] = df_mimic3['icd_codes'].replace(icd9_to_icd10_map)

# Check if any original codes did not get mapped
unmapped_mask = ~original_codes.isin(icd9_to_icd10_map.keys())
unmapped_codes = original_codes[unmapped_mask].unique()
if len(unmapped_codes) > 0:
    print(f"\nWARNING: Found {len(unmapped_codes)} unmapped ICD-9 codes: {unmapped_codes}")

# Display the mapping results
comparison_df = pd.DataFrame({
    'original_icd_codes': original_codes,
    'mapped_icd_codes': df_mimic3['icd_codes']
})
print("\nICD-9 to ICD-10 mapping results:")
print(comparison_df.head(10))

# Count frequencies of ICD-10 codes
icd10_counts = df_mimic3['icd_codes'].value_counts()
print("\nFrequency of ICD-10 codes:")
print(icd10_counts)

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error

# Assuming this definition is available or can be inferred from your create_matrices
# This value MUST match the one used when the original model's features were created.
NO_VISIT_SENTINEL = 999999

# Assuming these definitions are available (as they were for create_matrices)
# These help distinguish time-based features from base numerical features.
TIME_COLS_TO_PROCESS_DEF = [
    'days_from_first_to_second',
    'days_from_second_to_third',
    'length_of_stay_2',
    'length_of_stay_3',
    'days_between_visit_1_and_2',
    'days_between_visit_2_and_3'
]
CATEGORICAL_FEATURES_DEF = ['gender', 'race']


def calculate_map_at_k(recommended_indices, actual_index, k=3):
    """
    Calculate Average Precision at k for a single patient.
    """
    if not recommended_indices or actual_index not in recommended_indices[:k]:
        return 0.0
    try:
        rank = recommended_indices.index(actual_index) + 1
        return 1.0 / rank
    except ValueError: # Should be caught by the first check, but as a safeguard
        return 0.0


def load_and_test_cf_cosine_model(df_mimic3, model_path="/content/drive/MyDrive/saved_models/cf_cosine_k7_model.pkl",
                                 train_data_path="/content/drive/MyDrive/saved_models/cf_cosine_k7_train_data.pkl",
                                 K=7):
    """
    Load the trained CF (Cosine) model and test it on MIMIC-III data,
    respecting information availability for each visit.
    """
    print("Loading CF (Cosine) model and training data...")
    with open(model_path, 'rb') as f:
        model_components = pickle.load(f)
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)

    therapy_ids_map = model_components['therapy_ids_map']
    therapy_ids = model_components['therapy_ids']
    encoder = model_components['encoder']
    scaler = model_components['scaler']
    # feature_names from the loaded model are the numerical features the scaler expects, in order.
    numerical_feature_names_from_model = model_components['feature_names']

    X_train = train_data['X_train']
    A_all_train = train_data['A_all_train']
    # df_train = train_data['df_train'] # Not directly used in this testing script after loading X_train

    print(f"Model loaded. K={K}. Training X_train: {X_train.shape}, A_all_train: {A_all_train.shape}")

    # ====== DIAGNOSTIC CODE FOR THERAPY IDS ======
    print("\n=== Therapy IDs Map Analysis ===")
    print(f"Total therapies in model: {len(therapy_ids_map)}")
    print("Available therapies in model:")
    for therapy, idx in sorted(therapy_ids_map.items()):
        print(f"  {idx}: {therapy}")

    print(f"\nIs 'NO_MEDICATION' in therapy_ids_map? {'NO_MEDICATION' in therapy_ids_map}")
    # ====== END OF DIAGNOSTIC CODE ======

    df_mimic3_processed = df_mimic3.copy()

    if 'comorbidity_count' not in df_mimic3_processed.columns:
        print("Creating comorbidity_count feature for MIMIC-III data...")
        if 'comorbidities' in df_mimic3_processed.columns:
            df_mimic3_processed['comorbidity_count'] = df_mimic3_processed['comorbidities'].apply(
                lambda x: str(x).count(',') + 1 if pd.notna(x) and str(x).strip() else 0
            )
        else:
            df_mimic3_processed['comorbidity_count'] = 0
            print("Warning: 'comorbidities' column not found in MIMIC-III. Using dummy comorbidity_count=0.")

    print("Creating feature matrices for MIMIC-III test data (chronologically aware)...")

    # --- 1. Process Categorical Features ---
    # Ensure all defined categorical columns exist, fill NaNs with 'Unknown'
    df_categorical_mimic3 = pd.DataFrame(index=df_mimic3_processed.index)
    for col in CATEGORICAL_FEATURES_DEF:
        if col in df_mimic3_processed:
            df_categorical_mimic3[col] = df_mimic3_processed[col]
        else:
            print(f"Warning: Categorical feature '{col}' not found in df_mimic3. Filling with 'Unknown'.")
            df_categorical_mimic3[col] = 'Unknown'
    df_categorical_mimic3 = df_categorical_mimic3.fillna('Unknown')

    try:
        encoded_categorical_part_sparse = encoder.transform(df_categorical_mimic3[CATEGORICAL_FEATURES_DEF])
        if hasattr(encoded_categorical_part_sparse, 'toarray'):
            encoded_categorical_part = encoded_categorical_part_sparse.toarray()
        else: # Encoder was likely set to sparse_output=False
            encoded_categorical_part = encoded_categorical_part_sparse
        print(f"Categorical features encoded. Shape: {encoded_categorical_part.shape}")
    except Exception as e:
        print(f"Error encoding categorical features for MIMIC-III: {e}")
        # Fallback to zeros, matching number of features from encoder
        n_samples_mimic3 = df_mimic3_processed.shape[0]
        try:
            n_encoded_cat_features = encoder.transform(pd.DataFrame([['Unknown']*len(CATEGORICAL_FEATURES_DEF)], columns=CATEGORICAL_FEATURES_DEF)).shape[1]
        except: # If even that fails, try to infer from X_train
             n_encoded_cat_features = X_train.shape[1] - len(numerical_feature_names_from_model)

        encoded_categorical_part = np.zeros((n_samples_mimic3, n_encoded_cat_features))
        print(f"Created dummy categorical encoding. Shape: {encoded_categorical_part.shape}")


    # --- 2. Process Numerical Features (Chronologically Aware) ---
    # This part assumes df_mimic3_processed has a 'visit_number' column,
    # and that for a given visit_number, future data columns are NaN.
    all_numerical_rows_scaled = []

    for i in range(len(df_mimic3_processed)):
        current_row_mimic3 = df_mimic3_processed.iloc[i]
        current_visit_num = int(current_row_mimic3.get('visit_number', 1))
        if not (1 <= current_visit_num <= 3): # Ensure visit_num is within expected range
            current_visit_num = 1 # Default if out of bounds

        numerical_values_this_row_ordered = []
        for col_name in numerical_feature_names_from_model: # Iterate in the order scaler expects
            val_to_use = np.nan

            is_time_col = col_name in TIME_COLS_TO_PROCESS_DEF

            if is_time_col:
                val_to_use = NO_VISIT_SENTINEL # Default for time cols not yet known
                can_be_known = False
                # Define what's known at the START of each visit
                if current_visit_num == 1: # For Visit 1, no prior visit info is known
                    pass
                elif current_visit_num == 2: # For Visit 2, info from Visit 1 is known
                    if col_name in ['days_from_first_to_second', 'length_of_stay_1', 'days_between_visit_1_and_2']:
                        can_be_known = True
                elif current_visit_num >= 3: # For Visit 3, info from Visit 1 & 2 is known
                    if col_name in ['days_from_first_to_second', 'length_of_stay_1', 'days_between_visit_1_and_2',
                                   'days_from_second_to_third', 'length_of_stay_2', 'days_between_visit_2_and_3']:
                        can_be_known = True

                if can_be_known and col_name in current_row_mimic3:
                    raw_val = pd.to_numeric(current_row_mimic3.get(col_name), errors='coerce')
                    if pd.notna(raw_val):
                        val_to_use = raw_val
                    # else it remains NO_VISIT_SENTINEL (if NaN in data or not knowable for this visit)
            else: # Base numerical features (e.g., anchor_age, comorbidity_count)
                raw_val = pd.to_numeric(current_row_mimic3.get(col_name), errors='coerce')
                val_to_use = 0 if pd.isna(raw_val) else raw_val # Default to 0 if NaN

            numerical_values_this_row_ordered.append(val_to_use)

        # Reshape for scaler and transform
        scaled_numerical_for_row = scaler.transform(np.array(numerical_values_this_row_ordered).reshape(1, -1))
        all_numerical_rows_scaled.append(scaled_numerical_for_row.flatten())

    scaled_numerical_part = np.array(all_numerical_rows_scaled)
    print(f"Numerical features processed and scaled. Shape: {scaled_numerical_part.shape}")

    # --- 3. Combine Features ---
    # Original X_df was pd.concat([encoded_df, scaled_df], axis=1)
    # So, encoded categorical first, then scaled numerical.
    X_test = np.hstack([encoded_categorical_part, scaled_numerical_part])
    print(f"Final chronologically aware X_test shape for MIMIC-III: {X_test.shape}")

    if X_test.shape[1] != X_train.shape[1]:
        print(f"FATAL ERROR: X_test columns ({X_test.shape[1]}) do not match X_train columns ({X_train.shape[1]})!")
        print("This usually means a mismatch in feature definitions, categorical encoding, or numerical feature lists.")
        print(f"  Encoded categorical features in X_test: {encoded_categorical_part.shape[1]}")
        print(f"  Scaled numerical features in X_test: {scaled_numerical_part.shape[1]}")
        return {} # Cannot proceed

    # --- Create target matrix Y_test for evaluation ---
    n_test = len(df_mimic3_processed)
    n_therapies_model = len(therapy_ids) # Number of therapies known by the loaded model
    Y_test = np.zeros((n_test, n_therapies_model))
    therapy_found_count = 0
    no_med_count = 0

    # Assuming 'first_antipsychotic' is the target for MIMIC-III evaluation for consistency
    # with your previous use of this column.
    target_therapy_column = 'first_antipsychotic'
    if target_therapy_column not in df_mimic3_processed.columns:
        print(f"Warning: Target column '{target_therapy_column}' not found in df_mimic3. Y_test will be all zeros.")
        target_therapy_column = None # No target to evaluate against

    if target_therapy_column:
        # Add detailed logging to debug Y_test creation
        matched_medications = []
        unmatched_medications = []

        for i, idx in enumerate(df_mimic3_processed.index):
            original_therapy_name = df_mimic3_processed.loc[idx, target_therapy_column]
            therapy_name = original_therapy_name

            # Handle NaN values
            if pd.isna(therapy_name):
                if 'NO_MEDICATION' in therapy_ids_map:
                    therapy_name = 'NO_MEDICATION'
                    no_med_count += 1
                else:
                    print(f"Patient {i}: NaN therapy but NO_MEDICATION not in map - skipping")
                    continue
            else:
                # Normalize therapy name: convert to lowercase and strip whitespace
                therapy_name = str(therapy_name).lower().strip()

            # Check if therapy exists in map
            if therapy_name in therapy_ids_map:
                therapy_idx_mapped = therapy_ids_map[therapy_name]
                Y_test[i, therapy_idx_mapped] = 1
                therapy_found_count += 1
                matched_medications.append(f"Patient {i}: {original_therapy_name} -> {therapy_name} (idx {therapy_idx_mapped})")
            else:
                unmatched_medications.append(f"Patient {i}: {original_therapy_name} not in therapy_ids_map")
                print(f"Patient {i}: Therapy '{original_therapy_name}' not found in therapy_ids_map")

        print(f"Found {therapy_found_count} matches for '{target_therapy_column}' in therapy_ids_map.")
        if 'NO_MEDICATION' in therapy_ids_map:
             print(f"Treated {no_med_count} NaN target values as NO_MEDICATION.")

        print(f"\nDetailed Y_test creation results:")
        print(f"Matched medications: {len(matched_medications)}")
        print(f"Unmatched medications: {len(unmatched_medications)}")

        if matched_medications:
            print("First 10 matched medications:")
            for match in matched_medications[:10]:
                print(f"  {match}")

        if unmatched_medications:
            print("First 10 unmatched medications:")
            for unmatch in unmatched_medications[:10]:
                print(f"  {unmatch}")

        # Double-check Y_test creation
        patients_with_therapy_in_Y = (Y_test.sum(axis=1) > 0).sum()
        print(f"Verification: Patients with therapy marked in Y_test: {patients_with_therapy_in_Y}")

    # ====== DIAGNOSTIC CODE FOR Y_TEST ======
    print(f"\n=== Pre-Evaluation Diagnostics ===")
    print(f"Total rows in MIMIC-III processed: {len(df_mimic3_processed)}")
    print(f"Y_test shape: {Y_test.shape}")
    print(f"Number of patients with at least one therapy marked in Y_test: {(Y_test.sum(axis=1) > 0).sum()}")

    # Check what's actually in Y_test
    therapy_counts_in_Y = {}
    for i in range(Y_test.shape[0]):
        nonzero_indices = np.nonzero(Y_test[i])[0]
        if len(nonzero_indices) > 0:
            therapy_idx = nonzero_indices[0]  # Assuming one therapy per patient
            therapy_name = therapy_ids[therapy_idx]
            therapy_counts_in_Y[therapy_name] = therapy_counts_in_Y.get(therapy_name, 0) + 1

    print(f"\nTherapy distribution in Y_test:")
    for therapy, count in sorted(therapy_counts_in_Y.items()):
        print(f"  {therapy}: {count} patients")

    # Check visit number distribution
    visit_dist = df_mimic3_processed['visit_number'].value_counts().sort_index()
    print(f"\nVisit number distribution:")
    for visit_num, count in visit_dist.items():
        print(f"  Visit {visit_num}: {count} rows")
    # ====== END OF DIAGNOSTIC CODE ======

    # --- Calculate similarity matrix ---
    print("Calculating cosine similarity between MIMIC-III (X_test) and MIMIC-IV (X_train)...")
    similarity_matrix = cosine_similarity(X_test, X_train) # Shape: (n_mimic3, n_mimic4_train)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # --- Evaluate model performance ---
    visit_results = {1: [], 2: [], 3: []}
    all_visit_results_list = [] # Renamed to avoid conflict with module name
    skipped_patients = []
    print("Evaluating model performance on MIMIC-III data...")

    for i in range(X_test.shape[0]):
        current_row_mimic3 = df_mimic3_processed.iloc[i]
        visit_num = int(current_row_mimic3.get('visit_number', 1))
        if not (1 <= visit_num <= 3):
            visit_num = 1

        # Check if this patient has any therapy marked in Y_test
        actual_therapy_indices_in_Y = np.nonzero(Y_test[i])[0]
        if len(actual_therapy_indices_in_Y) == 0:
            # Track why this patient was skipped
            therapy_name = current_row_mimic3.get(target_therapy_column, 'MISSING')
            skipped_patients.append({
                'patient_idx': i,
                'visit_num': visit_num,
                'therapy_name': therapy_name,
                'reason': 'No therapy marked in Y_test'
            })
            continue

        similarity_row = similarity_matrix[i, :]
        if len(similarity_row) != X_train.shape[0]:
            skipped_patients.append({
                'patient_idx': i,
                'visit_num': visit_num,
                'reason': f'Similarity row length mismatch: {len(similarity_row)} != {X_train.shape[0]}'
            })
            continue

        # K-NN prediction logic (simplified from your recommend_therapy/predict_outcome)
        actual_K = min(K, X_train.shape[0])
        if actual_K <= 0:
            skipped_patients.append({
                'patient_idx': i,
                'visit_num': visit_num,
                'reason': 'K <= 0'
            })
            continue

        neighbor_indices = np.argsort(similarity_row)[-actual_K:]
        neighbor_similarities = similarity_row[neighbor_indices]

        # Ensure A_all_train is dense for easier indexing here if it's sparse
        A_all_train_dense = A_all_train.toarray() if hasattr(A_all_train, "toarray") else A_all_train
        neighbor_outcomes = A_all_train_dense[neighbor_indices]

        weights = neighbor_similarities.reshape(-1, 1)
        # Ensure no division by zero if all similarities are zero for some reason
        sum_weights = weights.sum()
        if sum_weights < 1e-9 : # Effectively zero
            predicted_scores = np.mean(neighbor_outcomes, axis=0) if neighbor_outcomes.shape[0] > 0 else np.zeros(A_all_train_dense.shape[1])
        else:
            weighted_outcomes = neighbor_outcomes * weights
            predicted_scores = weighted_outcomes.sum(axis=0) / sum_weights

        ranked_indices = np.argsort(predicted_scores)[::-1]
        top_3_indices = ranked_indices[:3].tolist()

        actual_idx = actual_therapy_indices_in_Y[0] # Assuming one actual therapy for evaluation
        # actual_therapy_name = therapy_ids[actual_idx] # therapy_ids is based on original model

        match = actual_idx in top_3_indices
        current_visit_metric = {
            'visit_number': visit_num, 'match': match, 'rank': float('inf'),
            'actual_therapy_idx': actual_idx, # 'actual_therapy_name': actual_therapy_name,
            'pred_score_for_actual': predicted_scores[actual_idx] if actual_idx < len(predicted_scores) else np.nan,
            'top1_match': False, 'map_score': 0.0, 'rmse': np.nan
        }

        if match:
            current_visit_metric['rank'] = top_3_indices.index(actual_idx) + 1
            current_visit_metric['top1_match'] = (current_visit_metric['rank'] == 1)
            # For MAP score, often only "good" actual outcomes contribute.
            # Here, Y_test[i, actual_idx] is 1, so we consider it relevant.
            current_visit_metric['map_score'] = 1.0 / current_visit_metric['rank']
            # RMSE: predicted score for actual therapy vs. ideal score (e.g., 1.0 if Y_test is binary)
            pred_outcome_for_actual = predicted_scores[actual_idx]
            current_visit_metric['rmse'] = np.sqrt((pred_outcome_for_actual - 1.0)**2) # Assuming target is 1.0

        visit_results[visit_num].append(current_visit_metric)
        all_visit_results_list.append(current_visit_metric)

    # ====== POST-EVALUATION DIAGNOSTICS ======
    print(f"\n=== Evaluation Summary ===")
    print(f"Total patients processed: {X_test.shape[0]}")
    print(f"Patients evaluated: {len(all_visit_results_list)}")
    print(f"Patients skipped: {len(skipped_patients)}")

    if skipped_patients:
        print(f"\nSkipped patients breakdown:")
        skip_reasons = {}
        for patient in skipped_patients:
            reason = patient['reason']
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        for reason, count in skip_reasons.items():
            print(f"  {reason}: {count} patients")

        # Show a few examples of skipped patients
        print(f"\nFirst 5 skipped patients:")
        for patient in skipped_patients[:5]:
            print(f"  Patient {patient['patient_idx']}, Visit {patient['visit_num']}: {patient['reason']}")
            if 'therapy_name' in patient:
                print(f"    Therapy: {patient['therapy_name']}")
    # ====== END OF POST-EVALUATION DIAGNOSTICS ======

    # --- Calculate aggregate metrics ---
    def calculate_aggregated_visit_metrics(results_list): # Renamed param to avoid conflict
        if not results_list:
            return {'n_samples': 0, 'n_matches': 0, 'overlap': 0, 'top_1_accuracy': 0, 'map@3': 0, 'rmse': np.nan}

        n_samples = len(results_list)
        n_matches = sum(1 for r in results_list if r['match'])
        overlap = n_matches / n_samples if n_samples > 0 else 0
        top_1_matches = sum(1 for r in results_list if r['top1_match'])
        # Top-1 Accuracy: Among all samples, how many times was the actual therapy the top recommendation?
        top_1_accuracy_overall = top_1_matches / n_samples if n_samples > 0 else 0

        map_scores = [r['map_score'] for r in results_list] # map_score is already AP@k for this sample
        mean_ap_at_3 = np.mean(map_scores) if map_scores else 0.0

        rmse_values = [r['rmse'] for r in results_list if pd.notna(r['rmse'])] # Changed from np.isfinite for pandas Series
        avg_rmse = np.mean(rmse_values) if rmse_values else np.nan

        return {
            'n_samples': n_samples, 'n_matches_top3': n_matches, 'overlap_top3': overlap,
            'top_1_accuracy': top_1_accuracy_overall, 'map@3': mean_ap_at_3, 'rmse': avg_rmse
        }

    final_metrics = {
        'visit1': calculate_aggregated_visit_metrics(visit_results[1]),
        'visit2': calculate_aggregated_visit_metrics(visit_results[2]),
        'visit3': calculate_aggregated_visit_metrics(visit_results[3]),
        'all_visits': calculate_aggregated_visit_metrics(all_visit_results_list)
    }

    print("\n--- MIMIC-III Evaluation Results (Chronologically Aware) ---")
    for visit_name, v_metrics in final_metrics.items():
        print(f"\n{visit_name.replace('_', ' ').replace('visit', 'Visit ').title()} Results:")
        print(f"  Number of samples: {v_metrics['n_samples']}")
        print(f"  Matches in Top-3: {v_metrics['n_matches_top3']}")
        print(f"  Overlap Rate (Top-3): {v_metrics['overlap_top3']:.4f}")
        print(f"  Top-1 Accuracy: {v_metrics['top_1_accuracy']:.4f}")
        print(f"  MAP@3: {v_metrics['map@3']:.4f}")
        if pd.notna(v_metrics['rmse']):
            print(f"  RMSE (of predicted score for actual): {v_metrics['rmse']:.4f}")
        else:
            print(f"  RMSE: N/A")

    return final_metrics

# Example of how you might call this:
# Assuming df_mimic3 is your preprocessed MIMIC-III DataFrame
# Make sure it has 'visit_number' and the necessary feature columns.
# For columns that represent future information for a given visit_number, ensure they are NaN.
metrics = load_and_test_cf_cosine_model(df_mimic3)
