import pandas as pd

# Read both datasets
df_mimic4 = df
df_mimicnw = pd.read_csv('/content/drive/MyDrive/Northwestern/schizophrenia_antipsychotic_analysis.csv')

# Compare categorical values
for df, name in [(df_mimic4, 'MIMIC-IV'), (df_mimicnw, 'MIMIC-NW')]:
    print(f"\n{name} unique values:")
    print("Gender:", df['gender'].unique())
    print("Race:", df['race'].unique())

# MIMIC-NW race frequencies
print("\nMIMIC-NW race frequencies:")
print(df_mimicnw['race'].value_counts())

  # Create mapping dictionaries for race and gender
race_mapping = {
    'WHITE': 'White',
    'HISPANIC OR LATINO': 'Hispanic/Latino',
    'BLACK/AFRICAN': 'Black/African American',
    'ASIAN': 'Asian',

}

# Gender is already similar (M/F), but let's ensure consistency in case of other values
gender_mapping = {
    'M': 'M',
    'F': 'F'
    # Add other mappings if needed
}

# Apply the mappings to MIMIC-III
df_mimicnw['race'] = df_mimicnw['race'].map(race_mapping)

# Drop rows with NaN race values
df_mimicnw = df_mimicnw.dropna(subset=['race'])

# Verify the NaN values are gone
print("Updated MIMIC-NW unique values after dropping NaN:")
print("Race:", df_mimicnw['race'].unique())
print("\nUpdated MIMIC-NW race frequencies:")
print(df_mimicnw['race'].value_counts())
print(f"\nNew dataset size: {len(df_mimicnw)} rows")

# Display the updated unique values
print("\nUpdated MIMIC-NW unique values:")
print("Gender:", df_mimicnw['gender'].unique())
print("Race:", df_mimicnw['race'].unique())

# Compare with MIMIC-IV
print("\nMIMIC-IV unique values:")
print("Gender:", df_mimic4['gender'].unique())
print("Race:", df_mimic4['race'].unique())
      # prompt: see cell above. frequency of all unique race column entries for mimic nw

print("\nMIMIC-NW race frequencies:")
print(df_mimicnw['race'].value_counts())

import pandas as pd

# MIMIC-IV standard antipsychotic names
mimic_iv_antipsychotics = [
    "chlorpromazine", "droperidol", "fluphenazine", "haloperidol", "loxapine",
    "perphenazine", "pimozide", "prochlorperazine", "thioridazine", "thiothixene",
    "trifluoperazine", "aripiprazole", "asenapine", "clozapine", "iloperidone",
    "lurasidone", "olanzapine", "paliperidone", "quetiapine", "risperidone",
    "ziprasidone", "amisulpride"
]

# Northwestern detailed antipsychotic names
northwestern_antipsychotics = [
    'ARIPIPRAZOLE   5 MG ORAL TAB',
    'HALOPERIDOL  0.5 MG ORAL TAB',
    'HALOPERIDOL  5 MG ORAL TAB',
    'HALOPERIDOL 10 MG ORAL TAB',
    'HALOPERIDOL DECANOATE  50 MG/ML IM SOLN',
    'HALOPERIDOL LACTATE 5 MG/ML INJ SOLN',
    'OLANZAPINE  2.5 MG ORAL TAB',
    'OLANZAPINE  5 MG ORAL TAB',
    'OLANZAPINE  5 MG ORAL TBDI',
    'OLANZAPINE 10 MG IM SOLR',
    'OLANZAPINE 10 MG ORAL TAB',
    'QUETIAPINE  12.5 MG ORAL SPLIT TAB',
    'QUETIAPINE  25 MG ORAL TAB',
    'QUETIAPINE 100 MG ORAL TAB',
    'QUETIAPINE 200 MG ORAL TAB',
    'QUETIAPINE 300 MG ORAL TAB',
    'QUETIAPINE 300 MG ORAL TB24',
    'RISPERIDONE 0.5 MG ORAL TAB',
    'RISPERIDONE 1 MG/ML ORAL SOLN',
]

# Create explicit mapping dictionary
northwestern_to_mimic_iv = {}

# Map each Northwestern drug to its MIMIC-IV equivalent
for nw_drug in northwestern_antipsychotics:
    if 'ARIPIPRAZOLE' in nw_drug:
        northwestern_to_mimic_iv[nw_drug] = 'aripiprazole'
    elif 'HALOPERIDOL' in nw_drug:
        northwestern_to_mimic_iv[nw_drug] = 'haloperidol'
    elif 'OLANZAPINE' in nw_drug:
        northwestern_to_mimic_iv[nw_drug] = 'olanzapine'
    elif 'QUETIAPINE' in nw_drug:
        northwestern_to_mimic_iv[nw_drug] = 'quetiapine'
    elif 'RISPERIDONE' in nw_drug:
        northwestern_to_mimic_iv[nw_drug] = 'risperidone'

# Function to standardize drug names
def standardize_antipsychotic_name(drug_name):
    """
    Convert Northwestern detailed drug names to MIMIC-IV standard names
    """
    if pd.isna(drug_name):
        return drug_name

    drug_str = str(drug_name).strip()

    # First check exact match in Northwestern mapping
    if drug_str in northwestern_to_mimic_iv:
        return northwestern_to_mimic_iv[drug_str]

    # Check partial match for Northwestern style names
    drug_upper = drug_str.upper()
    for nw_drug, standard_name in northwestern_to_mimic_iv.items():
        if nw_drug in drug_upper:
            return standard_name

    # Check if already in MIMIC-IV standard format
    drug_lower = drug_str.lower()
    if drug_lower in mimic_iv_antipsychotics:
        return drug_lower

    # Check partial match for MIMIC-IV names
    for standard_drug in mimic_iv_antipsychotics:
        if standard_drug in drug_lower:
            return standard_drug

    # Return original if not an antipsychotic
    return drug_name

print("=== APPLYING STANDARDIZATION TO MIMIC NORTHWESTERN DATASET ===")

# Define the antipsychotic columns
antipsychotic_columns = ['first_antipsychotic', 'second_antipsychotic', 'third_antipsychotic']

# Show before standardization
print(f"\n=== BEFORE STANDARDIZATION ===")
for col in antipsychotic_columns:
    if col in df_mimicnw.columns:
        unique_drugs = df_mimicnw[col].dropna().unique()
        print(f"\n{col}:")
        print(f"  Unique values: {len(unique_drugs)}")
        for drug in unique_drugs:
            count = df_mimicnw[col].value_counts()[drug]
            print(f"    - {drug} ({count} patients)")

# Store examples of what will change for reporting
standardization_examples = {}
for col in antipsychotic_columns:
    if col in df_mimicnw.columns:
        examples = []
        for drug in df_mimicnw[col].dropna().unique():
            standardized = standardize_antipsychotic_name(drug)
            if drug != standardized:
                examples.append((drug, standardized))
        standardization_examples[col] = examples[:5]  # Keep first 5 examples

# Apply standardization directly to the original columns
for col in antipsychotic_columns:
    if col in df_mimicnw.columns:
        df_mimicnw[col] = df_mimicnw[col].apply(standardize_antipsychotic_name)

print(f"\n=== AFTER STANDARDIZATION ===")
for col in antipsychotic_columns:
    if col in df_mimicnw.columns:
        unique_drugs = df_mimicnw[col].dropna().unique()
        print(f"\n{col} (standardized):")
        print(f"  Unique values: {len(unique_drugs)}")
        for drug in unique_drugs:
            count = df_mimicnw[col].value_counts()[drug]
            print(f"    - {drug} ({count} patients)")

# Show what changed
print(f"\n=== STANDARDIZATION EXAMPLES ===")
for col, examples in standardization_examples.items():
    if examples:
        print(f"\n{col} standardization examples:")
        for original, standardized in examples:
            print(f"  '{original}' → '{standardized}'")
    else:
        print(f"\n{col}: No changes needed (already standardized)")

# Summary statistics
print(f"\n=== STANDARDIZATION SUMMARY ===")
total_patients = len(df_mimicnw)
print(f"Total patients in MIMIC Northwestern dataset: {total_patients}")

for col in antipsychotic_columns:
    if col in df_mimicnw.columns:
        non_null_count = df_mimicnw[col].notna().sum()
        unique_count = df_mimicnw[col].nunique()
        print(f"\n{col}:")
        print(f"  Patients with this antipsychotic: {non_null_count} ({non_null_count/total_patients*100:.1f}%)")
        print(f"  Unique standardized drugs: {unique_count}")

print(f"\n=== DATASET UPDATED ===")
print("Antipsychotic columns have been standardized to MIMIC-IV format in place")
print("Original columns updated with standardized values")

      # Print all columns for both dataframes
print("MIMIC-IV Columns:")
for col in df_mimic4.columns:
    print(f"- {col}")

print("\nMIMIC-NW Columns:")
for col in df_mimicnw.columns:
    print(f"- {col}")

# Optional: Compare column counts
print(f"\nTotal columns in MIMIC-IV: {len(df_mimic4.columns)}")
print(f"Total columns in MIMIC-NW: {len(df_mimicnw.columns)}")

# Optional: Find common columns
common_cols = set(df_mimic4.columns).intersection(set(df_mimicnw.columns))
print(f"\nCommon columns between datasets: {len(common_cols)}")
print("Common columns:")
for col in sorted(common_cols):
    print(f"- {col}")

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


def load_and_test_cf_cosine_model(df_mimicnw, model_path="/content/drive/MyDrive/saved_models/cf_cosine_k7_model.pkl",
                                 train_data_path="/content/drive/MyDrive/saved_models/cf_cosine_k7_train_data.pkl",
                                 K=7):
    """
    Load the trained CF (Cosine) model and test it on Northwestern data,
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

    df_mimicnw_processed = df_mimicnw.copy()

    if 'comorbidity_count' not in df_mimicnw_processed.columns:
        print("Creating comorbidity_count feature for Northwestern data...")
        if 'comorbidities' in df_mimicnw_processed.columns:
            df_mimicnw_processed['comorbidity_count'] = df_mimicnw_processed['comorbidities'].apply(
                lambda x: str(x).count(',') + 1 if pd.notna(x) and str(x).strip() else 0
            )
        else:
            df_mimicnw_processed['comorbidity_count'] = 0
            print("Warning: 'comorbidities' column not found in Northwestern data. Using dummy comorbidity_count=0.")

    print("Creating feature matrices for Northwestern test data (chronologically aware)...")

    # --- 1. Process Categorical Features ---
    # Ensure all defined categorical columns exist, fill NaNs with 'Unknown'
    df_categorical_mimicnw = pd.DataFrame(index=df_mimicnw_processed.index)
    for col in CATEGORICAL_FEATURES_DEF:
        if col in df_mimicnw_processed:
            df_categorical_mimicnw[col] = df_mimicnw_processed[col]
        else:
            print(f"Warning: Categorical feature '{col}' not found in df_mimicnw. Filling with 'Unknown'.")
            df_categorical_mimicnw[col] = 'Unknown'
    df_categorical_mimicnw = df_categorical_mimicnw.fillna('Unknown')

    try:
        encoded_categorical_part_sparse = encoder.transform(df_categorical_mimicnw[CATEGORICAL_FEATURES_DEF])
        if hasattr(encoded_categorical_part_sparse, 'toarray'):
            encoded_categorical_part = encoded_categorical_part_sparse.toarray()
        else: # Encoder was likely set to sparse_output=False
            encoded_categorical_part = encoded_categorical_part_sparse
        print(f"Categorical features encoded. Shape: {encoded_categorical_part.shape}")
    except Exception as e:
        print(f"Error encoding categorical features for Northwestern data: {e}")
        # Fallback to zeros, matching number of features from encoder
        n_samples_mimicnw = df_mimicnw_processed.shape[0]
        try:
            n_encoded_cat_features = encoder.transform(pd.DataFrame([['Unknown']*len(CATEGORICAL_FEATURES_DEF)], columns=CATEGORICAL_FEATURES_DEF)).shape[1]
        except: # If even that fails, try to infer from X_train
             n_encoded_cat_features = X_train.shape[1] - len(numerical_feature_names_from_model)

        encoded_categorical_part = np.zeros((n_samples_mimicnw, n_encoded_cat_features))
        print(f"Created dummy categorical encoding. Shape: {encoded_categorical_part.shape}")


    # --- 2. Process Numerical Features (Chronologically Aware) ---
    # This part assumes df_mimicnw_processed has a 'visit_number' column,
    # and that for a given visit_number, future data columns are NaN.
    all_numerical_rows_scaled = []

    for i in range(len(df_mimicnw_processed)):
        current_row_mimicnw = df_mimicnw_processed.iloc[i]
        current_visit_num = int(current_row_mimicnw.get('visit_number', 1))
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

                if can_be_known and col_name in current_row_mimicnw:
                    raw_val = pd.to_numeric(current_row_mimicnw.get(col_name), errors='coerce')
                    if pd.notna(raw_val):
                        val_to_use = raw_val
                    # else it remains NO_VISIT_SENTINEL (if NaN in data or not knowable for this visit)
            else: # Base numerical features (e.g., anchor_age, comorbidity_count)
                raw_val = pd.to_numeric(current_row_mimicnw.get(col_name), errors='coerce')
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
    print(f"Final chronologically aware X_test shape for Northwestern: {X_test.shape}")

    if X_test.shape[1] != X_train.shape[1]:
        print(f"FATAL ERROR: X_test columns ({X_test.shape[1]}) do not match X_train columns ({X_train.shape[1]})!")
        print("This usually means a mismatch in feature definitions, categorical encoding, or numerical feature lists.")
        print(f"  Encoded categorical features in X_test: {encoded_categorical_part.shape[1]}")
        print(f"  Scaled numerical features in X_test: {scaled_numerical_part.shape[1]}")
        return {} # Cannot proceed

    # --- Create target matrix Y_test for evaluation ---
    n_test = len(df_mimicnw_processed)
    n_therapies_model = len(therapy_ids) # Number of therapies known by the loaded model
    Y_test = np.zeros((n_test, n_therapies_model))
    therapy_found_count = 0
    no_med_count = 0

    # Assuming 'first_antipsychotic' is the target for Northwestern evaluation for consistency
    # with your previous use of this column.
    target_therapy_column = 'first_antipsychotic'
    if target_therapy_column not in df_mimicnw_processed.columns:
        print(f"Warning: Target column '{target_therapy_column}' not found in df_mimicnw. Y_test will be all zeros.")
        target_therapy_column = None # No target to evaluate against

    if target_therapy_column:
        # Add detailed logging to debug Y_test creation
        matched_medications = []
        unmatched_medications = []

        for i, idx in enumerate(df_mimicnw_processed.index):
            original_therapy_name = df_mimicnw_processed.loc[idx, target_therapy_column]
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
    print(f"Total rows in Northwestern processed: {len(df_mimicnw_processed)}")
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
    visit_dist = df_mimicnw_processed['visit_number'].value_counts().sort_index()
    print(f"\nVisit number distribution:")
    for visit_num, count in visit_dist.items():
        print(f"  Visit {visit_num}: {count} rows")
    # ====== END OF DIAGNOSTIC CODE ======

    # --- Calculate similarity matrix ---
    print("Calculating cosine similarity between Northwestern (X_test) and MIMIC-IV (X_train)...")
    similarity_matrix = cosine_similarity(X_test, X_train) # Shape: (n_mimicnw, n_mimic4_train)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # --- Evaluate model performance ---
    visit_results = {1: [], 2: [], 3: []}
    all_visit_results_list = [] # Renamed to avoid conflict with module name
    skipped_patients = []
    print("Evaluating model performance on Northwestern data...")

    for i in range(X_test.shape[0]):
        current_row_mimicnw = df_mimicnw_processed.iloc[i]
        visit_num = int(current_row_mimicnw.get('visit_number', 1))
        if not (1 <= visit_num <= 3):
            visit_num = 1

        # Check if this patient has any therapy marked in Y_test
        actual_therapy_indices_in_Y = np.nonzero(Y_test[i])[0]
        if len(actual_therapy_indices_in_Y) == 0:
            # Track why this patient was skipped
            therapy_name = current_row_mimicnw.get(target_therapy_column, 'MISSING')
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

    print("\n--- Northwestern Evaluation Results (Chronologically Aware) ---")
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
# Assuming df_mimicnw is your preprocessed Northwestern DataFrame
# Make sure it has 'visit_number' and the necessary feature columns.
# For columns that represent future information for a given visit_number, ensure they are NaN.
metrics = load_and_test_cf_cosine_model(df_mimicnw)
