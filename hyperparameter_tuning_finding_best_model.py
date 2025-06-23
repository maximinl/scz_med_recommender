import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, precision_recall_curve, auc, f1_score
from sklearn.neighbors import LocalOutlierFactor, NeighborhoodComponentsAnalysis as NCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.sparse import lil_matrix, issparse, csr_matrix
from tqdm.notebook import tqdm
import json
import warnings
import traceback
import os


# Try importing optional packages
try:
    from gower import gower_matrix
    print("Successfully imported gower_matrix.")
except ImportError:
    print("Info: 'gower' package not found. Install it ('pip install gower') to use Gower similarity.")
    gower_matrix = None

try:
    from skrebate import ReliefF
    print("Successfully imported ReliefF from skrebate.")
except ImportError:
    print("Info: 'skrebate' package not found. Install it ('pip install skrebate') to use RBA weighting.")
    ReliefF = None

# Modified calculate_affinity function to prevent data leakage
def calculate_affinity(row, reference_df):
    """
    Updated affinity calculation that handles structural missing values
    and avoids division by zero/NaN warnings during normalization.
    Uses meaningful defaults (0.0 or 1.0) for zero/NaN normalization ranges.
    Uses the reference DataFrame ('reference_df') for calculating normalization bounds.
    """
    # --- Safely get values from row, default to NaN/appropriate value if column missing ---
    def get_val(r, col_name, default=np.nan):
        if isinstance(r, pd.Series):
            return r.get(col_name, default)
        elif isinstance(r, dict):
            return r.get(col_name, default)
        else:
            try:
                if hasattr(r, 'index') and col_name in r.index:
                    return r[col_name]
                elif hasattr(r, '__contains__') and col_name in r:
                    return r[col_name]
                else:
                    return default
            except Exception:
                return default

    # Use defaults consistent with preprocessing
    time_between_admissions_1_2 = get_val(row, 'days_between_visit_1_and_2', default=float('inf'))
    time_between_admissions_2_3 = get_val(row, 'days_between_visit_2_and_3', default=float('inf'))
    visit_number = get_val(row, 'visit_number', default=np.nan)
    time_from_first_to_second = get_val(row, 'days_from_first_to_second', default=float('inf'))
    time_from_second_to_third = get_val(row, 'days_from_second_to_third', default=float('inf'))
    second_med = get_val(row, 'second_antipsychotic', default='No_Second_Med')
    third_med = get_val(row, 'third_antipsychotic', default='No_Third_Med')
    length_of_stay_1 = get_val(row, 'length_of_stay_1', default=np.nan)
    length_of_stay_2 = get_val(row, 'length_of_stay_2', default=0)
    length_of_stay_3 = get_val(row, 'length_of_stay_3', default=0)

    # --- Handle cases where essential columns might be missing or all NaN ---
    if pd.isna(visit_number) or pd.isna(length_of_stay_1):
        return np.nan

    # --- Calculate Normalization Bounds (Safely handles missing columns in reference_df) ---
    time_cols = [col for col in ['days_between_visit_1_and_2', 'days_between_visit_2_and_3'] if col in reference_df.columns]
    switch_cols = [col for col in ['days_from_first_to_second', 'days_from_second_to_third'] if col in reference_df.columns]
    los_cols = [col for col in ['length_of_stay_1', 'length_of_stay_2', 'length_of_stay_3'] if col in reference_df.columns]

    # Calculate sums, replacing inf with NaN temporarily for min/max calculation if needed
    def calculate_safe_bounds(df, cols):
        min_val_out, max_val_out = np.nan, np.nan
        all_inf_or_nan = False

        if not cols:
            return min_val_out, max_val_out, all_inf_or_nan

        numeric_cols_df = df[cols].apply(pd.to_numeric, errors='coerce')
        row_sums = numeric_cols_df.replace(float('inf'), np.nan).sum(axis=1, skipna=True)

        valid_sums = row_sums.dropna()

        if valid_sums.empty:
             all_inf_or_nan = True
        else:
            min_val_out = valid_sums.min()
            max_val_out = valid_sums.max()

        return min_val_out, max_val_out, all_inf_or_nan

    min_time, max_time, all_time_inf = calculate_safe_bounds(reference_df, time_cols)
    min_switch_time, max_switch_time, all_switch_inf = calculate_safe_bounds(reference_df, switch_cols)

    # For LoS and Visit Number, 0 is a valid minimum. Max needs to be calculated.
    max_length_of_stay = 0
    if los_cols:
        try:
            numeric_los_df = reference_df[los_cols].apply(pd.to_numeric, errors='coerce')
            los_sums = numeric_los_df.sum(axis=1, skipna=True)
            if los_sums.notna().any():
                 max_length_of_stay = los_sums.max()
            else:
                 max_length_of_stay = np.nan
        except Exception as e:
            max_length_of_stay = np.nan

    # Ensure max_visit_num is calculated correctly even if column is missing
    max_visit_num = np.nan
    if 'visit_number' in reference_df.columns:
        numeric_visits = pd.to_numeric(reference_df['visit_number'], errors='coerce')
        if numeric_visits.notna().any():
            max_visit_num = numeric_visits.max(skipna=True)

    # If max_visit_num is still NaN, use current row's value if valid
    visit_number_num = pd.to_numeric(visit_number, errors='coerce')
    if pd.isna(max_visit_num) and pd.notna(visit_number_num):
        max_visit_num = visit_number_num
    elif pd.isna(max_visit_num):
         max_visit_num = 0

    # Define a small epsilon for checking division by zero in floating point numbers
    epsilon = 1e-9

    # Calculate ranges, check for NaN explicitly
    time_range = max_time - min_time if np.isfinite(max_time) and np.isfinite(min_time) else np.nan
    switch_time_range = max_switch_time - min_switch_time if np.isfinite(max_switch_time) and np.isfinite(min_switch_time) else np.nan
    # --- Calculate Normalized Components ---

    # Time between admissions
    # Sum current times, treating inf as inf
    current_time_sum = 0
    is_current_time_inf = False
    # Ensure times are numeric before comparison/addition
    t1_num = pd.to_numeric(time_between_admissions_1_2, errors='coerce')
    t2_num = pd.to_numeric(time_between_admissions_2_3, errors='coerce')

    if t1_num == float('inf') or t2_num == float('inf'):
        is_current_time_inf = True # Mark if either contributing time is infinite
        if t1_num == float('inf') and t2_num == float('inf'):
             current_time_sum = float('inf') # Both inf -> sum is inf
        elif np.isfinite(t1_num):
             current_time_sum = t1_num # Use the finite one
        elif np.isfinite(t2_num):
             current_time_sum = t2_num # Use the finite one
        else: # Handle cases like one inf, one NaN
             current_time_sum = float('inf') # Default to inf if cannot determine finite sum
    elif pd.notna(t1_num) and pd.notna(t2_num): # Both are finite and not NaN
        current_time_sum = t1_num + t2_num
    else: # One or both are NaN (but not inf)
        current_time_sum = np.nan # Result is NaN if any component is NaN


    # Normalize Time Between Admissions
    time_between_admissions_normalized = np.nan # Default to NaN
    if current_time_sum == float('inf'): # If the patient's time is infinite
        time_between_admissions_normalized = 1.0
    elif all_time_inf: # If all patients had infinite time
         time_between_admissions_normalized = 1.0
    elif pd.notna(current_time_sum): # Only proceed if current time is finite
        if pd.isna(time_range): # Should only happen if min/max were NaN but not all_inf (unlikely)
             pass # Keep NaN
        elif time_range <= epsilon: # Range is zero (all finite values were identical)
             time_between_admissions_normalized = 0.0 # Assign 0, representing the minimum observed time
        elif np.isfinite(min_time) and time_range > epsilon: # Ensure min_time is valid and range > 0 before division
             time_between_admissions_normalized = (current_time_sum - min_time) / time_range
        # else: min_time was NaN or range was invalid, keep normalized as NaN


    # Number of visits
    number_of_visits_normalized = np.nan # Default to NaN
    # visit_number_num already coerced above
    if pd.notna(visit_number_num):
        # Ensure max_visit_num is finite before division
        if pd.isna(max_visit_num) or not np.isfinite(max_visit_num) or max_visit_num <= 0:
            # If max is invalid or zero, normalized score depends on current value
            # If current value is also 0 or less, it represents the minimum possible, so score is 1.0 (1 - 0/max -> 1)
            # If current value is positive but max is 0, this is odd, treat score as 0 (worst)
            number_of_visits_normalized = 1.0 if visit_number_num <= 0 else 0.0
        else:
            # Normal case: Higher visit number -> lower score
            number_of_visits_normalized = 1.0 - (visit_number_num / max_visit_num)


    # Medication switch timing & penalty
    medication_switch_penalty = 0.0 # Start with no penalty
    time_between_medication_switch = float('inf') # Default to infinite time (no switch)
    has_second_med = (second_med != 'No_Second_Med')
    has_third_med = (third_med != 'No_Third_Med')

    # Ensure time components are numeric
    tfs_num = pd.to_numeric(time_from_first_to_second, errors='coerce')
    tst_num = pd.to_numeric(time_from_second_to_third, errors='coerce')

    # Determine penalty and calculate finite switch time if applicable
    if has_third_med:
        medication_switch_penalty = 0.2
        # Calculate time only if both components are finite
        if np.isfinite(tfs_num) and np.isfinite(tst_num):
            time_between_medication_switch = tfs_num + tst_num
        # If either is infinite, the total time is effectively infinite for switching *to third*
        # Keep time_between_medication_switch as float('inf')

    elif has_second_med:
        medication_switch_penalty = 0.1
        # Time is finite only if time_from_first_to_second is finite
        if np.isfinite(tfs_num):
            time_between_medication_switch = tfs_num
        # Otherwise, keep time_between_medication_switch as float('inf')

    else: # No second or third med
        medication_switch_penalty = -0.05 # Bonus
        # time_between_medication_switch remains float('inf')


    # Normalize medication switch time
    time_between_medication_switch_normalized = np.nan # Default to NaN
    if time_between_medication_switch == float('inf'): # If the patient's switch time is infinite
        time_between_medication_switch_normalized = 1.0
    elif all_switch_inf: # If all patients had infinite switch times
        time_between_medication_switch_normalized = 1.0
    elif pd.notna(time_between_medication_switch): # Only proceed if current switch time is finite
        if pd.isna(switch_time_range): # Error case
            pass # Keep NaN
        elif switch_time_range <= epsilon: # Range is zero (all finite switch times were identical)
            time_between_medication_switch_normalized = 0.0 # Assign 0, representing the minimum observed switch time
        elif np.isfinite(min_switch_time) and switch_time_range > epsilon: # Ensure min_switch_time is valid and range > 0
             time_between_medication_switch_normalized = (time_between_medication_switch - min_switch_time) / switch_time_range
        # else: min_switch_time was NaN or range invalid, keep normalized as NaN


    # Length of stay
    length_of_stay_normalized = np.nan # Default to NaN
    # Ensure LoS components are numeric
    los1_num = pd.to_numeric(length_of_stay_1, errors='coerce')
    los2_num = pd.to_numeric(length_of_stay_2, errors='coerce')
    los3_num = pd.to_numeric(length_of_stay_3, errors='coerce')

    # Calculate sum only if los1 is valid (others default to 0 if NaN)
    current_los_sum = np.nan
    if pd.notna(los1_num):
        # Ensure others are treated as 0 if NaN *before* summing
        los2_val = los2_num if pd.notna(los2_num) else 0
        los3_val = los3_num if pd.notna(los3_num) else 0
        current_los_sum = los1_num + los2_val + los3_val

    if pd.notna(current_los_sum): # Proceed only if sum is valid
        # Ensure max_length_of_stay is finite before division
        if pd.isna(max_length_of_stay) or not np.isfinite(max_length_of_stay) or max_length_of_stay < 0: # Handle NaN or negative max LoS
             pass # Keep NaN
        elif max_length_of_stay <= epsilon: # Max LoS is effectively zero
            if abs(current_los_sum) <= epsilon: # If current sum is also effectively zero
                 length_of_stay_normalized = 1.0 # Interpret as minimum relative stay (1 - 0/0 -> 1)
            else: # Max is zero, but current is positive? Treat as max relative stay.
                 length_of_stay_normalized = 0.0 # (1 - positive/0 -> treat as 0)
        else:
            # Normal case: Higher LoS -> lower score
            length_of_stay_normalized = 1.0 - (current_los_sum / max_length_of_stay)

    # --- Calculate Final Affinity Score ---
    # Ensure normalization didn't produce NaN/inf before averaging
    components = [
        time_between_admissions_normalized,
        number_of_visits_normalized,
        time_between_medication_switch_normalized,
        length_of_stay_normalized
    ]

    # Filter out potential NaN/inf values from components before averaging
    valid_components = [c for c in components if pd.notna(c) and np.isfinite(c)]

    if not valid_components:
        return np.nan # Return NaN if all components failed

    # Calculate average of valid components
    affinity_sum = sum(valid_components)
    affinity_avg = affinity_sum / len(valid_components) # Average only the valid ones

    # Apply penalty after averaging
    affinity = affinity_avg - medication_switch_penalty

    # Clip score to be between 0 and 1
    affinity = max(0.0, min(1.0, affinity))

    return affinity


# --- Data Preparation and Matrix Creation ---
# Modified create_matrices function to prevent data leakage
def create_matrices(df, therapy_ids_map=None, fit_transform=True, encoder=None, scaler=None, feature_names=None):
    """
    Creates feature and outcome matrices from the DataFrame.
    If fit_transform is True, fits and transforms the data (for training).
    If fit_transform is False, uses provided encoder and scaler (for test data).

    Parameters:
    - feature_names: List of feature names that were available during fit (for consistency)
    """
    print("Running: create_matrices...")
    # Make a copy to avoid modifying the original DataFrame
    df_processed = df.copy()
    N = len(df_processed)

    # --- Therapy IDs ---
    # Ensure all therapy names are strings and handle potential NaNs
    therapies_cleaned = df_processed['first_antipsychotic'].apply(
        lambda x: 'NO_MEDICATION' if pd.isna(x) or str(x).lower() == 'nan' else str(x)
    )
    all_therapies = therapies_cleaned.unique()
    # Ensure 'NO_MEDICATION' is definitely included and at index 0
    unique_therapies = ['NO_MEDICATION'] + sorted([t for t in all_therapies if t != 'NO_MEDICATION'])

    if therapy_ids_map is None:
        therapy_ids_map = {name: i for i, name in enumerate(unique_therapies)}
        print(f"  Created new therapy ID map with {len(therapy_ids_map)} therapies.")
        print(f"  Therapies: {unique_therapies}")
    else:
        # Ensure all therapies in the current data are in the map, add if necessary
        existing_therapies = set(therapy_ids_map.keys())
        current_unique_therapies = set(unique_therapies)
        new_therapies = list(current_unique_therapies - existing_therapies)
        if new_therapies:
            print(f"  Warning: Adding {len(new_therapies)} new therapies to existing map: {new_therapies}")
            start_index = len(therapy_ids_map)
            for idx, therapy in enumerate(new_therapies):
                therapy_ids_map[therapy] = start_index + idx
        print(f"  Using existing therapy ID map with {len(therapy_ids_map)} therapies.")
        print(f"  Therapies: {list(therapy_ids_map.keys())}")

    # Create the list in the correct order based on the map
    therapy_ids = [None] * len(therapy_ids_map)
    for name, i in therapy_ids_map.items():
        if i < len(therapy_ids):
             therapy_ids[i] = name
        else:
             while len(therapy_ids) <= i:
                 therapy_ids.append(None)
             therapy_ids[i] = name
    therapy_ids = [t for t in therapy_ids if t is not None]

    M = len(therapy_ids_map)
    print(f"  Number of unique therapies (M): {M}")

    # --- Feature Selection & Handling ---
    # Define features explicitly
    categorical_features = ['gender', 'race']
    # Base numerical features (these should be complete based on user info)
    numerical_features_base = ['anchor_age', 'comorbidity_count']
    # Time-based features (handle NaN specifically for event-not-occurred)
    numerical_features_time = []

    # Define a sentinel value to indicate "no visit occurred"
    NO_VISIT_SENTINEL = 999999  # A very large value that clearly indicates "no visit"

    # Expanded list of all time-based features that indicate missing visits when NaN
    time_cols_to_process = [
        'days_from_first_to_second',
        'days_from_second_to_third',
        'length_of_stay_2',
        'length_of_stay_3',
        'days_between_visit_1_and_2',
        'days_between_visit_2_and_3'
    ]

    # Print a clear explanation of how these features are handled
    print(f"  Time-based features with meaningful missing values will use {NO_VISIT_SENTINEL} to indicate no visit occurred")

    if not fit_transform and feature_names is not None:
        # For test data, ensure the same features as training are available
        time_features_needed = [col for col in time_cols_to_process if col in feature_names]
        for col_name in time_features_needed:
            if col_name not in df_processed.columns:
                print(f"  Adding required feature '{col_name}' with NO_VISIT_SENTINEL value to indicate no visit occurred.")
                df_processed[col_name] = NO_VISIT_SENTINEL
                numerical_features_time.append(col_name)
            elif col_name not in numerical_features_time:
                print(f"  Adding required feature '{col_name}' from feature_names list.")
                numeric_col = pd.to_numeric(df_processed[col_name], errors='coerce')
                # NaN in these columns indicate "no visit occurred", so use sentinel
                df_processed[col_name] = numeric_col.fillna(NO_VISIT_SENTINEL)
                numerical_features_time.append(col_name)
    else:
        # For training data, handle these features appropriately
        for col_name in time_cols_to_process:
            if col_name in df_processed.columns:
                numeric_col = pd.to_numeric(df_processed[col_name], errors='coerce')
                # For these time-based features, NaN means "no visit occurred"
                df_processed[col_name] = numeric_col.fillna(NO_VISIT_SENTINEL)
                numerical_features_time.append(col_name)
                print(f"  Added '{col_name}' as feature (NaNs set to {NO_VISIT_SENTINEL} to indicate no visit occurred)")
            else:
                print(f"  Warning: Time feature '{col_name}' not found in DataFrame.")

    # Combine numerical features
    numerical_features = numerical_features_base + numerical_features_time

    # Check if all selected features exist in the dataframe
    missing_cat = [f for f in categorical_features if f not in df_processed.columns]
    missing_num = [f for f in numerical_features if f not in df_processed.columns]
    if missing_cat:
        raise ValueError(f"Missing categorical feature columns: {missing_cat}")
    if missing_num:
        raise ValueError(f"Missing numerical feature columns: {missing_num}")

    # --- Feature Processing with proper handling for train/test ---
    print("  Processing features (OneHotEncoding, Scaling)...")

    # One-Hot Encode Categorical Features
    try:
        if fit_transform:
            # Training set: Fit and transform
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_categorical = encoder.fit_transform(df_processed[categorical_features])
        else:
            # Test set: Transform only using fitted encoder
            if encoder is None:
                raise ValueError("Encoder required when fit_transform is False")
            encoded_categorical = encoder.transform(df_processed[categorical_features])

        encoded_df = pd.DataFrame(encoded_categorical,
                                 index=df_processed.index,
                                 columns=encoder.get_feature_names_out(categorical_features))
    except ValueError as e:
        if "contains NaN" in str(e):
             print(f"ERROR: NaN found in categorical features {categorical_features} during OneHotEncoding.")
        raise e

    # Standardize Numerical Features
    existing_numerical_features = [f for f in numerical_features if f in df_processed.columns]
    if not existing_numerical_features:
        raise ValueError("No numerical features found to scale.")

    try:
        if fit_transform:
            # Training set: Fit and transform
            scaler = StandardScaler()
            scaled_numerical = scaler.fit_transform(df_processed[existing_numerical_features])
            # Save the feature names used during fit
            feature_names = existing_numerical_features.copy()
        else:
            # Test set: Transform only using fitted scaler
            if scaler is None:
                raise ValueError("Scaler required when fit_transform is False")

            # Ensure test data has all the features the scaler expects
            missing_features = set(feature_names) - set(existing_numerical_features)
            if missing_features:
                print(f"  Warning: Test data is missing features that were used during training: {missing_features}")
                for feature in missing_features:
                    print(f"  Adding missing feature '{feature}' with NO_VISIT_SENTINEL value.")
                    df_processed[feature] = NO_VISIT_SENTINEL  # Add missing feature with sentinel value

                # Update existing_numerical_features to include newly added features
                existing_numerical_features = [f for f in feature_names if f in df_processed.columns]

            # Make sure features are in the same order as they were during fit
            existing_numerical_features = [f for f in feature_names if f in df_processed.columns]

            scaled_numerical = scaler.transform(df_processed[existing_numerical_features])

        scaled_df = pd.DataFrame(scaled_numerical,
                                index=df_processed.index,
                                columns=existing_numerical_features)
    except ValueError as e:
         if "contains NaN" in str(e):
             print(f"ERROR: NaN found in numerical features {existing_numerical_features} during Scaling.")
         raise e

    # Combine Features
    X_df = pd.concat([encoded_df, scaled_df], axis=1)
    X = X_df.values
    print(f"  Shape of feature matrix X: {X.shape}")

    # Check for NaNs in final matrix
    if np.isnan(X).any():
        print("ERROR: NaNs found in final X matrix after processing!")
        raise ValueError("NaNs found in final feature matrix X.")

    # --- Outcome Matrices (A~hist, A~all, Y~) ---
    # Initialize with specified shape N x M
    A_hist = lil_matrix((N, M), dtype=float)
    A_all = lil_matrix((N, M), dtype=float)
    Y = lil_matrix((N, M), dtype=float)

    # Pre-calculate subject groups and indices for faster lookup
    subject_indices = {}
    for idx, subject_id in enumerate(df_processed['subject_id']):
        if subject_id not in subject_indices:
            subject_indices[subject_id] = []
        subject_indices[subject_id].append(df_processed.index[idx])

    # Iterate through the original DataFrame index to maintain order
    print("  Populating outcome matrices (Y, A_all, A_hist)...")
    for i in tqdm(range(N), desc=" Creating Outcome Matrices"):
        current_original_index = df_processed.index[i]
        row = df_processed.iloc[i]
        subject_id = row['subject_id']

        # --- Get current therapy and affinity score ---
        current_therapy_val = row['first_antipsychotic']
        if pd.isna(current_therapy_val) or str(current_therapy_val).lower() == 'nan':
            current_therapy_str = 'NO_MEDICATION'
        else:
            current_therapy_str = str(current_therapy_val)

        affinity_score = row['affinity_score']
        if pd.isna(affinity_score):
            affinity_score = 0

        # Get therapy index from map
        if current_therapy_str in therapy_ids_map:
            therapy_idx = therapy_ids_map[current_therapy_str]
        else:
            print(f"  Internal Warning: Therapy '{current_therapy_str}' from row index {current_original_index} still not in map. Skipping.")
            continue

        # --- Populate Y (Current consultation outcome) ---
        Y[i, therapy_idx] = affinity_score

        # --- Populate A_all and A_hist ---
        history_original_indices = subject_indices[subject_id]
        try:
            current_row_position_in_history = history_original_indices.index(current_original_index)
        except ValueError:
            print(f"  Internal Error: Could not find current index {current_original_index} in subject {subject_id}'s history. Skipping.")
            continue

        # Iterate through the subject's history up to and including the current row
        temp_A_all_row = {}
        for hist_idx_pos in range(current_row_position_in_history + 1):
            hist_original_index = history_original_indices[hist_idx_pos]
            hist_row = df_processed.loc[hist_original_index]

            hist_therapy_val = hist_row['first_antipsychotic']
            if pd.isna(hist_therapy_val) or str(hist_therapy_val).lower() == 'nan':
                hist_therapy_str = 'NO_MEDICATION'
            else:
                hist_therapy_str = str(hist_therapy_val)

            hist_affinity = hist_row['affinity_score']
            if pd.isna(hist_affinity):
                 hist_affinity = 0

            if hist_therapy_str in therapy_ids_map:
                hist_therapy_idx = therapy_ids_map[hist_therapy_str]
                temp_A_all_row[hist_therapy_idx] = hist_affinity

                if hist_idx_pos < current_row_position_in_history:
                    A_hist[i, hist_therapy_idx] = hist_affinity

        # Populate A_all for the current row i
        for hist_therapy_idx, score in temp_A_all_row.items():
            A_all[i, hist_therapy_idx] = score

    # Convert to csr_matrix for efficient computations
    A_hist = A_hist.tocsr()
    A_all = A_all.tocsr()
    Y = Y.tocsr()
    print("Finished: create_matrices.")

    # --- Final Checks ---
    if np.isnan(X).any():
        print("ERROR: NaNs detected in final X matrix after create_matrices.")
        raise ValueError("NaNs detected in final X matrix.")
    if hasattr(A_hist, 'data') and np.isnan(A_hist.data).any():
        print("Warning: NaNs detected in final A_hist matrix data (filling with 0).")
        A_hist.data = np.nan_to_num(A_hist.data, nan=0.0)
    if hasattr(A_all, 'data') and np.isnan(A_all.data).any():
        print("Warning: NaNs detected in final A_all matrix data (filling with 0).")
        A_all.data = np.nan_to_num(A_all.data, nan=0.0)
    if hasattr(Y, 'data') and np.isnan(Y.data).any():
        print("Warning: NaNs detected in final Y matrix data (filling with 0).")
        Y.data = np.nan_to_num(Y.data, nan=0.0)

    if fit_transform:
        return X, A_hist, A_all, Y, therapy_ids, therapy_ids_map, encoder, scaler, feature_names
    else:
        return X, A_hist, A_all, Y, therapy_ids, therapy_ids_map

# --- Helper Functions for Evaluation ---

def calculate_similarity_cosine(X_subset, X_test_subset=None):
    """Calculates cosine similarity. If X_test_subset is provided, calculates between train and test."""
    if X_test_subset is None:
        # Ensure no NaNs before calculation (should not happen if checks in create_matrices pass)
        if np.isnan(X_subset).any(): raise ValueError("NaN found in X_subset for cosine similarity")
        return cosine_similarity(X_subset)
    else:
        # Ensure no NaN values exist before calculating similarity
        if np.isnan(X_subset).any(): raise ValueError("NaN found in X_subset for cosine similarity")
        if np.isnan(X_test_subset).any(): raise ValueError("NaN found in X_test_subset for cosine similarity")
        # X_subset = np.nan_to_num(X_subset) # Keep error check instead of imputation here
        # X_test_subset = np.nan_to_num(X_test_subset)
        return cosine_similarity(X_test_subset, X_subset) # Shape: (n_test, n_train)

def calculate_similarity_euclidean(X_subset, X_test_subset=None):
    """Calculates Euclidean distance based similarity."""
    if X_test_subset is None:
         # Ensure no NaNs
        if np.isnan(X_subset).any(): raise ValueError("NaN found in X_subset for euclidean similarity")
        distances = euclidean_distances(X_subset)
    else:
        # Ensure no NaN values exist
        if np.isnan(X_subset).any(): raise ValueError("NaN found in X_subset for euclidean similarity")
        if np.isnan(X_test_subset).any(): raise ValueError("NaN found in X_test_subset for euclidean similarity")
        # X_subset = np.nan_to_num(X_subset)
        # X_test_subset = np.nan_to_num(X_test_subset)
        distances = euclidean_distances(X_test_subset, X_subset) # Shape: (n_test, n_train)
    # Add epsilon to avoid division by zero if distance is exactly 0
    return 1 / (1 + distances + 1e-9) # Convert distance to similarity

def calculate_similarity_gower(X_subset, X_test_subset=None):
    """
    Calculates Gower similarity using the imported function.
    Handles both pairwise (within X_subset) and train-test calculation.
    NOTE: Gower handles NaNs internally, but requires DataFrame input.
    """
    if gower_matrix is None:
        raise ImportError("Gower package not installed. Cannot use Gower similarity.")

    # Gower requires DataFrame, convert if necessary
    # It can handle NaNs internally, so no explicit check needed here unlike cosine/euclidean
    if not isinstance(X_subset, pd.DataFrame):
        X_subset_df = pd.DataFrame(X_subset)
    else:
        X_subset_df = X_subset

    if X_test_subset is None:
        # Calculate pairwise within the training set
        sim_matrix = 1 - gower_matrix(X_subset_df) # Gower returns distance, convert to similarity
    else:
        # Calculate similarity between test and train
        if not isinstance(X_test_subset, pd.DataFrame):
            X_test_subset_df = pd.DataFrame(X_test_subset)
        else:
            X_test_subset_df = X_test_subset

        n_test = X_test_subset_df.shape[0]
        n_train = X_subset_df.shape[0]

        # Check for empty dataframes
        if n_test == 0 or n_train == 0:
             return np.empty((n_test, n_train)) # Return empty array if either is empty


        # Combine test and train for pairwise calculation
        # Ensure column names match if they exist, otherwise rely on positional alignment
        if hasattr(X_test_subset_df, 'columns') and hasattr(X_subset_df, 'columns'):
             if not X_test_subset_df.columns.equals(X_subset_df.columns):
                  print("Warning: Column names mismatch between test and train for Gower. Relying on column order.")
                  # Consider aligning columns here if necessary, e.g.,
                  # X_subset_df = X_subset_df[X_test_subset_df.columns] or similar logic
        combined_df = pd.concat([X_test_subset_df, X_subset_df], ignore_index=True)

        # Calculate full distance matrix
        full_dist_matrix = gower_matrix(combined_df)

        # Extract the block corresponding to Test vs Train distances
        # This block is from row 0 to n_test-1, and col n_test to end
        dist_test_train = full_dist_matrix[:n_test, n_test:]

        # Convert distance to similarity
        sim_matrix = 1 - dist_test_train # Shape: (n_test, n_train)

    return sim_matrix


def calculate_similarity_euclidean_rbf(X_subset, X_test_subset=None, gamma=1.0):
    """Calculates RBF kernel similarity based on Euclidean distance."""
    if X_test_subset is None:
        # Ensure no NaNs
        if np.isnan(X_subset).any(): raise ValueError("NaN found in X_subset for RBF similarity")
        distances = euclidean_distances(X_subset)
    else:
        # Ensure no NaN values exist
        if np.isnan(X_subset).any(): raise ValueError("NaN found in X_subset for RBF similarity")
        if np.isnan(X_test_subset).any(): raise ValueError("NaN found in X_test_subset for RBF similarity")
        # X_subset = np.nan_to_num(X_subset)
        # X_test_subset = np.nan_to_num(X_test_subset)
        distances = euclidean_distances(X_test_subset, X_subset) # Shape: (n_test, n_train)
    return np.exp(-gamma * (distances ** 2))

def rba_weighting_and_selection(X_train, A_all_train, Y_train, K_RBA=5, thr_good=0.5, thr_w=0.01):
    """Performs Relief-based attribute weighting and selection using ReliefF."""
    if ReliefF is None:
        raise ImportError("skrebate package not installed. Cannot use ReliefF for RBA.")

    # Ensure no NaNs in input data for RBA
    if np.isnan(X_train).any(): raise ValueError("NaN found in X_train for RBA weighting")

    print("  Running RBA weighting...")
    # Ensure labels are calculated correctly
    rba_labels = create_rba_labels(A_all_train, Y_train, thr_good)
    if len(np.unique(rba_labels)) < 2:
        print("  Warning: RBA labels have less than 2 unique classes. Skipping weighting.")
        return X_train, np.ones(X_train.shape[1]) # Return unweighted and equal weights

    # Ensure ReliefF is initialized correctly
    rba = ReliefF(n_neighbors=K_RBA, n_features_to_select=None) # Select all features initially to get weights
    rba.fit(X_train, rba_labels)
    feature_weights = rba.feature_importances_

    if feature_weights is None:
         print("  Warning: RBA feature weights are None. Skipping weighting.")
         return X_train, np.ones(X_train.shape[1])

    # Apply threshold and weighting
    feature_weights_thresholded = np.copy(feature_weights)
    feature_weights_thresholded[feature_weights_thresholded < thr_w] = 0
    if np.sum(feature_weights_thresholded) == 0:
        print("  Warning: All RBA feature weights are zero after thresholding. Using unweighted features.")
        return X_train, np.ones(X_train.shape[1])

    X_train_weighted = X_train * feature_weights_thresholded
    print("  Finished RBA weighting.")
    return X_train_weighted, feature_weights_thresholded # Return weights as well

def calculate_similarity_gower_with_rba(X_train, A_all_train, Y_train, X_test=None):
    """Calculates Gower similarity after RBA weighting."""
    if gower_matrix is None:
        raise ImportError("Gower package not installed. Cannot use Gower similarity.")

    # RBA weighting should only happen on training data to learn weights
    X_train_weighted, feature_weights = rba_weighting_and_selection(X_train, A_all_train, Y_train)

    # Ensure X_train_weighted is a DataFrame for Gower
    # Gower handles NaNs, but RBA weighting step should have already checked/failed on NaNs
    X_train_weighted_df = pd.DataFrame(X_train_weighted)

    # Apply weighting to test set if provided
    if X_test is not None:
        if np.isnan(X_test).any(): raise ValueError("NaN found in X_test before applying RBA weights")
        # Ensure no division by zero if all weights are zero (handled in rba_weighting)
        X_test_weighted = X_test * feature_weights
        X_test_weighted_df = pd.DataFrame(X_test_weighted)
        # Use the corrected calculate_similarity_gower for train/test
        return calculate_similarity_gower(X_train_weighted_df, X_test_weighted_df) # Pass DFs
    else:
        # Calculate Gower pairwise within weighted train set
        return calculate_similarity_gower(X_train_weighted_df) # Pass DF


def calculate_similarity_euclidean_with_nca(X_subset, nca_model, X_test_subset=None):
    """Calculates Euclidean similarity after NCA transformation."""
    # Ensure no NaN values exist before transform
    if np.isnan(X_subset).any(): raise ValueError("NaN found in X_subset for NCA similarity")
    # X_subset = np.nan_to_num(X_subset) # Use check instead
    X_transformed_train = nca_model.transform(X_subset)

    if X_test_subset is None:
        distances = euclidean_distances(X_transformed_train)
    else:
        if np.isnan(X_test_subset).any(): raise ValueError("NaN found in X_test_subset for NCA similarity")
        # X_test_subset = np.nan_to_num(X_test_subset) # Use check instead
        X_transformed_test = nca_model.transform(X_test_subset)
        distances = euclidean_distances(X_transformed_test, X_transformed_train) # Shape: (n_test, n_train)
    # Add epsilon to avoid division by zero if distance is exactly 0
    return 1 / (1 + distances + 1e-9) # Convert distance to similarity

def create_rba_labels(A_all_train, Y_train, thr_good=0.5):
    """Creates labels for Relief-based algorithms."""
    labels = np.zeros(A_all_train.shape[0], dtype=int) # Default label 0 (neutral)
    # Convert to dense arrays for easier processing if sparse
    Y_train_dense = Y_train.toarray() if issparse(Y_train) else np.asarray(Y_train)
    A_all_train_dense = A_all_train.toarray() if issparse(A_all_train) else np.asarray(A_all_train)

    # Check for NaNs which shouldn't be present after create_matrices fixes
    if np.isnan(Y_train_dense).any() or np.isnan(A_all_train_dense).any():
        raise ValueError("NaN found in outcome matrices for RBA label creation.")

    for i in range(A_all_train_dense.shape[0]):
        # Find the therapy applied in the current consultation i
        applied_therapy_indices = np.nonzero(Y_train_dense[i])[0]
        if len(applied_therapy_indices) == 0:
            continue # Skip if no therapy recorded in Y for this row

        target_therapy = applied_therapy_indices[0] # Assume only one therapy per row in Y
        target_outcome_is_good = Y_train_dense[i, target_therapy] > thr_good

        # Find other consultations where the *same* therapy was applied (using A_all)
        same_therapy_consultation_indices = np.nonzero(A_all_train_dense[:, target_therapy])[0]

        # Compare outcomes for these consultations
        # This simplified version labels based on *any* comparison found
        found_comparison = False
        for j in same_therapy_consultation_indices:
            if i == j: continue # Don't compare to self
            outcome_j_is_good = A_all_train_dense[j, target_therapy] > thr_good
            if outcome_j_is_good == target_outcome_is_good:
                labels[i] = 1 # Similar outcome (hit)
            else:
                labels[i] = -1 # Different outcome (miss)
            found_comparison = True
            break # Simplified: label based on first comparison found
        # If no comparison found (e.g., first time therapy used), label remains 0

    return labels


def create_lmnn_labels(A_all_train, Y_train, thr_good=0.5):
    """Creates binary labels for LMNN/NCA based on outcome quality."""
    labels = np.zeros(A_all_train.shape[0], dtype=int) # Default label 0
    Y_train_dense = Y_train.toarray() if issparse(Y_train) else np.asarray(Y_train)

    # Check for NaNs
    if np.isnan(Y_train_dense).any():
        raise ValueError("NaN found in Y_train for LMNN/NCA label creation.")

    for i in range(A_all_train.shape[0]):
        applied_therapy_indices = np.nonzero(Y_train_dense[i])[0]
        if len(applied_therapy_indices) > 0:
            target_therapy = applied_therapy_indices[0]
            target_outcome_is_good = Y_train_dense[i, target_therapy] > thr_good
            labels[i] = 1 if target_outcome_is_good else 0 # Binary label: 1 for good, 0 for bad
        # else: keep label 0 if no therapy in Y

    return labels


def predict_outcome(target_consultation_idx_in_test, # Index within the test set being predicted
                    A_train,                       # Training outcomes (A_all_train_filtered)
                    similarity_row,                # Single row from similarity matrix (test_idx, train_indices)
                    K):                            # Number of neighbors
    """Predicts outcome scores for all therapies based on K nearest neighbors."""
    if K is None or K <= 0:
        # print(f"Warning: Invalid K={K} encountered in predict_outcome. Returning zeros.")
        if A_train is not None:
             return np.zeros(A_train.shape[1])
        else:
             return np.array([]) # Cannot determine number of therapies

    if A_train is None or similarity_row is None:
        raise ValueError("A_train and similarity_row cannot be None.")
    if not isinstance(A_train, csr_matrix): # Ensure A_train is CSR for efficiency
         A_train = csr_matrix(A_train)

    n_train = A_train.shape[0]
    n_therapies = A_train.shape[1]

    if n_train == 0: # Handle empty training set
        return np.zeros(n_therapies)

    # Check for NaNs in similarity row before proceeding
    if np.isnan(similarity_row).any():
        print(f"Error: NaNs found in similarity row for test index {target_consultation_idx_in_test}. Cannot predict.")
        # Option 1: Return zeros
        # return np.zeros(n_therapies)
        # Option 2: Raise error
        raise ValueError(f"NaNs found in similarity row for test index {target_consultation_idx_in_test}")

    if n_train < K:
        # print(f"Warning: Number of training samples ({n_train}) is less than K ({K}). Using K={n_train}.")
        K = n_train

    # Get similarities for the target test consultation with all training consultations
    if issparse(similarity_row):
        similarity_row = similarity_row.toarray().flatten() # Ensure it's a dense 1D array

    # Ensure similarity_row has the correct length
    if len(similarity_row) != n_train:
         # This can happen if similarity matrix calculation had issues (e.g., placeholder Gower)
         print(f"Error: Length of similarity_row ({len(similarity_row)}) does not match n_train ({n_train}). Returning zeros.")
         return np.zeros(n_therapies)

    # Find the indices of the K nearest neighbors in the training set
    # Argsort gives indices from lowest to highest similarity, so take the last K
    # Ensure K is not larger than n_train after potential adjustment
    actual_K = min(K, n_train)
    if actual_K <= 0: return np.zeros(n_therapies) # No neighbors to select

    neighbor_indices = np.argsort(similarity_row)[-actual_K:]

    # Get the outcomes and similarities of these neighbors
    neighbor_outcomes = A_train[neighbor_indices] # Shape: (K, n_therapies)
    neighbor_similarities = similarity_row[neighbor_indices] # Shape: (K,)

    # Handle potential zero similarities (avoid division by zero in weighted average)
    valid_weights_mask = neighbor_similarities > 1e-9 # Use a small epsilon
    if not np.any(valid_weights_mask):
        # If all similarities are effectively zero, return mean of neighbor outcomes (or zeros)
        if neighbor_outcomes.nnz > 0: # Check if there are any non-zero outcomes
             # Calculate simple average if weights are zero
             predicted_scores = neighbor_outcomes.mean(axis=0).A1 # .A1 converts matrix row to array
        else:
             predicted_scores = np.zeros(n_therapies) # All outcomes were zero
    else:
        # Calculate weighted average of neighbor outcomes using only valid weights
        valid_neighbor_outcomes = neighbor_outcomes[valid_weights_mask]
        valid_neighbor_similarities = neighbor_similarities[valid_weights_mask]

        # Check if sparse matrix multiplication is feasible
        try:
            # Element-wise multiplication requires broadcasting or explicit handling
            # Convert similarities to a column vector for broadcasting
            weights_col = valid_neighbor_similarities.reshape(-1, 1)
            # Use multiply method for sparse matrices
            weighted_outcomes = valid_neighbor_outcomes.multiply(weights_col)
            sum_of_weights = np.sum(valid_neighbor_similarities)
            # Add epsilon to denominator to prevent division by zero if sum_of_weights is tiny
            predicted_scores = weighted_outcomes.sum(axis=0).A1 / (sum_of_weights + 1e-9) # .A1 converts matrix row to array
        except Exception as e:
            print(f"Error during weighted average calculation: {e}. Falling back to simple average.")
            predicted_scores = neighbor_outcomes.mean(axis=0).A1 # Fallback

    # Ensure output is always a dense numpy array of the correct shape
    if not isinstance(predicted_scores, np.ndarray) or predicted_scores.shape != (n_therapies,):
        print(f"Error: Predicted scores calculation resulted in unexpected type/shape for test index {target_consultation_idx_in_test}. Returning zeros.")
        predicted_scores = np.zeros(n_therapies) # Fallback if something went wrong
    # Final check for NaNs in prediction output
    if np.isnan(predicted_scores).any():
        print(f"Error: NaN values found in final predicted_scores for test index {target_consultation_idx_in_test}. Returning zeros.")
        predicted_scores = np.zeros(n_therapies)


    return predicted_scores


def recommend_therapy(target_consultation_idx_in_test, # Index within the test set
                      A_train,                       # Training outcomes (A_all_train_filtered)
                      similarity_matrix,             # Full similarity matrix (test_indices x train_indices)
                      K,                             # Number of neighbors
                      top_n=3):                      # Number of recommendations
    """Recommends top N therapies based on predicted outcomes."""
    try:
        if target_consultation_idx_in_test >= similarity_matrix.shape[0]:
             raise IndexError(f"target_consultation_idx_in_test ({target_consultation_idx_in_test}) out of bounds for similarity matrix rows ({similarity_matrix.shape[0]})")

        # Get the similarity row for the specific test consultation
        similarity_row = similarity_matrix[target_consultation_idx_in_test, :]

        # Predict outcomes for all therapies
        predicted_outcomes = predict_outcome(
            target_consultation_idx_in_test,
            A_train,
            similarity_row,
            K
        )

        # Handle NaN predictions before sorting (should not happen if predict_outcome handles NaNs)
        if np.isnan(predicted_outcomes).any():
            print(f"Warning: NaN values found in predicted outcomes for index {target_consultation_idx_in_test} AFTER predict_outcome. Replacing with -inf for ranking.")
            predicted_outcomes = np.nan_to_num(predicted_outcomes, nan=-np.inf)


        # Rank therapies based on predicted outcomes (higher is better)
        # Argsort gives indices from lowest to highest, so reverse it [::-1]
        ranked_therapy_indices = np.argsort(predicted_outcomes)[::-1]

        # Select top N therapies
        actual_top_n = min(top_n, len(ranked_therapy_indices)) # Ensure top_n doesn't exceed available therapies
        top_n_indices = ranked_therapy_indices[:actual_top_n]
        top_n_scores = predicted_outcomes[top_n_indices]

        return top_n_indices, top_n_scores

    except Exception as e:
        print(f"Error in recommend_therapy for target_consultation index {target_consultation_idx_in_test}: {str(e)}")
        # print(traceback.format_exc()) # Uncomment for detailed traceback
        # Return default values when an error occurs
        return np.array([]), np.array([])

def inner_cross_validation_fixed(df_train, X_train, A_hist_train, A_all_train, Y_train,
                                K_values, similarity_func, groups_train, similarity_name,
                                all_data_points):
    """
    Performs inner cross-validation to find the best K, with fixes to prevent data leakage.
    """
    kf = GroupKFold(n_splits=5)
    best_K = K_values[0]
    best_score = float('inf')
    all_K_metrics = {}
    fold_best_k_values = []

    print(f"--- Starting Inner CV for {similarity_name} ---")

    for K in K_values:
        fold_metrics = []
        print(f"  Testing K={K}...")
        fold_iter = 0

        for inner_train_idx, inner_val_idx in kf.split(X_train, groups=groups_train):
            fold_iter += 1

            # Get inner training and validation data
            df_train_inner = df_train.iloc[inner_train_idx].copy()
            df_val_inner = df_train.iloc[inner_val_idx].copy()

            # Calculate affinity scores for inner validation using inner training stats
            df_val_inner['affinity_score'] = df_val_inner.apply(
                lambda row: calculate_affinity(row, df_train_inner), axis=1)

            # Get inner matrices (using the split from the outer fold)
            X_train_inner, X_val_inner = X_train[inner_train_idx], X_train[inner_val_idx]
            A_all_train_inner, A_all_val_inner = A_all_train[inner_train_idx], A_all_train[inner_val_idx]
            Y_train_inner, Y_val_inner = Y_train[inner_train_idx], Y_train[inner_val_idx]

            # For simplicity, use unfiltered data
            X_train_inner_f = X_train_inner
            A_all_train_inner_f = A_all_train_inner
            Y_train_inner_f = Y_train_inner
            X_val_inner_f = X_val_inner
            Y_val_inner_f = Y_val_inner

            if X_train_inner_f.shape[0] == 0 or X_val_inner_f.shape[0] == 0:
                continue

            # Calculate Similarity based on method
            try:
                if similarity_name == 'DR-NCA (Euclidean)':
                    # Train NCA model on inner training data only
                    nca_model = train_nca_model(X_train_inner_f, A_all_train_inner_f, Y_train_inner_f)
                    if nca_model is None:
                        similarity_matrix_inner = calculate_similarity_euclidean(X_train_inner_f, X_val_inner_f)
                    else:
                        similarity_matrix_inner = calculate_similarity_euclidean_with_nca(
                            X_train_inner_f, nca_model, X_val_inner_f)
                elif similarity_name == 'DR-RBA (Gower)':
                    similarity_matrix_inner = calculate_similarity_gower_with_rba(
                        X_train_inner_f, A_all_train_inner_f, Y_train_inner_f, X_val_inner_f)
                else:
                    X_train_arg = pd.DataFrame(X_train_inner_f) if similarity_func == calculate_similarity_gower else X_train_inner_f
                    X_val_arg = pd.DataFrame(X_val_inner_f) if similarity_func == calculate_similarity_gower else X_val_inner_f
                    similarity_matrix_inner = similarity_func(X_train_arg, X_val_arg)
            except ImportError as ie:
                 print(f"  ImportError calculating similarity in inner fold {fold_iter} for K={K}: {ie}. Skipping fold.")
                 continue
            except ValueError as ve:
                 print(f"  ValueError calculating similarity in inner fold {fold_iter} for K={K}: {ve}. Skipping fold.")
                 continue
            except Exception as e:
                 print(f"  Error calculating similarity in inner fold {fold_iter} for K={K}: {e}")
                 continue

            # Ensure similarity matrix has correct shape
            expected_inner_shape = (X_val_inner_f.shape[0], X_train_inner_f.shape[0])
            if not hasattr(similarity_matrix_inner, 'shape') or similarity_matrix_inner.shape != expected_inner_shape:
                 print(f"  Warning: Inner similarity matrix shape mismatch. Skipping fold.")
                 continue
            if np.isnan(similarity_matrix_inner).any():
                print(f"  Warning: NaN values found in inner similarity matrix. Skipping fold.")
                continue


            # --- Evaluate Predictions on Inner Validation Set ---
            y_true, y_pred = [], []
            y_true_binary, y_pred_binary = [], []
            map_scores = []
            overlap_count = 0
            coverage_count = 0
            total_suggestions = 0
            accurate_suggestions = 0

            for i in range(X_val_inner_f.shape[0]): # Iterate through validation samples
                # Get recommendations using inner training data and calculated similarity
                recommended_therapies, predicted_outcomes = recommend_therapy(
                    i, # Index within the validation set
                    A_all_train_inner_f, # Use inner training outcomes (CSR)
                    similarity_matrix_inner, # Use val x train similarity (dense)
                    K)

                if len(recommended_therapies) == 0:
                    continue # Skip if no recommendations made

                # --- Get Actual Outcome from Inner Validation Set (Y_val_inner_f) ---
                # FIX for IndexError: Check if nonzero returns results
                actual_therapy_indices = Y_val_inner_f[i].nonzero()[1] if issparse(Y_val_inner_f) else np.nonzero(Y_val_inner_f[i])[0]

                if len(actual_therapy_indices) == 0:
                    # print(f"  Warning: No actual therapy found in Y_val_inner_f for index {i} (K={K}). Skipping sample.")
                    continue # Skip this validation sample if no actual therapy recorded

                actual_therapy = actual_therapy_indices[0] # Assume first non-zero is the one
                # Ensure actual_outcome is a scalar
                actual_outcome_sparse = Y_val_inner_f[i, actual_therapy]
                actual_outcome = actual_outcome_sparse.item() if hasattr(actual_outcome_sparse, 'item') else actual_outcome_sparse


                total_suggestions += len(recommended_therapies) # Count total recommendations made

                # --- Compare Recommendation with Actual ---
                if actual_therapy in recommended_therapies:
                    try:
                         rec_index = list(recommended_therapies).index(actual_therapy)
                         pred_outcome_for_actual = predicted_outcomes[rec_index]

                         overlap_count += 1
                         y_true.append(actual_outcome)
                         y_pred.append(pred_outcome_for_actual)
                         y_true_binary.append(1 if actual_outcome > 0.5 else 0)
                         y_pred_binary.append(1 if pred_outcome_for_actual > 0.5 else 0)

                         # MAP calculation
                         rank = rec_index + 1
                         if actual_outcome > 0.5: # Only count for good actual outcomes
                             map_scores.append(1 / rank)

                         # Accuracy calculation (Top-1 match)
                         # Check if the therapy with the highest predicted score matches the actual therapy
                         if recommended_therapies[0] == actual_therapy: # Index 0 is highest score
                              accurate_suggestions += 1
                    except IndexError:
                         print(f"  IndexError during metric calculation for inner val sample {i}, K={K}.")
                         continue # Skip this sample if indexing fails
                else:
                    # Therapy not in top N recommendations, contributes to coverage calculation
                    coverage_count += len(recommended_therapies) # Example: count suggestions made
                    # Alternative: coverage_count += 1 (just count the instance)


            # --- Calculate Metrics for this Inner Fold ---
            num_val_samples = X_val_inner_f.shape[0]
            rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if y_true else np.inf
            overlap = overlap_count / num_val_samples if num_val_samples > 0 else 0
            map_at_3 = np.mean(map_scores) if map_scores else 0
            # Define Coverage based on overlap
            coverage = (num_val_samples - overlap_count) / num_val_samples if num_val_samples > 0 else 0 # Proportion Missed
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0) if (y_true_binary or y_pred_binary) else 0
            auc_pr = 0
            # Ensure there are samples and both classes exist for AUC-PR
            if y_true_binary and len(np.unique(y_true_binary)) > 1:
                 try:
                      # Ensure y_pred_binary has same length and is valid
                      # Use predicted outcome score directly for PR curve
                      valid_preds_scores = [p for t, p in zip(y_true, y_pred)] # Get predicted scores for overlapped cases
                      if len(valid_preds_scores) == len(y_true_binary): # Ensure lengths match
                           precision, recall, _ = precision_recall_curve(y_true_binary, valid_preds_scores)
                           auc_pr = auc(recall, precision)
                      else:
                           # Fallback if lengths mismatch (should ideally not happen often)
                           # print(f"  Warning: Length mismatch for AUC-PR in inner fold {fold_iter}, K={K}. Using binary predictions.")
                           valid_preds_binary = y_pred_binary if len(y_pred_binary) == len(y_true_binary) else [0]*len(y_true_binary)
                           precision, recall, _ = precision_recall_curve(y_true_binary, valid_preds_binary)
                           auc_pr = auc(recall, precision)

                 except Exception as e_auc:
                      # print(f"  Warning: Could not calculate AUC-PR for K={K}, fold {fold_iter}. Error: {e_auc}")
                      auc_pr = 0 # Assign 0 if calculation fails
            accuracy = accurate_suggestions / overlap_count if overlap_count > 0 else 0 # Accuracy among overlapped cases where top-1 could match

            fold_metrics.append({
                'RMSE': rmse, 'MAP@3': map_at_3, 'Overlap': overlap,
                'Coverage': coverage, 'F1': f1, 'AUC-PR': auc_pr, 'Accuracy': accuracy
            })

            # Record individual data point
            all_data_points.append({
                'similarity_name': similarity_name, 'K': K, 'rmse': rmse,
                'map_at_3': map_at_3, 'overlap': overlap, 'coverage': coverage,
                'f1': f1, 'auc_pr': auc_pr, 'accuracy': accuracy, 'loop': 'inner',
                'fold': fold_iter # Track inner fold number
            })
            # End of inner fold loop

        # --- Average Metrics for this K across Inner Folds ---
        if fold_metrics: # Check if any folds completed for this K
            # Calculate mean only for finite values to avoid NaN propagation
            avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics if np.isfinite(fold[metric])])
                           if any(np.isfinite(fold[metric]) for fold in fold_metrics) else np.nan
                           for metric in fold_metrics[0]}
            all_K_metrics[K] = avg_metrics

            # print(f"  Inner CV Avg for K={K}: RMSE={avg_metrics.get('RMSE', float('nan')):.3f}, MAP@3={avg_metrics.get('MAP@3', float('nan')):.3f}, Overlap={avg_metrics.get('Overlap', float('nan')):.3f}")

            # --- Update Best K based on RMSE ---
            current_k_rmse = avg_metrics.get('RMSE', float('inf'))
            if np.isfinite(current_k_rmse) and current_k_rmse < best_score:
                best_score = current_k_rmse
                best_K = K
        else:
            # print(f"  No valid inner folds completed for K={K}.")
            all_K_metrics[K] = {metric: np.nan for metric in ['RMSE', 'MAP@3', 'Overlap', 'Coverage', 'F1', 'AUC-PR', 'Accuracy']} # Store NaNs if no folds worked

    # --- End of K loop ---
    if best_score == float('inf'):
        print(f"  Warning: No suitable K value found for {similarity_name} in inner CV. Defaulting to K={K_values[0]}.")
        best_K = K_values[0] # Default if no K worked
        # Try to get RMSE for default K, handle if it's also NaN
        default_k_metrics = all_K_metrics.get(best_K, {})
        best_score = default_k_metrics.get('RMSE', float('inf'))


    print(f"--- Finished Inner CV for {similarity_name}. Best K based on RMSE: {best_K} (RMSE: {best_score:.3f}) ---")
    # Return the best K found and the dictionary of metrics per K
    return best_K, all_K_metrics

def train_nca_model(X_train, A_all_train, Y_train):
    """Helper function to train NCA model using only training data"""
    try:
        nca_labels = create_lmnn_labels(A_all_train, Y_train)
        if len(np.unique(nca_labels)) < 2:
             print("  Warning: NCA labels have only one class. Skipping NCA training.")
             return None
        else:
             nca_model = NCA(max_iter=100, random_state=42, tol=1e-3)
             try:
                  nca_model.fit(X_train, nca_labels)
                  return nca_model
             except ValueError as nca_ve:
                  print(f"  NCA fit failed: {nca_ve}. Proceeding without NCA transformation.")
                  return None
    except Exception as e_nca:
        print(f"  Error training NCA model: {e_nca}. Proceeding without NCA transformation.")
        return None

def evaluate_model_fixed(df, K_values, similarity_func, similarity_name, all_data_points, groups):
    """
    Evaluates the model using Leave-One-Group-Out cross-validation with inner CV for K selection.
    Prevents data leakage by properly separating preprocessing between train and test sets.
    """
    logo = LeaveOneGroupOut()
    outer_fold_metrics = []
    inner_fold_results_agg = []
    best_k_per_outer_fold = []

    print(f"===== Starting Evaluation for {similarity_name} =====")

    if groups is None:
         raise ValueError("Groups must be provided for LeaveOneGroupOut CV.")

    unique_groups = np.unique(groups)
    outer_fold_count = len(unique_groups)
    print(f"  Number of outer folds (groups): {outer_fold_count}")

    if outer_fold_count < 2:
        print("Warning: Less than 2 groups found. LOGO CV requires at least 2 groups. Skipping evaluation.")
        # Return empty dictionaries to handle case
        metrics_keys = ['RMSE', 'MAP@3', 'Overlap', 'F1', 'AUC-PR', 'Accuracy', 'Coverage', 'Best K Avg', 'Best K Std']
        nan_metrics = {m: np.nan for m in metrics_keys}
        for m in metrics_keys:
             if not m.endswith('Std') and not m.endswith('Avg'):
                  nan_metrics[f"{m} Std"] = np.nan

        inner_keys = ['RMSE', 'MAP@3', 'Overlap', 'F1', 'AUC-PR', 'Accuracy', 'Coverage']
        nan_inner = {m: np.nan for m in inner_keys}
        for m in inner_keys:
            nan_inner[f"{m} Std"] = np.nan

        return nan_metrics, nan_inner

    # Iterate through each fold (group)
    for fold_num, (train_index, test_index) in enumerate(tqdm(
        logo.split(df, groups=groups), total=outer_fold_count,
        desc=f"Outer CV ({similarity_name})"
    )):
        # --- Split Data into Train/Test at DataFrame Level ---
        df_train = df.iloc[train_index].copy()
        df_test = df.iloc[test_index].copy()
        groups_train = groups[train_index]

        # --- Calculate Affinity Scores SEPARATELY for train and test ---
        # Use only training data statistics for normalization bounds
        print(f"  Calculating affinity scores for training data (fold {fold_num+1})")
        df_train['affinity_score'] = df_train.apply(
            lambda row: calculate_affinity(row, df_train), axis=1)

        # Use training data statistics to calculate test affinity scores
        print(f"  Calculating affinity scores for test data (fold {fold_num+1})")
        df_test['affinity_score'] = df_test.apply(
            lambda row: calculate_affinity(row, df_train), axis=1)

        # --- Create Features and Matrices SEPARATELY for train and test ---
        # Create training matrices (fit_transform=True to learn encoders/scalers)
        print(f"  Creating matrices for training data (fold {fold_num+1})")
        X_train, A_hist_train, A_all_train, Y_train, therapy_ids, therapy_ids_map, encoder, scaler, feature_names = create_matrices(
            df_train, therapy_ids_map=None, fit_transform=True)

        # Create test matrices using encoders/scalers fitted on training data
        print(f"  Creating matrices for test data (fold {fold_num+1})")
        X_test, A_hist_test, A_all_test, Y_test, _, _ = create_matrices(
            df_test, therapy_ids_map=therapy_ids_map, fit_transform=False,
            encoder=encoder, scaler=scaler, feature_names=feature_names)

        # --- Inner Cross-Validation for K Selection ---
        best_K_fold, inner_metrics_fold_dict = inner_cross_validation_fixed(
            df_train, X_train, A_hist_train, A_all_train, Y_train,
            K_values, similarity_func, groups_train,
            similarity_name, all_data_points)

        if best_K_fold is None:
            print(f"  Warning: Inner CV failed to find a best K for outer fold {fold_num + 1}. Skipping outer fold.")
            continue

        # Store the best K and inner metrics
        best_k_per_outer_fold.append(best_K_fold)
        if best_K_fold in inner_metrics_fold_dict:
            inner_fold_results_agg.append(inner_metrics_fold_dict[best_K_fold])
        else:
            inner_fold_results_agg.append({metric: np.nan for metric in ['RMSE', 'MAP@3', 'Overlap', 'Coverage', 'F1', 'AUC-PR', 'Accuracy']})

        K = best_K_fold

        # --- Calculate Similarity for Outer Fold (Test vs Train) ---
        print(f"  Calculating outer similarity (Test vs Train) using K={K}...")
        try:
            if similarity_name == 'DR-NCA (Euclidean)':
                # Train NCA model using training data only
                nca_model = train_nca_model(X_train, A_all_train, Y_train)
                if nca_model is None:
                    similarity_matrix_outer = calculate_similarity_euclidean(X_train, X_test)
                else:
                    similarity_matrix_outer = calculate_similarity_euclidean_with_nca(
                        X_train, nca_model, X_test)
            elif similarity_name == 'DR-RBA (Gower)':
                similarity_matrix_outer = calculate_similarity_gower_with_rba(
                     X_train, A_all_train, Y_train, X_test)
            else:
                X_train_arg = pd.DataFrame(X_train) if similarity_func == calculate_similarity_gower else X_train
                X_test_arg = pd.DataFrame(X_test) if similarity_func == calculate_similarity_gower else X_test
                similarity_matrix_outer = similarity_func(X_train_arg, X_test_arg)
        except ImportError as ie:
             print(f"  ImportError calculating outer similarity matrix: {ie}. Skipping outer fold.")
             continue
        except ValueError as ve:
             print(f"  ValueError calculating outer similarity matrix: {ve}. Skipping outer fold.")
             continue
        except Exception as e:
             print(f"  Error calculating outer similarity matrix: {e}")
             continue

        # Check similarity matrix shape and NaNs
        expected_shape = (X_test.shape[0], X_train.shape[0])
        if not hasattr(similarity_matrix_outer, 'shape') or similarity_matrix_outer.shape != expected_shape:
             print(f"  Warning: Outer similarity matrix shape mismatch. Expected {expected_shape}, Got {getattr(similarity_matrix_outer, 'shape', 'N/A')}. Skipping outer fold.")
             continue
        if np.isnan(similarity_matrix_outer).any():
            print(f"  Error: NaN values found in outer similarity matrix for fold {fold_num + 1}. Skipping outer fold.")
            continue

        # --- Evaluate Predictions on Outer Test Set ---
        y_true, y_pred = [], []
        y_true_binary, y_pred_binary = [], []
        map_scores = []
        overlap_count = 0
        coverage_count = 0
        total_suggestions_outer = 0
        accurate_suggestions_outer = 0

        for i in range(X_test.shape[0]):
            recommended_therapies, predicted_outcomes = recommend_therapy(
                i, A_all_train, similarity_matrix_outer, K)

            if len(recommended_therapies) == 0:
                continue

            actual_therapy_indices = Y_test[i].nonzero()[1] if issparse(Y_test) else np.nonzero(Y_test[i])[0]

            if len(actual_therapy_indices) == 0:
                continue

            actual_therapy = actual_therapy_indices[0]
            actual_outcome_sparse = Y_test[i, actual_therapy]
            actual_outcome = actual_outcome_sparse.item() if hasattr(actual_outcome_sparse, 'item') else actual_outcome_sparse

            total_suggestions_outer += len(recommended_therapies)

            if actual_therapy in recommended_therapies:
                try:
                     rec_index = list(recommended_therapies).index(actual_therapy)
                     pred_outcome_for_actual = predicted_outcomes[rec_index]

                     overlap_count += 1
                     y_true.append(actual_outcome)
                     y_pred.append(pred_outcome_for_actual)
                     y_true_binary.append(1 if actual_outcome > 0.5 else 0)
                     y_pred_binary.append(1 if pred_outcome_for_actual > 0.5 else 0)

                     rank = rec_index + 1
                     if actual_outcome > 0.5:
                         map_scores.append(1 / rank)

                     if recommended_therapies[0] == actual_therapy:
                          accurate_suggestions_outer += 1
                except IndexError:
                     print(f"  IndexError during metric calculation for outer test sample {i}, K={K}.")
                     continue
            else:
                coverage_count += 1

        # --- Calculate Metrics for this Outer Fold ---
        num_test_samples = X_test.shape[0]
        accuracy_outer = accurate_suggestions_outer / overlap_count if overlap_count > 0 else 0

        if num_test_samples > 0:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if y_true else np.inf
            overlap = overlap_count / num_test_samples
            map_at_3 = np.mean(map_scores) if map_scores else 0
            coverage = (num_test_samples - overlap_count) / num_test_samples
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0) if (y_true_binary or y_pred_binary) else 0
            auc_pr = 0

            if y_true_binary and len(np.unique(y_true_binary)) > 1:
                 try:
                      valid_preds_scores = [p for t, p in zip(y_true, y_pred)]
                      if len(valid_preds_scores) == len(y_true_binary):
                           precision, recall, _ = precision_recall_curve(y_true_binary, valid_preds_scores)
                           auc_pr = auc(recall, precision)
                      else:
                           valid_preds_binary = y_pred_binary if len(y_pred_binary) == len(y_true_binary) else [0]*len(y_true_binary)
                           precision, recall, _ = precision_recall_curve(y_true_binary, valid_preds_binary)
                           auc_pr = auc(recall, precision)
                 except Exception as e_auc:
                      auc_pr = 0

            fold_outer_metrics = {
                'RMSE': rmse, 'MAP@3': map_at_3, 'Overlap': overlap, 'Coverage': coverage,
                'F1': f1, 'AUC-PR': auc_pr, 'Accuracy': accuracy_outer, 'Best K': K
            }
            outer_fold_metrics.append(fold_outer_metrics)

            # Record individual data point for outer fold
            all_data_points.append({
                'similarity_name': similarity_name, 'K': K, 'rmse': rmse,
                'map_at_3': map_at_3, 'overlap': overlap, 'coverage': coverage,
                'f1': f1, 'auc_pr': auc_pr, 'accuracy': accuracy_outer, 'loop': 'outer',
                'fold': fold_num + 1
            })
        else:
            print(f"  Skipping metrics calculation for outer fold {fold_num + 1} due to zero test samples.")

    # --- Aggregate Outer Fold Metrics ---
    overall_outer_metrics = {}
    if outer_fold_metrics:
        metrics_to_process = ['RMSE', 'MAP@3', 'Overlap', 'F1', 'AUC-PR', 'Accuracy', 'Coverage']

        for metric in metrics_to_process:
            valid_values = [fold[metric] for fold in outer_fold_metrics if metric in fold and np.isfinite(fold.get(metric, np.nan))]
            if valid_values:
                 overall_outer_metrics[metric] = np.mean(valid_values)
                 overall_outer_metrics[f"{metric} Std"] = np.std(valid_values)
            else:
                 overall_outer_metrics[metric] = np.nan
                 overall_outer_metrics[f"{metric} Std"] = np.nan

        # Handle Best K separately
        valid_k_values = [k for k in best_k_per_outer_fold if np.isfinite(k)]
        if valid_k_values:
            overall_outer_metrics['Best K Avg'] = np.mean(valid_k_values)
            overall_outer_metrics['Best K Std'] = np.std(valid_k_values)
        else:
            overall_outer_metrics['Best K Avg'] = np.nan
            overall_outer_metrics['Best K Std'] = np.nan

        print(f"\n===== Aggregated Outer Results for {similarity_name} =====")
        for metric in metrics_to_process:
             if metric in overall_outer_metrics and np.isfinite(overall_outer_metrics[metric]):
                  mean_val = overall_outer_metrics[metric]
                  std_val_key = f"{metric} Std"
                  std_val = overall_outer_metrics.get(std_val_key, np.nan)
                  if np.isfinite(std_val):
                       print(f"  {metric}: {mean_val:.4f} (± {std_val:.4f})")
                  else:
                       print(f"  {metric}: {mean_val:.4f}")
             else:
                  print(f"  {metric}: NaN")

        if 'Best K Avg' in overall_outer_metrics and np.isfinite(overall_outer_metrics['Best K Avg']):
             mean_k = overall_outer_metrics['Best K Avg']
             std_k_key = 'Best K Std'
             std_k = overall_outer_metrics.get(std_k_key, np.nan)
             if np.isfinite(std_k):
                  print(f"  Best K: {mean_k:.2f} (± {std_k:.2f})")
             else:
                  print(f"  Best K: {mean_k:.2f}")
        else:
             print(f"  Best K: NaN")

        # Append overall outer average to all_data_points
        overall_outer_avg_record = overall_outer_metrics.copy()
        overall_outer_avg_record['similarity_name'] = similarity_name
        overall_outer_avg_record['metrics_type'] = 'Overall Outer Average'
        all_data_points.append(overall_outer_avg_record)
    else:
        print(f"\n===== No valid outer folds completed for {similarity_name} =====")
        metrics_keys = ['RMSE', 'MAP@3', 'Overlap', 'F1', 'AUC-PR', 'Accuracy', 'Coverage', 'Best K Avg', 'Best K Std']
        overall_outer_metrics = {m: np.nan for m in metrics_keys}
        for m in metrics_keys:
             if not m.endswith('Std') and not m.endswith('Avg'):
                  overall_outer_metrics[f"{m} Std"] = np.nan

    # --- Aggregate Inner Fold Metrics ---
    overall_inner_metrics = {}
    if inner_fold_results_agg:
        first_valid_inner_result = next((item for item in inner_fold_results_agg if isinstance(item, dict) and item), None)

        if first_valid_inner_result:
             metrics_to_process_inner = list(first_valid_inner_result.keys())

             for metric in metrics_to_process_inner:
                  if not metric.endswith(' Std'):
                       valid_values = [fold[metric] for fold in inner_fold_results_agg if isinstance(fold, dict) and metric in fold and np.isfinite(fold.get(metric, np.nan))]
                       if valid_values:
                            overall_inner_metrics[metric] = np.mean(valid_values)
                            overall_inner_metrics[f"{metric} Std"] = np.std(valid_values)
                       else:
                            overall_inner_metrics[metric] = np.nan
                            overall_inner_metrics[f"{metric} Std"] = np.nan

             print(f"\n===== Aggregated Inner Results (at Best K per fold) for {similarity_name} =====")
             for metric in metrics_to_process_inner:
                  if not metric.endswith(' Std'):
                       if metric in overall_inner_metrics and np.isfinite(overall_inner_metrics[metric]):
                            mean_val = overall_inner_metrics[metric]
                            std_val_key = f"{metric} Std"
                            std_val = overall_inner_metrics.get(std_val_key, np.nan)
                            if np.isfinite(std_val):
                                 print(f"  {metric}: {mean_val:.4f} (± {std_val:.4f})")
                            else:
                                 print(f"  {metric}: {mean_val:.4f}")
                       else:
                            print(f"  {metric}: NaN")

             # Append overall inner average to all_data_points
             overall_inner_avg_record = overall_inner_metrics.copy()
             overall_inner_avg_record['similarity_name'] = similarity_name
             overall_inner_avg_record['metrics_type'] = 'Overall Inner Average (Best K)'
             all_data_points.append(overall_inner_avg_record)
        else:
             print(f"\n===== No valid inner fold dictionary results collected for {similarity_name} =====")
             metrics_to_process_inner = ['RMSE', 'MAP@3', 'Overlap', 'F1', 'AUC-PR', 'Accuracy', 'Coverage']
             overall_inner_metrics = {m: np.nan for m in metrics_to_process_inner}
             for m in metrics_to_process_inner:
                  overall_inner_metrics[f"{m} Std"] = np.nan
    else:
        print(f"\n===== No valid inner fold results collected for {similarity_name} =====")
        metrics_to_process_inner = ['RMSE', 'MAP@3', 'Overlap', 'F1', 'AUC-PR', 'Accuracy', 'Coverage']
        overall_inner_metrics = {m: np.nan for m in metrics_to_process_inner}
        for m in metrics_to_process_inner:
             overall_inner_metrics[f"{m} Std"] = np.nan

    return overall_outer_metrics, overall_inner_metrics

def aggregate_visit_metrics(visit_results, visit_num, fold, best_K):
    """
    Aggregates metrics for a specific visit number in a fold.
    Now includes the best K value in the output metrics.

    Parameters:
    - visit_results: List of dictionaries containing metrics for each patient visit
    - visit_num: Visit number (1, 2, 3, or 'all')
    - fold: Current fold number
    - best_K: The best K value used for this fold

    Returns:
    - Dictionary of aggregated metrics
    """
    if not visit_results:
        return {}

    # Initialize metrics
    n_samples = len(visit_results)
    n_matches = sum(1 for v in visit_results if v.get('match', False))
    n_top1_matches = sum(1 for v in visit_results if v.get('top1_match', False))

    # Calculate metrics
    overlap = n_matches / n_samples if n_samples > 0 else 0
    coverage = 1 - overlap
    accuracy = n_top1_matches / n_matches if n_matches > 0 else 0
    map_scores = [v.get('map_score', 0) for v in visit_results]
    map_at_3 = sum(map_scores) / n_samples if n_samples > 0 else 0
    valid_rmse_values = [v.get('rmse', np.nan) for v in visit_results if v.get('match', False)]
    valid_rmse_values = [r for r in valid_rmse_values if np.isfinite(r)]
    rmse = np.mean(valid_rmse_values) if valid_rmse_values else np.nan

    # Return metrics dictionary with best_K included
    return {
        'n_samples': n_samples,
        'overlap': overlap,
        'coverage': coverage,
        'accuracy': accuracy,
        'map_at_3': map_at_3,
        'rmse': rmse,
        'visit_num': visit_num,
        'fold': fold,
        'best_K': best_K  # Add best K to the output
    }

# --- End of evaluate_model function ---
def evaluate_all_visits(df, K_values, similarity_func, similarity_name, all_data_points):
    """
    Comprehensive evaluation that makes recommendations for all visits,
    using only information that would be available at the time of each visit.
    Enhanced to properly track and report K values.
    """
    print(f"===== Starting Comprehensive Visit-Based Evaluation for {similarity_name} =====")

    # Get unique subject IDs for patient-based splitting
    unique_subjects = df['subject_id'].unique()
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Store metrics separately for each visit number
    visit1_metrics = []
    visit2_metrics = []
    visit3_metrics = []
    all_visits_metrics = []  # Combined metrics across all visits

    # Store best K values for each fold
    best_K_values = []

    # Initialize overall patient counter and total
    total_patients_to_process = 0
    for _, test_subject_idx in kf.split(unique_subjects):
        total_patients_to_process += len(unique_subjects[test_subject_idx])

    print(f"  Total patients to be evaluated: {total_patients_to_process}")
    patients_processed = 0

    for fold, (train_subject_idx, test_subject_idx) in enumerate(tqdm(
        kf.split(unique_subjects), total=n_folds,
        desc=f"Visit-Based CV ({similarity_name})"
    )):
        # Get subject IDs for this fold
        train_subjects = unique_subjects[train_subject_idx]
        test_subjects = unique_subjects[test_subject_idx]
        test_subject_count = len(test_subjects)

        # Get all training data for these subjects
        train_df = df[df['subject_id'].isin(train_subjects)].copy()

        # Calculate affinity scores for training data
        print(f"  Calculating affinity scores for training data (fold {fold+1})")
        train_df['affinity_score'] = train_df.apply(lambda row: calculate_affinity(row, train_df), axis=1)

        # Find best K using proper inner cross-validation
        best_K, inner_K_metrics = find_best_K_all_visits(train_df, K_values, similarity_func, similarity_name)
        best_K_values.append(best_K)

        # Store inner CV K results in all_data_points
        for k, metrics in inner_K_metrics.items():
            all_data_points.append({
                'similarity_name': similarity_name,
                'K': k,
                'rmse': metrics.get('RMSE', np.nan),
                'map_at_3': metrics.get('MAP@3', np.nan),
                'overlap': metrics.get('Overlap', np.nan),
                'coverage': metrics.get('Coverage', np.nan),
                'f1': metrics.get('F1', np.nan),
                'auc_pr': metrics.get('AUC-PR', np.nan),
                'accuracy': metrics.get('Accuracy', np.nan),
                'loop': 'inner_visit_based',
                'fold': fold + 1
            })

        # Metrics for this fold, separated by visit number
        fold_visit1_results = []
        fold_visit2_results = []
        fold_visit3_results = []
        fold_all_visits_results = []

        # Initialize patient progress bar for this fold
        fold_patients_processed = 0
        print(f"  Processing {test_subject_count} patients in fold {fold+1}...")

        # Process each test subject individually
        for test_subject in tqdm(test_subjects, desc=f"  Patients in fold {fold+1}",
                                leave=False, miniters=max(1, len(test_subjects)//20)):
            # Get all visits for this test subject
            subject_visits = df[df['subject_id'] == test_subject].sort_values('visit_number').copy()

            if subject_visits.empty:
                fold_patients_processed += 1
                patients_processed += 1
                continue

            # Track metrics for this subject
            subject_metrics = []

            # Process each visit for this subject
            for visit_idx, visit_row in subject_visits.iterrows():
                visit_num = int(visit_row['visit_number'])

                if visit_num < 1 or visit_num > 3:
                    print(f"  Warning: Unexpected visit number {visit_num} for subject {test_subject}. Skipping.")
                    continue

                # Process based on visit number
                if visit_num == 1:
                    # For first visits, use all training data
                    reference_df = train_df.copy()
                else:
                    # For later visits, use training data plus this subject's previous visits
                    previous_visits = subject_visits[subject_visits['visit_number'] < visit_num].copy()

                    # Calculate affinity for previous visits using training data as reference
                    previous_visits['affinity_score'] = previous_visits.apply(
                        lambda row: calculate_affinity(row, train_df), axis=1)

                    # Combine training data with this subject's previous visits
                    reference_df = pd.concat([train_df, previous_visits])

                # Create a single-row DataFrame for the current visit
                current_visit_df = pd.DataFrame([visit_row])

                # Calculate affinity for current visit
                current_visit_df['affinity_score'] = current_visit_df.apply(
                    lambda row: calculate_affinity(row, reference_df), axis=1)

                # Create matrices for reference data and current visit
                X_ref, A_hist_ref, A_all_ref, Y_ref, therapy_ids, therapy_ids_map, encoder, scaler, feature_names = create_matrices(
                    reference_df, therapy_ids_map=None, fit_transform=True)

                X_current, _, _, Y_current, _, _ = create_matrices(
                    current_visit_df, therapy_ids_map=therapy_ids_map, fit_transform=False,
                    encoder=encoder, scaler=scaler, feature_names=feature_names)

                # Calculate similarity
                try:
                    if similarity_name == 'DR-NCA (Euclidean)':
                        # Simplified for example - implement actual NCA if available
                        similarity_matrix = calculate_similarity_euclidean(X_ref, X_current)
                    elif similarity_name == 'DR-RBA (Gower)':
                        # Simplified for example - implement actual RBA if available
                        similarity_matrix = calculate_similarity_euclidean(X_ref, X_current)
                    else:
                        X_ref_arg = pd.DataFrame(X_ref) if similarity_func == calculate_similarity_gower else X_ref
                        X_current_arg = pd.DataFrame(X_current) if similarity_func == calculate_similarity_gower else X_current
                        similarity_matrix = similarity_func(X_ref_arg, X_current_arg)
                except Exception as e:
                    print(f"  Error calculating similarity for subject {test_subject}, visit {visit_num}: {e}")
                    continue

                # Get recommendations
                recommended_therapies, predicted_outcomes = recommend_therapy(
                    0, A_all_ref, similarity_matrix, best_K)

                if len(recommended_therapies) == 0:
                    print(f"  No recommendations generated for subject {test_subject}, visit {visit_num}")
                    continue

                # Get actual therapy and outcome
                actual_therapy_indices = Y_current[0].nonzero()[1] if issparse(Y_current) else np.nonzero(Y_current[0])[0]

                if len(actual_therapy_indices) == 0:
                    print(f"  No actual therapy found for subject {test_subject}, visit {visit_num}")
                    continue

                actual_therapy = actual_therapy_indices[0]
                actual_outcome = Y_current[0, actual_therapy].item() if hasattr(Y_current[0, actual_therapy], 'item') else Y_current[0, actual_therapy]

                # Calculate metrics for this visit
                match = actual_therapy in recommended_therapies
                if match:
                    rec_index = list(recommended_therapies).index(actual_therapy)
                    pred_outcome = predicted_outcomes[rec_index]

                    rank = rec_index + 1
                    top1_match = (rank == 1)
                    map_score = 1 / rank if actual_outcome > 0.5 else 0

                    # Calculate predicted vs actual outcome metrics
                    rmse = np.sqrt((pred_outcome - actual_outcome) ** 2)

                    visit_metrics = {
                        'subject_id': test_subject,
                        'visit_number': visit_num,
                        'match': match,
                        'rank': rank,
                        'actual_therapy': actual_therapy,
                        'actual_outcome': actual_outcome,
                        'pred_outcome': pred_outcome,
                        'top1_match': top1_match,
                        'map_score': map_score,
                        'rmse': rmse
                    }
                else:
                    visit_metrics = {
                        'subject_id': test_subject,
                        'visit_number': visit_num,
                        'match': match,
                        'rank': float('inf'),
                        'actual_therapy': actual_therapy,
                        'actual_outcome': actual_outcome,
                        'pred_outcome': np.nan,
                        'top1_match': False,
                        'map_score': 0,
                        'rmse': np.nan
                    }

                # Store metrics based on visit number
                if visit_num == 1:
                    fold_visit1_results.append(visit_metrics)
                elif visit_num == 2:
                    fold_visit2_results.append(visit_metrics)
                elif visit_num == 3:
                    fold_visit3_results.append(visit_metrics)

                # Also store in combined results
                fold_all_visits_results.append(visit_metrics)

            # Update patient counters
            fold_patients_processed += 1
            patients_processed += 1

            # Update overall progress periodically
            if patients_processed % 10 == 0 or patients_processed == total_patients_to_process:
                percentage = (patients_processed / total_patients_to_process) * 100
                progress_bar = f"[{'=' * int(percentage // 2)}{' ' * (50 - int(percentage // 2))}]"
                print(f"\r  Overall Patient Progress: {progress_bar} {patients_processed}/{total_patients_to_process} ({percentage:.1f}%)", end="")

        print()  # New line after patient progress for this fold

        # Calculate aggregate metrics for each visit number in this fold
        if fold_visit1_results:
            visit1_fold_metrics = aggregate_visit_metrics(fold_visit1_results, visit_num=1, fold=fold, best_K=best_K)
            visit1_metrics.append(visit1_fold_metrics)

            # Add to all_data_points
            all_data_points.append({
                'similarity_name': similarity_name,
                'K': best_K,
                **{k: v for k, v in visit1_fold_metrics.items()},
                'loop': 'visit1',
                'fold': fold + 1
            })

        if fold_visit2_results:
            visit2_fold_metrics = aggregate_visit_metrics(fold_visit2_results, visit_num=2, fold=fold, best_K=best_K)
            visit2_metrics.append(visit2_fold_metrics)

            # Add to all_data_points
            all_data_points.append({
                'similarity_name': similarity_name,
                'K': best_K,
                **{k: v for k, v in visit2_fold_metrics.items()},
                'loop': 'visit2',
                'fold': fold + 1
            })

        if fold_visit3_results:
            visit3_fold_metrics = aggregate_visit_metrics(fold_visit3_results, visit_num=3, fold=fold, best_K=best_K)
            visit3_metrics.append(visit3_fold_metrics)

            # Add to all_data_points
            all_data_points.append({
                'similarity_name': similarity_name,
                'K': best_K,
                **{k: v for k, v in visit3_fold_metrics.items()},
                'loop': 'visit3',
                'fold': fold + 1
            })

        if fold_all_visits_results:
            all_visits_fold_metrics = aggregate_visit_metrics(fold_all_visits_results, visit_num='all', fold=fold, best_K=best_K)
            all_visits_metrics.append(all_visits_fold_metrics)

            # Add to all_data_points
            all_data_points.append({
                'similarity_name': similarity_name,
                'K': best_K,
                **{k: v for k, v in all_visits_fold_metrics.items()},
                'loop': 'all_visits',
                'fold': fold + 1
            })

        # Print fold results
        print(f"  Fold {fold+1} results:")
        if fold_visit1_results:
            print(f"    Visit 1: {len(fold_visit1_results)} samples, " +
                  f"Overlap={visit1_fold_metrics['overlap']:.3f}, RMSE={visit1_fold_metrics['rmse']:.3f}")
        if fold_visit2_results:
            print(f"    Visit 2: {len(fold_visit2_results)} samples, " +
                  f"Overlap={visit2_fold_metrics['overlap']:.3f}, RMSE={visit2_fold_metrics['rmse']:.3f}")
        if fold_visit3_results:
            print(f"    Visit 3: {len(fold_visit3_results)} samples, " +
                  f"Overlap={visit3_fold_metrics['overlap']:.3f}, RMSE={visit3_fold_metrics['rmse']:.3f}")
        if fold_all_visits_results:
            print(f"    All Visits: {len(fold_all_visits_results)} samples, " +
                  f"Overlap={all_visits_fold_metrics['overlap']:.3f}, RMSE={all_visits_fold_metrics['rmse']:.3f}")

    # Calculate overall metrics for each visit number
    visit1_overall = calculate_overall_visit_metrics(visit1_metrics, 'Visit 1')
    visit2_overall = calculate_overall_visit_metrics(visit2_metrics, 'Visit 2')
    visit3_overall = calculate_overall_visit_metrics(visit3_metrics, 'Visit 3')
    all_visits_overall = calculate_overall_visit_metrics(all_visits_metrics, 'All Visits')

    # After collecting all best_K_values_for_folds
    # After collecting all best_K_values
    print(f"\n*** DEBUG: Raw K values across folds for {similarity_name}: {best_K_values}")
    print(f"    Mean: {np.mean(best_K_values)}")
    print(f"    Std:  {np.std(best_K_values)}")

    # Add overall metrics to all_data_points
    for visit_name, metrics in [
        ('visit1_overall', visit1_overall),
        ('visit2_overall', visit2_overall),
        ('visit3_overall', visit3_overall),
        ('all_visits_overall', all_visits_overall)
    ]:
        if metrics:
            all_data_points.append({
                'similarity_name': similarity_name,
                'metrics_type': visit_name,
                **metrics
            })

    # Print overall results
    print(f"\n===== Overall Results for {similarity_name} =====")

    print("\nVisit 1 Results:")
    print_visit_metrics(visit1_overall)

    print("\nVisit 2 Results:")
    print_visit_metrics(visit2_overall)

    print("\nVisit 3 Results:")
    print_visit_metrics(visit3_overall)

    print("\nAll Visits Results:")
    print_visit_metrics(all_visits_overall)

    return {
        'visit1': visit1_overall,
        'visit2': visit2_overall,
        'visit3': visit3_overall,
        'all_visits': all_visits_overall
    }


def calculate_overall_visit_metrics(fold_metrics, visit_name):
    """Aggregate metrics across all folds for a specific visit number, including K."""
    if not fold_metrics:
        return {}

    overall_metrics = {}

    # Metrics to process (now including best_K)
    metric_keys = ['rmse', 'map_at_3', 'overlap', 'coverage', 'accuracy', 'n_samples', 'best_K']

    # Calculate mean and std for each metric
    for metric in metric_keys:
        values = [fold[metric] for fold in fold_metrics if metric in fold and np.isfinite(fold.get(metric, np.nan))]
        if values:
            overall_metrics[metric] = np.mean(values)
            overall_metrics[f"{metric}_std"] = np.std(values)
        else:
            overall_metrics[metric] = np.nan
            overall_metrics[f"{metric}_std"] = np.nan

    # Add visit name
    overall_metrics['visit_name'] = visit_name

    return overall_metrics

def print_visit_metrics(metrics):
    """Print metrics in a formatted way, including K information."""
    if not metrics:
        print("  No metrics available")
        return

    # Original metrics
    for metric in ['rmse', 'map_at_3', 'overlap', 'coverage', 'accuracy']:
        if metric in metrics and np.isfinite(metrics[metric]):
            mean_val = metrics[metric]
            std_key = f"{metric}_std"
            std_val = metrics.get(std_key, np.nan)

            if np.isfinite(std_val):
                print(f"  {metric.upper()}: {mean_val:.4f} (± {std_val:.4f})")
            else:
                print(f"  {metric.upper()}: {mean_val:.4f}")
        else:
            print(f"  {metric.upper()}: NaN")

    # Add K information
    if 'best_K' in metrics and np.isfinite(metrics['best_K']):
        mean_k = metrics['best_K']
        std_key = 'best_K_std'
        std_val = metrics.get(std_key, np.nan)
        if np.isfinite(std_val):
            print(f"  BEST K: {mean_k:.1f} (± {std_val:.1f})")
        else:
            print(f"  BEST K: {mean_k:.1f}")
    elif 'best_K_mean' in metrics and np.isfinite(metrics['best_K_mean']):
        mean_k = metrics['best_K_mean']
        std_val = metrics.get('best_K_std', np.nan)
        if np.isfinite(std_val):
            print(f"  BEST K: {mean_k:.1f} (± {std_val:.1f})")
        else:
            print(f"  BEST K: {mean_k:.1f}")
    else:
        print(f"  BEST K: NaN")

    if 'n_samples' in metrics:
        print(f"  SAMPLES: {metrics['n_samples']:.1f} avg per fold")

def find_best_K_all_visits(train_df, K_values, similarity_func, similarity_name):
    """
    Find the best K value using inner cross-validation on the training set.
    This is a more robust implementation than the simplified approach.
    """
    print(f"  Finding best K for {similarity_name} using inner CV...")

    # Create groups for inner CV based on subject_id
    groups_train = train_df['subject_id'].values

    # Create features and matrices for training data
    train_df['affinity_score'] = train_df.apply(lambda row: calculate_affinity(row, train_df), axis=1)
    X_train, A_hist_train, A_all_train, Y_train, _, _, _, _, _ = create_matrices(train_df, fit_transform=True)

    # Use the inner_cross_validation_fixed function which already exists
    best_K, inner_metrics = inner_cross_validation_fixed(
        train_df, X_train, A_hist_train, A_all_train, Y_train,
        K_values, similarity_func, groups_train, similarity_name, [])

    print(f"  Selected K={best_K} for {similarity_name} using inner CV (RMSE: {inner_metrics.get(best_K, {}).get('RMSE', 'N/A')})")
    return best_K, inner_metrics


# --- Main Execution Code ---
## --- Modified Main Execution Code ---

# Make sure 'visit_number' is appropriate type (do this early)
if 'visit_number' in df.columns:
    df['visit_number'] = pd.to_numeric(df['visit_number'], errors='coerce')
else:
     print("Warning: 'visit_number' column not found.")
     df['visit_number'] = 1 # Assign a default if missing? Or handle appropriately

# --- Feature Engineering for comorbidity_count ---
print("Starting feature engineering for comorbidity_count...")
df_processed = df.copy()

if 'comorbidities' in df_processed.columns:
    df_processed['comorbidity_count'] = df_processed['comorbidities'].apply(
        lambda x: len([item for item in str(x).split(',') if item.strip()]) if pd.notna(x) else 0
    )
    print("Feature 'comorbidity_count' created.")
    if df_processed['comorbidity_count'].isnull().any():
        print("Warning: NaNs found in 'comorbidity_count' after calculation.")
        df_processed['comorbidity_count'].fillna(0, inplace=True)
else:
    print("Warning: 'comorbidities' column not found. Using dummy comorbidity_count.")
    df_processed['comorbidity_count'] = 0 # Ensure column exists even if dummy


# Define the range of K values to explore
K_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# Define a dictionary of similarity functions to evaluate
similarity_functions = {
    'CF (Cosine)': calculate_similarity_cosine,
    'CF (Euclidean)': calculate_similarity_euclidean,
    # 'DR (Gower)': calculate_similarity_gower,             # Requires 'gower' package
    'DR-RBA (Gower)': calculate_similarity_gower_with_rba,# Requires 'gower' and 'skrebate'
    'DR (Euclidean RBF)': calculate_similarity_euclidean_rbf,
    'DR-NCA (Euclidean)': calculate_similarity_euclidean_with_nca, # Requires NCA training
}

# Filter out functions that cannot be used due to missing imports
if 'gower_matrix' in globals() and gower_matrix is None:
    print("Excluding Gower-based similarities due to missing package.")
    similarity_functions = {k: v for k, v in similarity_functions.items() if 'Gower' not in k}
if 'ReliefF' in globals() and ReliefF is None:
    print("Excluding RBA-based similarities due to missing package.")
    similarity_functions = {k: v for k, v in similarity_functions.items() if 'RBA' not in k}

# --- Results Storage for Visit-Based Evaluation ---
# Will store results from evaluate_all_visits, keyed by similarity name
visit_based_results = {}

# For tracking all detailed results (used by evaluate_all_visits internally)
all_data_points = []


# --- Run Visit-Based Evaluation (`evaluate_all_visits`) ---
print("\n--- Starting Comprehensive Visit-Based Evaluation ---")

for similarity_name, similarity_func in similarity_functions.items():
    print(f"\nEvaluating similarity function: {similarity_name} with visit-based approach")

    # Run the visit-based evaluation
    # evaluate_all_visits appends individual fold and visit results to all_data_points internally
    results = evaluate_all_visits(
        df_processed, K_values, similarity_func, similarity_name, all_data_points)

    # Store the overall aggregated visit-based results
    visit_based_results[similarity_name] = results


# --- Display Visit-Based Results Summaries ---
print("\n--- Final Visit-Based Evaluation Results Summary (Averaged Across Folds) ---")

# Define the order of visit types for printing
visit_types_order = ['visit1', 'visit2', 'visit3', 'all_visits']
# Define the order of metrics for printing
metrics_order = ['rmse', 'overlap', 'map_at_3', 'accuracy', 'best_K', 'coverage'] # Include best_K

# Create and print a summary table for each visit type
for visit_type in visit_types_order:
    print(f"\n===== {visit_type.replace('_', ' ').title()} Results =====")

    summary_data = {}
    # Collect data for this visit type across all models
    for model_name, results_dict in visit_based_results.items():
        if visit_type in results_dict and results_dict[visit_type]:
            summary_data[model_name] = results_dict[visit_type]

    if summary_data:
        summary_df = pd.DataFrame(summary_data).transpose()

        # Define columns to display and their order (Mean and Std Dev)
        display_cols = []
        for metric in metrics_order:
            mean_col = f"{metric}" # Use the metric name directly as key in visit_based_results
            std_col = f"{metric}_std"
            if mean_col in summary_df.columns:
                 display_cols.append(mean_col)
            if std_col in summary_df.columns:
                 display_cols.append(std_col)

        # Define formatters for better display
        formatters = {
            col: (lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A") if '_std' not in col else (lambda x: f"± {x:.4f}" if pd.notnull(x) else "")
            for col in display_cols
        }
         # Special formatting for best_K (usually integer or few decimals)
        if 'best_K' in formatters:
            formatters['best_K'] = lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A"
        if 'best_K_std' in formatters:
            formatters['best_K_std'] = lambda x: f"± {x:.1f}" if pd.notnull(x) else ""


        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1200): # Increased width
             # Custom print loop to combine Mean ± Std for desired metrics
            header_line1 = f"{'Model':<20s} | "
            header_line2 = f"{'-'*20} |-"
            data_format_strings = []

            # Build headers and data format strings based on display_cols and metrics_order
            for metric in metrics_order:
                mean_col = f"{metric}"
                std_col = f"{metric}_std"
                col_label = metric.replace('_', ' ').title()

                if mean_col in summary_df.columns and std_col in summary_df.columns:
                     header_line1 += f"{col_label:<20s} | "
                     header_line2 += f"{'-'*20}-|"
                     data_format_strings.append((mean_col, std_col, metric, formatters[mean_col], formatters[std_col]))
                elif mean_col in summary_df.columns:
                     header_line1 += f"{col_label:<10s} | " # Use smaller width if no std
                     header_line2 += f"{'-'*10}-|"
                     data_format_strings.append((mean_col, None, metric, formatters[mean_col], None))


            # Remove trailing separators
            header_line1 = header_line1.rstrip(' | ')
            header_line2 = header_line2.rstrip('-|')

            print(header_line1)
            print(header_line2)

            for index, row in summary_df[display_cols].iterrows(): # Select columns in display_order
                row_string = f"{index:<20s} | "
                for mean_col, std_col, metric, mean_formatter, std_formatter in data_format_strings:
                    mean_val = row.get(mean_col, np.nan)
                    std_val = row.get(std_col, np.nan)

                    if std_col is not None:
                         value_str = f"{mean_formatter(mean_val)} {std_formatter(std_val)}".strip()
                         row_string += f"{value_str:<20s} | "
                    else:
                         value_str = f"{mean_formatter(mean_val)}".strip()
                         row_string += f"{value_str:<10s} | " # Match smaller width


                print(row_string.rstrip(' | ')) # Remove trailing separator

            print("-" * len(header_line1)) # Print separator matching header width


    else:
        print("  No data available for this visit type.")


# --- Optional: Display Best Overall Model (based on All Visits RMSE and Overlap > 0.75) ---
# Find the best model based on the 'all_visits' aggregated results
best_overall_rmse = float('inf')
best_overall_model = None
best_overall_k_avg = np.nan

# REMOVE the incorrect 'if 'all_visits' in visit_based_results.values():' line
# Start the loop directly
for model_name, results in visit_based_results.items():
    # The inner check is correct to ensure 'all_visits' results exist and are not empty
    if 'all_visits' in results and results['all_visits']:
        current_rmse = results['all_visits'].get('rmse', float('inf'))
        current_overlap = results['all_visits'].get('overlap', np.nan)
        current_k_avg = results['all_visits'].get('best_K', np.nan) # Use best_K from all_visits overall

        # Check if the current model meets the overlap criteria and has a valid RMSE
        if pd.notna(current_overlap) and current_overlap > 0.75 and pd.notna(current_rmse):
             # If it meets the overlap criteria, check if its RMSE is better than the current best
             if current_rmse < best_overall_rmse:
                 # This code will now execute for models meeting the criteria
                 best_overall_rmse = current_rmse
                 best_overall_model = model_name
                 best_overall_k_avg = current_k_avg

# After the loop, check if a best model was found (which should happen now)
if best_overall_model:
     print(f"\n*** Best Overall Model (based on All Visits RMSE < {best_overall_rmse:.4f} and Overlap > 0.75): {best_overall_model} (Avg K: {best_overall_k_avg:.2f}) ***")
else:
     # This message should now only print if genuinely no model met the criteria
     print("\n*** No model met the criteria (All Visits RMSE and Overlap > 0.75) to be declared best overall. ***")

# --- Save all_data_points to JSON ---
# Ensure the directory exists before trying to save
output_file_path = "/content/drive/MyDrive/model_evaluation_data_points_visit_based.json" # Changed filename
output_dir = os.path.dirname(output_file_path)
if output_dir and not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    except Exception as e:
        print(f"Error creating directory {output_dir}: {e}")
        # Decide if you want to stop here or try saving anyway
        pass

try:
    # Convert NumPy types (like NaN, inf) to standard Python types for JSON serialization
    def convert_to_json_serializable(obj):
          """Convert NumPy types and other non-JSON serializable types to standard Python types."""
          if isinstance(obj, np.ndarray):
              return [convert_to_json_serializable(x) for x in obj]
          if isinstance(obj, (np.float64, float)):
              if np.isnan(obj): return None  # JSON null for NaN
              if np.isinf(obj): return str(obj)  # Represent inf as string
              return float(obj)
          if isinstance(obj, (np.int64, int)):
              return int(obj)
          if isinstance(obj, dict):
              return {k: convert_to_json_serializable(v) for k, v in obj.items()}
          if isinstance(obj, list):
              return [convert_to_json_serializable(item) for item in obj]
          # Handle other NumPy scalar types
          if np.isscalar(obj) and not isinstance(obj, (str, bool, type(None))):
              try:
                  return obj.item()  # Convert NumPy scalar to Python scalar
              except:
                  return str(obj)  # Fallback to string representation
          return obj

    serializable_data_points = convert_to_json_serializable(all_data_points)

    # Open the file in write mode
    with open(output_file_path, 'w') as f:
        # Use json.dump to write the list of dictionaries to the file
        # indent=4 makes the JSON file human-readable
        json.dump(serializable_data_points, f, indent=4)

    print(f"Successfully saved all data points to {output_file_path}")

except Exception as e:
    print(f"Error saving data points to JSON at {output_file_path}: {e}")
    print("Please ensure Google Drive is mounted and the path is correct.")
    print(f"Traceback:\n{traceback.format_exc()}") # Print detailed traceback
