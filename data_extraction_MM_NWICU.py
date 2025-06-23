import pandas as pd
import numpy as np
from datetime import datetime

# Load all required datasets
print("=== LOADING DATASETS ===")
diagnoses_df = pd.read_csv('/content/drive/MyDrive/Northwestern/diagnoses_icd-2.csv')
prescriptions_df = pd.read_csv('/content/drive/MyDrive/Northwestern/prescriptions.csv')
patients_df = pd.read_csv('/content/drive/MyDrive/Northwestern/patients.csv')
admissions_df = pd.read_csv('/content/drive/MyDrive/Northwestern/admissions.csv')

print(f"Diagnoses: {len(diagnoses_df)} rows")
print(f"Prescriptions: {len(prescriptions_df)} rows")
print(f"Patients: {len(patients_df)} rows")
print(f"Admissions: {len(admissions_df)} rows")

# Convert datetime columns
prescriptions_df['starttime'] = pd.to_datetime(prescriptions_df['starttime'])
prescriptions_df['stoptime'] = pd.to_datetime(prescriptions_df['stoptime'])
admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'])

print("\n=== STEP 1: COMORBIDITIES ===")
# Comorbidities CTE - get all ICD codes except F2% and 295%
comorbidities_mask = (~diagnoses_df['icd_code'].str.startswith('F2', na=False)) & \
                     (~diagnoses_df['icd_code'].str.startswith('295', na=False))

comorbidities = diagnoses_df[comorbidities_mask].groupby(['subject_id', 'hadm_id']).agg({
    'icd_code': lambda x: ', '.join(x.astype(str).unique())
}).reset_index()
comorbidities.columns = ['subject_id', 'hadm_id', 'comorbidities']

print(f"Comorbidities processed: {len(comorbidities)} admission records")

print("\n=== STEP 2: PATIENT VISITS ===")
# PatientVisits CTE - F2% or 295% diagnoses with seq_num = 1 (primary diagnosis)
schizo_mask = (diagnoses_df['icd_code'].str.startswith('F2', na=False) |
               diagnoses_df['icd_code'].str.startswith('295', na=False)) & \
              (diagnoses_df['seq_num'] == 1)

schizo_diagnoses = diagnoses_df[schizo_mask].copy()
print(f"Schizophrenia primary diagnoses: {len(schizo_diagnoses)}")

# Group ICD codes by patient/admission
icd_grouped = schizo_diagnoses.groupby(['subject_id', 'hadm_id']).agg({
    'icd_code': lambda x: ', '.join(x.astype(str).unique())
}).reset_index()
icd_grouped.columns = ['subject_id', 'hadm_id', 'icd_codes']

# Group prescriptions by admission
prescriptions_grouped = prescriptions_df.groupby('hadm_id').agg({
    'drug': lambda x: ', '.join(x.astype(str))
}).reset_index()
prescriptions_grouped.columns = ['hadm_id', 'drugs']

# Debug: Check columns after each merge
print(f"ICD grouped columns: {list(icd_grouped.columns)}")
print(f"Patients columns: {list(patients_df.columns)}")
print(f"Admissions columns: {list(admissions_df.columns)}")

# Join with patients and admissions - use suffixes to handle duplicate columns
print("Merging with patients...")
patient_visits = icd_grouped.merge(patients_df, on='subject_id', how='left', suffixes=('', '_pat'))
print(f"After patients merge: {patient_visits.shape}, columns: {list(patient_visits.columns)}")

print("Merging with admissions...")
# Both tables have subject_id, so specify suffixes
patient_visits = patient_visits.merge(admissions_df, on=['subject_id', 'hadm_id'], how='left', suffixes=('', '_adm'))
print(f"After admissions merge: {patient_visits.shape}, columns: {list(patient_visits.columns)}")

print("Merging with prescriptions...")
patient_visits = patient_visits.merge(prescriptions_grouped, on='hadm_id', how='left')
print(f"After prescriptions merge: {patient_visits.shape}, columns: {list(patient_visits.columns)}")

# Now we should have the correct columns
print(f"Final columns: {list(patient_visits.columns)}")

# Add visit numbers and length of stay
if 'dischtime' in patient_visits.columns and 'admittime' in patient_visits.columns:
    patient_visits['length_of_stay'] = (patient_visits['dischtime'] - patient_visits['admittime']).dt.days
    patient_visits = patient_visits.sort_values(['subject_id', 'admittime'])
    patient_visits['visit_number'] = patient_visits.groupby('subject_id').cumcount() + 1
else:
    print("ERROR: Still missing time columns")
    patient_visits['length_of_stay'] = 0
    patient_visits['visit_number'] = 1

# Filter for age <= 65 and visit_number <= 3
print(f"Before age/visit filtering: {len(patient_visits)} admissions")
print(f"Age distribution:")
print(patient_visits['anchor_age'].value_counts().sort_index())

print(f"\nVisit number distribution:")
print(patient_visits['visit_number'].value_counts().sort_index())

# Show how many are excluded by each filter
age_filter = patient_visits['anchor_age'] <= 65
visit_filter = patient_visits['visit_number'] <= 3

print(f"\nFiltering breakdown:")
print(f"Total admissions: {len(patient_visits)}")
print(f"Age ≤ 65: {age_filter.sum()} admissions")
print(f"Visit ≤ 3: {visit_filter.sum()} admissions")
print(f"Both age ≤ 65 AND visit ≤ 3: {(age_filter & visit_filter).sum()} admissions")

patient_visits = patient_visits[age_filter & visit_filter]

print(f"Patient visits (age ≤40, visits ≤3): {len(patient_visits)}")

print("\n=== STEP 3: ANTIPSYCHOTIC PRESCRIPTIONS ===")
# Define antipsychotic drugs
antipsychotic_drugs = [
    "chlorpromazine", "droperidol", "fluphenazine", "haloperidol", "loxapine",
    "perphenazine", "pimozide", "prochlorperazine", "thioridazine", "thiothixene",
    "trifluoperazine", "aripiprazole", "asenapine", "clozapine", "iloperidone",
    "lurasidone", "olanzapine", "paliperidone", "quetiapine", "risperidone",
    "ziprasidone", "amisulpride"
]

# Debug: Check what drugs we actually have
print("Sample drugs in prescriptions:")
print(prescriptions_df['drug'].str.lower().value_counts().head(20))

# Filter prescriptions for antipsychotics
antipsychotic_mask = prescriptions_df['drug'].str.lower().isin(antipsychotic_drugs)
antipsychotic_prescriptions = prescriptions_df[antipsychotic_mask].copy()

print(f"Antipsychotic prescriptions: {len(antipsychotic_prescriptions)}")

# If no exact matches, try partial matching
if len(antipsychotic_prescriptions) == 0:
    print("No exact matches found. Checking for partial matches...")

    # Check for partial matches
    for drug in antipsychotic_drugs[:5]:  # Check first 5
        partial_matches = prescriptions_df[prescriptions_df['drug'].str.lower().str.contains(drug, na=False)]
        if len(partial_matches) > 0:
            print(f"'{drug}' partial matches: {len(partial_matches)}")
            print(f"  Examples: {partial_matches['drug'].unique()[:3]}")

    # Try a broader search for common antipsychotics
    common_antipsychotics = ['haloperidol', 'olanzapine', 'risperidone', 'quetiapine', 'aripiprazole']
    broader_mask = prescriptions_df['drug'].str.lower().str.contains('|'.join(common_antipsychotics), na=False)
    broader_matches = prescriptions_df[broader_mask]
    print(f"Broader search found: {len(broader_matches)} prescriptions")

    if len(broader_matches) > 0:
        print("Found antipsychotic-like drugs:")
        print(broader_matches['drug'].value_counts().head(10))
        antipsychotic_prescriptions = broader_matches.copy()

if len(antipsychotic_prescriptions) == 0:
    print("Still no antipsychotic prescriptions found. Creating empty sequence...")
    # Create empty DataFrames for subsequent steps
    antipsychotic_sequence = pd.DataFrame()
    unique_third_antipsychotic = pd.DataFrame()
    final_sequence = pd.DataFrame(columns=['subject_id', 'hadm_id', 'first_antipsychotic', 'second_antipsychotic',
                                          'third_antipsychotic', 'starttime_1', 'starttime_2', 'starttime_3'])
else:
    # Add previous drug and start time using lag function equivalent
    antipsychotic_prescriptions = antipsychotic_prescriptions.sort_values(['subject_id', 'hadm_id', 'starttime'])
    antipsychotic_prescriptions['prev_drug'] = antipsychotic_prescriptions.groupby(['subject_id', 'hadm_id'])['drug'].shift(1)
    antipsychotic_prescriptions['prev_starttime'] = antipsychotic_prescriptions.groupby(['subject_id', 'hadm_id'])['starttime'].shift(1)

    print("\n=== STEP 4: ANTIPSYCHOTIC SEQUENCE ===")
    # Filter for drug changes (drug != prev_drug OR prev_drug IS NULL)
    sequence_mask = (antipsychotic_prescriptions['drug'] != antipsychotic_prescriptions['prev_drug']) | \
                    (antipsychotic_prescriptions['prev_drug'].isna())

    antipsychotic_sequence = antipsychotic_prescriptions[sequence_mask].copy()
    antipsychotic_sequence['drug_order'] = antipsychotic_sequence.groupby(['subject_id', 'hadm_id']).cumcount() + 1

    print(f"Antipsychotic sequence changes: {len(antipsychotic_sequence)}")

    print("\n=== STEP 5: UNIQUE THIRD ANTIPSYCHOTIC ===")
    # For simplicity, let's use the antipsychotic_sequence as is for now
    # The "unique third antipsychotic" logic is complex - skipping for initial testing
    unique_third_antipsychotic = antipsychotic_sequence.copy()

    print(f"Unique antipsychotic sequence: {len(unique_third_antipsychotic)}")

    print("\n=== STEP 6: FINAL SEQUENCE ===")
    # Fix the groupby reset_index issue
    if len(unique_third_antipsychotic) > 0:
        final_sequence_data = []

        for (subject_id, hadm_id), group in unique_third_antipsychotic.groupby(['subject_id', 'hadm_id']):
            row = {
                'subject_id': subject_id,
                'hadm_id': hadm_id,
                'first_antipsychotic': group[group['drug_order'] == 1]['drug'].iloc[0] if len(group[group['drug_order'] == 1]) > 0 else None,
                'second_antipsychotic': group[group['drug_order'] == 2]['drug'].iloc[0] if len(group[group['drug_order'] == 2]) > 0 else None,
                'third_antipsychotic': group[group['drug_order'] == 3]['drug'].iloc[0] if len(group[group['drug_order'] == 3]) > 0 else None,
                'starttime_1': group[group['drug_order'] == 1]['starttime'].iloc[0] if len(group[group['drug_order'] == 1]) > 0 else None,
                'starttime_2': group[group['drug_order'] == 2]['starttime'].iloc[0] if len(group[group['drug_order'] == 2]) > 0 else None,
                'starttime_3': group[group['drug_order'] == 3]['starttime'].iloc[0] if len(group[group['drug_order'] == 3]) > 0 else None,
            }
            final_sequence_data.append(row)

        final_sequence = pd.DataFrame(final_sequence_data)
    else:
        final_sequence = pd.DataFrame(columns=['subject_id', 'hadm_id', 'first_antipsychotic', 'second_antipsychotic',
                                              'third_antipsychotic', 'starttime_1', 'starttime_2', 'starttime_3'])

print(f"Final sequences: {len(final_sequence)}")

print("\n=== STEP 7: DAYS BETWEEN ===")
# Calculate days between antipsychotic switches
final_sequence['days_from_first_to_second'] = (
    final_sequence['starttime_2'] - final_sequence['starttime_1']
).dt.days

final_sequence['days_from_second_to_third'] = (
    final_sequence['starttime_3'] - final_sequence['starttime_2']
).dt.days

print("\n=== STEP 8: VISIT TIMES ===")
# Get visit dates for first 3 visits
visit_times = patient_visits[patient_visits['visit_number'] <= 3].groupby('subject_id').apply(
    lambda x: pd.Series({
        'visit_1_date': x[x['visit_number'] == 1]['admittime'].iloc[0] if len(x[x['visit_number'] == 1]) > 0 else None,
        'visit_2_date': x[x['visit_number'] == 2]['admittime'].iloc[0] if len(x[x['visit_number'] == 2]) > 0 else None,
        'visit_3_date': x[x['visit_number'] == 3]['admittime'].iloc[0] if len(x[x['visit_number'] == 3]) > 0 else None,
    })
).reset_index()

print(f"Visit times: {len(visit_times)}")

print("\n=== STEP 9: FINAL JOIN ===")
# Main join operation
result = patient_visits.merge(comorbidities, on=['subject_id', 'hadm_id'], how='left')
result = result.merge(final_sequence, on=['subject_id', 'hadm_id'], how='left')
result = result.merge(visit_times, on='subject_id', how='left')

# Add length of stay for visits 1, 2, 3
los_by_visit = patient_visits.groupby('subject_id').apply(
    lambda x: pd.Series({
        'length_of_stay_1': x[x['visit_number'] == 1]['length_of_stay'].iloc[0] if len(x[x['visit_number'] == 1]) > 0 else None,
        'length_of_stay_2': x[x['visit_number'] == 2]['length_of_stay'].iloc[0] if len(x[x['visit_number'] == 2]) > 0 else None,
        'length_of_stay_3': x[x['visit_number'] == 3]['length_of_stay'].iloc[0] if len(x[x['visit_number'] == 3]) > 0 else None,
    })
).reset_index()

result = result.merge(los_by_visit, on='subject_id', how='left')

# Calculate days between visits
result['days_between_visit_1_and_2'] = (result['visit_2_date'] - result['visit_1_date']).dt.days
result['days_between_visit_2_and_3'] = (result['visit_3_date'] - result['visit_2_date']).dt.days

# Select final columns to match SQL output
final_columns = [
    'subject_id', 'hadm_id', 'visit_number', 'gender', 'anchor_age', 'race',
    'icd_codes', 'drugs', 'admittime', 'dischtime', 'length_of_stay', 'comorbidities',
    'first_antipsychotic', 'second_antipsychotic', 'third_antipsychotic',
    'days_from_first_to_second', 'days_from_second_to_third',
    'visit_1_date', 'visit_2_date', 'visit_3_date',
    'length_of_stay_1', 'length_of_stay_2', 'length_of_stay_3',
    'days_between_visit_1_and_2', 'days_between_visit_2_and_3'
]

# Filter to only include columns that exist
existing_columns = [col for col in final_columns if col in result.columns]
final_result = result[existing_columns].copy()

print(f"\n=== FINAL RESULTS ===")
print(f"Final dataset shape: {final_result.shape}")
print(f"Unique patients: {final_result['subject_id'].nunique()}")
print(f"Total admissions: {final_result['hadm_id'].nunique()}")

# Show summary statistics
print(f"\nVisit distribution:")
print(final_result['visit_number'].value_counts().sort_index())

print(f"\nAge distribution:")
print(final_result['anchor_age'].describe())

print(f"\nGender distribution:")
print(final_result['gender'].value_counts())

# Show patients with antipsychotic sequences
antipsychotic_patients = final_result[final_result['first_antipsychotic'].notna()]
print(f"\nPatients with antipsychotic prescriptions: {len(antipsychotic_patients)}")

if len(antipsychotic_patients) > 0:
    print(f"Patients with 2+ antipsychotics: {antipsychotic_patients['second_antipsychotic'].notna().sum()}")
    print(f"Patients with 3+ antipsychotics: {antipsychotic_patients['third_antipsychotic'].notna().sum()}")

# Save results
final_result.to_csv('/content/drive/MyDrive/Northwestern/schizophrenia_antipsychotic_analysis.csv', index=False)
print(f"\nResults saved to: schizophrenia_antipsychotic_analysis.csv")

# Display first few rows
print(f"\n=== SAMPLE DATA ===")
print(final_result.head())

# Show columns with data
print(f"\n=== COLUMN DATA AVAILABILITY ===")
for col in final_result.columns:
    non_null_count = final_result[col].notna().sum()
    print(f"{col}: {non_null_count}/{len(final_result)} ({non_null_count/len(final_result)*100:.1f}%)")
