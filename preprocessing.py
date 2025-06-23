!pip install metric-learn
!pip install surprise
!pip install gower
!pip install skrebate
!pip install tabulate

 #mimic 4

# Read the data from the CSV file
df = pd.read_csv('/content/drive/MyDrive/bq-results-20250509-065907-1746774021891/bq-results-20250509-065907-1746774021891.csv')

import pandas as pd
import numpy as np

# Set pandas display options for better output
pd.set_option('display.max_rows', None)

# 1. Standardize Antipsychotic Names (Convert to Lowercase)
print("Cleaning antipsychotic columns...")

antipsychotic_columns = ['first_antipsychotic', 'second_antipsychotic', 'third_antipsychotic']
for col in antipsychotic_columns:
    if col in df.columns:
        print(f"   - Cleaning {col}")
        df[col] = df[col].astype(str).str.lower()
        # Replace 'nan' strings with actual np.nan
        df[col] = df[col].replace('nan', np.nan)
    else:
        print(f"   - Column '{col}' not found.")

print("Antipsychotic cleaning finished.")

# 2. Extract and Process ICD Codes
print("\nProcessing ICD codes...")

# Function to extract the first code
def get_first_code(codes):
    if pd.isna(codes) or codes == '':
        return np.nan
    codes_str = str(codes)
    try:
        first_code = codes_str.split(',', 1)[0].strip()
        return first_code
    except Exception:
        return codes_str.strip()

# Extract the first code and replace in the original column
df['icd_codes'] = df['icd_codes'].apply(get_first_code)
df['icd_codes'] = df['icd_codes'].replace('', np.nan)

# Store a copy for frequency analysis before mapping
first_code_temp = df['icd_codes'].copy()

# 3. Apply Race Grouping
print("\nGrouping 'race' categories...")
mapping_dict = {
    # White variations
    'WHITE': 'White', 'WHITE - RUSSIAN': 'White', 'WHITE - OTHER EUROPEAN': 'White',
    'WHITE - EASTERN EUROPEAN': 'White', 'PORTUGUESE': 'White',
    # Black variations
    'BLACK/AFRICAN AMERICAN': 'Black/African American', 'BLACK/CARIBBEAN ISLAND': 'Black/African American',
    'BLACK/AFRICAN': 'Black/African American', 'BLACK/CAPE VERDEAN': 'Black/African American',
    # Hispanic/Latino variations
    'HISPANIC OR LATINO': 'Hispanic/Latino', 'HISPANIC/LATINO - CUBAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic/Latino', 'HISPANIC/LATINO - DOMINICAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - CENTRAL AMERICAN': 'Hispanic/Latino', 'HISPANIC/LATINO - GUATEMALAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - MEXICAN': 'Hispanic/Latino', 'HISPANIC/LATINO - COLUMBIAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - HONDURAN': 'Hispanic/Latino', 'HISPANIC/LATINO - SALVADORAN': 'Hispanic/Latino',
    'SOUTH AMERICAN': 'Hispanic/Latino', 'WHITE - BRAZILIAN': 'Hispanic/Latino',
    # Asian variations
    'ASIAN': 'Asian', 'ASIAN - CHINESE': 'Asian', 'ASIAN - ASIAN INDIAN': 'Asian',
    'ASIAN - KOREAN': 'Asian', 'ASIAN - SOUTH EAST ASIAN': 'Asian',
    # Native American
    'AMERICAN INDIAN/ALASKA NATIVE': 'American Indian/Alaska Native',
    # Pacific Islander
    'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Native Hawaiian/Other Pacific Islander',
    # Multiracial
    'MULTIPLE RACE/ETHNICITY': 'Multiracial',
    # Unknown / Other / Declined
    'UNKNOWN': 'Unknown/Declined/Other', 'OTHER': 'Unknown/Declined/Other',
    'PATIENT DECLINED TO ANSWER': 'Unknown/Declined/Other', 'UNABLE TO OBTAIN': 'Unknown/Declined/Other',
}

# Apply the mapping to the race column
df['race'] = df['race'].map(mapping_dict).fillna('Unknown/Declined/Other')

# Filter out specified race categories and store back in df
races_to_exclude = ['Unknown/Declined/Other', 'American Indian/Alaska Native',
                   'Native Hawaiian/Other Pacific Islander', 'Multiracial']
df = df[~df['race'].isin(races_to_exclude)]

print("\nValue counts for final updated 'race' column after filtering:")
print(df['race'].value_counts())

# 4. ICD-9 to ICD-10 Mapping
print("\nCreating ICD-9 to ICD-10 mapping for frequency analysis...")

# Define the mapping based on the observed ICD-9 codes
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

# Create the ICD-10 mapping and replace in the original column
df['icd_codes'] = df['icd_codes'].replace(icd9_to_icd10_map)

import pandas as pd
import numpy as np
from collections import Counter

# Assume df is your DataFrame, loaded elsewhere

# --- Pre-processing: Get unique patients and aggregate data ---
# First, let's get visit counts per patient
visit_counts_per_patient = df.groupby('subject_id').size()

# Get one row per patient (using first visit for most characteristics)
df_patients = df.groupby('subject_id').first().reset_index()

# Add visit counts to patient data
df_patients['num_visits'] = df_patients['subject_id'].map(visit_counts_per_patient)

# For antipsychotics, we want to get all unique antipsychotics per patient across all visits
antipsychotic_cols = ['first_antipsychotic', 'second_antipsychotic', 'third_antipsychotic']
existing_antipsychotic_cols = [col for col in antipsychotic_cols if col in df.columns]

# Get all antipsychotics per patient across all their visits
patient_antipsychotics = {}
all_antipsychotics_list = []

for subject_id in df['subject_id'].unique():
    patient_visits = df[df['subject_id'] == subject_id]
    patient_meds = []

    for _, visit in patient_visits.iterrows():
        for col in existing_antipsychotic_cols:
            if pd.notna(visit[col]):
                patient_meds.append(visit[col])
                all_antipsychotics_list.append(visit[col])

    # Remove duplicates for this patient
    unique_patient_meds = list(set(patient_meds))
    patient_antipsychotics[subject_id] = unique_patient_meds

# Add number of unique antipsychotics per patient
df_patients['num_antipsychotics'] = df_patients['subject_id'].apply(
    lambda x: len(patient_antipsychotics.get(x, []))
)

# Create number of comorbidities variable (using first visit data)
if 'comorbidities' in df_patients.columns:
    df_patients['num_comorbidities'] = df_patients['comorbidities'].fillna('').apply(
        lambda x: len([c.strip() for c in str(x).split(',') if c.strip()]) if str(x) != 'nan' and str(x) != '' else 0
    )
else:
    df_patients['num_comorbidities'] = 0

# Get top 5 antipsychotics across all visits
top_antipsychotics = Counter(all_antipsychotics_list).most_common(5)

# --- Table 1 generation ---
table = []

# === PATIENT OVERVIEW ===
total_patients = len(df_patients)
table.append({"Characteristic": "Total patients", "Value": f"{total_patients}"})
table.append({"Characteristic": "", "Value": ""})

# Age (from first visit)
if 'anchor_age' in df_patients.columns:
    age_mean = df_patients['anchor_age'].mean()
    age_std = df_patients['anchor_age'].std()
    age_n = df_patients['anchor_age'].notna().sum()
    table.append({"Characteristic": "Age (SD)", "Value": f"{age_mean:.1f} ± {age_std:.1f}   {age_n}"})

# Length of Stay (average across all visits per patient)
if 'length_of_stay' in df.columns:
    los_per_patient = df.groupby('subject_id')['length_of_stay'].mean()
    los_mean = los_per_patient.mean()
    los_std = los_per_patient.std()
    los_n = los_per_patient.notna().sum()
    table.append({"Characteristic": "Length of Stay (SD)", "Value": f"{los_mean:.1f} ± {los_std:.1f}   {los_n}"})

table.append({"Characteristic": "", "Value": ""})

# === DEMOGRAPHICS ===
# Gender (from first visit)
if 'gender' in df_patients.columns:
    table.append({"Characteristic": "Gender", "Value": ""})
    gender_counts = df_patients['gender'].value_counts()

    # Sort gender for consistent order (M first, then F)
    gender_order = ['M', 'F'] if 'M' in gender_counts.index and 'F' in gender_counts.index else gender_counts.index
    for gender in gender_order:
        if gender in gender_counts.index:
            count = gender_counts[gender]
            pct = (count / total_patients) * 100
            table.append({"Characteristic": gender, "Value": f"{pct:.1f}%\t{count}"})

    table.append({"Characteristic": "", "Value": ""})

# Race (from first visit) - sort by frequency
if 'race' in df_patients.columns:
    table.append({"Characteristic": "Race", "Value": ""})
    race_counts = df_patients['race'].value_counts()

    for race in race_counts.index:
        count = race_counts[race]
        pct = (count / total_patients) * 100
        table.append({"Characteristic": str(race), "Value": f"{pct:.1f}%\t{count}"})

    table.append({"Characteristic": "", "Value": ""})

# === CLINICAL CHARACTERISTICS ===
# Number of Comorbidities
table.append({"Characteristic": "Number of Comorbidities", "Value": ""})
comorbidity_counts = df_patients['num_comorbidities'].value_counts().sort_index()

# Group comorbidities in logical order (fixed grouping)
comorbidity_groups = [
    ('None', [0]),
    ('1-2', [1, 2]),
    ('3-5', [3, 4, 5]),
    ('6-9', [6, 7, 8, 9]),
    ('10+', list(range(10, df_patients['num_comorbidities'].max() + 1)) if df_patients['num_comorbidities'].max() >= 10 else [])
]

for group_name, values in comorbidity_groups:
    if not values:  # Skip empty groups
        continue
    count = sum(comorbidity_counts.get(v, 0) for v in values)
    if count > 0:
        pct = (count / total_patients) * 100
        table.append({"Characteristic": group_name, "Value": f"{pct:.1f}%\t{count}"})

table.append({"Characteristic": "", "Value": ""})

# === TREATMENT PATTERNS ===
# Number of Antipsychotics (unique per patient)
table.append({"Characteristic": "Number of Antipsychotics", "Value": ""})
antipsychotic_counts = df_patients['num_antipsychotics'].value_counts().sort_index()

for num_meds in sorted(antipsychotic_counts.index):
    count = antipsychotic_counts[num_meds]
    pct = (count / total_patients) * 100
    label = "None" if num_meds == 0 else str(num_meds)
    table.append({"Characteristic": label, "Value": f"{pct:.1f}%\t{count}"})

table.append({"Characteristic": "", "Value": ""})

# Number of Visits per Patient
table.append({"Characteristic": "Number of Visits", "Value": ""})
visit_frequency = visit_counts_per_patient.value_counts().sort_index()

for num_visits in sorted(visit_frequency.index):
    count = visit_frequency[num_visits]
    pct = (count / total_patients) * 100
    table.append({"Characteristic": str(num_visits), "Value": f"{pct:.1f}%\t{count}"})

table.append({"Characteristic": "", "Value": ""})

# Prescribed Antipsychotics (including no medication + top 5 + others to sum to 100%)
if top_antipsychotics:
    table.append({"Characteristic": "Prescribed Antipsychotics", "Value": ""})

    # First, add patients with no antipsychotics
    no_meds_count = (df_patients['num_antipsychotics'] == 0).sum()
    no_meds_pct = (no_meds_count / total_patients) * 100
    table.append({"Characteristic": "None", "Value": f"{no_meds_pct:.1f}%\t{no_meds_count}"})

    # Calculate how many PATIENTS received each of the top 5 medications
    patients_per_medication = {}
    for subject_id in df['subject_id'].unique():
        patient_visits = df[df['subject_id'] == subject_id]
        patient_meds = []

        for _, visit in patient_visits.iterrows():
            for col in existing_antipsychotic_cols:
                if pd.notna(visit[col]):
                    patient_meds.append(visit[col])

        # Get unique medications for this patient
        unique_patient_meds = list(set(patient_meds))
        for med in unique_patient_meds:
            if med not in patients_per_medication:
                patients_per_medication[med] = 0
            patients_per_medication[med] += 1

    # Get top 5 medications by number of patients
    top_5_by_patients = sorted(patients_per_medication.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_med_names = [med[0] for med in top_5_by_patients]

    # Add top 5 medications
    for i, (medication, patient_count) in enumerate(top_5_by_patients, 1):
        pct = (patient_count / total_patients) * 100
        table.append({"Characteristic": f"{i}. {medication.title()}", "Value": f"{pct:.1f}%\t{patient_count}"})

    # Calculate "Others" - patients who received antipsychotics but NONE of the top 5
    patients_with_others = 0
    for subject_id in df['subject_id'].unique():
        patient_visits = df[df['subject_id'] == subject_id]
        patient_meds = []

        for _, visit in patient_visits.iterrows():
            for col in existing_antipsychotic_cols:
                if pd.notna(visit[col]):
                    patient_meds.append(visit[col])

        unique_patient_meds = list(set(patient_meds))

        # Check if this patient received any antipsychotics AND none of them are in top 5
        if len(unique_patient_meds) > 0:
            has_top_5 = any(med in top_5_med_names for med in unique_patient_meds)
            if not has_top_5:
                patients_with_others += 1

    if patients_with_others > 0:
        others_pct = (patients_with_others / total_patients) * 100
        table.append({"Characteristic": "6. Others", "Value": f"{others_pct:.1f}%\t{patients_with_others}"})

    table.append({"Characteristic": "", "Value": ""})

# Create clean, formatted output
output_lines = []
output_lines.append("Table 1: Patient Cohort")
output_lines.append("=" * 60)
output_lines.append("")

# Create formatted table with logical sections
table1_df = pd.DataFrame(table)

for _, row in table1_df.iterrows():
    characteristic = row['Characteristic']
    value = row['Value']

    if characteristic == "" and value == "":
        output_lines.append("")  # Add blank line
    elif value == "":
        output_lines.append(f"{characteristic}")  # Section header
        output_lines.append("")  # Add space after section header
    else:
        # Handle the spacing for the table format
        if '\t' in str(value):
            parts = str(value).split('\t')
            if len(parts) == 2:
                # Format: "  Item name                     12.3%      123"
                output_lines.append(f"  {characteristic:<35} {parts[0]:>8} {parts[1]:>8}")
            else:
                output_lines.append(f"  {characteristic:<35} {value}")
        else:
            # For continuous variables like Age, Length of Stay
            output_lines.append(f"  {characteristic:<35} {value}")

# Print the formatted table
formatted_output = "\n".join(output_lines)
print(formatted_output)

# Save to text file
with open('table1_patient_cohort.txt', 'w') as f:
    f.write(formatted_output)

# Also create a clean CSV version
csv_data = []
current_section = ""

for _, row in table1_df.iterrows():
    characteristic = row['Characteristic']
    value = row['Value']

    if value == "" and characteristic != "":
        current_section = characteristic
    elif '\t' in str(value):
        parts = str(value).split('\t')
        if len(parts) == 2:
            csv_data.append({
                'Section': current_section,
                'Characteristic': characteristic,
                'Percentage': parts[0],
                'Count': parts[1]
            })
    elif value != "" and characteristic != "":
        csv_data.append({
            'Section': current_section,
            'Characteristic': characteristic,
            'Value': value,
            'Count': ''
        })

csv_df = pd.DataFrame(csv_data)
csv_df.to_csv('table1_patient_cohort.csv', index=False)

print("\n" + "="*60)
print("✓ Table saved to: table1_patient_cohort.txt")
print("✓ Data saved to: table1_patient_cohort.csv")
print("="*60)

# Brief summary
print(f"\nSummary Statistics:")
print(f"  • Total unique patients: {total_patients:,}")
print(f"  • Total visits: {len(df):,}")
print(f"  • Average visits per patient: {visit_counts_per_patient.mean():.1f}")
if 'anchor_age' in df_patients.columns:
    print(f"  • Mean age: {df_patients['anchor_age'].mean():.1f} ± {df_patients['anchor_age'].std():.1f} years")
print(f"  • Total antipsychotic prescriptions: {len(all_antipsychotics_list):,}")
print(f"  • Unique antipsychotics prescribed: {len(set(all_antipsychotics_list))}")
