WITH Comorbidities AS (
  SELECT
    d.subject_id,
    d.hadm_id,
    STRING_AGG(DISTINCT d.icd_code, ', ') AS comorbidities
  FROM
    `physionet-data.mimiciv_hosp.diagnoses_icd` d
  WHERE
    d.icd_code NOT LIKE 'F2%' AND d.icd_code NOT LIKE '295%'
  GROUP BY
    d.subject_id, d.hadm_id
),
PatientVisits AS (
  SELECT
    d.subject_id,
    d.hadm_id,
    STRING_AGG(DISTINCT d.icd_code, ', ') AS icd_codes,
    p.dod,
    p.gender,
    p.anchor_age,
    a.admittime,
    a.dischtime,
    a.race,
    pr.drugs,
    ROW_NUMBER() OVER (PARTITION BY d.subject_id ORDER BY a.admittime) AS visit_number,
    DATE_DIFF(a.dischtime, a.admittime, DAY) AS length_of_stay
  FROM
    `physionet-data.mimiciv_hosp.diagnoses_icd` d
  LEFT JOIN
    `physionet-data.mimiciv_hosp.patients` p ON d.subject_id = p.subject_id
  LEFT JOIN
    `physionet-data.mimiciv_hosp.admissions` a ON d.hadm_id = a.hadm_id
  LEFT JOIN (
    SELECT
      hadm_id,
      STRING_AGG(drug, ', ' ORDER BY starttime, stoptime) AS drugs
    FROM
      `physionet-data.mimiciv_hosp.prescriptions`
    GROUP BY
      hadm_id
  ) pr ON d.hadm_id = pr.hadm_id
  WHERE
    (d.icd_code LIKE 'F2%' OR d.icd_code LIKE '295%') -- Keep the schizophrenia diagnosis filter
    AND d.seq_num = 1 -- **Add this condition to filter for primary diagnosis**
  GROUP BY
    d.subject_id, d.hadm_id, p.dod, p.gender, p.anchor_age, a.admittime, a.dischtime, a.race, pr.drugs
),
AntipsychoticPrescriptions AS (
  SELECT
    pr.subject_id,
    pr.hadm_id,
    pr.drug,
    pr.starttime,
    LAG(pr.drug) OVER (
      PARTITION BY pr.subject_id, pr.hadm_id
      ORDER BY pr.starttime
    ) AS prev_drug,
    LAG(pr.starttime) OVER (
      PARTITION BY pr.subject_id, pr.hadm_id
      ORDER BY pr.starttime
    ) AS prev_starttime
  FROM
    `physionet-data.mimiciv_hosp.prescriptions` pr
  WHERE
    LOWER(pr.drug) IN (
      "chlorpromazine", "droperidol", "fluphenazine", "haloperidol", "loxapine",
      "perphenazine", "pimozide", "prochlorperazine", "thioridazine", "thiothixene",
      "trifluoperazine", "aripiprazole", "asenapine", "clozapine", "iloperidone",
      "lurasidone", "olanzapine", "paliperidone", "quetiapine", "risperidone", "ziprasidone","amisulpride"
    )
),
AntipsychoticSequence AS (
  SELECT
    ap.subject_id,
    ap.hadm_id,
    ap.drug,
    ap.starttime,
    ROW_NUMBER() OVER (
      PARTITION BY ap.subject_id, ap.hadm_id
      ORDER BY ap.starttime
    ) AS drug_order
  FROM
    AntipsychoticPrescriptions ap
  WHERE
    ap.drug != ap.prev_drug OR ap.prev_drug IS NULL
),
-- NEW CTE to identify unique third antipsychotics
UniqueThirdAntipsychotic AS (
  SELECT
    ap.subject_id,
    ap.hadm_id,
    ap.drug,
    ap.starttime,
    ROW_NUMBER() OVER (
      PARTITION BY ap.subject_id, ap.hadm_id
      ORDER BY ap.starttime
    ) AS drug_order
  FROM
    AntipsychoticPrescriptions ap
  WHERE
    (ap.drug != ap.prev_drug OR ap.prev_drug IS NULL)
    AND ap.drug NOT IN (
      SELECT drug FROM AntipsychoticPrescriptions ap2
      WHERE ap2.subject_id = ap.subject_id
      AND ap2.starttime < ap.starttime
    )
),
-- Modified FinalSequence to use the new CTE
FinalSequence AS (
  SELECT
    subject_id,
    hadm_id,
    MAX(CASE WHEN drug_order = 1 THEN drug END) AS first_antipsychotic,
    MAX(CASE WHEN drug_order = 2 THEN drug END) AS second_antipsychotic,
    MAX(CASE WHEN drug_order = 3 THEN drug END) AS third_antipsychotic,
    MAX(CASE WHEN drug_order = 1 THEN starttime END) AS starttime_1,
    MAX(CASE WHEN drug_order = 2 THEN starttime END) AS starttime_2,
    MAX(CASE WHEN drug_order = 3 THEN starttime END) AS starttime_3
  FROM
    UniqueThirdAntipsychotic
  GROUP BY
    subject_id, hadm_id
),
DaysBetween AS (
  SELECT
    subject_id,
    hadm_id,
    DATE_DIFF(starttime_2, starttime_1, DAY) AS days_from_first_to_second,
    DATE_DIFF(starttime_3, starttime_2, DAY) AS days_from_second_to_third
  FROM
    FinalSequence
),
VisitTimes AS (
  SELECT
    subject_id,
    MAX(CASE WHEN visit_number = 1 THEN admittime END) AS visit_1_date,
    MAX(CASE WHEN visit_number = 2 THEN admittime END) AS visit_2_date,
    MAX(CASE WHEN visit_number = 3 THEN admittime END) AS visit_3_date
  FROM
    PatientVisits
  WHERE
    visit_number <= 3
  GROUP BY
    subject_id
)
SELECT
  pv.subject_id,
  pv.hadm_id,
  pv.visit_number,
  pv.gender,
  pv.anchor_age,
  pv.race,
  pv.icd_codes,
  pv.drugs,
  pv.admittime,
  pv.dischtime,
  pv.length_of_stay,
  c.comorbidities,
  fs.first_antipsychotic,
  fs.second_antipsychotic,
  fs.third_antipsychotic,
  db.days_from_first_to_second,
  db.days_from_second_to_third,
  vt.visit_1_date,
  vt.visit_2_date,
  vt.visit_3_date,
  MAX(CASE WHEN pv.visit_number = 1 THEN pv.length_of_stay END) OVER (PARTITION BY pv.subject_id) AS length_of_stay_1,
  MAX(CASE WHEN pv.visit_number = 2 THEN pv.length_of_stay END) OVER (PARTITION BY pv.subject_id) AS length_of_stay_2,
  MAX(CASE WHEN pv.visit_number = 3 THEN pv.length_of_stay END) OVER (PARTITION BY pv.subject_id) AS length_of_stay_3,
  DATE_DIFF(vt.visit_2_date, vt.visit_1_date, DAY) AS days_between_visit_1_and_2,
  DATE_DIFF(vt.visit_3_date, vt.visit_2_date, DAY) AS days_between_visit_2_and_3
FROM
  PatientVisits pv
LEFT JOIN
  Comorbidities c ON pv.subject_id = c.subject_id AND pv.hadm_id = c.hadm_id
LEFT JOIN
  FinalSequence fs ON pv.subject_id = fs.subject_id AND pv.hadm_id = fs.hadm_id
LEFT JOIN
  DaysBetween db ON pv.subject_id = db.subject_id AND pv.hadm_id = db.hadm_id
LEFT JOIN
  VisitTimes vt ON pv.subject_id = vt.subject_id
WHERE
  pv.visit_number <= 3
  AND pv.anchor_age <= 40;
