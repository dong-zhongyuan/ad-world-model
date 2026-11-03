import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "clinical"))
from adni_visits import build_index_cohort


def test_sample_visit_crosswalk_and_post_blood_outcome(tmp_path):
    samples = pd.DataFrame({
        "sample_column": ["profile1", "profile2"],
        "subject": ["001_S_0001", "001_S_0002"],
        "sample_visit": ["v03", "bl"],
        "phase": ["ADNI2", "ADNIGO"],
    })
    diagnosis = pd.DataFrame({
        "PTID": samples.subject,
        "VISCODE": ["v03", "bl"],
        "EXAMDATE": ["2020-02-01", "2020-02-01"],
        "DIAGNOSIS": [2, 1],
    })
    mmse = pd.DataFrame([
        ("ADNI2", "001_S_0001", "v01", "sc", "2020-01-15", 30),
        ("ADNI2", "001_S_0001", "v05", "m06", "2020-01-20", 26),
        ("ADNI2", "001_S_0001", "v11", "m12", "2021-01-15", 27),
        ("ADNIGO", "001_S_0002", "sc", "sc", "2020-01-15", 30),
        ("ADNIGO", "001_S_0002", "m12", "m12", "2021-01-15", 28),
    ], columns=["PHASE", "PTID", "VISCODE", "VISCODE2", "VISDATE", "MMSCORE"])
    diagnosis_path = tmp_path / "dx.csv"
    mmse_path = tmp_path / "mmse.csv"
    diagnosis.to_csv(diagnosis_path, index=False)
    mmse.to_csv(mmse_path, index=False)
    cohort, _ = build_index_cohort(mmse_path, diagnosis_path, samples)
    assert cohort.subject.tolist() == samples.subject.tolist()
    assert cohort.event.tolist() == [1, 0]
    assert cohort.baseline_mmse.tolist() == [30, 30]
    assert cohort.mmse_visit.tolist() == ["v01", "sc"]
    expected = (pd.Timestamp("2021-01-15") - pd.Timestamp("2020-02-01")).days / 30.4375
    assert cohort.time.iloc[0] == pytest.approx(expected)
    assert (cohort.time > 0).all()
