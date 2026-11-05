#!/usr/bin/env python3
"""Compare OGT control and patient data using shared route-level metrics.

This script:
1. Selects control participants from ``control_demographics_sorted.csv``,
   keeping only first sessions when a participant has both ``_1`` and ``_2``.
2. Loads control task data from JATOS exports in ``data_controls``.
3. Loads patient task data from raw OCTAL TXT files and raw JATOS exports in
   ``data_patients/OGT_*``.
4. Computes basic route metrics for practice and test.
5. Computes Ang-style uniqueness metrics on test routes.
6. Writes participant-level metrics and group comparison summaries to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SESSION_SUFFIX_RE = re.compile(r"^(.*?)(?:_(0*1|0*2))$")
CONTROL_DEMOGRAPHIC_KEYS = {"subID", "age", "sex", "hand", "timestamp"}
PATIENT_TEXT_METADATA_REJECT = {"test"}
CONTROL_EXTRA_ID_RE = re.compile(r"^N[CH]_\d+$", re.IGNORECASE)
X_CENTER = 400.0
Y_CENTER = 600.0
CANVAS_WIDTH = 2 * X_CENTER
CANVAS_HEIGHT = 2 * Y_CENTER
TARGET_TOP_Y = 0.15 * CANVAS_HEIGHT
TARGET_BOTTOM_Y = 0.90 * CANVAS_HEIGHT
TARGET_VERTICAL_SPAN = TARGET_BOTTOM_Y - TARGET_TOP_Y
PLOT_METRICS = [
    "practice_mean_speed_px_per_s",
    "test_route_count",
    "test_total_distance_px",
    "test_uniqueness_mean",
    "test_mds_hull_area",
]
EXPECTED_TEST_DURATION_MS = 120000.0
SHORT_TEST_THRESHOLD_MS = 115000.0
PRACTICE_CALIBRATION_MIN_ROUTES = 3
SOURCE_FRAME_FALLBACKS: dict[str, tuple[float, float, float]] = {
    "JATOS": (400.0, 228.0, 1199.0),
    "OCTAL": (1000.0, 195.0, 1005.0),
    "OGT_Mandarin": (667.0, 158.0, 714.0),
    "OGT_Spanish": (400.0, 232.0, 1197.0),
}
CLINICAL_CORRELATION_FIELDS = [
    "BACS_VM_t",
    "BACS_DS_t",
    "BACS_TMT_t",
    "BACS_VF_t",
    "BACS_SC_t",
    "BACS_ToL_t",
    "BACS_comp_t",
    "PSP_score",
    "AMI_behavioural",
    "AMI_social",
    "AMI_emotional",
    "MAP",
    "EXP",
    "CDS_total",
    "PANSS_marder_positive",
]


@dataclass
class ParticipantRoutes:
    group: str
    participant_id: str
    source_id: str
    source_platform: str
    practice_routes: list[np.ndarray]
    test_routes: list[np.ndarray]
    metadata: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-csv",
        type=Path,
        default=Path("control_demographics_sorted.csv"),
        help="Cleaned control demographics CSV.",
    )
    parser.add_argument(
        "--control-dir",
        type=Path,
        default=Path("data_controls"),
        help="Root folder for control JATOS study_result folders.",
    )
    parser.add_argument(
        "--patient-octal-dir",
        type=Path,
        default=Path("data_patients/OCTAL"),
        help="Folder containing raw patient OCTAL TXT files.",
    )
    parser.add_argument(
        "--patient-jatos-dirs",
        nargs="+",
        type=Path,
        default=[
            Path("data_patients/OGT_English"),
            Path("data_patients/OGT_Mandarin"),
            Path("data_patients/OGT_Spanish"),
        ],
        help="One or more raw patient JATOS root folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory where output CSVs will be written.",
    )
    parser.add_argument(
        "--clinical-csv",
        type=Path,
        default=Path("CHANSS_clinical_demog.csv"),
        help="Clinical/demographic CSV for patient correlations.",
    )
    parser.add_argument(
        "--clinical-id-corrections-csv",
        type=Path,
        default=Path("analysis_outputs/ogt_patients_missing_clinical_data_corrected.csv"),
        help="Optional CSV mapping unmatched OGT participant IDs to corrected clinical IDs.",
    )
    parser.add_argument(
        "--np-points",
        type=int,
        default=200,
        help="Number of evenly spaced points used to resample each route.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=40,
        help="Moving-average window used for velocity/acceleration features.",
    )
    parser.add_argument(
        "--velocity-downsample",
        type=int,
        default=10,
        help="Downsampling factor applied to velocity/acceleration features.",
    )
    parser.add_argument(
        "--uniqueness-reference",
        choices=["pooled", "within_group"],
        default="pooled",
        help=(
            "Whether uniqueness is computed against all test routes combined "
            "or separately within each group."
        ),
    )
    parser.add_argument(
        "--write-route-points",
        action="store_true",
        help="Also write a long-format CSV of extracted route points.",
    )
    return parser.parse_args()


def strip_session_suffix(participant_id: str) -> tuple[str, int | None]:
    match = SESSION_SUFFIX_RE.match(participant_id.strip())
    if not match:
        return participant_id.strip(), None
    base_id, session_str = match.groups()
    return base_id, int(session_str)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_control_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    unsuffixed_rows: list[dict[str, str]] = []
    suffixed_rows: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)

    for row in rows:
        participant_id = row.get("subID", "").strip()
        base_id, session = strip_session_suffix(participant_id)
        if session is None:
            unsuffixed_rows.append(row)
        else:
            suffixed_rows[base_id][session] = row

    selected = list(unsuffixed_rows)
    dropped_second_only: list[str] = []

    for base_id, session_rows in sorted(suffixed_rows.items()):
        if 1 in session_rows:
            selected.append(session_rows[1])
        elif 2 in session_rows:
            dropped_second_only.append(base_id)

    if dropped_second_only:
        warnings.warn(
            "Dropped control participants with only a session-2 record and no "
            f"session-1 match: {', '.join(dropped_second_only)}"
        )

    return sorted(selected, key=lambda row: row["study_result_id"])


def parse_json_file(path: Path) -> object:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"Empty file: {path}")
    return parse_json_text(text)


def parse_json_text(text: str) -> object:
    text = text.strip()
    if not text:
        raise ValueError("Empty JSON text")
    if text.startswith("["):
        return json.loads(text.replace("][", ","))
    if text.startswith('{"round"'):
        normalized = f"[{text.replace('}{', '},{')}]"
        return json.loads(normalized)
    return json.loads(text)


def is_control_demographics_payload(payload: object) -> bool:
    return isinstance(payload, dict) and bool(CONTROL_DEMOGRAPHIC_KEYS & set(payload))


def group_jatos_samples(samples: Iterable[dict], round_number: int) -> list[np.ndarray]:
    grouped: dict[int, list[list[float]]] = defaultdict(list)

    for sample in samples:
        if sample.get("round") != round_number or sample.get("complete") != 1:
            continue
        path_number = int(sample["path"])
        grouped[path_number].append(
            [
                float(sample["time_since_round_started"]),
                float(sample["x"]),
                float(sample["y"]),
            ]
        )

    routes: list[np.ndarray] = []
    for path_number in sorted(grouped):
        route = np.asarray(grouped[path_number], dtype=float)
        if route.size == 0:
            continue
        route = route[np.argsort(route[:, 0])]
        routes.append(route)
    return routes


def extract_control_routes(study_result_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    task_payloads: list[list[dict]] = []

    for data_file in sorted(study_result_dir.glob("comp-result_*/data.txt")):
        try:
            payload = parse_json_file(data_file)
        except (ValueError, json.JSONDecodeError):
            continue

        if is_control_demographics_payload(payload):
            continue
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            task_payloads.append(payload)

    if not task_payloads:
        raise FileNotFoundError(f"No control task JSON found in {study_result_dir}")

    # If duplicate task exports exist, prefer the longest payload.
    task_samples = max(task_payloads, key=len)
    practice_routes = group_jatos_samples(task_samples, round_number=0)
    test_routes = group_jatos_samples(task_samples, round_number=1)
    return practice_routes, test_routes


def extract_control_metadata(study_result_dir: Path) -> dict[str, str]:
    for data_file in sorted(study_result_dir.glob("comp-result_*/data.txt")):
        text = data_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text or text.startswith("[") or text.startswith('{"round"'):
            continue

        if text.startswith("{"):
            try:
                payload = parse_json_text(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                metadata = {key: str(value) for key, value in payload.items()}
                metadata.setdefault("source_file", str(data_file))
                metadata.setdefault("study_result_id", study_result_dir.name)
                metadata.setdefault("component_result_id", data_file.parent.name)
                return metadata
        else:
            participant_id = parse_metadata_id_from_text(text)
            if participant_id:
                return {
                    "subID": participant_id,
                    "study_result_id": study_result_dir.name,
                    "component_result_id": data_file.parent.name,
                    "source_file": str(data_file),
                }

    return {}


def canonicalize_patient_id(patient_id: str) -> str:
    cleaned = patient_id.strip().replace("\ufeff", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.rstrip(",")
    return cleaned.upper()


def parse_metadata_id_from_text(text: str) -> str | None:
    first_token = text.split(",")[0].strip()
    if not first_token:
        return None
    if first_token.lower() in PATIENT_TEXT_METADATA_REJECT:
        return None
    return canonicalize_patient_id(first_token)


def select_best_task_payload(study_result_dir: Path) -> tuple[list[dict], str] | tuple[None, None]:
    task_candidates: list[tuple[list[dict], Path]] = []
    for data_file in sorted(study_result_dir.glob("comp-result_*/data.txt")):
        text = data_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        if not (text.startswith("[") or text.startswith('{"round"')):
            continue
        try:
            payload = parse_json_text(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            task_candidates.append((payload, data_file))

    if not task_candidates:
        return None, None

    best_payload, best_file = max(task_candidates, key=lambda item: len(item[0]))
    return best_payload, str(best_file)


def parse_delimited_line(line: str) -> list[str]:
    parts = re.split(r"[\t,]+", line.strip())
    return [part.strip() for part in parts if part.strip()]


def load_octal_file(path: Path) -> list[np.ndarray]:
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("OptionGeneration") or line.startswith("Path Number"):
                continue
            parts = parse_delimited_line(line)
            if len(parts) < 4:
                continue
            try:
                path_number = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                time_ms = float(parts[3])
            except ValueError:
                continue
            grouped[path_number].append([time_ms, x, y])

    routes = []
    for path_number in sorted(grouped):
        route = np.asarray(grouped[path_number], dtype=float)
        if route.size == 0:
            continue
        route = route[np.argsort(route[:, 0])]
        routes.append(route)
    return routes


def classify_octal_duration(routes: list[np.ndarray]) -> float:
    if not routes:
        return math.nan
    start_time = min(route[0, 0] for route in routes if route.shape[0] > 0)
    end_time = max(route[-1, 0] for route in routes if route.shape[0] > 0)
    return (end_time - start_time) / 1000.0


def copy_and_transform_routes(
    routes: list[np.ndarray],
    *,
    x_center_raw: float,
    y_top_raw: float,
    x_scale: float,
    y_scale: float,
) -> list[np.ndarray]:
    transformed: list[np.ndarray] = []
    for route in routes:
        route_copy = np.asarray(route, dtype=float).copy()
        if route_copy.size and route_copy.shape[1] >= 3:
            route_copy[:, 1] = (route_copy[:, 1] - x_center_raw) * x_scale + X_CENTER
            route_copy[:, 2] = (route_copy[:, 2] - y_top_raw) * y_scale + TARGET_TOP_Y
        transformed.append(route_copy)
    return transformed


def estimate_practice_coordinate_calibration(participant: ParticipantRoutes) -> dict[str, object]:
    """Estimate a participant-specific task-frame transform from practice routes.

    The practice condition is a repeated straight movement between the two
    fixed red circles. In task code these circles are horizontally centred and
    placed at 15% and 90% of the canvas height, so the practice endpoints give
    a per-participant calibration for raw x/y frame, device, and browser scale.
    """
    endpoint_x: list[float] = []
    low_y: list[float] = []
    high_y: list[float] = []
    for route in participant.practice_routes:
        route_arr = np.asarray(route, dtype=float)
        if route_arr.shape[0] < 2 or route_arr.shape[1] < 3:
            continue
        start = route_arr[0, 1:3]
        end = route_arr[-1, 1:3]
        if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
            continue
        endpoint_x.extend([float(start[0]), float(end[0])])
        low_y.append(float(min(start[1], end[1])))
        high_y.append(float(max(start[1], end[1])))

    n_routes = len(low_y)
    method = "practice_run0"
    if n_routes >= PRACTICE_CALIBRATION_MIN_ROUTES:
        x_center_raw = float(np.median(endpoint_x))
        y_top_raw = float(np.median(low_y))
        y_bottom_raw = float(np.median(high_y))
    else:
        fallback = SOURCE_FRAME_FALLBACKS.get(participant.source_platform)
        if fallback is None:
            fallback = (X_CENTER, TARGET_TOP_Y, TARGET_BOTTOM_Y)
        x_center_raw, y_top_raw, y_bottom_raw = fallback
        method = f"source_fallback_{participant.source_platform}"

    y_span_raw = y_bottom_raw - y_top_raw
    valid = (
        math.isfinite(x_center_raw)
        and math.isfinite(y_top_raw)
        and math.isfinite(y_bottom_raw)
        and x_center_raw > 0
        and y_span_raw > 0
    )
    if not valid:
        x_center_raw = X_CENTER
        y_top_raw = TARGET_TOP_Y
        y_bottom_raw = TARGET_BOTTOM_Y
        y_span_raw = TARGET_VERTICAL_SPAN
        method = "identity_fallback"

    raw_width_estimate = 2.0 * x_center_raw
    x_scale = CANVAS_WIDTH / raw_width_estimate if raw_width_estimate > 0 else 1.0
    y_scale = TARGET_VERTICAL_SPAN / y_span_raw if y_span_raw > 0 else 1.0
    return {
        "coordinate_transform": "practice-calibrated affine to 800x1200",
        "coordinate_calibration_method": method,
        "calibration_practice_routes": n_routes,
        "calibration_x_center_raw": x_center_raw,
        "calibration_y_top_raw": y_top_raw,
        "calibration_y_bottom_raw": y_bottom_raw,
        "calibration_raw_width_estimate": raw_width_estimate,
        "calibration_raw_height_from_span": y_span_raw / 0.75 if y_span_raw > 0 else math.nan,
        "calibration_x_scale": x_scale,
        "calibration_y_scale": y_scale,
    }


def harmonise_participant_coordinate_frame(participant: ParticipantRoutes) -> ParticipantRoutes:
    """Map raw participant routes to the common 800x1200 active task frame."""
    calibration = estimate_practice_coordinate_calibration(participant)
    metadata = dict(participant.metadata)
    metadata.update({key: str(value) for key, value in calibration.items()})
    return ParticipantRoutes(
        group=participant.group,
        participant_id=participant.participant_id,
        source_id=participant.source_id,
        source_platform=participant.source_platform,
        practice_routes=copy_and_transform_routes(
            participant.practice_routes,
            x_center_raw=float(calibration["calibration_x_center_raw"]),
            y_top_raw=float(calibration["calibration_y_top_raw"]),
            x_scale=float(calibration["calibration_x_scale"]),
            y_scale=float(calibration["calibration_y_scale"]),
        ),
        test_routes=copy_and_transform_routes(
            participant.test_routes,
            x_center_raw=float(calibration["calibration_x_center_raw"]),
            y_top_raw=float(calibration["calibration_y_top_raw"]),
            x_scale=float(calibration["calibration_x_scale"]),
            y_scale=float(calibration["calibration_y_scale"]),
        ),
        metadata=metadata,
    )


def load_patients_from_octal(patient_octal_dir: Path) -> list[ParticipantRoutes]:
    grouped_files: dict[str, list[Path]] = defaultdict(list)
    for txt_path in sorted(patient_octal_dir.glob("*.txt")):
        if txt_path.name.startswith("."):
            continue
        participant_id = txt_path.name.split("_")[0]
        grouped_files[participant_id].append(txt_path)

    patients: list[ParticipantRoutes] = []
    for participant_id in sorted(grouped_files):
        practice_candidates: list[tuple[float, list[np.ndarray], Path]] = []
        test_candidates: list[tuple[float, list[np.ndarray], Path]] = []

        for txt_path in sorted(grouped_files[participant_id]):
            routes = load_octal_file(txt_path)
            duration_seconds = classify_octal_duration(routes)
            if math.isnan(duration_seconds):
                continue
            if 20 <= duration_seconds <= 45:
                practice_candidates.append((abs(duration_seconds - 30.0), routes, txt_path))
            if 90 <= duration_seconds <= 135:
                test_candidates.append((abs(duration_seconds - 120.0), routes, txt_path))

        practice_routes = min(practice_candidates, default=(None, [], None), key=lambda item: item[0])[1]
        test_routes = min(test_candidates, default=(None, [], None), key=lambda item: item[0])[1]
        kept_practice = min(practice_candidates, default=(None, [], None), key=lambda item: item[0])[2]
        kept_test = min(test_candidates, default=(None, [], None), key=lambda item: item[0])[2]

        if not practice_routes and not test_routes:
            continue

        patients.append(
            ParticipantRoutes(
                group="patient",
                participant_id=canonicalize_patient_id(participant_id),
                source_id=participant_id,
                source_platform="OCTAL",
                practice_routes=practice_routes,
                test_routes=test_routes,
                metadata={
                    "raw_source": "OCTAL",
                    "raw_practice_file": str(kept_practice) if kept_practice else "",
                    "raw_test_file": str(kept_test) if kept_test else "",
                },
            )
        )

    return patients


def load_patients_from_jatos(patient_jatos_dirs: list[Path]) -> list[ParticipantRoutes]:
    candidate_rows: list[dict[str, object]] = []

    for root_dir in patient_jatos_dirs:
        language = root_dir.name
        for study_dir in sorted(root_dir.glob("study_result_*")):
            payload, task_file = select_best_task_payload(study_dir)
            if payload is None:
                continue

            participant_id = None
            saw_invalid_placeholder_id = False
            for data_file in sorted(study_dir.glob("comp-result_*/data.txt")):
                text = data_file.read_text(encoding="utf-8", errors="ignore").strip()
                if not text or text.startswith("[") or text.startswith('{"round"'):
                    continue
                if text.startswith("{"):
                    try:
                        metadata_payload = parse_json_text(text)
                    except json.JSONDecodeError:
                        metadata_payload = None
                    if isinstance(metadata_payload, dict):
                        for key in ("subID", "participant_id", "participantId", "id"):
                            if key in metadata_payload and str(metadata_payload[key]).strip():
                                participant_id = canonicalize_patient_id(str(metadata_payload[key]))
                                break
                else:
                    first_token = text.split(",")[0].strip()
                    if first_token.lower() in PATIENT_TEXT_METADATA_REJECT:
                        saw_invalid_placeholder_id = True
                    participant_id = parse_metadata_id_from_text(text)
                if participant_id:
                    break

            # Do not keep placeholder "test" exports as anonymous study IDs.
            if saw_invalid_placeholder_id and not participant_id:
                continue
            if not participant_id:
                participant_id = canonicalize_patient_id(study_dir.name)

            practice_routes = group_jatos_samples(payload, round_number=0)
            test_routes = group_jatos_samples(payload, round_number=1)
            if not practice_routes and not test_routes:
                continue

            candidate_rows.append(
                {
                    "participant_id": participant_id,
                    "source_id": study_dir.name,
                    "source_platform": language,
                    "practice_routes": practice_routes,
                    "test_routes": test_routes,
                    "metadata": {
                        "raw_source": language,
                        "raw_task_file": task_file,
                    },
                    "quality_score": sum(route.shape[0] for route in practice_routes + test_routes),
                }
            )

    best_by_id: dict[str, dict[str, object]] = {}
    for row in candidate_rows:
        participant_id = str(row["participant_id"])
        if participant_id not in best_by_id or row["quality_score"] > best_by_id[participant_id]["quality_score"]:
            best_by_id[participant_id] = row

    patients = []
    for participant_id in sorted(best_by_id):
        row = best_by_id[participant_id]
        patients.append(
            ParticipantRoutes(
                group="patient",
                participant_id=participant_id,
                source_id=str(row["source_id"]),
                source_platform=str(row["source_platform"]),
                practice_routes=list(row["practice_routes"]),
                test_routes=list(row["test_routes"]),
                metadata=dict(row["metadata"]),
            )
        )

    return patients


ADDITIONAL_CONTROL_DEMOGRAPHICS_CSV = Path("control_demographics_additional.csv")
EXTRA_CONTROL_ATTEMPT_LOG: list[dict[str, object]] = []
# Analysis sample: reference participants tested before the data freeze (1 May 2026, UTC).
CONTROL_DATA_FREEZE_MS = 1777593600000


def control_attempt_timestamp_ms(study_dir: Path) -> int | None:
    """Session start time (ms since epoch) stored with the participant ID in the JATOS export."""
    for data_file in sorted(study_dir.glob("comp-result_*/data.txt")):
        text = data_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text or text.startswith("[") or text.startswith("{"):
            continue
        parts = re.split(r"[,\t]+", text)
        if len(parts) > 1 and parts[1].strip().isdigit():
            return int(parts[1].strip())
    return None


def normalise_extra_control_id(participant_id: str) -> str:
    """NH_01, NH_001 and nh_1 all map to NH_001."""
    match = re.match(r"^\s*(N[CH])[_-]?0*(\d+)\s*$", str(participant_id), flags=re.IGNORECASE)
    if not match:
        return str(participant_id).strip().upper()
    return f"{match.group(1).upper()}_{int(match.group(2)):03d}"


def load_additional_control_demographics(path: Path) -> dict[str, dict[str, str]]:
    """Demographics for NC_*/NH_* controls (subID, age, sex, hand, education_years, testing_date)."""
    if not path.exists():
        return {}
    return {
        normalise_extra_control_id(row["subID"]): {k: (v or "").strip() for k, v in row.items()}
        for row in read_csv_rows(path)
        if row.get("subID", "").strip()
    }


def load_controls(control_csv: Path, control_dir: Path) -> list[ParticipantRoutes]:
    rows = select_control_rows(read_csv_rows(control_csv))
    controls: list[ParticipantRoutes] = []
    selected_ids = set()
    selected_study_results = set()

    for row in rows:
        study_result_id = row["study_result_id"].strip()
        participant_id = row["subID"].strip()
        study_dir = control_dir / study_result_id
        practice_routes, test_routes = extract_control_routes(study_dir)
        controls.append(
            ParticipantRoutes(
                group="control",
                participant_id=participant_id,
                source_id=study_result_id,
                source_platform="JATOS",
                practice_routes=practice_routes,
                test_routes=test_routes,
                metadata=row,
            )
        )
        selected_ids.add(canonicalize_patient_id(participant_id))
        selected_study_results.add(study_result_id)

    # Add extra controls (NC_* and NH_* IDs) that are not in the curated demographics CSV.
    # Some participants restarted the task, leaving several study_result folders. Keep the
    # attempt with the most completed test paths (ties: more practice paths, then the later
    # attempt) and log the attempts that were skipped.
    extra_demographics = load_additional_control_demographics(ADDITIONAL_CONTROL_DEMOGRAPHICS_CSV)
    EXTRA_CONTROL_ATTEMPT_LOG.clear()
    attempts: dict[str, list[dict[str, object]]] = defaultdict(list)
    for study_dir in sorted(control_dir.glob("study_result_*")):
        if not study_dir.is_dir() or study_dir.name in selected_study_results:
            continue

        metadata = extract_control_metadata(study_dir)
        participant_id = metadata.get("subID", "").strip()
        canonical_id = canonicalize_patient_id(participant_id) if participant_id else ""
        if not canonical_id or not CONTROL_EXTRA_ID_RE.match(canonical_id):
            continue
        if canonical_id in selected_ids:
            continue

        started_ms = control_attempt_timestamp_ms(study_dir)
        if started_ms is not None and started_ms >= CONTROL_DATA_FREEZE_MS:
            EXTRA_CONTROL_ATTEMPT_LOG.append({
                "participant_id": canonical_id, "study_result_id": study_dir.name,
                "practice_paths": "", "test_paths": "", "selected": 0, "excluded_after_data_freeze": 1,
            })
            continue

        try:
            practice_routes, test_routes = extract_control_routes(study_dir)
        except FileNotFoundError:
            practice_routes, test_routes = [], []
        attempts[canonical_id].append({
            "participant_id": participant_id,
            "study_dir": study_dir,
            "metadata": metadata,
            "practice_routes": practice_routes,
            "test_routes": test_routes,
        })

    for canonical_id in sorted(attempts):
        candidates = attempts[canonical_id]
        best = max(
            candidates,
            key=lambda a: (
                len(a["test_routes"]),
                len(a["practice_routes"]),
                int(str(a["study_dir"].name).rsplit("_", 1)[-1]),
            ),
        )
        usable = len(best["test_routes"]) > 0
        for a in candidates:
            EXTRA_CONTROL_ATTEMPT_LOG.append({
                "participant_id": canonical_id,
                "study_result_id": a["study_dir"].name,
                "practice_paths": len(a["practice_routes"]),
                "test_paths": len(a["test_routes"]),
                "selected": int(usable and a is best),
                "excluded_after_data_freeze": 0,
            })
        if not usable:
            warnings.warn(f"Control {canonical_id} has no completed test paths in any attempt; excluded.")
            continue

        metadata = dict(best["metadata"])
        demo = extra_demographics.get(normalise_extra_control_id(canonical_id))
        if demo:
            for key in ("age", "sex", "hand", "education_years", "testing_date"):
                if demo.get(key, "") != "":
                    metadata[key] = demo[key]
        else:
            warnings.warn(f"No demographic record for extra control {canonical_id}.")

        study_dir = best["study_dir"]
        controls.append(
            ParticipantRoutes(
                group="control",
                participant_id=str(best["participant_id"]),
                source_id=study_dir.name,
                source_platform="JATOS",
                practice_routes=best["practice_routes"],
                test_routes=best["test_routes"],
                metadata=metadata,
            )
        )
        selected_ids.add(canonical_id)
        selected_study_results.add(study_dir.name)

    return [harmonise_participant_coordinate_frame(control) for control in controls]


def load_patients(patient_octal_dir: Path, patient_jatos_dirs: list[Path]) -> list[ParticipantRoutes]:
    patients = load_patients_from_octal(patient_octal_dir)
    patients.extend(load_patients_from_jatos(patient_jatos_dirs))
    return [harmonise_participant_coordinate_frame(patient) for patient in patients]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def resample_route(route: np.ndarray, np_points: int) -> np.ndarray | None:
    z = route[:, 1] + 1j * route[:, 2]
    if z.size < 2:
        return None

    s = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(z)))])
    unique_mask = np.concatenate([[True], np.diff(s) > 0])
    s = s[unique_mask]
    z = z[unique_mask]

    if s.size < 2 or s[-1] <= 0:
        return None

    grid = np.linspace(0.0, 1.0, np_points)
    normalized_s = s / s[-1]
    real_interp = np.interp(grid, normalized_s, z.real)
    imag_interp = np.interp(grid, normalized_s, z.imag)
    return real_interp + 1j * imag_interp


def build_route_feature_vector(
    route: np.ndarray,
    np_points: int,
    smooth_window: int,
    velocity_downsample: int,
) -> tuple[np.ndarray, int] | tuple[None, None]:
    z = resample_route(route, np_points=np_points)
    if z is None:
        return None, None

    real_part = np.real(z)
    imag_part = np.imag(z)

    vel_real = moving_average(np.diff(real_part), smooth_window)
    vel_imag = moving_average(np.diff(imag_part), smooth_window)
    acc_real = moving_average(np.diff(vel_real), smooth_window)
    acc_imag = moving_average(np.diff(vel_imag), smooth_window)

    vel_real = vel_real[::velocity_downsample]
    vel_imag = vel_imag[::velocity_downsample]
    acc_real = acc_real[::velocity_downsample]
    acc_imag = acc_imag[::velocity_downsample]

    feature_vector = np.concatenate(
        [
            real_part - X_CENTER,
            imag_part - Y_CENTER,
            vel_real,
            vel_imag,
            acc_real,
            acc_imag,
        ]
    )
    n_xy = real_part.size
    n_vel = vel_real.size
    n_acc = acc_real.size
    flippable_mask = np.concatenate(
        [
            np.ones(n_xy, dtype=bool),
            np.zeros(n_xy, dtype=bool),
            np.ones(n_vel, dtype=bool),
            np.zeros(n_vel, dtype=bool),
            np.ones(n_acc, dtype=bool),
            np.zeros(n_acc, dtype=bool),
        ]
    )
    feature_vector_flipped = feature_vector.copy()
    feature_vector_flipped[flippable_mask] *= -1
    return np.vstack([feature_vector, feature_vector_flipped]), n_xy


def safe_nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return math.nan
    return float(np.nanmean(array))


def compute_route_summary(routes: list[np.ndarray]) -> dict[str, float]:
    if not routes:
        return {
            "route_count": 0,
            "valid_route_count": 0,
            "session_duration_ms": math.nan,
            "total_draw_time_ms": math.nan,
            "mean_path_duration_ms": math.nan,
            "mean_interpath_interval_ms": math.nan,
            "total_distance_px": math.nan,
            "mean_path_length_px": math.nan,
            "mean_speed_px_per_s": math.nan,
        }

    starts = np.array([route[0, 0] for route in routes], dtype=float)
    ends = np.array([route[-1, 0] for route in routes], dtype=float)
    durations = ends - starts
    interpath = starts[1:] - ends[:-1] if len(routes) > 1 else np.array([], dtype=float)

    path_lengths = []
    for route in routes:
        if route.shape[0] < 2:
            path_lengths.append(0.0)
            continue
        xy = route[:, 1:3]
        path_lengths.append(float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))

    total_distance = float(np.sum(path_lengths))
    total_draw_time = float(np.sum(durations))
    mean_speed = (
        total_distance / (total_draw_time / 1000.0)
        if total_draw_time > 0
        else math.nan
    )

    return {
        "route_count": int(len(routes)),
        "valid_route_count": int(sum(route.shape[0] > 1 for route in routes)),
        "session_duration_ms": float(np.max(ends) - np.min(starts)),
        "total_draw_time_ms": total_draw_time,
        "mean_path_duration_ms": safe_nanmean(durations),
        "mean_interpath_interval_ms": safe_nanmean(interpath),
        "total_distance_px": total_distance,
        "mean_path_length_px": safe_nanmean(path_lengths),
        "mean_speed_px_per_s": mean_speed,
    }


def classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray | None:
    if distance_matrix.shape[0] < n_components:
        return None

    n_items = distance_matrix.shape[0]
    squared = distance_matrix ** 2
    centering = np.eye(n_items) - np.ones((n_items, n_items)) / n_items
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive = eigenvalues > 0
    if np.count_nonzero(positive) < n_components:
        return None

    selected_values = eigenvalues[positive][:n_components]
    selected_vectors = eigenvectors[:, positive][:, :n_components]
    return selected_vectors * np.sqrt(selected_values)


def compute_uniqueness_metrics(
    participants: list[ParticipantRoutes],
    np_points: int,
    smooth_window: int,
    velocity_downsample: int,
) -> dict[str, dict[str, float]]:
    route_records: list[dict[str, object]] = []

    for participant in participants:
        for route_index, route in enumerate(participant.test_routes):
            feature_pair, _ = build_route_feature_vector(
                route,
                np_points=np_points,
                smooth_window=smooth_window,
                velocity_downsample=velocity_downsample,
            )
            if feature_pair is None:
                continue
            route_records.append(
                {
                    "participant_id": participant.participant_id,
                    "group": participant.group,
                    "route_index": route_index,
                    "feature": feature_pair[0],
                    "feature_flipped": feature_pair[1],
                }
            )

    if not route_records:
        return {}

    features = np.vstack([record["feature"] for record in route_records])
    features_flipped = np.vstack([record["feature_flipped"] for record in route_records])
    distance_matrix = squareform(pdist(features, metric="euclidean"))
    flip_distance_matrix = cdist(features_flipped, features, metric="euclidean")

    by_participant: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(route_records):
        by_participant[str(record["participant_id"])].append(idx)

    results: dict[str, dict[str, float]] = {}
    for participant in participants:
        indices = by_participant.get(participant.participant_id, [])
        if not indices:
            results[participant.participant_id] = {
                "test_uniqueness_mean": math.nan,
                "test_log_uniqueness_mean": math.nan,
                "test_mds_hull_area": math.nan,
                "test_uniqueness_route_count": 0,
            }
            continue

        uniqueness_values = []
        for idx in indices:
            d_orig = distance_matrix[:, idx].astype(float).copy()
            d_flip = flip_distance_matrix[:, idx].astype(float).copy()
            d_orig[idx] = np.nan
            d_flip[idx] = np.nan
            combined = np.concatenate([d_orig, d_flip])
            uniqueness_values.append(float(np.nanmin(combined)))

        subject_distances = distance_matrix[np.ix_(indices, indices)]
        coords = classical_mds(subject_distances, n_components=2)
        hull_area = math.nan
        if coords is not None and coords.shape[0] >= 3:
            try:
                hull_area = float(ConvexHull(coords).volume)
            except QhullError:
                hull_area = math.nan

        positive_uniqueness = [value for value in uniqueness_values if value > 0]
        results[participant.participant_id] = {
            "test_uniqueness_mean": safe_nanmean(uniqueness_values),
            "test_log_uniqueness_mean": safe_nanmean(np.log(positive_uniqueness)),
            "test_mds_hull_area": hull_area,
            "test_uniqueness_route_count": len(uniqueness_values),
        }

    return results


def compare_metric(values_control: np.ndarray, values_patient: np.ndarray) -> dict[str, float]:
    values_control = values_control[~np.isnan(values_control)]
    values_patient = values_patient[~np.isnan(values_patient)]

    result = {
        "n_control": int(values_control.size),
        "n_patient": int(values_patient.size),
        "mean_control": float(np.mean(values_control)) if values_control.size else math.nan,
        "mean_patient": float(np.mean(values_patient)) if values_patient.size else math.nan,
        "median_control": float(np.median(values_control)) if values_control.size else math.nan,
        "median_patient": float(np.median(values_patient)) if values_patient.size else math.nan,
        "ttest_p": math.nan,
        "mannwhitney_p": math.nan,
    }

    if values_control.size >= 2 and values_patient.size >= 2:
        result["ttest_p"] = float(
            ttest_ind(values_control, values_patient, equal_var=False, nan_policy="omit").pvalue
        )
        result["mannwhitney_p"] = float(
            mannwhitneyu(values_control, values_patient, alternative="two-sided").pvalue
        )

    return result


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text in ("", "NA", "NaN", "nan"):
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def save_raincloud_plot(
    path: Path,
    participant_rows: list[dict[str, object]],
    metric: str,
) -> None:
    groups = ["control", "patient"]
    labels = ["Controls", "Patients"]
    colors = ["#5A7D9A", "#C26D52"]
    data = []
    for group in groups:
        values = np.asarray(
            [
                float(row.get(metric, math.nan))
                for row in participant_rows
                if row["group"] == group
            ],
            dtype=float,
        )
        values = values[~np.isnan(values)]
        data.append(values)

    if not any(values.size for values in data):
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = np.arange(len(groups), dtype=float)

    parts = ax.violinplot(
        data,
        positions=positions - 0.12,
        widths=0.5,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[idx])
        body.set_edgecolor("none")
        body.set_alpha(0.28)

    box = ax.boxplot(
        data,
        positions=positions + 0.06,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
    )
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor("white")
        patch.set_edgecolor(colors[idx])
        patch.set_linewidth(1.4)
    for element in ("whiskers", "caps", "medians"):
        for artist in box[element]:
            artist.set_color("#333333")
            artist.set_linewidth(1.1)

    rng = np.random.default_rng(42)
    for idx, values in enumerate(data):
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.06, 0.06, size=values.size)
        ax.scatter(
            np.full(values.size, positions[idx] + 0.22) + jitter,
            values,
            s=18,
            alpha=0.55,
            color=colors[idx],
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(metric.replace("_", " "))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_group_plots(output_dir: Path, participant_rows: list[dict[str, object]]) -> None:
    for metric in PLOT_METRICS:
        save_raincloud_plot(output_dir / f"{metric}_raincloud.png", participant_rows, metric)


def maybe_write_route_points(path: Path, participants: list[ParticipantRoutes]) -> None:
    rows: list[dict[str, object]] = []
    for participant in participants:
        for condition_name, routes in (
            ("practice", participant.practice_routes),
            ("test", participant.test_routes),
        ):
            for route_index, route in enumerate(routes):
                for sample_index, sample in enumerate(route):
                    rows.append(
                        {
                            "group": participant.group,
                            "participant_id": participant.participant_id,
                            "source_id": participant.source_id,
                            "condition": condition_name,
                            "route_index": route_index,
                            "sample_index": sample_index,
                            "time_ms": float(sample[0]),
                            "x": float(sample[1]),
                            "y": float(sample[2]),
                        }
                    )

    write_csv(
        path,
        rows,
        [
            "group",
            "participant_id",
            "source_id",
            "condition",
            "route_index",
            "sample_index",
            "time_ms",
            "x",
            "y",
        ],
    )


def write_coordinate_calibration(path: Path, participants: list[ParticipantRoutes]) -> None:
    rows = []
    for participant in participants:
        row = {
            "group": participant.group,
            "participant_id": participant.participant_id,
            "source_id": participant.source_id,
            "source_platform": participant.source_platform,
            "raw_source": participant.metadata.get("raw_source", ""),
            "coordinate_transform": participant.metadata.get("coordinate_transform", ""),
            "coordinate_calibration_method": participant.metadata.get("coordinate_calibration_method", ""),
            "calibration_practice_routes": participant.metadata.get("calibration_practice_routes", ""),
            "calibration_x_center_raw": participant.metadata.get("calibration_x_center_raw", ""),
            "calibration_y_top_raw": participant.metadata.get("calibration_y_top_raw", ""),
            "calibration_y_bottom_raw": participant.metadata.get("calibration_y_bottom_raw", ""),
            "calibration_raw_width_estimate": participant.metadata.get("calibration_raw_width_estimate", ""),
            "calibration_raw_height_from_span": participant.metadata.get("calibration_raw_height_from_span", ""),
            "calibration_x_scale": participant.metadata.get("calibration_x_scale", ""),
            "calibration_y_scale": participant.metadata.get("calibration_y_scale", ""),
        }
        rows.append(row)

    write_csv(
        path,
        rows,
        [
            "group",
            "participant_id",
            "source_id",
            "source_platform",
            "raw_source",
            "coordinate_transform",
            "coordinate_calibration_method",
            "calibration_practice_routes",
            "calibration_x_center_raw",
            "calibration_y_top_raw",
            "calibration_y_bottom_raw",
            "calibration_raw_width_estimate",
            "calibration_raw_height_from_span",
            "calibration_x_scale",
            "calibration_y_scale",
        ],
    )


def build_participant_metric_rows(
    participants: list[ParticipantRoutes],
    uniqueness_metrics: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for participant in participants:
        practice_summary = compute_route_summary(participant.practice_routes)
        test_summary = compute_route_summary(participant.test_routes)
        uniqueness_summary = uniqueness_metrics.get(participant.participant_id, {})

        row: dict[str, object] = {
            "group": participant.group,
            "participant_id": participant.participant_id,
            "source_id": participant.source_id,
            "source_platform": participant.source_platform,
            "raw_source": participant.metadata.get("raw_source", ""),
            "raw_practice_file": participant.metadata.get("raw_practice_file", ""),
            "raw_test_file": participant.metadata.get("raw_test_file", ""),
            "raw_task_file": participant.metadata.get("raw_task_file", ""),
            "coordinate_transform": participant.metadata.get("coordinate_transform", ""),
            "coordinate_calibration_method": participant.metadata.get("coordinate_calibration_method", ""),
            "calibration_practice_routes": participant.metadata.get("calibration_practice_routes", ""),
            "calibration_x_center_raw": participant.metadata.get("calibration_x_center_raw", ""),
            "calibration_y_top_raw": participant.metadata.get("calibration_y_top_raw", ""),
            "calibration_y_bottom_raw": participant.metadata.get("calibration_y_bottom_raw", ""),
            "calibration_raw_width_estimate": participant.metadata.get("calibration_raw_width_estimate", ""),
            "calibration_raw_height_from_span": participant.metadata.get("calibration_raw_height_from_span", ""),
            "calibration_x_scale": participant.metadata.get("calibration_x_scale", ""),
            "calibration_y_scale": participant.metadata.get("calibration_y_scale", ""),
            "has_practice": int(bool(participant.practice_routes)),
            "has_test": int(bool(participant.test_routes)),
        }

        for key, value in practice_summary.items():
            row[f"practice_{key}"] = value
        for key, value in test_summary.items():
            row[f"test_{key}"] = value
        row.update(uniqueness_summary)
        test_duration = float(row.get("test_session_duration_ms", math.nan))
        row["test_expected_duration_ms"] = EXPECTED_TEST_DURATION_MS
        row["test_short_threshold_ms"] = SHORT_TEST_THRESHOLD_MS
        row["test_duration_gap_ms"] = (
            EXPECTED_TEST_DURATION_MS - test_duration if not math.isnan(test_duration) else math.nan
        )
        row["test_duration_gap_s"] = (
            row["test_duration_gap_ms"] / 1000.0
            if not math.isnan(row["test_duration_gap_ms"])
            else math.nan
        )
        row["flag_test_short_under_115s"] = int(
            (not math.isnan(test_duration)) and (test_duration < SHORT_TEST_THRESHOLD_MS)
        )

        if participant.group == "control":
            for metadata_key in (
                "study_result_id",
                "component_result_id",
                "age",
                "sex",
                "hand",
                "timestamp_utc",
                "timestamp_local",
            ):
                row[metadata_key] = participant.metadata.get(metadata_key, "")
        rows.append(row)

    return rows


def build_group_summary_rows(participant_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric_metrics = [
        "practice_route_count",
        "practice_valid_route_count",
        "practice_session_duration_ms",
        "practice_total_draw_time_ms",
        "practice_mean_path_duration_ms",
        "practice_mean_interpath_interval_ms",
        "practice_total_distance_px",
        "practice_mean_path_length_px",
        "practice_mean_speed_px_per_s",
        "test_route_count",
        "test_valid_route_count",
        "test_session_duration_ms",
        "test_total_draw_time_ms",
        "test_mean_path_duration_ms",
        "test_mean_interpath_interval_ms",
        "test_total_distance_px",
        "test_mean_path_length_px",
        "test_mean_speed_px_per_s",
        "test_uniqueness_mean",
        "test_log_uniqueness_mean",
        "test_mds_hull_area",
        "test_uniqueness_route_count",
    ]

    controls = [row for row in participant_rows if row["group"] == "control"]
    patients = [row for row in participant_rows if row["group"] == "patient"]
    summary_rows: list[dict[str, object]] = []

    for metric in numeric_metrics:
        values_control = np.asarray([float(row.get(metric, math.nan)) for row in controls], dtype=float)
        values_patient = np.asarray([float(row.get(metric, math.nan)) for row in patients], dtype=float)
        result = compare_metric(values_control, values_patient)
        result["metric"] = metric
        summary_rows.append(result)

    return summary_rows


def build_short_test_rows(participant_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in participant_rows:
        if int(row.get("flag_test_short_under_115s", 0)) != 1:
            continue
        rows.append(
            {
                "group": row["group"],
                "participant_id": row["participant_id"],
                "source_id": row["source_id"],
                "source_platform": row["source_platform"],
                "test_session_duration_ms": row.get("test_session_duration_ms", math.nan),
                "test_session_duration_s": (
                    float(row["test_session_duration_ms"]) / 1000.0
                    if str(row.get("test_session_duration_ms", "")) not in ("", "nan", "NaN")
                    else math.nan
                ),
                "test_duration_gap_s": row.get("test_duration_gap_s", math.nan),
                "test_route_count": row.get("test_route_count", math.nan),
                "raw_source": row.get("raw_source", ""),
                "raw_test_file": row.get("raw_test_file", ""),
                "raw_task_file": row.get("raw_task_file", ""),
            }
        )
    return rows


def load_clinical_rows(path: Path) -> list[dict[str, object]]:
    rows = read_csv_rows(path)
    enriched = []
    for row in rows:
        enriched_row: dict[str, object] = dict(row)
        enriched_row["ID_subject"] = row.get("ID_subject", "").strip().upper()
        map_terms = [to_float(row.get(key, "")) for key in ["BNSS_1", "BNSS_2", "BNSS_3", "BNSS_5", "BNSS_6", "BNSS_7", "BNSS_8"]]
        exp_terms = [to_float(row.get(key, "")) for key in ["BNSS_9", "BNSS_10", "BNSS_11", "BNSS_12", "BNSS_13"]]
        ami_scored = {}
        for i in range(1, 19):
            raw_value = to_float(row.get(f"AMI_{i}", ""))
            ami_scored[i] = raw_value if not math.isnan(raw_value) else math.nan  # CHANSS database already apathy-scored; no reversal

        enriched_row["MAP"] = float(np.nansum(map_terms)) if not all(math.isnan(x) for x in map_terms) else math.nan
        enriched_row["EXP"] = float(np.nansum(exp_terms)) if not all(math.isnan(x) for x in exp_terms) else math.nan
        behavioural_items = [ami_scored[i] for i in [5, 9, 10, 11, 12, 15]]
        social_items = [ami_scored[i] for i in [2, 3, 4, 8, 14, 17]]
        emotional_items = [ami_scored[i] for i in [1, 6, 7, 13, 16, 18]]
        enriched_row["AMI_behavioural"] = (
            float(np.nanmean(behavioural_items))
            if not all(math.isnan(x) for x in behavioural_items)
            else math.nan
        )
        enriched_row["AMI_social"] = (
            float(np.nanmean(social_items))
            if not all(math.isnan(x) for x in social_items)
            else math.nan
        )
        enriched_row["AMI_emotional"] = (
            float(np.nanmean(emotional_items))
            if not all(math.isnan(x) for x in emotional_items)
            else math.nan
        )

        # Calgary Depression Scale (CDSS): sum of CDS_1 .. CDS_9 (0-3 each,
        # range 0-27).  Missing if all nine items are missing.
        cds_items = [to_float(row.get(f"CDS_{i}", "")) for i in range(1, 10)]
        enriched_row["CDS_total"] = (
            float(np.nansum(cds_items))
            if not all(math.isnan(x) for x in cds_items)
            else math.nan
        )

        # PANSS Marder positive factor: P1 (delusions), P3 (hallucinations),
        # PG9 (unusual thought content), PG10 (disorientation).  NB: the user
        # spec listed g9/g10; the CSV uses the `pg` naming.
        marder_items = [
            to_float(row.get(key, ""))
            for key in ("PANSS_p1", "PANSS_p3", "PANSS_pg9", "PANSS_pg10")
        ]
        enriched_row["PANSS_marder_positive"] = (
            float(np.nansum(marder_items))
            if not all(math.isnan(x) for x in marder_items)
            else math.nan
        )

        enriched.append(enriched_row)
    return enriched


def load_clinical_id_corrections(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = read_csv_rows(path)
    corrections = {}
    for row in rows:
        ogt_id = str(row.get("participant_id", "")).strip().upper()
        corrected_id = str(row.get("participant_id_corrected", "")).strip().upper()
        if ogt_id and corrected_id:
            corrections[ogt_id] = corrected_id
    return corrections


def build_clinical_match_rows(
    participant_rows: list[dict[str, object]],
    clinical_rows: list[dict[str, object]],
    id_corrections: dict[str, str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    clinical_by_id = {str(row["ID_subject"]).upper(): row for row in clinical_rows if str(row["ID_subject"]).strip()}
    id_corrections = id_corrections or {}
    matched_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []

    for row in participant_rows:
        if row["group"] != "patient":
            continue
        participant_id = str(row["participant_id"]).strip().upper()
        clinical_lookup_id = id_corrections.get(participant_id, participant_id)
        clinical = clinical_by_id.get(clinical_lookup_id)
        if clinical is None:
            missing_rows.append(
                {
                    "participant_id": row["participant_id"],
                    "participant_id_corrected": id_corrections.get(participant_id, ""),
                    "source_id": row["source_id"],
                    "source_platform": row["source_platform"],
                    "raw_source": row.get("raw_source", ""),
                    "raw_test_file": row.get("raw_test_file", ""),
                    "raw_task_file": row.get("raw_task_file", ""),
                }
            )
            continue

        merged = dict(row)
        merged["clinical_lookup_id"] = clinical_lookup_id
        for field in CLINICAL_CORRELATION_FIELDS:
            merged[field] = clinical.get(field, math.nan)
        matched_rows.append(merged)

    return matched_rows, missing_rows


def build_clinical_correlation_rows(
    matched_rows: list[dict[str, object]],
    uniqueness_field: str = "test_uniqueness_mean",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    uniqueness_values = np.asarray([to_float(row.get(uniqueness_field, math.nan)) for row in matched_rows], dtype=float)

    for field in CLINICAL_CORRELATION_FIELDS:
        clinical_values = np.asarray([to_float(row.get(field, math.nan)) for row in matched_rows], dtype=float)
        valid = (~np.isnan(uniqueness_values)) & (~np.isnan(clinical_values))
        rho = math.nan
        p_value = math.nan
        if np.count_nonzero(valid) >= 3:
            rho, p_value = spearmanr(uniqueness_values[valid], clinical_values[valid])
        rows.append(
            {
                "ogt_metric": uniqueness_field,
                "clinical_metric": field,
                "n": int(np.count_nonzero(valid)),
                "spearman_rho": float(rho) if not math.isnan(rho) else math.nan,
                "spearman_p": float(p_value) if not math.isnan(p_value) else math.nan,
            }
        )

    return rows


def save_clinical_correlation_heatmap(path: Path, correlation_rows: list[dict[str, object]]) -> None:
    metrics = [row["clinical_metric"] for row in correlation_rows]
    rho_values = np.asarray([[to_float(row["spearman_rho"]) for row in correlation_rows]], dtype=float)
    p_values = [to_float(row["spearman_p"]) for row in correlation_rows]
    n_values = [int(row["n"]) for row in correlation_rows]

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 0.9), 2.8))
    im = ax.imshow(rho_values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_yticks([0])
    ax.set_yticklabels(["Uniqueness"])
    ax.set_title("Spearman correlations: OGT uniqueness vs clinical measures")

    for idx, row in enumerate(correlation_rows):
        rho = to_float(row["spearman_rho"])
        p_value = p_values[idx]
        n = n_values[idx]
        label = f"rho={rho:.2f}\np={p_value:.3g}\nn={n}" if not math.isnan(rho) else f"n={n}"
        ax.text(idx, 0, label, ha="center", va="center", color="black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Spearman rho")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    controls = load_controls(args.control_csv, args.control_dir)
    patients = load_patients(args.patient_octal_dir, args.patient_jatos_dirs)
    participants = controls + patients

    if args.uniqueness_reference == "pooled":
        uniqueness_metrics = compute_uniqueness_metrics(
            participants,
            np_points=args.np_points,
            smooth_window=args.smooth_window,
            velocity_downsample=args.velocity_downsample,
        )
    else:
        uniqueness_metrics = {}
        for group_name in ("control", "patient"):
            group_participants = [p for p in participants if p.group == group_name]
            uniqueness_metrics.update(
                compute_uniqueness_metrics(
                    group_participants,
                    np_points=args.np_points,
                    smooth_window=args.smooth_window,
                    velocity_downsample=args.velocity_downsample,
                )
            )

    participant_rows = build_participant_metric_rows(participants, uniqueness_metrics)
    group_rows = build_group_summary_rows(participant_rows)
    short_test_rows = build_short_test_rows(participant_rows)
    clinical_rows = load_clinical_rows(args.clinical_csv) if args.clinical_csv.exists() else []
    clinical_id_corrections = load_clinical_id_corrections(args.clinical_id_corrections_csv)
    matched_clinical_rows, missing_clinical_rows = build_clinical_match_rows(
        participant_rows,
        clinical_rows,
        clinical_id_corrections,
    )
    clinical_correlation_rows = build_clinical_correlation_rows(matched_clinical_rows) if clinical_rows else []

    participant_fieldnames = list(participant_rows[0].keys()) if participant_rows else []
    group_fieldnames = list(group_rows[0].keys()) if group_rows else []
    short_test_fieldnames = list(short_test_rows[0].keys()) if short_test_rows else [
        "group",
        "participant_id",
        "source_id",
        "source_platform",
        "test_session_duration_ms",
        "test_session_duration_s",
        "test_duration_gap_s",
        "test_route_count",
        "raw_source",
        "raw_test_file",
        "raw_task_file",
    ]
    missing_clinical_fieldnames = list(missing_clinical_rows[0].keys()) if missing_clinical_rows else [
        "participant_id",
        "source_id",
        "source_platform",
        "raw_source",
        "raw_test_file",
        "raw_task_file",
    ]
    clinical_correlation_fieldnames = list(clinical_correlation_rows[0].keys()) if clinical_correlation_rows else [
        "ogt_metric",
        "clinical_metric",
        "n",
        "spearman_rho",
        "spearman_p",
    ]

    write_csv(args.output_dir / "ogt_participant_metrics.csv", participant_rows, participant_fieldnames)
    write_csv(args.output_dir / "ogt_group_comparison_summary.csv", group_rows, group_fieldnames)
    write_csv(args.output_dir / "ogt_short_test_sessions_under_115s.csv", short_test_rows, short_test_fieldnames)
    write_csv(args.output_dir / "ogt_patients_missing_clinical_data.csv", missing_clinical_rows, missing_clinical_fieldnames)
    write_csv(args.output_dir / "ogt_uniqueness_clinical_correlations.csv", clinical_correlation_rows, clinical_correlation_fieldnames)
    write_coordinate_calibration(args.output_dir / "ogt_coordinate_calibration.csv", participants)
    save_group_plots(args.output_dir / "figures", participant_rows)
    if clinical_correlation_rows:
        save_clinical_correlation_heatmap(
            args.output_dir / "figures" / "uniqueness_clinical_correlation_heatmap.png",
            clinical_correlation_rows,
        )

    if args.write_route_points:
        maybe_write_route_points(args.output_dir / "ogt_route_points.csv", participants)

    n_controls = sum(row["group"] == "control" for row in participant_rows)
    n_patients = sum(row["group"] == "patient" for row in participant_rows)
    print(f"Wrote participant metrics for {n_controls} controls and {n_patients} patients.")
    if clinical_rows:
        print(
            "Matched "
            f"{len(matched_clinical_rows)} patients to clinical data; "
            f"{len(missing_clinical_rows)} patients had no clinical match."
        )
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
