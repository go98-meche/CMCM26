"""Load the two Excel maps selected in scenario_config.py.

Supports either two separate .xlsx files or two worksheets in one workbook.
The numeric cell values are preserved as terrain codes.
"""

import numpy as np
import openpyxl

from config import (
    EMPTY_AS,
    EXCEL_TO_TERRAIN,
    get_scenario_map_sources,
    VENUE_TYPE,
    CONCERT_EVENT_TYPE,
    OBSTACLE_CASE,
    LED_ENABLED,
    MAP_STORAGE_MODE,
    MAP_WORKBOOK_PATH,
    BASE_DIR,
)


def _read_worksheet_matrix(path, sheet_name=None):
    workbook = openpyxl.load_workbook(
        path,
        data_only=True,
        read_only=True
    )

    try:
        if sheet_name is None:
            worksheet = workbook.active
        else:
            if sheet_name not in workbook.sheetnames:
                raise ValueError(
                    f"Worksheet {sheet_name!r} not found in {path}. "
                    f"Available sheets: {workbook.sheetnames}"
                )

            worksheet = workbook[sheet_name]

        values = [
            list(row)
            for row in worksheet.iter_rows(values_only=True)
        ]

    finally:
        workbook.close()

    # Remove completely empty trailing rows and columns.
    while values and all(value is None for value in values[-1]):
        values.pop()

    if not values:
        raise ValueError(
            f"No non-empty map cells found in {path}."
        )

    last_col = max(
        (
            c
            for row in values
            for c, value in enumerate(row)
            if value is not None
        ),
        default=-1
    )

    values = [
        row[:last_col + 1]
        for row in values
    ]

    terrain = np.full(
        (len(values), last_col + 1),
        EMPTY_AS,
        dtype=np.int16
    )

    unknown = {}

    for r, row in enumerate(values):

        for c, value in enumerate(row):

            if value is None or (
                isinstance(value, str) and not value.strip()
            ):
                continue

            if isinstance(value, bool):
                unknown[str(value)] = (
                    unknown.get(str(value), 0) + 1
                )
                continue

            try:
                number = float(value)

                if not number.is_integer():
                    raise ValueError

                code = int(number)

            except (TypeError, ValueError):
                unknown[str(value)] = (
                    unknown.get(str(value), 0) + 1
                )
                continue

            if code not in EXCEL_TO_TERRAIN:
                unknown[code] = unknown.get(code, 0) + 1
                continue

            terrain[r, c] = EXCEL_TO_TERRAIN[code]

    if unknown:
        raise ValueError(
            f"Unknown cell values in {path}: {unknown}. "
            f"Allowed values: {sorted(EXCEL_TO_TERRAIN)}"
        )

    return terrain


def load_scenario_terrain():
    """Return {1: floor1_matrix, 2: floor2_matrix}."""

    sources = get_scenario_map_sources(
        venue_type=VENUE_TYPE,
        concert_event_type=CONCERT_EVENT_TYPE,
        obstacle_case=OBSTACLE_CASE,
        storage_mode=MAP_STORAGE_MODE,
        workbook_path=MAP_WORKBOOK_PATH,
        base_dir=BASE_DIR,
    )

    terrain = {}

    for floor in (1, 2):

        source = sources[floor]
        path = source["path"]
        sheet_name = source["sheet_name"]

        if not path.exists():
            raise FileNotFoundError(
                f"Selected scenario map for Floor {floor} "
                f"was not found: {path}\n"
                "Check scenario_config.py, or set "
                "MAP_STORAGE_MODE='sheets' and provide "
                "MAP_WORKBOOK_PATH."
            )

        terrain[floor] = _read_worksheet_matrix(
            path,
            sheet_name=sheet_name
        )

    print("\nSelected evacuation scenario:")
    print(f"Venue: {VENUE_TYPE}")

    if VENUE_TYPE == "concert":
        print(f"Concert type: {CONCERT_EVENT_TYPE}")

        if CONCERT_EVENT_TYPE == "emergency":
            print(f"Obstacle case: {OBSTACLE_CASE}")

    print(f"LED enabled: {LED_ENABLED}")
    print(f"Floor 1 map shape: {terrain[1].shape}")
    print(f"Floor 2 map shape: {terrain[2].shape}")

    return terrain