"""
Identify floor-specific exits and pair the two-floor stair cells.
"""

import numpy as np

from config import *


# ============================================================
# 1. Find connected components
# ============================================================

def _find_connected_components(
    mask
):
    """
    Find connected components in a Boolean mask using
    the original Moore neighborhood.

    Returns:
        list of components
    """

    rows, cols = mask.shape

    visited = np.zeros(
        mask.shape,
        dtype=bool
    )

    components = []

    for r in range(rows):

        for c in range(cols):

            if not mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]

            visited[r, c] = True

            component = []

            while stack:

                cr, cc = stack.pop()

                component.append(
                    (cr, cc)
                )

                for dr, dc in MOORE_OFFSETS:

                    nr = cr + dr
                    nc = cc + dc

                    if not (
                        0 <= nr < rows
                        and 0 <= nc < cols
                    ):
                        continue

                    if (
                        mask[nr, nc]
                        and not visited[nr, nc]
                    ):

                        visited[nr, nc] = True

                        stack.append(
                            (nr, nc)
                        )

            components.append(component)

    return components


# ============================================================
# 2. Pair stair cells between floors
# ============================================================

def _pair_stair_cells(
    terrain
):
    """
    Pair each Floor-2 stair cell with its corresponding Floor-1 cell.

    Pairing is performed separately for code 6 and code 7.

    Default rule:
        Sort both floors' cells by (row, column), then pair
        cells with the same sorted index.

    Returns:
        {
            (2, upper_row, upper_col):
                (1, lower_row, lower_col)
        }
    """

    stair_pairs = {}

    for stair_code in sorted(STAIR_CODES):

        floor1_cells = [
            tuple(map(int, p))
            for p in np.argwhere(
                terrain[1] == stair_code
            )
        ]

        floor2_cells = [
            tuple(map(int, p))
            for p in np.argwhere(
                terrain[2] == stair_code
            )
        ]

        floor1_cells.sort()
        floor2_cells.sort()

        if len(floor1_cells) != len(floor2_cells):

            raise ValueError(
                f"Stair code {stair_code} has different "
                f"numbers of cells on the two floors."
            )

        if len(floor1_cells) != 2:

            raise ValueError(
                f"Expected exactly two stair cells with "
                f"code {stair_code} on each floor."
            )

        for upper_position, lower_position in zip(
            floor2_cells,
            floor1_cells
        ):

            key = (
                2,
                upper_position[0],
                upper_position[1]
            )

            value = (
                1,
                lower_position[0],
                lower_position[1]
            )

            stair_pairs[key] = value

    print("\nStair pairing:")

    for upper, lower in stair_pairs.items():

        print(
            f"Floor 2 {upper[1:]} "
            f"-> Floor 1 {lower[1:]}"
        )

    return stair_pairs


# ============================================================
# 3. Identify all floor-specific destinations
# ============================================================

def identify_exits(
    terrain
):
    """
    Identify target destinations on both floors.

    Floor 1:
        Code 4 = ground-floor exit.

    Floor 2:
        Code 5 = upper-floor evacuation exit.
        Code 6 = stairway target 6.
        Code 7 = stairway target 7.

    Returns
    -------
    exits : list
        Each target has:
            id
            floor
            code
            kind
            cells

    exit_map : dict
        exit_map[floor] is a 2D array.
        Each target cell stores its target ID.
        Non-target cells store -1.

    stair_pairs : dict
        Maps a Floor-2 stair cell to its corresponding
        Floor-1 landing cell.
    """

    if set(terrain.keys()) != set(FLOORS):

        raise ValueError(
            f"Terrain must contain floors {FLOORS}."
        )

    exits = []

    exit_map = {
        floor: np.full(
            terrain[floor].shape,
            -1,
            dtype=int
        )
        for floor in FLOORS
    }

    next_target_id = 0

    # ========================================================
    # Floor 1: ground-floor exits
    # ========================================================

    floor1 = terrain[1]

    ground_components = _find_connected_components(
        floor1 == GROUND_EXIT
    )

    for component in ground_components:

        target_id = next_target_id

        next_target_id += 1

        component = [
            (int(r), int(c))
            for r, c in component
        ]

        for r, c in component:

            exit_map[1][r, c] = target_id

        exits.append({
            "id": target_id,
            "floor": 1,
            "code": GROUND_EXIT,
            "kind": "ground_exit",
            "cells": component
        })

    # ========================================================
    # Floor 2: upper-floor exits
    # ========================================================

    floor2 = terrain[2]

    upper_components = _find_connected_components(
        floor2 == UPPER_EXIT
    )

    for component in upper_components:

        target_id = next_target_id

        next_target_id += 1

        component = [
            (int(r), int(c))
            for r, c in component
        ]

        for r, c in component:

            exit_map[2][r, c] = target_id

        exits.append({
            "id": target_id,
            "floor": 2,
            "code": UPPER_EXIT,
            "kind": "upper_exit",
            "cells": component
        })

    # ========================================================
    # Floor 2: stairway targets
    # ========================================================

    for stair_code in sorted(STAIR_CODES):

        cells = [
            (int(r), int(c))
            for r, c in np.argwhere(
                floor2 == stair_code
            )
        ]

        target_id = next_target_id

        next_target_id += 1

        for r, c in cells:

            exit_map[2][r, c] = target_id

        exits.append({
            "id": target_id,
            "floor": 2,
            "code": stair_code,
            "kind": "stair",
            "cells": cells
        })

    # ========================================================
    # Pair stair cells
    # ========================================================

    stair_pairs = _pair_stair_cells(
        terrain
    )

    if not any(
        item["floor"] == 1
        and item["kind"] == "ground_exit"
        for item in exits
    ):

        raise ValueError(
            "No ground-floor exits detected."
        )

    if not any(
        item["floor"] == 2
        and item["kind"] == "upper_exit"
        for item in exits
    ):

        raise ValueError(
            "No upper-floor exits detected."
        )

    print("\nDetected destinations:")

    for floor in FLOORS:

        print(f"\nFloor {floor}:")

        for item in exits:

            if item["floor"] != floor:
                continue

            print(
                f"  Target {item['id']} | "
                f"Code {item['code']} | "
                f"Type {item['kind']} | "
                f"{len(item['cells'])} cells"
            )

    return exits, exit_map, stair_pairs