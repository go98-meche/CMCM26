"""
Regional Stream-based Pedestrian Fundamental Diagram (SbFD).

Each floor is divided independently into REGION_SIZE x REGION_SIZE
macroscopic regions.
"""

import numpy as np

from config import *

from pathfinding import get_valid_neighbors


# ============================================================
# 1. Direction vectors
# ============================================================

DIRECTION_VECTORS = {
    "E": np.array([1.0, 0.0]),
    "W": np.array([-1.0, 0.0]),
    "N": np.array([0.0, 1.0]),
    "S": np.array([0.0, -1.0])
}


# ============================================================
# 2. Determine movement direction
# ============================================================

def direction_from_positions(
    old_position,
    new_position
):
    """
    Convert one orthogonal CA move into E, W, N, or S.

    Positions are stored as (row, column).
    """

    old_r, old_c = old_position
    new_r, new_c = new_position

    dr = new_r - old_r
    dc = new_c - old_c

    if dr == 0 and dc == 1:
        return "E"

    if dr == 0 and dc == -1:
        return "W"

    if dr == -1 and dc == 0:
        return "N"

    if dr == 1 and dc == 0:
        return "S"

    raise ValueError(
        f"Invalid orthogonal movement: "
        f"{old_position} -> {new_position}"
    )


# ============================================================
# 3. Infer initial stream direction
# ============================================================

def infer_initial_stream_direction(
    agent,
    terrain,
    exit_map,
    distance_fields,
    rng
):
    """
    Infer a temporary direction for an Agent that has
    not moved yet.
    """

    if agent.target_exit is None:
        return None

    floor = agent.floor

    r, c = agent.position

    current_distance = (
        distance_fields[floor][agent.target_exit][r, c]
    )

    if not np.isfinite(current_distance):
        return None

    candidates = []

    for (nr, nc), _ in get_valid_neighbors(
        terrain[floor],
        exit_map[floor],
        agent.position,
        agent.target_exit
    ):

        next_distance = (
            distance_fields[floor][agent.target_exit][nr, nc]
        )

        if (
            np.isfinite(next_distance)
            and next_distance < current_distance - 1e-10
        ):

            candidates.append(
                (nr, nc, next_distance)
            )

    if not candidates:
        return None

    best_distance = min(
        item[2]
        for item in candidates
    )

    best_positions = [
        (nr, nc)
        for nr, nc, distance in candidates
        if np.isclose(distance, best_distance)
    ]

    selected_position = best_positions[
        int(rng.integers(len(best_positions)))
    ]

    return direction_from_positions(
        agent.position,
        selected_position
    )


# ============================================================
# 4. Calculate regional speeds
# ============================================================

def calculate_region_speeds(
    agents,
    terrain,
    exit_map,
    distance_fields,
    rng
):
    """
    Calculate regional SbFD speeds independently on each floor.

    Returns:
        {
            1: {(region_row, region_col): speed, ...},
            2: {(region_row, region_col): speed, ...}
        }

    Agents in transit are excluded from density calculations.
    """

    agents_by_region = {
        floor: {}
        for floor in FLOORS
    }

    for agent in agents:

        if (
            agent.status != "active"
            or agent.position is None
        ):
            continue

        floor = agent.floor

        r, c = agent.position

        region_key = (
            r // REGION_SIZE,
            c // REGION_SIZE
        )

        agents_by_region[floor].setdefault(
            region_key,
            []
        ).append(agent)

    region_speeds = {
        floor: {}
        for floor in FLOORS
    }

    # ========================================================
    # Calculate each floor independently
    # ========================================================

    for floor in FLOORS:

        floor_terrain = terrain[floor]

        rows, cols = floor_terrain.shape

        n_region_rows = (
            rows + REGION_SIZE - 1
        ) // REGION_SIZE

        n_region_cols = (
            cols + REGION_SIZE - 1
        ) // REGION_SIZE

        for region_row in range(n_region_rows):

            for region_col in range(n_region_cols):

                region_key = (
                    region_row,
                    region_col
                )

                r0 = region_row * REGION_SIZE
                c0 = region_col * REGION_SIZE

                r1 = min(
                    r0 + REGION_SIZE,
                    rows
                )

                c1 = min(
                    c0 + REGION_SIZE,
                    cols
                )

                region_terrain = floor_terrain[
                    r0:r1,
                    c0:c1
                ]

                walkable_cells = np.count_nonzero(
                    region_terrain != WALL
                )

                area_m2 = (
                    walkable_cells * CELL_SIZE_M ** 2
                )

                region_agents = (
                    agents_by_region[floor].get(
                        region_key,
                        []
                    )
                )

                population = len(region_agents)

                if population == 0:

                    region_speeds[floor][region_key] = (
                        FREE_FLOW_SPEED
                    )

                    continue

                if area_m2 <= 0:

                    raise ValueError(
                        f"Floor {floor}, region {region_key} "
                        "contains Agents but no walkable area."
                    )

                # ------------------------------------------------
                # Overall density
                # ------------------------------------------------

                total_density = (
                    population / area_m2
                )

                # ------------------------------------------------
                # Count directional streams
                # ------------------------------------------------

                stream_counts = {
                    direction: 0
                    for direction in DIRECTION_VECTORS
                }

                for agent in region_agents:

                    direction = agent.last_direction

                    if direction not in stream_counts:

                        direction = infer_initial_stream_direction(
                            agent,
                            terrain,
                            exit_map,
                            distance_fields,
                            rng
                        )

                    if direction in stream_counts:

                        stream_counts[direction] += 1

                present_streams = [
                    direction
                    for direction, count in stream_counts.items()
                    if count > 0
                ]

                # ------------------------------------------------
                # Stream-specific speeds
                # ------------------------------------------------

                stream_speeds = {}

                for stream in present_streams:

                    exponent = (
                        -THETA * total_density ** 2
                    )

                    vector_s = DIRECTION_VECTORS[stream]

                    for other_stream in present_streams:

                        if other_stream == stream:
                            continue

                        vector_other = (
                            DIRECTION_VECTORS[other_stream]
                        )

                        cosine_phi = float(
                            np.clip(
                                np.dot(
                                    vector_s,
                                    vector_other
                                ),
                                -1.0,
                                1.0
                            )
                        )

                        phi = float(
                            np.arccos(cosine_phi)
                        )

                        angular_interference = (
                            1.0 - np.cos(ALPHA * phi)
                        )

                        other_density = (
                            stream_counts[other_stream]
                            / area_m2
                        )

                        exponent -= (
                            BETA
                            * other_density
                            * angular_interference
                        )

                    stream_speeds[stream] = (
                        FREE_FLOW_SPEED
                        * np.exp(exponent)
                    )

                # ------------------------------------------------
                # Regional average speed
                # ------------------------------------------------

                if present_streams:

                    regional_speed = sum(
                        stream_counts[stream]
                        * stream_speeds[stream]
                        for stream in present_streams
                    ) / population

                else:

                    regional_speed = FREE_FLOW_SPEED

                # Keep speed positive.
                regional_speed = float(
                    np.clip(
                        regional_speed,
                        MIN_SPEED_MPS,
                        FREE_FLOW_SPEED
                    )
                )

                region_speeds[floor][region_key] = (
                    regional_speed
                )

                # Assign speed to every Agent in the region.
                for agent in region_agents:

                    agent.speed_mps = regional_speed

    return region_speeds


# ============================================================
# 5. Convert regional speeds into cell speed fields
# ============================================================

def build_cell_speed_fields(
    terrain,
    region_speeds
):
    """
    Create one speed matrix for each floor.

    Every cell receives the speed of its macroscopic region.

    Empty regions retain FREE_FLOW_SPEED.
    """

    cell_speed_fields = {}

    for floor in FLOORS:

        floor_terrain = terrain[floor]

        rows, cols = floor_terrain.shape

        speeds = np.full(
            (rows, cols),
            FREE_FLOW_SPEED,
            dtype=float
        )

        for r in range(rows):

            for c in range(cols):

                region_key = (
                    r // REGION_SIZE,
                    c // REGION_SIZE
                )

                speeds[r, c] = (
                    region_speeds[floor].get(
                        region_key,
                        FREE_FLOW_SPEED
                    )
                )

        cell_speed_fields[floor] = speeds

    return cell_speed_fields