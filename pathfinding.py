"""
Pathfinding for the two-floor CA evacuation model.

Supports:
1. Floor-specific target destinations.
2. Four-direction movement.
3. Horizontal/vertical seat constraints.
4. Geometric distance fields.
5. Speed-weighted travel-time fields.
"""

import heapq
import numpy as np

from config import *


# ============================================================
# 1. Check whether a cell is passable
# ============================================================

def is_passable(
    terrain,
    exit_map,
    r,
    c,
    target_exit
):
    """
    Check whether a cell can be used for a particular target.

    Another target destination is blocked.

    Stair landing cells on Floor 1 are not targets,
    so they remain ordinary passable cells.
    """

    if terrain[r, c] == WALL:
        return False

    cell_target = exit_map[r, c]

    if (
        cell_target >= 0
        and cell_target != target_exit
    ):
        return False

    return True


# ============================================================
# 2. Check seat-direction restrictions
# ============================================================

def movement_allowed(
    terrain,
    source,
    destination
):
    """
    Check whether movement between two adjacent cells is allowed.

    Assumptions:
        Horizontal Seat (2):
            Only horizontal movement is allowed.

        Vertical Seat (3):
            Only vertical movement is allowed.

    Both source and destination seat cells are checked.
    """

    sr, sc = source
    dr, dc = destination

    delta_r = dr - sr
    delta_c = dc - sc

    if abs(delta_r) + abs(delta_c) != 1:
        return False

    source_code = terrain[sr, sc]

    destination_code = terrain[dr, dc]

    # Leave Horizontal Seat only horizontally.
    if (
        source_code == HORIZONTAL_SEAT
        and delta_r != 0
    ):
        return False

    # Leave Vertical Seat only vertically.
    if (
        source_code == VERTICAL_SEAT
        and delta_c != 0
    ):
        return False

    # Enter Horizontal Seat only horizontally.
    if (
        destination_code == HORIZONTAL_SEAT
        and delta_r != 0
    ):
        return False

    # Enter Vertical Seat only vertically.
    if (
        destination_code == VERTICAL_SEAT
        and delta_c != 0
    ):
        return False

    return True


# ============================================================
# 3. Get valid neighbors
# ============================================================

def get_valid_neighbors(
    terrain,
    exit_map,
    position,
    target_exit
):
    """
    Get valid four-direction neighbors.

    Returns:
        [((row, col), movement_cost), ...]
    """

    rows, cols = terrain.shape

    r, c = position

    neighbors = []

    for dr, dc in ORTHOGONAL_OFFSETS:

        nr = r + dr
        nc = c + dc

        if not (
            0 <= nr < rows
            and 0 <= nc < cols
        ):
            continue

        if not is_passable(
            terrain,
            exit_map,
            nr,
            nc,
            target_exit
        ):
            continue

        if not movement_allowed(
            terrain,
            (r, c),
            (nr, nc)
        ):
            continue

        neighbors.append(
            (
                (nr, nc),
                ORTHOGONAL_COST
            )
        )

    return neighbors


# ============================================================
# 4. Build a reverse graph
# ============================================================

def _build_reverse_graph(
    terrain,
    exit_map,
    target_exit,
    cell_speeds=None
):
    """
    Build a reverse graph while preserving the cost of
    each original forward edge.

    For a forward movement u -> v:

        Geometric cost:
            ORTHOGONAL_COST

        Travel-time cost:
            CELL_SIZE_M / speed[u]

    The speed belongs to the cell being left.
    """

    rows, cols = terrain.shape

    reverse_graph = {}

    if cell_speeds is not None:

        cell_speeds = np.asarray(
            cell_speeds,
            dtype=float
        )

        if cell_speeds.shape != terrain.shape:

            raise ValueError(
                "cell_speeds must have the same shape as terrain."
            )

    for r in range(rows):

        for c in range(cols):

            if not is_passable(
                terrain,
                exit_map,
                r,
                c,
                target_exit
            ):
                continue

            for (nr, nc), step_cost in get_valid_neighbors(
                terrain,
                exit_map,
                (r, c),
                target_exit
            ):

                if cell_speeds is None:

                    edge_cost = step_cost

                else:

                    speed = cell_speeds[r, c]

                    if (
                        not np.isfinite(speed)
                        or speed <= 0
                    ):
                        continue

                    edge_cost = (
                        CELL_SIZE_M
                        * step_cost
                        / speed
                    )

                # Original edge:
                # (r,c) -> (nr,nc)
                #
                # Reverse edge:
                # (nr,nc) -> (r,c)

                reverse_graph.setdefault(
                    (nr, nc),
                    []
                ).append(
                    (
                        (r, c),
                        float(edge_cost)
                    )
                )

    return reverse_graph


# ============================================================
# 5. Calculate fields for one target
# ============================================================

def _calculate_fields_for_target(
    terrain,
    target_info,
    exit_map,
    cell_speeds=None
):
    """
    Reverse multi-source Dijkstra for one target destination.
    """

    rows, cols = terrain.shape

    target_id = target_info["id"]

    distances = np.full(
        (rows, cols),
        np.inf,
        dtype=float
    )

    reverse_graph = _build_reverse_graph(
        terrain,
        exit_map,
        target_id,
        cell_speeds
    )

    queue = []

    for r, c in target_info["cells"]:

        distances[r, c] = 0.0

        heapq.heappush(
            queue,
            (0.0, r, c)
        )

    while queue:

        current_cost, r, c = heapq.heappop(
            queue
        )

        if (
            current_cost
            > distances[r, c] + 1e-12
        ):
            continue

        for (pr, pc), edge_cost in reverse_graph.get(
            (r, c),
            []
        ):

            candidate = (
                current_cost + edge_cost
            )

            if candidate < distances[pr, pc]:

                distances[pr, pc] = candidate

                heapq.heappush(
                    queue,
                    (candidate, pr, pc)
                )

    return distances


# ============================================================
# 6. Calculate geometric distance fields for both floors
# ============================================================

def calculate_distance_fields(
    terrain,
    exits,
    exit_map
):
    """
    Calculate geometric shortest-path distances.

    Returns:
        {
            1: {target_id: distance_matrix, ...},
            2: {target_id: distance_matrix, ...}
        }

    Distances are measured in grid-cell units.
    """

    distance_fields = {
        floor: {}
        for floor in FLOORS
    }

    for target_info in exits:

        floor = target_info["floor"]

        target_id = target_info["id"]

        distances = _calculate_fields_for_target(
            terrain[floor],
            target_info,
            exit_map[floor],
            cell_speeds=None
        )

        distance_fields[floor][target_id] = distances

    print("\nGeometric distance fields calculated.")

    return distance_fields


# ============================================================
# 7. Calculate travel-time fields for both floors
# ============================================================

def calculate_travel_time_fields(
    terrain,
    exits,
    exit_map,
    cell_speeds
):
    """
    Calculate minimum travel time to each target.

    For a forward movement u -> v:

        edge_time(u,v) = CELL_SIZE_M / speed[u]

    Returns:
        {
            1: {target_id: time_matrix, ...},
            2: {target_id: time_matrix, ...}
        }

    All values are measured in seconds.
    """

    travel_time_fields = {
        floor: {}
        for floor in FLOORS
    }

    for target_info in exits:

        floor = target_info["floor"]

        target_id = target_info["id"]

        travel_times = _calculate_fields_for_target(
            terrain[floor],
            target_info,
            exit_map[floor],
            cell_speeds=cell_speeds[floor]
        )

        travel_time_fields[floor][target_id] = (
            travel_times
        )

    print(
        "\nTwo-floor travel-time fields calculated."
    )

    return travel_time_fields