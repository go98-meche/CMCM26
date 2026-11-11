"""
Initial target-exit selection for the two-floor CA model.

Implements:
1. Rational satisficing search in concert.
2. Rational minimum-travel-time choice in lecture.
3. Habit-based familiarity and travel-time choice.
4. Peer influence and travel-time choice using RUM.
"""

import numpy as np

from config import *


# ============================================================
# 1. Get reachable exits on one floor
# ============================================================

def get_reachable_exits(
    position,
    travel_time_fields,
    floor
):
    """
    Return reachable target IDs on the Agent's current floor.
    """

    r, c = position

    floor_fields = travel_time_fields[floor]

    return [
        target_id
        for target_id, times in floor_fields.items()
        if np.isfinite(times[r, c])
    ]


# ============================================================
# 2. Choose nearest target by travel time
# ============================================================

def choose_nearest_exit(
    position,
    reachable_exits,
    travel_time_fields,
    floor,
    rng
):
    """
    Choose the target with minimum estimated travel time.

    Randomly resolve ties.
    """

    r, c = position

    times = {
        target_id: travel_time_fields[floor][target_id][r, c]
        for target_id in reachable_exits
    }

    minimum = min(times.values())

    best = [
        target_id
        for target_id, value in times.items()
        if np.isclose(value, minimum)
    ]

    return int(
        rng.choice(best)
    )


# ============================================================
# 3. Bresenham line algorithm
# ============================================================

def _bresenham_line(
    start,
    end
):
    """
    Return grid cells along a straight line.

    Positions are represented by (row, col).
    """

    r0, c0 = start
    r1, c1 = end

    x0, y0 = c0, r0
    x1, y1 = c1, r1

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    error = dx - dy

    cells = []

    while True:

        cells.append(
            (y0, x0)
        )

        if x0 == x1 and y0 == y1:
            break

        doubled_error = 2 * error

        if doubled_error > -dy:
            error -= dy
            x0 += sx

        if doubled_error < dx:
            error += dx
            y0 += sy

    return cells


# ============================================================
# 4. Check line of sight
# ============================================================

def _line_of_sight_to_target(
    position,
    target_cells,
    floor_terrain
):
    """
    A target is considered visible if at least one target cell
    has an unobstructed straight line.

    Walls block visibility.
    Seat cells do not block visibility in this version.
    """

    r, c = position

    ordered_cells = sorted(
        target_cells,
        key=lambda cell: (
            (cell[0] - r) ** 2
            + (cell[1] - c) ** 2
        )
    )

    for target_cell in ordered_cells:

        line = _bresenham_line(
            position,
            target_cell
        )

        intermediate_cells = line[1:-1]

        blocked = any(
            floor_terrain[lr, lc] == WALL
            for lr, lc in intermediate_cells
        )

        if not blocked:
            return True

    return False


# ============================================================
# 5. Perceived straight-line distance
# ============================================================

def _perceived_distance(
    position,
    target_cells
):
    """
    Minimum Euclidean distance to a target region, in meters.
    """

    r, c = position

    minimum_cells = min(
        np.hypot(
            tr - r,
            tc - c
        )
        for tr, tc in target_cells
    )

    return float(
        minimum_cells * CELL_SIZE_M
    )


# ============================================================
# 6. Determine known targets
# ============================================================

def get_known_exits(
    agent,
    reachable_exits,
    exits,
    terrain,
    scenario
):
    """
    Lecture:
        All reachable targets on the current floor are known.

    Concert:
        Only reachable targets visible from the current position
        are initially known.

    If no target is visible, use all reachable targets when
    CONCERT_FALLBACK_TO_ALL_REACHABLE is True.
    """

    floor = agent.floor

    if scenario == "lecture":

        return list(reachable_exits), False

    if scenario != "concert":

        raise ValueError(
            "Scenario must be 'concert' or 'lecture'."
        )

    if not CONCERT_REQUIRE_LINE_OF_SIGHT:

        return list(reachable_exits), False

    target_lookup = {
        item["id"]: item
        for item in exits
    }

    visible_targets = []

    for target_id in reachable_exits:

        target_info = target_lookup[target_id]

        visible = _line_of_sight_to_target(
            agent.position,
            target_info["cells"],
            terrain[floor]
        )

        if visible:

            visible_targets.append(target_id)

    if visible_targets:

        return visible_targets, False

    if CONCERT_FALLBACK_TO_ALL_REACHABLE:

        return list(reachable_exits), True

    raise ValueError(
        f"Agent {agent.agent_id} cannot see any reachable target."
    )


# ============================================================
# 7. Random Utility Model
# ============================================================

def _sample_rum_choice(
    agent,
    candidate_exits,
    travel_time_fields,
    rng,
    peer_shares=None
):
    """
    Sample an exit using the type-specific Random Utility Model.

    V(i,e) =
        -alpha_r * T_hat(i,e)
        + beta_r * q_local(i,e)
        + gamma_r * F(i,e)

    P(i,e) =
        exp(lambda_r * V(i,e))
        / sum_k exp(lambda_r * V(i,k))
    """

    if not candidate_exits:

        raise ValueError(
            "RUM choice received an empty candidate set."
        )

    if peer_shares is None:

        peer_shares = {}

    floor = agent.floor

    r, c = agent.position

    alpha = CHOICE_ALPHA[agent.agent_type]

    beta = CHOICE_BETA[agent.agent_type]

    gamma = CHOICE_GAMMA[agent.agent_type]

    lam = CHOICE_LAMBDA[agent.agent_type]

    if lam < 0:

        raise ValueError(
            "CHOICE_LAMBDA must be non-negative."
        )

    utilities = []

    for target_id in candidate_exits:

        travel_time = float(
            travel_time_fields[floor][target_id][r, c]
        )

        q_local = float(
            peer_shares.get(target_id, 0.0)
        )

        familiarity = (
            1.0
            if agent.familiar_exit == target_id
            else 0.0
        )

        utility = (
            -alpha * travel_time
            + beta * q_local
            + gamma * familiarity
        )

        utilities.append(utility)

    utilities = np.asarray(
        utilities,
        dtype=float
    )

    logits = lam * utilities

    # Numerical stability.
    logits -= np.max(logits)

    weights = np.exp(logits)

    probabilities = (
        weights / weights.sum()
    )

    selected = rng.choice(
        candidate_exits,
        p=probabilities
    )

    return int(selected)


# ============================================================
# 8. Rational Agent — Lecture
# ============================================================

def _choose_rational_lecture(
    agent,
    candidate_exits,
    travel_time_fields,
    rng
):
    """
    Rational lecture students know every reachable target
    on their current floor and choose minimum travel time.
    """

    floor = agent.floor

    r, c = agent.position

    times = {
        target_id:
            travel_time_fields[floor][target_id][r, c]
        for target_id in candidate_exits
    }

    minimum = min(
        times.values()
    )

    best = [
        target_id
        for target_id, value in times.items()
        if np.isclose(value, minimum)
    ]

    return int(
        rng.choice(best)
    )


# ============================================================
# 9. Rational Agent — Concert
# ============================================================

def _choose_rational_concert(
    agent,
    candidate_exits,
    exits,
    travel_time_fields
):
    """
    Rational concert audience use satisficing search.

    1. Sort known targets by perceived straight-line distance.
    2. Search from the nearest target.
    3. Accept the first target with travel time <= threshold.
    4. Increase threshold if no target is satisfactory.
    """

    floor = agent.floor

    r, c = agent.position

    target_lookup = {
        item["id"]: item
        for item in exits
    }

    ordered_targets = sorted(
        candidate_exits,
        key=lambda target_id: (
            _perceived_distance(
                agent.position,
                target_lookup[target_id]["cells"]
            ),
            target_id
        )
    )

    travel_times = {
        target_id:
            float(
                travel_time_fields[floor][target_id][r, c]
            )
        for target_id in ordered_targets
    }

    threshold = float(
        CONCERT_INITIAL_SATISFACTION_TIME_S
    )

    increment = float(
        CONCERT_THRESHOLD_INCREMENT_S
    )

    if threshold < 0 or increment <= 0:

        raise ValueError(
            "Satisfaction threshold must be non-negative "
            "and increment must be positive."
        )

    maximum_time = max(
        travel_times.values()
    )

    max_iterations = (
        int(
            np.ceil(
                max(0.0, maximum_time - threshold)
                / increment
            )
        )
        + 2
    )

    for _ in range(max_iterations):

        for target_id in ordered_targets:

            if (
                travel_times[target_id]
                <= threshold + 1e-12
            ):

                agent.satisfaction_threshold_s = threshold

                return int(target_id)

        threshold += increment

    # Numerical safety fallback.
    agent.satisfaction_threshold_s = threshold

    return int(ordered_targets[0])


# ============================================================
# 10. Assign familiarity
# ============================================================

def _assign_familiar_exit(
    agent,
    candidate_exits,
    travel_time_fields,
    rng
):
    """
    Assign a fixed familiar target to a Habit-Based Agent.

    By default, one known reachable target is chosen randomly.
    """

    if agent.familiar_exit in candidate_exits:

        return

    if HABIT_FAMILIARITY_ASSIGNMENT == "random_known_exit":

        agent.familiar_exit = int(
            rng.choice(candidate_exits)
        )

    elif HABIT_FAMILIARITY_ASSIGNMENT == "nearest_time":

        agent.familiar_exit = choose_nearest_exit(
            agent.position,
            candidate_exits,
            travel_time_fields,
            agent.floor,
            rng
        )

    else:

        raise ValueError(
            "Unknown HABIT_FAMILIARITY_ASSIGNMENT."
        )


# ============================================================
# 11. Assign targets
# ============================================================

def assign_target_exits(
    agents,
    exits,
    distance_fields,
    rng,
    travel_time_fields,
    scenario=EVACUATION_SCENARIO,
    terrain=None,
    all_agents=None
):
    """
    Assign target destinations to the supplied Agents.

    The function can be called:

        1. Once for all initial Agents.
        2. Again for Agents who arrive on Floor 1 from Floor 2.

    all_agents provides neighborhood context for peer influence.
    """

    if scenario not in {"concert", "lecture"}:

        raise ValueError(
            "scenario must be 'concert' or 'lecture'."
        )

    if terrain is None:

        raise ValueError(
            "terrain must be supplied."
        )

    if all_agents is None:

        all_agents = agents

    if not agents:

        return {}

    target_lookup = {
        item["id"]: item
        for item in exits
    }

    # ========================================================
    # STAGE 1. Determine reachable and known targets
    # ========================================================

    reachable_by_agent = {}
    known_by_agent = {}

    baseline_load = {
        item["id"]: 0
        for item in exits
    }

    fallback_count = 0

    for agent in agents:

        if (
            agent.status != "active"
            or agent.position is None
        ):

            raise ValueError(
                f"Agent {agent.agent_id} is not active on a floor."
            )

        floor = agent.floor

        reachable = get_reachable_exits(
            agent.position,
            travel_time_fields,
            floor
        )

        if not reachable:

            raise ValueError(
                f"Agent {agent.agent_id} on floor {floor} "
                "cannot reach any target."
            )

        reachable_by_agent[agent.agent_id] = reachable

        nearest = choose_nearest_exit(
            agent.position,
            reachable,
            travel_time_fields,
            floor,
            rng
        )

        baseline_load[nearest] += 1

        known, used_fallback = get_known_exits(
            agent,
            reachable,
            exits,
            terrain,
            scenario
        )

        known_by_agent[agent.agent_id] = known

        fallback_count += int(used_fallback)

    # ========================================================
    # STAGE 2. Preliminary preferences
    # ========================================================

    preliminary_targets = {}

    for agent in agents:

        candidates = known_by_agent[
            agent.agent_id
        ]

        # ----------------------------------------------------
        # Rational
        # ----------------------------------------------------

        if agent.agent_type == "rational":

            if scenario == "concert":

                selected = _choose_rational_concert(
                    agent,
                    candidates,
                    exits,
                    travel_time_fields
                )

            else:

                selected = _choose_rational_lecture(
                    agent,
                    candidates,
                    travel_time_fields,
                    rng
                )

            preliminary_targets[
                agent.agent_id
            ] = selected

        # ----------------------------------------------------
        # Habit
        # ----------------------------------------------------

        elif agent.agent_type == "habit":

            _assign_familiar_exit(
                agent,
                candidates,
                travel_time_fields,
                rng
            )

            # Preliminary choice considers familiarity and time.
            preliminary_targets[
                agent.agent_id
            ] = _sample_rum_choice(
                agent,
                candidates,
                travel_time_fields,
                rng,
                peer_shares={}
            )

        # ----------------------------------------------------
        # Peer
        # ----------------------------------------------------

        elif agent.agent_type == "peer":

            preliminary_targets[
                agent.agent_id
            ] = choose_nearest_exit(
                agent.position,
                candidates,
                travel_time_fields,
                agent.floor,
                rng
            )

        else:

            raise ValueError(
                f"Unknown Agent type: {agent.agent_type}"
            )

    # ========================================================
    # STAGE 3. Peer influence
    # ========================================================

    final_targets = preliminary_targets.copy()

    for agent in agents:

        if agent.agent_type != "peer":

            continue

        r, c = agent.position

        floor = agent.floor

        candidates = known_by_agent[
            agent.agent_id
        ]

        candidate_set = set(candidates)

        target_counts = {
            target_id: 0
            for target_id in candidates
        }

        # ----------------------------------------------------
        # Observe neighbors on the SAME floor only.
        # ----------------------------------------------------

        for neighbor in all_agents:

            if neighbor.agent_id == agent.agent_id:
                continue

            if neighbor.floor != floor:
                continue

            if (
                neighbor.status != "active"
                or neighbor.position is None
            ):
                continue

            nr, nc = neighbor.position

            if (
                abs(nr - r) > PEER_RADIUS
                or abs(nc - c) > PEER_RADIUS
            ):
                continue

            if nr == r and nc == c:
                continue

            # Use the neighbor's existing target if available.
            neighbor_target = neighbor.target_exit

            # Otherwise use its preliminary preference.
            if neighbor_target is None:

                neighbor_target = preliminary_targets.get(
                    neighbor.agent_id
                )

            if neighbor_target in candidate_set:

                target_counts[neighbor_target] += 1

        total_neighbors = sum(
            target_counts.values()
        )

        if total_neighbors > 0:

            peer_shares = {
                target_id: count / total_neighbors
                for target_id, count in target_counts.items()
            }

        else:

            peer_shares = {
                target_id: 0.0
                for target_id in candidates
            }

        # ----------------------------------------------------
        # RUM: peer influence + travel time
        # ----------------------------------------------------

        final_targets[
            agent.agent_id
        ] = _sample_rum_choice(
            agent,
            candidates,
            travel_time_fields,
            rng,
            peer_shares
        )

    # ========================================================
    # STAGE 4. Save final targets
    # ========================================================

    for agent in agents:

        agent.target_exit = final_targets[
            agent.agent_id
        ]

    # ========================================================
    # STAGE 5. Print results
    # ========================================================

    print("\nTarget selection completed.")

    print("Scenario:", scenario)

    for floor in FLOORS:

        floor_agents = [
            agent
            for agent in agents
            if agent.floor == floor
        ]

        if not floor_agents:
            continue

        print(f"\nFloor {floor} target distribution:")

        floor_targets = [
            item
            for item in exits
            if item["floor"] == floor
        ]

        for target_info in floor_targets:

            target_id = target_info["id"]

            count = sum(
                agent.target_exit == target_id
                for agent in floor_agents
            )

            print(
                f"  Target {target_id} "
                f"(code {target_info['code']}): {count}"
            )

    if scenario == "concert":

        print(
            "\nAgents using all-reachable fallback:",
            fallback_count
        )

    return baseline_load