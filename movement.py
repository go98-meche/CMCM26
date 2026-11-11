"""
One synchronous CA update for the two-floor evacuation model.

Includes:
1. Regional SbFD speed calculation.
2. Dynamic decision changes.
3. Synchronous cell movement.
4. Destination conflict resolution.
5. Occupied-cell dependencies.
6. Head-on exchanges.
7. Stair transit and floor transitions.
8. Per-timestep movement diagnostics.

Conflict-resolution principle
-----------------------------
Once an Agent has a valid movement whose destination is unique and
whose occupancy dependency can be resolved, that movement is reserved
for the current timestep. Agents whose proposed movements fail then
try their next candidate. This avoids repeatedly reallocating already
valid movements while resolving another Agent's conflict.
"""

import numpy as np

from config import *

from pathfinding import get_valid_neighbors

from speed_model import (
    calculate_region_speeds,
    direction_from_positions
)

from decision_change import apply_decision_changes


# ============================================================
# Diagnostics settings
# ============================================================

# Print one summary line for every CA timestep.
SHOW_MOVEMENT_DIAGNOSTICS = True

# Print a few Agent IDs that did not move despite having candidates.
DIAGNOSTICS_MAX_AGENT_IDS = 8


# ============================================================
# 1. Build candidate destinations
# ============================================================

def _build_candidate_lists(
    agents,
    terrain,
    exit_map,
    distance_fields,
    rng,
    diagnostics=None
):
    """
    Build ordered candidate destinations for active Agents.

    Each destination is represented by:
        (floor, row, column)

    Candidates must strictly reduce the remaining distance to the
    Agent's current target exit.

    The optional diagnostics dictionary records why an Agent did not
    receive any candidate destination.
    """

    candidate_lists = {}

    if diagnostics is None:
        diagnostics = {}

    diagnostics.setdefault("unreachable_current_distance", 0)
    diagnostics.setdefault("no_descending_neighbor", 0)
    diagnostics.setdefault("unreachable_agent_ids", [])
    diagnostics.setdefault("no_descending_agent_ids", [])

    for agent in agents:

        if (
            agent.status != "active"
            or agent.position is None
            or agent.target_exit is None
        ):
            continue

        floor = agent.floor
        r, c = agent.position

        current_distance = (
            distance_fields[floor][agent.target_exit][r, c]
        )

        if not np.isfinite(current_distance):

            diagnostics["unreachable_current_distance"] += 1
            diagnostics["unreachable_agent_ids"].append(
                agent.agent_id
            )
            continue

        candidates = []

        neighbors = get_valid_neighbors(
            terrain[floor],
            exit_map[floor],
            agent.position,
            agent.target_exit
        )

        for (nr, nc), _ in neighbors:

            next_distance = (
                distance_fields[floor][agent.target_exit][nr, nc]
            )

            if (
                np.isfinite(next_distance)
                and next_distance < current_distance - 1e-10
            ):

                candidates.append(
                    (
                        float(next_distance),
                        (floor, nr, nc)
                    )
                )

        if not candidates:

            diagnostics["no_descending_neighbor"] += 1
            diagnostics["no_descending_agent_ids"].append(
                agent.agent_id
            )
            continue

        # Sort by remaining distance.
        # Randomize equal-distance destinations.

        distance_levels = sorted(
            set(
                distance
                for distance, _ in candidates
            )
        )

        ordered = []

        for distance_level in distance_levels:

            tied = [
                destination
                for distance, destination in candidates
                if np.isclose(distance, distance_level)
            ]

            rng.shuffle(tied)

            ordered.extend(tied)

        candidate_lists[agent.agent_id] = ordered

    return candidate_lists


# ============================================================
# 2. Allocate unique destination cells
# ============================================================

def _assign_unique_destinations(
    candidate_lists,
    candidate_indices,
    rng,
    reserved_destinations=None
):
    """
    Allocate unique destination cells among the Agents being considered.

    Destinations already reserved by previously validated movements
    cannot be claimed again.

    When multiple Agents request the same destination:
        - One Agent wins randomly.
        - Losing Agents advance to their next candidate.
        - Losing Agents continue proposing within this allocation pass.

    Returns
    -------
    selected : dict
        agent_id -> (floor, row, column)

    stats : dict
        Destination-conflict diagnostics.
    """

    selected = {}

    claimed = set(reserved_destinations or ())

    stats = {
        "destination_conflict_groups": 0,
        "destination_conflict_agents": 0,
        "destination_conflict_losers": 0,
        "candidate_skips_reserved": 0
    }

    pending = {
        agent_id
        for agent_id, options in candidate_lists.items()
        if candidate_indices[agent_id] < len(options)
    }

    while pending:

        proposals = {}

        exhausted = []

        for agent_id in list(pending):

            options = candidate_lists[agent_id]

            index = candidate_indices[agent_id]

            # Skip destinations already reserved by another movement.
            while (
                index < len(options)
                and options[index] in claimed
            ):

                index += 1

                stats["candidate_skips_reserved"] += 1

            candidate_indices[agent_id] = index

            if index >= len(options):

                exhausted.append(agent_id)

                continue

            destination = options[index]

            proposals.setdefault(
                destination,
                []
            ).append(agent_id)

        for agent_id in exhausted:

            pending.discard(agent_id)

        if not proposals:

            break

        for destination, contenders in proposals.items():

            contenders = list(contenders)

            if len(contenders) > 1:

                stats["destination_conflict_groups"] += 1

                stats["destination_conflict_agents"] += len(
                    contenders
                )

                stats["destination_conflict_losers"] += (
                    len(contenders) - 1
                )

            rng.shuffle(contenders)

            winner = contenders[0]

            selected[winner] = destination

            claimed.add(destination)

            pending.discard(winner)

            # Losing Agents try their next candidate.
            for loser in contenders[1:]:

                candidate_indices[loser] += 1

    return selected, stats


# ============================================================
# 3. Resolve occupied-cell dependencies
# ============================================================

def _find_movable_agents(
    selected,
    occupied
):
    """
    Find all Agents whose selected movement can be completed.

    An Agent can move if:
        1. Its destination is empty; or
        2. The destination's current occupant also moves; or
        3. It belongs to a movement cycle (including a direct exchange).

    A chain ending at an Agent that does not move is blocked.

    This implementation is iterative rather than recursive, so a long
    line of Agents will not hit Python's recursion-depth limit.

    Returns
    -------
    set
        Agent IDs whose movement dependencies are valid.
    """

    movable = {}

    for start_agent_id in selected:

        if start_agent_id in movable:

            continue

        path = []

        path_positions = {}

        current_agent_id = start_agent_id

        while True:

            # The chain has reached a node whose result is known.
            if current_agent_id in movable:

                can_move = movable[current_agent_id]

                break

            # A cycle has been found.
            # Every Agent in this chain can move.
            if current_agent_id in path_positions:

                can_move = True

                break

            path_positions[current_agent_id] = len(path)

            path.append(current_agent_id)

            destination = selected[current_agent_id]

            occupant_id = occupied.get(destination)

            # The destination is empty.
            if occupant_id is None:

                can_move = True

                break

            # Defensive handling.
            if occupant_id == current_agent_id:

                can_move = True

                break

            # The occupant is not selected to move this timestep.
            if occupant_id not in selected:

                can_move = False

                break

            # Follow the dependency chain.
            current_agent_id = occupant_id

        # Every node in the traversed chain has the same outcome.
        for agent_id in reversed(path):

            movable[agent_id] = can_move

    return {
        agent_id
        for agent_id, can_move in movable.items()
        if can_move
    }


def _can_move(
    agent_id,
    selected,
    occupied,
    visiting=None
):
    """Compatibility wrapper for checking one Agent's dependency."""

    return agent_id in _find_movable_agents(
        selected,
        occupied
    )


# ============================================================
# 4. Identify direct head-on exchanges
# ============================================================

def _find_head_on_pairs(
    selected,
    occupied
):
    """
    Identify two-Agent position exchanges.

    A -> B's original cell
    B -> A's original cell
    """

    original_position = {
        agent_id: position
        for position, agent_id in occupied.items()
    }

    pairs = set()

    for agent_id, destination in selected.items():

        occupant_id = occupied.get(destination)

        if (
            occupant_id is None
            or occupant_id == agent_id
        ):
            continue

        if occupant_id not in selected:

            continue

        if (
            selected[occupant_id]
            == original_position[agent_id]
        ):

            pair = tuple(
                sorted(
                    (agent_id, occupant_id),
                    key=str
                )
            )

            pairs.add(pair)

    return pairs


# ============================================================
# 5. Resolve movement conflicts and dependencies
# ============================================================

def _resolve_movements(
    candidate_lists,
    occupied,
    rng
):
    """
    Resolve unique destinations and occupied-cell dependencies.

    Improved procedure:
        1. Allocate unique destinations.
        2. Identify movements that can be completed through empty-cell
           chains or movement cycles.
        3. Reserve those valid movements for this timestep.
        4. Agents whose movements fail try their next candidates.
        5. Repeat until no further candidate can be assigned.

    Previously validated movements are not reallocated when another
    Agent fails.

    Direct head-on exchanges retain the existing passing probability:
        HEAD_ON_SLOWDOWN_FACTOR

    Returns
    -------
    selected_movements : dict
        agent_id -> destination

    stats : dict
        Diagnostics about destination conflicts and occupancy blocking.
    """

    candidate_indices = {
        agent_id: 0
        for agent_id in candidate_lists
    }

    # Movements validated in earlier rounds are fixed for this timestep.
    fixed_movements = {}

    # Store one head-on decision per pair.
    head_on_decisions = {}

    stats = {
        "resolution_rounds": 0,
        "destination_conflict_groups": 0,
        "destination_conflict_agents": 0,
        "destination_conflict_losers": 0,
        "candidate_skips_reserved": 0,
        "dependency_blocked": 0,
        "candidate_retries": 0,
        "head_on_pairs": 0,
        "head_on_delayed_pairs": 0,
        "head_on_delayed_agents": 0
    }

    while True:

        # Only Agents with remaining candidates need another proposal.
        unresolved = {
            agent_id
            for agent_id, options in candidate_lists.items()
            if (
                candidate_indices[agent_id] < len(options)
                and agent_id not in fixed_movements
            )
        }

        if not unresolved:

            break

        current_candidate_lists = {
            agent_id: candidate_lists[agent_id]
            for agent_id in unresolved
        }

        new_selected, allocation_stats = _assign_unique_destinations(
            current_candidate_lists,
            candidate_indices,
            rng,
            reserved_destinations=set(fixed_movements.values())
        )

        for key, value in allocation_stats.items():

            stats[key] += value

        if not new_selected:

            break

        stats["resolution_rounds"] += 1

        # Combine previously reserved movements with this round's proposals.
        combined_selected = dict(fixed_movements)

        combined_selected.update(new_selected)

        # Find direct head-on exchanges.
        head_on_pairs = _find_head_on_pairs(
            combined_selected,
            occupied
        )

        blocked_by_head_on = set()

        for pair in head_on_pairs:

            if not set(pair).issubset(new_selected):

                continue

            if pair not in head_on_decisions:

                stats["head_on_pairs"] += 1

                passes = (
                    rng.random() < HEAD_ON_SLOWDOWN_FACTOR
                )

                head_on_decisions[pair] = passes

                if not passes:

                    stats["head_on_delayed_pairs"] += 1

                    stats["head_on_delayed_agents"] += 2

            if not head_on_decisions[pair]:

                blocked_by_head_on.update(pair)

        # Remove head-on pairs delayed during this timestep.
        effective_selected = {
            agent_id: destination
            for agent_id, destination in combined_selected.items()
            if agent_id not in blocked_by_head_on
        }

        final_movable = _find_movable_agents(
            effective_selected,
            occupied
        )

        # Newly proposed movements that can be completed are reserved.
        newly_valid = {
            agent_id
            for agent_id in new_selected
            if agent_id in final_movable
        }

        fixed_movements.update(
            {
                agent_id: new_selected[agent_id]
                for agent_id in newly_valid
            }
        )

        # Failed proposals advance to their next candidate.
        failed = set(new_selected) - newly_valid

        stats["dependency_blocked"] += len(
            failed - blocked_by_head_on
        )

        for agent_id in failed:

            candidate_indices[agent_id] += 1

            stats["candidate_retries"] += 1

        if not failed:

            break

    stats["unresolved_after_resolution"] = (
        len(set(candidate_lists) - set(fixed_movements))
    )

    return fixed_movements, stats


# ============================================================
# 6. Process existing transit Agents
# ============================================================

def _process_transit_agents(
    agents,
    occupied,
    time_step
):
    """
    Advance Agents currently in stair transit.

    Upper-floor exit:
        Agent is evacuated when countdown reaches zero.

    Stair 6/7:
        Agent arrives at the paired Floor-1 landing cell.

    If the landing cell is occupied:
        Agent waits until it becomes available.

    Returns:
        agents_to_keep
        newly_evacuated
        newly_arrived_floor1
    """

    agents_to_keep = []

    newly_evacuated = []

    newly_arrived_floor1 = []

    for agent in agents:

        if agent.status != "transit":

            agents_to_keep.append(agent)

            continue

        # Transit countdown.
        agent.transit_remaining -= 1

        if agent.transit_remaining > 0:

            agents_to_keep.append(agent)

            continue

        # ----------------------------------------------------
        # Upper-floor exit (code 5)
        # ----------------------------------------------------

        if agent.transit_kind == "upper_exit":

            agent.evacuation_time = time_step

            agent.position = None

            agent.status = "evacuated"

            newly_evacuated.append(agent)

            continue

        # ----------------------------------------------------
        # Stair transfer (code 6 or 7)
        # ----------------------------------------------------

        if agent.transit_kind == "stair":

            destination = agent.transit_destination

            if destination is None:

                raise ValueError(
                    f"Agent {agent.agent_id} "
                    "has no stair destination."
                )

            floor, row, col = destination

            destination_key = (
                floor,
                int(row),
                int(col)
            )

            # Check Floor-1 landing occupancy.
            if destination_key in occupied:

                # Retry landing next timestep.
                agent.transit_remaining = 1

                agents_to_keep.append(agent)

                continue

            # Transfer to Floor 1.
            agent.floor = floor

            agent.position = (
                int(row),
                int(col)
            )

            agent.status = "active"

            agent.transit_remaining = 0

            agent.transit_destination = None

            agent.transit_kind = None

            # The old target belonged to Floor 2.
            # A new Floor-1 target will be assigned by simulation.py.
            agent.target_exit = None

            agent.familiar_exit = None

            agent.satisfaction_threshold_s = None

            agent.last_direction = None

            agent.speed_mps = FREE_FLOW_SPEED

            agents_to_keep.append(agent)

            newly_arrived_floor1.append(agent)

            occupied[destination_key] = agent.agent_id

            continue

        raise ValueError(
            f"Unknown transit kind: {agent.transit_kind}"
        )

    return (
        agents_to_keep,
        newly_evacuated,
        newly_arrived_floor1
    )


# ============================================================
# 7. Build free-flow regional speed fields
# ============================================================

def _build_free_flow_region_speeds(
    terrain
):
    """
    At the initial CA step, all Agents retain free-flow speed.

    Construct the corresponding region-speed dictionary for both floors.
    """

    region_speeds = {}

    for floor in FLOORS:

        rows, cols = terrain[floor].shape

        n_region_rows = (
            rows + REGION_SIZE - 1
        ) // REGION_SIZE

        n_region_cols = (
            cols + REGION_SIZE - 1
        ) // REGION_SIZE

        region_speeds[floor] = {}

        for region_row in range(n_region_rows):

            for region_col in range(n_region_cols):

                region_speeds[floor][
                    (region_row, region_col)
                ] = FREE_FLOW_SPEED

    return region_speeds


# ============================================================
# 8. Print movement diagnostics
# ============================================================

def _print_movement_diagnostics(
    time_step,
    active_count,
    target_count,
    targetless_count,
    eligible_count,
    candidate_count,
    candidate_diagnostics,
    selected_count,
    moved_count,
    resolver_stats
):
    """Print a compact diagnostic summary for the current timestep."""

    if not SHOW_MOVEMENT_DIAGNOSTICS:

        return

    speed_wait_count = max(
        0,
        target_count - eligible_count
    )

    no_candidate_count = max(
        0,
        eligible_count - candidate_count
    )

    unresolved_ids = candidate_diagnostics.get(
        "unresolved_agent_ids",
        []
    )

    stuck_preview = unresolved_ids[
        :DIAGNOSTICS_MAX_AGENT_IDS
    ]

    time_seconds = time_step * TIME_STEP_SECONDS

    print(
        f"[Movement t={time_step:4d} | {time_seconds:7.2f}s] "
        f"active={active_count:4d} "
        f"targeted={target_count:4d} "
        f"targetless={targetless_count:3d} "
        f"eligible={eligible_count:4d} "
        f"speed_wait={speed_wait_count:4d} "
        f"with_candidates={candidate_count:4d} "
        f"no_candidates={no_candidate_count:4d} "
        f"moved={moved_count:4d}/{selected_count:4d}"
    )

    print(
        "    Path: "
        f"unreachable={candidate_diagnostics.get('unreachable_current_distance', 0)} "
        f"no_descending_neighbor={candidate_diagnostics.get('no_descending_neighbor', 0)}"
        " | Conflicts: "
        f"destination_groups={resolver_stats.get('destination_conflict_groups', 0)} "
        f"losers={resolver_stats.get('destination_conflict_losers', 0)} "
        f"dependency_blocked={resolver_stats.get('dependency_blocked', 0)} "
        f"retries={resolver_stats.get('candidate_retries', 0)} "
        f"head_on_delayed={resolver_stats.get('head_on_delayed_pairs', 0)} "
        f"rounds={resolver_stats.get('resolution_rounds', 0)}"
    )

    if stuck_preview:

        print(
            "    Unmoved Agent IDs with candidate paths: "
            f"{stuck_preview}"
        )

    no_path_ids = (
        candidate_diagnostics.get("unreachable_agent_ids", [])
        + candidate_diagnostics.get("no_descending_agent_ids", [])
    )

    if no_path_ids:

        print(
            "    Agent IDs without a descending candidate: "
            f"{no_path_ids[:DIAGNOSTICS_MAX_AGENT_IDS]}"
        )


# ============================================================
# 9. Main update function
# ============================================================

def update_agents(
    agents,
    terrain,
    exit_map,
    distance_fields,
    rng,
    time_step,
    stair_pairs,
    decision_state
):
    """
    Execute one synchronous CA time step on both floors.

    Update sequence:
        1. Process previous stair transit.
        2. Calculate regional SbFD speeds.
        3. Apply dynamic decision changes.
        4. Determine movement eligibility.
        5. Build candidate destinations.
        6. Resolve destination conflicts and occupancy dependencies.
        7. Apply positions simultaneously.
        8. Enter stair/exit transit when necessary.
        9. Print movement diagnostics.

    Returns:
        remaining_agents
        newly_evacuated
        moved_count
        newly_arrived_floor1
        decision_change_events
    """

    if not agents:

        return [], [], 0, [], []

    # ========================================================
    # 1. Snapshot current active occupancy
    # ========================================================

    occupied = {
        (
            agent.floor,
            agent.position[0],
            agent.position[1]
        ): agent.agent_id

        for agent in agents

        if (
            agent.status == "active"
            and agent.position is not None
        )
    }

    # ========================================================
    # 2. Process Agents already in transit
    # ========================================================

    (
        remaining_agents,
        newly_evacuated,
        newly_arrived_floor1
    ) = _process_transit_agents(
        agents,
        occupied,
        time_step
    )

    # ========================================================
    # 3. Separate active Agents
    # ========================================================

    active_agents = [
        agent
        for agent in remaining_agents

        if (
            agent.status == "active"
            and agent.position is not None
        )
    ]

    # ========================================================
    # 4. Calculate regional speed
    # ========================================================

    if time_step == 1:

        # Initial condition:
        # All pedestrians start at free-flow speed.

        region_speeds = _build_free_flow_region_speeds(
            terrain
        )

        for agent in active_agents:

            agent.speed_mps = FREE_FLOW_SPEED

    else:

        region_speeds = calculate_region_speeds(
            active_agents,
            terrain,
            exit_map,
            distance_fields,
            rng
        )

    # ========================================================
    # 5. Apply dynamic decision changes
    # ========================================================

    decision_change_events = apply_decision_changes(
        agents=active_agents,
        region_speeds=region_speeds,
        distance_fields=distance_fields,
        decision_state=decision_state,
        time_step=time_step,
        rng=rng
    )

    # Agent targets have now been updated.
    # Subsequent movement uses the new target exits.

    # ========================================================
    # 6. Rebuild current occupancy
    # ========================================================

    occupied = {
        (
            agent.floor,
            agent.position[0],
            agent.position[1]
        ): agent.agent_id

        for agent in active_agents

        if (
            agent.status == "active"
            and agent.position is not None
        )
    }

    # ========================================================
    # 7. Determine which Agents may attempt movement
    # ========================================================

    target_agents = [
        agent
        for agent in active_agents
        if agent.target_exit is not None
    ]

    targetless_active_count = (
        len(active_agents) - len(target_agents)
    )

    eligible_agents = []

    for agent in target_agents:

        probability = (
            agent.speed_mps / FREE_FLOW_SPEED
        )

        probability = float(
            np.clip(
                probability,
                MIN_SPEED_RATIO,
                1.0
            )
        )

        if rng.random() < probability:

            eligible_agents.append(agent)

    # ========================================================
    # 8. Build movement candidate lists
    # ========================================================

    candidate_diagnostics = {}

    candidate_lists = _build_candidate_lists(
        eligible_agents,
        terrain,
        exit_map,
        distance_fields,
        rng,
        diagnostics=candidate_diagnostics
    )

    # ========================================================
    # 9. Resolve movement conflicts and dependencies
    # ========================================================

    selected_movements, resolver_stats = _resolve_movements(
        candidate_lists,
        occupied,
        rng
    )

    # Agents that had at least one candidate but could not move.
    candidate_diagnostics["unresolved_agent_ids"] = [
        agent_id
        for agent_id in candidate_lists
        if agent_id not in selected_movements
    ]

    agent_by_id = {
        agent.agent_id: agent
        for agent in active_agents
    }

    # ========================================================
    # 10. Apply movements simultaneously
    # ========================================================

    moved_count = 0

    for agent_id, destination in selected_movements.items():

        agent = agent_by_id[agent_id]

        destination_floor, nr, nc = destination

        if destination_floor != agent.floor:

            raise RuntimeError(
                "Normal CA movement cannot directly change floors."
            )

        old_position = agent.position

        new_position = (
            int(nr),
            int(nc)
        )

        agent.position = new_position

        agent.last_direction = direction_from_positions(
            old_position,
            new_position
        )

        moved_count += 1

        destination_code = terrain[agent.floor][nr, nc]

        # ----------------------------------------------------
        # Floor 1: Ground-floor Exit
        # ----------------------------------------------------

        if (
            agent.floor == 1
            and destination_code == GROUND_EXIT
        ):

            agent.evacuation_time = time_step

            agent.position = None

            agent.status = "evacuated"

            newly_evacuated.append(agent)

            continue

        # ----------------------------------------------------
        # Floor 2: Upper-floor Exit
        # ----------------------------------------------------

        if (
            agent.floor == 2
            and destination_code == UPPER_EXIT
        ):

            agent.position = None

            agent.status = "transit"

            agent.transit_remaining = STAIR_TRANSIT_STEPS

            agent.transit_kind = "upper_exit"

            agent.transit_destination = None

            continue

        # ----------------------------------------------------
        # Floor 2: Stairway 6/7
        # ----------------------------------------------------

        if (
            agent.floor == 2
            and destination_code in STAIR_CODES
        ):

            stair_key = (
                2,
                int(nr),
                int(nc)
            )

            if stair_key not in stair_pairs:

                raise ValueError(
                    f"No Floor-1 stair match for {stair_key}."
                )

            agent.position = None

            agent.status = "transit"

            agent.transit_remaining = STAIR_TRANSIT_STEPS

            agent.transit_kind = "stair"

            agent.transit_destination = stair_pairs[stair_key]

            continue

    # ========================================================
    # 11. Print per-timestep diagnostics
    # ========================================================

    _print_movement_diagnostics(
        time_step=time_step,
        active_count=len(active_agents),
        target_count=len(target_agents),
        targetless_count=targetless_active_count,
        eligible_count=len(eligible_agents),
        candidate_count=len(candidate_lists),
        candidate_diagnostics=candidate_diagnostics,
        selected_count=len(selected_movements),
        moved_count=moved_count,
        resolver_stats=resolver_stats
    )

    # ========================================================
    # 12. Rebuild the remaining Agent list
    # ========================================================

    final_remaining = []

    included_ids = set()

    for agent in remaining_agents:

        if agent.status == "evacuated":

            continue

        if agent.agent_id not in included_ids:

            final_remaining.append(agent)

            included_ids.add(agent.agent_id)

    return (
        final_remaining,
        newly_evacuated,
        moved_count,
        newly_arrived_floor1,
        decision_change_events
    )