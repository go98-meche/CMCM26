"""
Dynamic decision-change model.

Agents can reconsider their target exit when the region they
occupy becomes congested.

The probability of reconsideration depends on:
1. Regional congestion severity.
2. The Agent's behavioral type.

Each Agent:
1. Can attempt only once during one continuous congestion
   event in a particular region.
2. Must satisfy the minimum interval since the most recent
   successful target change.

When an Agent decides to reconsider, the new target is selected
according to the most popular target among other active Agents
in the same region.
"""

import numpy as np

from config import *


# ============================================================
# 1. Initialize decision-change state
# ============================================================

def initialize_decision_change_state():
    """
    Initialize persistent congestion-event information.

    Region key:
        (floor, region_row, region_col)

    Each region stores:
        congested
        episode
        speed
        severity

    Episode increases whenever a region transitions from
    non-congested to congested.
    """

    return {
        "regions": {},

        "total_attempts": 0,

        "total_changes": 0
    }


# ============================================================
# 2. Update congestion events
# ============================================================

def update_congestion_events(
    region_speeds,
    decision_state
):
    """
    Update the congestion state of every macroscopic region.

    A new congestion episode starts when:

        previous state = not congested
        current state  = congested

    If a region remains congested, its episode number
    does not change.

    If the region becomes uncongested and later congested
    again, a new episode is created.

    Returns
    -------
    dict
        Current congestion information for every region.
    """

    threshold = DECISION_CHANGE_SPEED_THRESHOLD

    if threshold <= 0:

        raise ValueError(
            "DECISION_CHANGE_SPEED_THRESHOLD must be positive."
        )

    current_regions = {}

    region_records = decision_state["regions"]

    for floor, floor_regions in region_speeds.items():

        for region_position, speed in floor_regions.items():

            region_row, region_col = region_position

            region_key = (
                int(floor),
                int(region_row),
                int(region_col)
            )

            speed = float(speed)

            # ------------------------------------------------
            # Determine whether the region is congested
            # ------------------------------------------------

            congested = (
                speed < threshold
            )

            previous_record = region_records.get(
                region_key
            )

            if previous_record is None:

                previous_congested = False

                episode = 0

            else:

                previous_congested = (
                    previous_record["congested"]
                )

                episode = (
                    previous_record["episode"]
                )

            # ------------------------------------------------
            # Start a new congestion episode
            # ------------------------------------------------

            if congested and not previous_congested:

                episode += 1

            # ------------------------------------------------
            # Congestion severity
            # ------------------------------------------------

            if congested:

                severity = (
                    1.0 - speed / threshold
                )

                severity = float(
                    np.clip(
                        severity,
                        0.0,
                        1.0
                    )
                )

            else:

                severity = 0.0

            # ------------------------------------------------
            # Store current state
            # ------------------------------------------------

            record = {
                "congested": congested,
                "episode": episode,
                "speed": speed,
                "severity": severity
            }

            region_records[region_key] = record

            current_regions[region_key] = record

    return current_regions


# ============================================================
# 3. Get regional agents
# ============================================================

def group_agents_by_region(
    agents
):
    """
    Group active Agents by floor and macroscopic region.

    Agents in stair transit are excluded.

    Returns:
        {
            (floor, region_row, region_col): [Agent, ...]
        }
    """

    agents_by_region = {}

    for agent in agents:

        if (
            agent.status != "active"
            or agent.position is None
        ):
            continue

        r, c = agent.position

        region_key = (
            agent.floor,
            r // REGION_SIZE,
            c // REGION_SIZE
        )

        agents_by_region.setdefault(
            region_key,
            []
        ).append(agent)

    return agents_by_region


# ============================================================
# 4. Get majority target inside a region
# ============================================================

def choose_majority_target(
    agent,
    region_agents,
    target_snapshot,
    distance_fields,
    rng
):
    """
    Select the most frequently preferred reachable target
    among other active Agents in the same region.

    The Agent itself is excluded from voting.

    Only targets reachable by this Agent are considered.

    If several targets have equal maximum votes, select one
    randomly among the tied targets.

    Returns:
        selected_target, vote_counts

    If no valid neighbor votes exist:
        None, vote_counts
    """

    floor = agent.floor

    r, c = agent.position

    # --------------------------------------------------------
    # Find targets reachable by this Agent
    # --------------------------------------------------------

    reachable_targets = {
        target_id

        for target_id, distances
        in distance_fields[floor].items()

        if np.isfinite(
            distances[r, c]
        )
    }

    if not reachable_targets:

        return None, {}

    # --------------------------------------------------------
    # Count preferences among other Agents
    # --------------------------------------------------------

    vote_counts = {
        target_id: 0
        for target_id in reachable_targets
    }

    for neighbor in region_agents:

        # Do not count the Agent's own preference.
        if neighbor.agent_id == agent.agent_id:
            continue

        # Only active Agents on the same floor and in
        # the same region are included in region_agents.
        target = target_snapshot.get(
            neighbor.agent_id
        )

        if target in reachable_targets:

            vote_counts[target] += 1

    # --------------------------------------------------------
    # No available votes
    # --------------------------------------------------------

    total_votes = sum(
        vote_counts.values()
    )

    if total_votes == 0:

        return None, vote_counts

    # --------------------------------------------------------
    # Find the modal target
    # --------------------------------------------------------

    maximum_votes = max(
        vote_counts.values()
    )

    most_popular_targets = [
        target_id

        for target_id, count in vote_counts.items()

        if count == maximum_votes
    ]

    # --------------------------------------------------------
    # Resolve ties randomly
    # --------------------------------------------------------

    selected_target = int(
        rng.choice(most_popular_targets)
    )

    return selected_target, vote_counts


# ============================================================
# 5. Apply dynamic decision changes
# ============================================================

def apply_decision_changes(
    agents,
    region_speeds,
    distance_fields,
    decision_state,
    time_step,
    rng
):
    """
    Apply the dynamic decision-change model.

    For each active Agent:

        1. Identify its current region.
        2. Check whether the region is congested.
        3. Check whether the Agent has already attempted
           reconsideration during this congestion episode.
        4. Check the minimum time interval since the last
           successful decision change.
        5. Mark the attempt as consumed.
        6. Calculate congestion-dependent probability.
        7. Perform Bernoulli sampling.
        8. If successful, choose the modal target among
           other Agents in the same region.

    Important:
        An attempt is consumed even if the Bernoulli trial
        fails or the majority vote does not produce a
        different target.

    The cooldown timestamp is updated only when the
    target exit actually changes.

    Returns
    -------
    list
        Decision-change event records for this timestep.
    """

    # ========================================================
    # A. Update congestion status and episode numbers
    # ========================================================

    current_regions = update_congestion_events(
        region_speeds,
        decision_state
    )

    # ========================================================
    # B. Group active Agents by region
    # ========================================================

    agents_by_region = group_agents_by_region(
        agents
    )

    # ========================================================
    # C. Snapshot every Agent's existing target
    # ========================================================

    # This guarantees simultaneous peer voting.

    target_snapshot = {
        agent.agent_id: agent.target_exit
        for agent in agents
    }

    # ========================================================
    # D. Prepare event log
    # ========================================================

    events = []

    # ========================================================
    # E. Check each congested region
    # ========================================================

    for region_key, region_record in current_regions.items():

        if not region_record["congested"]:

            continue

        if region_key not in agents_by_region:

            continue

        region_agents = agents_by_region[
            region_key
        ]

        floor, region_row, region_col = region_key

        regional_speed = region_record["speed"]

        severity = region_record["severity"]

        episode = region_record["episode"]

        event_key = (
            floor,
            region_row,
            region_col,
            episode
        )

        # ====================================================
        # F. Check Agents inside this region
        # ====================================================

        for agent in region_agents:

            if agent.target_exit is None:

                continue

            # ------------------------------------------------
            # Restriction 1:
            # Only one attempt per congestion episode
            # ------------------------------------------------

            if event_key in agent.attempted_congestion_events:

                continue

            # ------------------------------------------------
            # Restriction 2:
            # Minimum interval between successful changes
            # ------------------------------------------------

            if agent.last_decision_change_time is not None:

                elapsed_seconds = (
                    time_step
                    - agent.last_decision_change_time
                ) * TIME_STEP_SECONDS

                if (
                    elapsed_seconds
                    < MIN_DECISION_CHANGE_INTERVAL_SECONDS
                ):

                    # The attempt is not allowed yet.
                    #
                    # Do NOT consume this congestion episode.
                    # The Agent may still attempt later if
                    # the current congestion event continues.

                    continue

            # =================================================
            # G. Both restrictions satisfied:
            #    consume the attempt before Bernoulli sampling
            # =================================================

            agent.attempted_congestion_events.add(
                event_key
            )

            agent.decision_change_attempts += 1

            decision_state["total_attempts"] += 1

            # ------------------------------------------------
            # Calculate individual probability
            # ------------------------------------------------

            p_max = DECISION_CHANGE_P_MAX[
                agent.agent_type
            ]

            if not 0.0 <= p_max <= 1.0:

                raise ValueError(
                    f"Invalid p_max for {agent.agent_type}: "
                    f"{p_max}"
                )

            probability = (
                p_max * severity
            )

            probability = float(
                np.clip(
                    probability,
                    0.0,
                    1.0
                )
            )

            old_target = agent.target_exit

            # ------------------------------------------------
            # Build the attempt record
            # ------------------------------------------------

            event = {
                "time_step": time_step,

                "time_seconds": (
                    time_step * TIME_STEP_SECONDS
                ),

                "agent_id": agent.agent_id,

                "agent_type": agent.agent_type,

                "floor": floor,

                "region_row": region_row,

                "region_col": region_col,

                "congestion_episode": episode,

                "regional_speed_mps": regional_speed,

                "congestion_threshold_mps": (
                    DECISION_CHANGE_SPEED_THRESHOLD
                ),

                "congestion_severity": severity,

                "p_max": p_max,

                "decision_change_probability": probability,

                "old_target": old_target,

                "new_target": old_target,

                "bernoulli_success": False,

                "changed": False,

                "reason": ""
            }

            # =================================================
            # H. Bernoulli experiment
            # =================================================

            reconsider = (
                rng.random() < probability
            )

            if not reconsider:

                event["reason"] = "bernoulli_failed"

                events.append(event)

                continue

            event["bernoulli_success"] = True

            # =================================================
            # I. Treat the decision as Peer-Influenced
            # =================================================

            new_target, vote_counts = choose_majority_target(
                agent,
                region_agents,
                target_snapshot,
                distance_fields,
                rng
            )

            if new_target is None:

                event["reason"] = "no_valid_peer_votes"

                events.append(event)

                continue

            event["new_target"] = new_target

            # ------------------------------------------------
            # If majority agrees with existing target
            # ------------------------------------------------

            if new_target == old_target:

                event["reason"] = "same_target"

                events.append(event)

                continue

            # =================================================
            # J. Apply a successful decision change
            # =================================================

            agent.target_exit = new_target

            agent.decision_changes += 1

            agent.last_decision_change_time = time_step

            agent.last_decision_change_from = old_target

            agent.last_decision_change_to = new_target

            decision_state["total_changes"] += 1

            event["changed"] = True

            event["reason"] = "changed"

            events.append(event)

    return events