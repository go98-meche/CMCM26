"""
Main two-floor Cellular Automaton simulation loop.

Includes dynamic decision changes.
"""

import numpy as np

from movement import update_agents

from decision import assign_target_exits

from decision_change import initialize_decision_change_state

from config import *


# ============================================================
# Run the complete two-floor CA simulation
# ============================================================

def simulate(
    agents,
    terrain,
    exits,
    exit_map,
    distance_fields,
    travel_time_fields,
    stair_pairs,
    rng,
    animator=None,
    max_steps=MAX_STEPS,
    scenario=EVACUATION_SCENARIO
):
    """
    Run the complete two-floor evacuation simulation.

    At each time step:

        1. Update transit Agents.
        2. Calculate regional speeds.
        3. Detect congestion events.
        4. Allow eligible Agents to reconsider their targets.
        5. Move Agents.
        6. Transfer Agents through stairways.
        7. Reassign targets to newly arrived Floor-1 Agents.
        8. Record simulation statistics.

    Stop when:

        - Every Agent has evacuated.
        - No movement, transit progress, or successful
          decision changes occur for NO_MOVEMENT_LIMIT
          consecutive steps.
        - Maximum time steps are reached.
    """

    initial_agents = len(agents)

    all_agents = list(agents)

    active_agents = list(agents)

    evacuated_agents = []

    decision_change_events = []

    # Persistent state for congestion episodes.
    # Initialized ONCE, not inside the timestep loop.

    decision_state = initialize_decision_change_state()

    consecutive_no_movement = 0

    history = [
        {
            "time_step": 0,
            "time_seconds": 0.0,

            "evacuated": 0,
            "remaining": initial_agents,

            "moved": 0,

            "floor1_active": sum(
                agent.status == "active"
                and agent.floor == 1
                for agent in agents
            ),

            "floor2_active": sum(
                agent.status == "active"
                and agent.floor == 2
                for agent in agents
            ),

            "in_transit": 0,

            "average_speed_mps": FREE_FLOW_SPEED,

            "decision_change_attempts": 0,

            "decision_changes": 0,

            "total_decision_change_attempts": 0,

            "total_decision_changes": 0
        }
    ]

    status = (
        "completed"
        if initial_agents == 0
        else "max_steps"
    )

    print("\nStarting two-floor CA simulation...")

    print(
        f"Physical time per CA step: "
        f"{TIME_STEP_SECONDS:.4f} s"
    )

    print(
        f"Congestion speed threshold: "
        f"{DECISION_CHANGE_SPEED_THRESHOLD:.3f} m/s"
    )

    print(
        f"Minimum interval between successful changes: "
        f"{MIN_DECISION_CHANGE_INTERVAL_SECONDS:.2f} s"
    )

    print("\nMaximum decision-change probabilities:")

    for agent_type, probability in DECISION_CHANGE_P_MAX.items():

        print(
            f"  {agent_type}: {probability:.2f}"
        )

    # --------------------------------------------------------
    # 1. Display initial state
    # --------------------------------------------------------

    if animator is not None:

        animator.update(
            agents=active_agents,
            time_step=0,
            evacuated=0,
            total=initial_agents,
            moved=0
        )

    # --------------------------------------------------------
    # 2. Time-step loop
    # --------------------------------------------------------

    if initial_agents > 0:

        for time_step in range(
            1,
            max_steps + 1
        ):

            # ------------------------------------------------
            # Store transit countdowns before the update.
            #
            # Used to distinguish genuine stair progress
            # from waiting for an occupied landing cell.
            # ------------------------------------------------

            transit_before = {
                agent.agent_id: agent.transit_remaining
                for agent in active_agents
                if agent.status == "transit"
            }

            # ------------------------------------------------
            # 3. Update Agents
            # ------------------------------------------------

            (
                active_agents,
                newly_evacuated,
                moved_count,
                newly_arrived_floor1,
                new_decision_events
            ) = update_agents(
                agents=active_agents,
                terrain=terrain,
                exit_map=exit_map,
                distance_fields=distance_fields,
                rng=rng,
                time_step=time_step,
                stair_pairs=stair_pairs,
                decision_state=decision_state
            )

            # ------------------------------------------------
            # 4. Save decision-change events
            # ------------------------------------------------

            decision_change_events.extend(
                new_decision_events
            )

            decision_attempts_this_step = len(
                new_decision_events
            )

            decision_changes_this_step = sum(
                event["changed"]
                for event in new_decision_events
            )

            # ------------------------------------------------
            # 5. Reassign target exits after stair arrival
            # ------------------------------------------------

            if newly_arrived_floor1:

                print(
                    f"\nStep {time_step}: "
                    f"{len(newly_arrived_floor1)} Agent(s) "
                    "arrived on Floor 1."
                )

                assign_target_exits(
                    agents=newly_arrived_floor1,

                    exits=exits,

                    distance_fields=distance_fields,

                    rng=rng,

                    travel_time_fields=travel_time_fields,

                    scenario=scenario,

                    terrain=terrain,

                    all_agents=active_agents
                )

            # ------------------------------------------------
            # 6. Update evacuated population
            # ------------------------------------------------

            evacuated_agents.extend(
                newly_evacuated
            )

            total_evacuated = len(
                evacuated_agents
            )

            remaining = len(
                active_agents
            )

            # ------------------------------------------------
            # 7. Count Agents on both floors
            # ------------------------------------------------

            floor1_active = sum(
                agent.status == "active"
                and agent.floor == 1
                and agent.position is not None
                for agent in active_agents
            )

            floor2_active = sum(
                agent.status == "active"
                and agent.floor == 2
                and agent.position is not None
                for agent in active_agents
            )

            in_transit = sum(
                agent.status == "transit"
                for agent in active_agents
            )

            # ------------------------------------------------
            # 8. Check whether transit made progress
            # ------------------------------------------------

            transit_progress = any(
                agent.status == "transit"
                and agent.agent_id in transit_before
                and agent.transit_remaining
                < transit_before[agent.agent_id]
                for agent in active_agents
            )

            # ------------------------------------------------
            # 9. Update no-progress counter
            # ------------------------------------------------

            if (
                moved_count > 0
                or transit_progress
                or newly_arrived_floor1
                or newly_evacuated
                or decision_changes_this_step > 0
            ):

                consecutive_no_movement = 0

            else:

                consecutive_no_movement += 1

            # ------------------------------------------------
            # 10. Calculate average assigned speed
            # ------------------------------------------------

            moving_agents = [
                agent
                for agent in active_agents
                if (
                    agent.status == "active"
                    and agent.position is not None
                )
            ]

            if moving_agents:

                average_speed = float(
                    np.mean([
                        agent.speed_mps
                        for agent in moving_agents
                    ])
                )

            else:

                average_speed = FREE_FLOW_SPEED

            physical_time = (
                time_step * TIME_STEP_SECONDS
            )

            # ------------------------------------------------
            # 11. Record cumulative decision-change statistics
            # ------------------------------------------------

            total_decision_attempts = (
                decision_state["total_attempts"]
            )

            total_decision_changes = (
                decision_state["total_changes"]
            )

            # ------------------------------------------------
            # 12. Record history
            # ------------------------------------------------

            history.append({

                "time_step": time_step,

                "time_seconds": physical_time,

                "evacuated": total_evacuated,

                "remaining": remaining,

                "moved": moved_count,

                "floor1_active": floor1_active,

                "floor2_active": floor2_active,

                "in_transit": in_transit,

                "average_speed_mps": average_speed,

                "decision_change_attempts": (
                    decision_attempts_this_step
                ),

                "decision_changes": (
                    decision_changes_this_step
                ),

                "total_decision_change_attempts": (
                    total_decision_attempts
                ),

                "total_decision_changes": (
                    total_decision_changes
                ),

                "consecutive_no_movement": (
                    consecutive_no_movement
                )
            })

            # ------------------------------------------------
            # 13. Update visualization
            # ------------------------------------------------

            if animator is not None:

                animator.update(
                    agents=active_agents,
                    time_step=time_step,
                    evacuated=total_evacuated,
                    total=initial_agents,
                    moved=moved_count
                )

            # ------------------------------------------------
            # 14. Print progress
            # ------------------------------------------------

            if time_step % 10 == 0:

                print(
                    f"Step {time_step:4d} | "
                    f"Time: {physical_time:7.2f} s | "
                    f"Evacuated: {total_evacuated:4d} | "
                    f"Floor 1: {floor1_active:4d} | "
                    f"Floor 2: {floor2_active:4d} | "
                    f"Transit: {in_transit:4d} | "
                    f"Moved: {moved_count:4d} | "
                    f"Decision attempts: "
                    f"{decision_attempts_this_step:3d} | "
                    f"Decision changes: "
                    f"{decision_changes_this_step:3d}"
                )

            # ------------------------------------------------
            # 15. Stop conditions
            # ------------------------------------------------

            if remaining == 0:

                status = "completed"

                break

            if (
                consecutive_no_movement
                >= NO_MOVEMENT_LIMIT
            ):

                status = "stalled"

                print(
                    "\nSimulation stalled after "
                    f"{NO_MOVEMENT_LIMIT} consecutive "
                    "steps without physical or decision progress."
                )

                print("\nRemaining Agent diagnostics:")

                for agent in active_agents[:30]:

                    print(
                        f"Agent {agent.agent_id} | "
                        f"Floor: {agent.floor} | "
                        f"Position: {agent.position} | "
                        f"Status: {agent.status} | "
                        f"Target: {agent.target_exit} | "
                        f"Speed: {agent.speed_mps:.3f} | "
                        f"Attempts: {agent.decision_change_attempts} | "
                        f"Changes: {agent.decision_changes}"
                    )

                break

        else:

            status = "max_steps"

    # --------------------------------------------------------
    # 16. Collect final results
    # --------------------------------------------------------

    total_steps = history[-1]["time_step"]

    total_time_seconds = (
        total_steps * TIME_STEP_SECONDS
    )

    results = {

        "status": status,

        "initial_agents": initial_agents,

        "all_agents": all_agents,

        "evacuated_agents": evacuated_agents,

        "remaining_agents": active_agents,

        "history": history,

        "total_steps": total_steps,

        "total_time_seconds": total_time_seconds,

        "time_step_seconds": TIME_STEP_SECONDS,

        "total_evacuated": len(evacuated_agents),

        "total_remaining": len(active_agents),

        "consecutive_no_movement": (
            consecutive_no_movement
        ),

        "decision_change_events": decision_change_events,

        "total_decision_change_attempts": (
            decision_state["total_attempts"]
        ),

        "total_decision_changes": (
            decision_state["total_changes"]
        )
    }

    print("\nTwo-floor CA simulation finished.")

    print(
        "\nTotal decision-change attempts:",
        results["total_decision_change_attempts"]
    )

    print(
        "Total successful target changes:",
        results["total_decision_changes"]
    )

    return results