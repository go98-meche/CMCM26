"""
Save and visualize two-floor simulation results,
including dynamic decision-change history.
"""

import csv
import matplotlib.pyplot as plt

from config import *


# ============================================================
# 1. Save simulation history
# ============================================================

def _save_history(results):

    with open(
        HISTORY_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "time_step",
            "time_seconds",
            "evacuated",
            "remaining",
            "floor1_active",
            "floor2_active",
            "in_transit",
            "moved",
            "average_speed_mps",
            "decision_change_attempts",
            "decision_changes",
            "total_decision_change_attempts",
            "total_decision_changes"
        ])

        for item in results["history"]:

            writer.writerow([
                item["time_step"],
                item["time_seconds"],
                item["evacuated"],
                item["remaining"],
                item["floor1_active"],
                item["floor2_active"],
                item["in_transit"],
                item["moved"],
                item["average_speed_mps"],
                item.get("decision_change_attempts", 0),
                item.get("decision_changes", 0),
                item.get("total_decision_change_attempts", 0),
                item.get("total_decision_changes", 0)
            ])


# ============================================================
# 2. Save individual Agent results
# ============================================================

def _save_agent_results(results):

    evacuated_ids = {
        agent.agent_id
        for agent in results["evacuated_agents"]
    }

    with open(
        AGENT_RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "agent_id",
            "agent_type",

            "initial_floor",
            "initial_row",
            "initial_col",

            "final_floor",
            "final_row",
            "final_col",

            "familiar_exit",
            "initial_target_exit",
            "final_target_exit",

            "satisfaction_threshold_s",

            "decision_change_attempts",
            "decision_changes",

            "last_decision_change_time_step",
            "last_decision_change_time_seconds",

            "last_decision_change_from",
            "last_decision_change_to",

            "evacuation_time_step",
            "evacuation_time_seconds",

            "speed_mps",
            "last_direction",

            "status"
        ])

        for agent in results["all_agents"]:

            # ------------------------------------------------
            # Determine final position
            # ------------------------------------------------

            if agent.position is None:

                final_row = None
                final_col = None

            else:

                final_row = agent.position[0]
                final_col = agent.position[1]

            # ------------------------------------------------
            # Target IDs are displayed using 1-based numbering.
            # ------------------------------------------------

            familiar_exit = (
                agent.familiar_exit + 1
                if agent.familiar_exit is not None
                else None
            )

            final_target_exit = (
                agent.target_exit + 1
                if agent.target_exit is not None
                else None
            )

            # The first target is reconstructed from the
            # Agent's decision-change history when available.
            #
            # If the Agent never changed its target, its
            # final target is also its initial target.

            if agent.last_decision_change_from is not None:

                initial_target_exit = (
                    agent.last_decision_change_from + 1
                )

            else:

                initial_target_exit = final_target_exit

            # ------------------------------------------------
            # Decision-change timing
            # ------------------------------------------------

            if agent.last_decision_change_time is not None:

                last_change_time_seconds = (
                    agent.last_decision_change_time
                    * TIME_STEP_SECONDS
                )

            else:

                last_change_time_seconds = None

            # ------------------------------------------------
            # Evacuation timing
            # ------------------------------------------------

            if agent.evacuation_time is not None:

                evacuation_time_seconds = (
                    agent.evacuation_time
                    * TIME_STEP_SECONDS
                )

            else:

                evacuation_time_seconds = None

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            if agent.status == "transit":

                final_status = "In Transit"

            elif agent.agent_id in evacuated_ids:

                final_status = "Evacuated"

            else:

                final_status = "Remaining"

            # ------------------------------------------------
            # Write row
            # ------------------------------------------------

            writer.writerow([
                agent.agent_id,
                agent.agent_type,

                agent.initial_floor,
                agent.initial_position[0],
                agent.initial_position[1],

                agent.floor,
                final_row,
                final_col,

                familiar_exit,
                initial_target_exit,
                final_target_exit,

                agent.satisfaction_threshold_s,

                agent.decision_change_attempts,
                agent.decision_changes,

                agent.last_decision_change_time,
                last_change_time_seconds,

                (
                    agent.last_decision_change_from + 1
                    if agent.last_decision_change_from is not None
                    else None
                ),

                (
                    agent.last_decision_change_to + 1
                    if agent.last_decision_change_to is not None
                    else None
                ),

                agent.evacuation_time,
                evacuation_time_seconds,

                agent.speed_mps,
                agent.last_direction,

                final_status
            ])


# ============================================================
# 3. Save decision-change event history
# ============================================================

def _save_decision_change_history(results):

    with open(
        DECISION_CHANGE_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "time_step",
            "time_seconds",

            "agent_id",
            "agent_type",
            "floor",

            "region_row",
            "region_col",
            "congestion_episode",

            "regional_speed_mps",
            "congestion_threshold_mps",
            "congestion_severity",

            "p_max",
            "decision_change_probability",

            "old_target",
            "new_target",

            "bernoulli_success",
            "changed",

            "reason"
        ])

        for event in results["decision_change_events"]:

            old_target = (
                event["old_target"] + 1
                if event["old_target"] is not None
                else None
            )

            new_target = (
                event["new_target"] + 1
                if event["new_target"] is not None
                else None
            )

            writer.writerow([
                event["time_step"],
                event["time_seconds"],

                event["agent_id"],
                event["agent_type"],
                event["floor"],

                event["region_row"],
                event["region_col"],
                event["congestion_episode"],

                event["regional_speed_mps"],
                event["congestion_threshold_mps"],
                event["congestion_severity"],

                event["p_max"],
                event["decision_change_probability"],

                old_target,
                new_target,

                event["bernoulli_success"],
                event["changed"],

                event["reason"]
            ])


# ============================================================
# 4. Save all results
# ============================================================

def save_simulation_results(results):

    _save_history(results)

    _save_agent_results(results)

    _save_decision_change_history(results)

    print("\nResult files saved:")

    print(HISTORY_CSV)

    print(AGENT_RESULTS_CSV)

    print(DECISION_CHANGE_CSV)


# ============================================================
# 5. Plot evacuation and decision-change curves
# ============================================================

def plot_evacuation_curve(results):

    history = results["history"]

    time_seconds = [
        item["time_seconds"]
        for item in history
    ]

    evacuated = [
        item["evacuated"]
        for item in history
    ]

    remaining = [
        item["remaining"]
        for item in history
    ]

    floor1_active = [
        item["floor1_active"]
        for item in history
    ]

    floor2_active = [
        item["floor2_active"]
        for item in history
    ]

    in_transit = [
        item["in_transit"]
        for item in history
    ]

    cumulative_changes = [
        item.get("total_decision_changes", 0)
        for item in history
    ]

    # --------------------------------------------------------
    # Create three panels
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 11),
        sharex=True
    )

    # --------------------------------------------------------
    # Plot 1: Total evacuation
    # --------------------------------------------------------

    axes[0].plot(
        time_seconds,
        evacuated,
        label="Evacuated",
        linewidth=2
    )

    axes[0].plot(
        time_seconds,
        remaining,
        label="Remaining",
        linewidth=2
    )

    axes[0].set_ylabel(
        "Number of Agents"
    )

    axes[0].set_title(
        "Total Evacuation Progress",
        weight="bold"
    )

    axes[0].grid(alpha=0.3)

    axes[0].legend()

    # --------------------------------------------------------
    # Plot 2: Distribution by floor
    # --------------------------------------------------------

    axes[1].plot(
        time_seconds,
        floor1_active,
        label="Floor 1",
        linewidth=2
    )

    axes[1].plot(
        time_seconds,
        floor2_active,
        label="Floor 2",
        linewidth=2
    )

    axes[1].plot(
        time_seconds,
        in_transit,
        label="In Stair Transit",
        linewidth=2
    )

    axes[1].set_ylabel(
        "Number of Agents"
    )

    axes[1].set_title(
        "Agent Distribution by Floor",
        weight="bold"
    )

    axes[1].grid(alpha=0.3)

    axes[1].legend()

    # --------------------------------------------------------
    # Plot 3: Cumulative successful decision changes
    # --------------------------------------------------------

    axes[2].plot(
        time_seconds,
        cumulative_changes,
        label="Successful Decision Changes",
        linewidth=2
    )

    axes[2].set_xlabel(
        "Simulated Time (seconds)"
    )

    axes[2].set_ylabel(
        "Number of Changes"
    )

    axes[2].set_title(
        "Dynamic Decision Changes",
        weight="bold"
    )

    axes[2].grid(alpha=0.3)

    axes[2].legend()

    plt.tight_layout()

    plt.savefig(
        CURVE_PNG,
        dpi=220,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)

    print(
        "Evacuation curve saved:",
        CURVE_PNG
    )