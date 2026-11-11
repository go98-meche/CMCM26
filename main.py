"""
Bailey Hall — Two-Floor Main CA Program
CMCM 2026
"""

import numpy as np

from config import *

from terrain import load_and_visualize_terrain

from agents import initialize_agents

from exits import identify_exits

from pathfinding import (
    calculate_distance_fields,
    calculate_travel_time_fields
)

from decision import assign_target_exits

from simulation import simulate

from visualization import EvacuationAnimator

from results import (
    save_simulation_results,
    plot_evacuation_curve
)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 0. Initialize random generator
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # 1. Load both floor maps
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("STEP 1: TWO-FLOOR TERRAIN INITIALIZATION")
    print("=" * 55)

    terrain = load_and_visualize_terrain(
        show=False
    )

    for floor in FLOORS:

        print(
            f"Floor {floor} terrain shape:",
            terrain[floor].shape
        )

    # --------------------------------------------------------
    # 2. Identify exits and stair connectors
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("STEP 2: EXIT AND STAIR IDENTIFICATION")
    print("=" * 55)

    exits, exit_map, stair_pairs = identify_exits(
        terrain
    )

    # --------------------------------------------------------
    # 3. Calculate geometric distance fields
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("STEP 3: TWO-FLOOR PATHFINDING")
    print("=" * 55)

    distance_fields = calculate_distance_fields(
        terrain,
        exits,
        exit_map
    )

    # --------------------------------------------------------
    # 4. Initialize Agents on both floors
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("STEP 4: TWO-FLOOR AGENT INITIALIZATION")
    print("=" * 55)

    agents = initialize_agents(
        terrain,
        rng=rng
    )

    # --------------------------------------------------------
    # 5. Calculate initial travel-time fields
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("STEP 5: INITIAL TRAVEL-TIME CALCULATION")
    print("=" * 55)

    # At time zero, every Agent has free-flow speed.

    initial_cell_speeds = {
        floor: np.full(
            terrain[floor].shape,
            FREE_FLOW_SPEED,
            dtype=float
        )
        for floor in FLOORS
    }

    travel_time_fields = calculate_travel_time_fields(
        terrain,
        exits,
        exit_map,
        initial_cell_speeds
    )

    # --------------------------------------------------------
    # 6. Assign initial target destinations
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("STEP 6: INITIAL TARGET SELECTION")
    print("=" * 55)

    baseline_load = assign_target_exits(
        agents=agents,
        exits=exits,
        distance_fields=distance_fields,
        rng=rng,
        travel_time_fields=travel_time_fields,
        scenario=EVACUATION_SCENARIO,
        terrain=terrain
    )

    # --------------------------------------------------------
    # 7. Display initial target distributions by floor
    # --------------------------------------------------------

    print("\nInitial target distributions:")

    for floor in FLOORS:

        print(f"\nFloor {floor}:")

        floor_agents = [
            agent
            for agent in agents
            if agent.floor == floor
        ]

        floor_targets = [
            target
            for target in exits
            if target["floor"] == floor
        ]

        for target in floor_targets:

            target_id = target["id"]

            count = sum(
                agent.target_exit == target_id
                for agent in floor_agents
            )

            print(
                f"Target {target_id} "
                f"(code {target['code']}): {count}"
            )

    # --------------------------------------------------------
    # 8. Initialize dynamic visualization
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("STEP 8: INITIALIZE TWO-FLOOR VISUALIZATION")
    print("=" * 55)

    if ANIMATE:

        animator = EvacuationAnimator(
            terrain,
            interval=FRAME_INTERVAL
        )

    else:

        animator = None

    # --------------------------------------------------------
    # 9. Print physical parameters
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("MODEL PARAMETERS")
    print("=" * 55)

    print(
        f"Cell size: {CELL_SIZE_M:.2f} m"
    )

    print(
        f"Free-flow speed: {FREE_FLOW_SPEED:.3f} m/s"
    )

    print(
        f"Physical time per step: "
        f"{TIME_STEP_SECONDS:.4f} s"
    )

    print(
        f"Region size: "
        f"{REGION_SIZE} x {REGION_SIZE} cells"
    )

    print(
        f"Stair transit: "
        f"{STAIR_TRANSIT_STEPS} steps"
    )

    print(
        f"Stair transit physical time: "
        f"{STAIR_TRANSIT_STEPS * TIME_STEP_SECONDS:.2f} s"
    )

    print(
        f"Scenario: {EVACUATION_SCENARIO}"
    )

    # --------------------------------------------------------
    # 10. Run two-floor CA
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("STEP 10: TWO-FLOOR CELLULAR AUTOMATON")
    print("=" * 55)

    results = simulate(
        agents=agents,
        terrain=terrain,
        exits=exits,
        exit_map=exit_map,
        distance_fields=distance_fields,
        travel_time_fields=travel_time_fields,
        stair_pairs=stair_pairs,
        rng=rng,
        animator=animator,
        max_steps=MAX_STEPS,
        scenario=EVACUATION_SCENARIO
    )

    # --------------------------------------------------------
    # 11. Save results
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("STEP 11: SAVE RESULTS")
    print("=" * 55)

    save_simulation_results(
        results
    )

    # --------------------------------------------------------
    # 12. Print final summary
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("FINAL TWO-FLOOR SIMULATION RESULTS")
    print("=" * 55)

    total = results["initial_agents"]

    evacuated = results["total_evacuated"]

    remaining = results["total_remaining"]

    floor1_remaining = sum(
        agent.status == "active"
        and agent.floor == 1
        for agent in results["remaining_agents"]
    )

    floor2_remaining = sum(
        agent.status == "active"
        and agent.floor == 2
        for agent in results["remaining_agents"]
    )

    in_transit = sum(
        agent.status == "transit"
        for agent in results["remaining_agents"]
    )

    print(
        f"Initial population: {total}"
    )

    print(
        f"Evacuated: {evacuated}"
    )

    print(
        f"Remaining: {remaining}"
    )

    print(
        f"Floor 1 active: {floor1_remaining}"
    )

    print(
        f"Floor 2 active: {floor2_remaining}"
    )

    print(
        f"In transit: {in_transit}"
    )

    print(
        f"Time steps: {results['total_steps']}"
    )

    print(
        f"Simulated time: "
        f"{results['total_time_seconds']:.2f} seconds"
    )

    print(
        f"Status: {results['status']}"
    )

    if total > 0:

        print(
            f"Evacuation rate: "
            f"{evacuated / total * 100:.2f}%"
        )

    if results["status"] == "stalled":

        print(
            "WARNING: Simulation stalled. "
            "Check floor connectivity, stair pairing, "
            "and congestion."
        )

    elif results["status"] == "max_steps":

        print(
            "WARNING: Maximum time steps reached."
        )

    else:

        print(
            "All Agents successfully evacuated."
        )

    # --------------------------------------------------------
    # 13. Keep final animation frame on screen
    # --------------------------------------------------------

    if animator is not None:

        animator.finish()

    # --------------------------------------------------------
    # 14. Plot evacuation curves
    # --------------------------------------------------------

    plot_evacuation_curve(
        results
    )


# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":

    main()