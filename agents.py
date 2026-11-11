"""
Agent definition and initialization for the two-floor CA model.
"""

from dataclasses import dataclass, field
import numpy as np

from config import *


@dataclass
class Agent:

    # Unique ID.
    agent_id: int

    # Current position: (row, col).
    # None when the Agent is in transit or has evacuated.
    position: tuple

    # Original position.
    initial_position: tuple

    # rational / habit / peer.
    agent_type: str

    # Current and initial floor.
    floor: int = 1
    initial_floor: int = 1

    # active / transit / evacuated.
    status: str = "active"

    # Current selected target ID.
    target_exit: int = None

    # Familiar target for Habit-Based Agents.
    familiar_exit: int = None

    # Rational concert satisfaction threshold.
    satisfaction_threshold_s: float = None

    # Evacuation time step.
    evacuation_time: int = None

    # Current assigned speed.
    speed_mps: float = FREE_FLOW_SPEED

    # Most recent actual movement direction.
    last_direction: str = None

    # Transit information.
    transit_remaining: int = 0

    # (destination_floor, destination_row, destination_col)
    transit_destination: tuple = None

    # "stair" or "upper_exit".
    transit_kind: str = None

    # --------------------------------------------------------
    # Decision-change history
    # --------------------------------------------------------

    # Each item:
    # (floor, region_row, region_col, congestion_episode)
    #
    # An Agent can attempt reconsideration only once
    # for each continuous congestion event in each region.

    attempted_congestion_events: set = field(
        default_factory=set
    )

    # Total number of eligible reconsideration attempts.
    # Includes Bernoulli failures.

    decision_change_attempts: int = 0

    # Number of times the actual target exit was changed.

    decision_changes: int = 0

    # Most recent successful target-change time step.

    last_decision_change_time: int = None

    # Previous target before the most recent successful change.

    last_decision_change_from: int = None

    # New target after the most recent successful change.

    last_decision_change_to: int = None


# ============================================================
# Initialize Agents on both floors
# ============================================================

def initialize_agents(
    terrain,
    rng,
    type_proportions=TYPE_PROPORTIONS
):
    """
    Create one Agent for every Horizontal Seat or Vertical Seat
    cell on both floors.
    """

    probabilities = np.array(
        [
            type_proportions.get(t, 0)
            for t in AGENT_TYPES
        ],
        dtype=float
    )

    if (
        np.any(probabilities < 0)
        or probabilities.sum() <= 0
    ):

        raise ValueError(
            "Invalid agent proportions."
        )

    probabilities /= probabilities.sum()

    agents = []

    next_agent_id = 1

    for floor in FLOORS:

        floor_terrain = terrain[floor]

        seat_positions = np.argwhere(
            np.isin(
                floor_terrain,
                list(SEAT_CODES)
            )
        )

        if len(seat_positions) == 0:

            raise ValueError(
                f"No Seat cells found on floor {floor}."
            )

        assigned_types = rng.choice(
            AGENT_TYPES,
            size=len(seat_positions),
            p=probabilities
        )

        for position, agent_type in zip(
            seat_positions,
            assigned_types
        ):

            position = (
                int(position[0]),
                int(position[1])
            )

            agents.append(
                Agent(
                    agent_id=next_agent_id,
                    position=position,
                    initial_position=position,
                    agent_type=str(agent_type),
                    floor=floor,
                    initial_floor=floor,
                    status="active",
                    speed_mps=FREE_FLOW_SPEED
                )
            )

            next_agent_id += 1

    rng.shuffle(agents)

    print("\nAgent initialization:")

    print("Total agents:", len(agents))

    for floor in FLOORS:

        count = sum(
            agent.initial_floor == floor
            for agent in agents
        )

        print(
            f"Initial Floor {floor}: {count} Agents"
        )

    for agent_type in AGENT_TYPES:

        count = sum(
            agent.agent_type == agent_type
            for agent in agents
        )

        print(
            f"{agent_type}: {count} "
            f"({count / len(agents) * 100:.1f}%)"
        )

    return agents