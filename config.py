"""
Global configuration for the two-floor CA evacuation model.
"""

from pathlib import Path
import math

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# FILE PATHS
# ============================================================

FLOOR_MAPS = {
    1: BASE_DIR / "map_floor1_concert.xlsx",
    2: BASE_DIR / "map_floor2_concert.xlsx"
}

FLOOR_SHEET_NAMES = {
    1: "Sheet1",
    2: "Sheet1"
}

FLOOR_MAP_RANGES = {
    1: None,
    2: None
}

XLSX_PATH = FLOOR_MAPS[1]
SHEET_NAME = FLOOR_SHEET_NAMES[1]
MAP_RANGE = FLOOR_MAP_RANGES[1]

TERRAIN_CSVS = {
    1: BASE_DIR / "map_floor1_terrain.csv",
    2: BASE_DIR / "map_floor2_terrain.csv"
}

TERRAIN_PNGS = {
    1: BASE_DIR / "map_floor1_terrain.png",
    2: BASE_DIR / "map_floor2_terrain.png"
}

TERRAIN_CSV = TERRAIN_CSVS[1]
TERRAIN_PNG = TERRAIN_PNGS[1]

HISTORY_CSV = BASE_DIR / "evacuation_history.csv"

AGENT_RESULTS_CSV = BASE_DIR / "agent_results.csv"

CURVE_PNG = BASE_DIR / "evacuation_curve.png"

# New output file for decision-change events.
DECISION_CHANGE_CSV = (
    BASE_DIR / "decision_change_history.csv"
)


# ============================================================
# FLOOR IDENTIFIERS
# ============================================================

FLOORS = (1, 2)

GROUND_FLOOR = 1
UPPER_FLOOR = 2


# ============================================================
# TERRAIN CLASSIFICATION
# ============================================================

# 0 = Wall
# 1 = Open Space
# 2 = Horizontal Seat
# 3 = Vertical Seat
# 4 = Ground-floor Exit
# 5 = Upper-floor Exit
# 6 = Stairway 6
# 7 = Stairway 7

WALL = 0
OBSTACLE = WALL

OPEN_SPACE = 1
HORIZONTAL_SEAT = 2
VERTICAL_SEAT = 3

GROUND_EXIT = 4
UPPER_EXIT = 5

STAIR_6 = 6
STAIR_7 = 7

EXIT = GROUND_EXIT

SEAT_CODES = {
    HORIZONTAL_SEAT,
    VERTICAL_SEAT
}

STAIR_CODES = {
    STAIR_6,
    STAIR_7
}

EXCEL_TO_TERRAIN = {
    0: WALL,
    1: OPEN_SPACE,
    2: HORIZONTAL_SEAT,
    3: VERTICAL_SEAT,
    4: GROUND_EXIT,
    5: UPPER_EXIT,
    6: STAIR_6,
    7: STAIR_7
}

EMPTY_AS = WALL

TARGET_CODES_BY_FLOOR = {
    1: {GROUND_EXIT},
    2: {UPPER_EXIT, STAIR_6, STAIR_7}
}


# ============================================================
# STAIR CONNECTORS
# ============================================================

STAIR_TRANSIT_SECONDS = 5.0

STAIR_TRANSIT_STEPS = max(
    1,
    math.ceil(
        STAIR_TRANSIT_SECONDS / (0.5 / 1.074)
    )
)

STAIR_PAIRING_METHOD = "row_major"


# ============================================================
# AGENT TYPES
# ============================================================

AGENT_TYPES = [
    "rational",
    "habit",
    "peer"
]

TYPE_PROPORTIONS = {
    "rational": 9 / 10,
    "habit": 0.5 / 10,
    "peer": 0.5 / 10
}


# ============================================================
# INITIAL DECISION PARAMETERS
# ============================================================

DISTANCE_WEIGHT = 0.5
CROWD_WEIGHT = 0.5

PEER_RADIUS = 1


# ============================================================
# CELLULAR AUTOMATON PARAMETERS
# ============================================================

MOORE_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

ORTHOGONAL_OFFSETS = [
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1)
]

ORTHOGONAL_COST = 1.0
DIAGONAL_COST = 2 ** 0.5

CELL_CAPACITY = 1

MAX_STEPS = 5000

NO_MOVEMENT_LIMIT = 100

HEAD_ON_SLOWDOWN_FACTOR = 0.5


# ============================================================
# SPEED MODEL PARAMETERS
# ============================================================

CELL_SIZE_M = 0.5

FREE_FLOW_SPEED = 1.5

THETA = 0.062
BETA = 0.072
ALPHA = 1.271

REGION_SIZE = 10

TIME_STEP_SECONDS = (
    CELL_SIZE_M / FREE_FLOW_SPEED
)

MIN_SPEED_RATIO = 0.05

MIN_SPEED_MPS = (
    FREE_FLOW_SPEED * MIN_SPEED_RATIO
)


# ============================================================
# INITIAL EXIT-CHOICE SCENARIO
# ============================================================

EVACUATION_SCENARIO = "concert"

CONCERT_REQUIRE_LINE_OF_SIGHT = True

CONCERT_FALLBACK_TO_ALL_REACHABLE = True

CONCERT_INITIAL_SATISFACTION_TIME_S = 15.0

CONCERT_THRESHOLD_INCREMENT_S = 5.0


# ============================================================
# HABIT-BASED PARAMETERS
# ============================================================

HABIT_FAMILIARITY_ASSIGNMENT = "random_known_exit"


# ============================================================
# RANDOM UTILITY MODEL PARAMETERS
# ============================================================

CHOICE_ALPHA = {
    "rational": 1.0,
    "habit": 0.05,
    "peer": 0.05
}

CHOICE_BETA = {
    "rational": 0.0,
    "habit": 0.0,
    "peer": 2.0
}

CHOICE_GAMMA = {
    "rational": 0.0,
    "habit": 2.0,
    "peer": 0.0
}

CHOICE_LAMBDA = {
    "rational": 1.0,
    "habit": 1.0,
    "peer": 1.0
}


# ============================================================
# DECISION CHANGE MODEL PARAMETERS
# ============================================================

# A region is congested if:
#
# regional_average_speed < DECISION_CHANGE_SPEED_THRESHOLD

DECISION_CHANGE_SPEED_THRESHOLD = 0.70

# Maximum probability of reconsidering the target exit.
#
# Required ordering:
#
# Rational > Peer > Habit

DECISION_CHANGE_P_MAX = {
    "rational": 0.80,
    "peer": 0.50,
    "habit": 0.20
}

# Minimum physical time between two successful
# decision changes by the same Agent.

MIN_DECISION_CHANGE_INTERVAL_SECONDS = 10.0


# ============================================================
# VISUALIZATION PARAMETERS
# ============================================================

TERRAIN_COLORS = {
    WALL: "#303030",
    OPEN_SPACE: "#F2F2F2",
    HORIZONTAL_SEAT: "#F4C542",
    VERTICAL_SEAT: "#F28C28",
    GROUND_EXIT: "#2E8B57",
    UPPER_EXIT: "#00A878",
    STAIR_6: "#4EA5D9",
    STAIR_7: "#8A65C2"
}

TERRAIN_LABELS = {
    WALL: "Wall",
    OPEN_SPACE: "Open Space",
    HORIZONTAL_SEAT: "Horizontal Seat",
    VERTICAL_SEAT: "Vertical Seat",
    GROUND_EXIT: "Floor 1 Exit",
    UPPER_EXIT: "Floor 2 Exit",
    STAIR_6: "Stairway 6",
    STAIR_7: "Stairway 7"
}

FRAME_INTERVAL = 0.5

ANIMATE = True

RANDOM_SEED = 42