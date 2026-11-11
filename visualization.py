"""
Dynamic two-floor visualization of the CA simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from config import *


# ============================================================
# Agent visualization codes
# ============================================================

AGENT_DISPLAY_CODES = {
    "rational": 8,
    "habit": 9,
    "peer": 10
}


# ============================================================
# Evacuation Animator
# ============================================================

class EvacuationAnimator:
    """
    Display Floor 1 and Floor 2 side by side.
    """

    def __init__(
        self,
        terrain,
        interval=FRAME_INTERVAL
    ):

        self.terrain = {
            floor: terrain[floor].copy()
            for floor in FLOORS
        }

        self.interval = interval

        # Terrain codes: 0-7.
        # Agent codes: 8-10.

        colors = [
            TERRAIN_COLORS[0],
            TERRAIN_COLORS[1],
            TERRAIN_COLORS[2],
            TERRAIN_COLORS[3],
            TERRAIN_COLORS[4],
            TERRAIN_COLORS[5],
            TERRAIN_COLORS[6],
            TERRAIN_COLORS[7],
            "#E63946",  # Rational
            "#9B5DE5",  # Habit
            "#F72585"   # Peer
        ]

        self.cmap = ListedColormap(colors)

        self.norm = BoundaryNorm(
            np.arange(-0.5, 11.5, 1),
            self.cmap.N
        )

        plt.ion()

        self.fig, axes = plt.subplots(
            1,
            2,
            figsize=(15, 7)
        )

        self.axes = {
            1: axes[0],
            2: axes[1]
        }

        self.images = {}

        for floor in FLOORS:

            initial_grid = self.build_display_grid(
                [],
                floor
            )

            ax = self.axes[floor]

            self.images[floor] = ax.imshow(
                initial_grid,
                cmap=self.cmap,
                norm=self.norm,
                interpolation="nearest",
                origin="upper"
            )

            ax.set_title(
                f"Bailey Hall — Floor {floor}",
                weight="bold"
            )

            ax.set_axis_off()

            ax.set_aspect("equal")

        # ----------------------------------------------------
        # Shared legend
        # ----------------------------------------------------

        legend_items = [
            Patch(
                facecolor=colors[code],
                label=TERRAIN_LABELS[code]
            )
            for code in range(8)
        ]

        legend_items.extend([
            Patch(
                facecolor=colors[8],
                label="Rational Agent"
            ),
            Patch(
                facecolor=colors[9],
                label="Habit Agent"
            ),
            Patch(
                facecolor=colors[10],
                label="Peer Agent"
            )
        ])

        self.fig.legend(
            handles=legend_items,
            loc="center left",
            bbox_to_anchor=(0.88, 0.5),
            frameon=False,
            title="Terrain / Agent Type"
        )

        self.fig.suptitle(
            "Bailey Hall — Two-Floor Cellular Automaton",
            weight="bold"
        )

        plt.tight_layout(
            rect=[0, 0, 0.86, 0.93]
        )

        plt.show(block=False)

        plt.pause(0.1)

    # ========================================================
    # Build display grid for one floor
    # ========================================================

    def build_display_grid(
        self,
        agents,
        floor
    ):
        """
        Overlay active Agents on a copy of one floor's terrain.

        Agents in transit have no grid position and are hidden.
        """

        display_grid = self.terrain[floor].copy()

        for agent in agents:

            if agent.status != "active":
                continue

            if agent.position is None:
                continue

            if agent.floor != floor:
                continue

            r, c = agent.position

            display_grid[r, c] = (
                AGENT_DISPLAY_CODES[agent.agent_type]
            )

        return display_grid

    # ========================================================
    # Update one animation frame
    # ========================================================

    def update(
        self,
        agents,
        time_step,
        evacuated,
        total,
        moved
    ):
        """
        Refresh both floor panels.
        """

        if not plt.fignum_exists(self.fig.number):
            return

        for floor in FLOORS:

            display_grid = self.build_display_grid(
                agents,
                floor
            )

            self.images[floor].set_data(
                display_grid
            )

            floor_count = sum(
                agent.status == "active"
                and agent.floor == floor
                and agent.position is not None
                for agent in agents
            )

            self.axes[floor].set_title(
                f"Floor {floor} — "
                f"Active: {floor_count}",
                weight="bold"
            )

        in_transit = sum(
            agent.status == "transit"
            for agent in agents
        )

        physical_time = (
            time_step * TIME_STEP_SECONDS
        )

        self.fig.suptitle(
            f"Bailey Hall — Two-Floor CA | "
            f"Step: {time_step} | "
            f"Time: {physical_time:.2f} s | "
            f"Evacuated: {evacuated}/{total} | "
            f"Remaining: {len(agents)} | "
            f"In Transit: {in_transit}",
            weight="bold"
        )

        self.fig.canvas.draw_idle()

        self.fig.canvas.flush_events()

        plt.pause(self.interval)

    # ========================================================
    # Keep the final frame visible
    # ========================================================

    def finish(self):
        """
        Keep the final animation window open.
        """

        if plt.fignum_exists(self.fig.number):

            plt.ioff()

            plt.show(block=True)