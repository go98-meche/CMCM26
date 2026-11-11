
"""
Bailey Hall — Excel Terrain Map Converter
CMCM 2026

Purpose:
    Read a manually constructed Excel map,
    convert its numerical values into a NumPy terrain matrix,
    and visualize the resulting terrain.

Excel input classification:
    1 = Green  = Exit
    2 = Black  = Obstacle
    3 = Blue   = Open Space
    4 = Orange = Seat

Python terrain classification:
    0 = Obstacle
    1 = Open Space
    2 = Seat
    3 = Exit

Dependencies:
    pip install openpyxl numpy matplotlib

Input:
    map_easy.xlsx

Output:
    map_easy_terrain.csv
    map_easy_terrain.png
"""

import numpy as np
import matplotlib.pyplot as plt
import openpyxl

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# 1.1 Excel input
# ------------------------------------------------------------

# Excel file should be in the same folder as this Python script.
XLSX_PATH = Path(__file__).with_name("map_easy.xlsx")

# Worksheet name.
SHEET_NAME = "Sheet1"

# Excel map range.
#
# None:
#     Automatically detect the smallest area containing data.
#
# Alternatively, specify a range manually.
#
# Example:
#     MAP_RANGE = "B2:BI65"

MAP_RANGE = None


# ------------------------------------------------------------
# 1.2 Python terrain classification
# ------------------------------------------------------------

OBSTACLE = 0
OPEN_SPACE = 1
SEAT = 2
EXIT = 3


# ------------------------------------------------------------
# 1.3 Excel-to-Python mapping
# ------------------------------------------------------------

# Excel numerical values -> Python terrain values.
#
# Excel:
#     1 = Green  = Exit
#     2 = Black  = Obstacle
#     3 = Blue   = Open Space
#     4 = Orange = Seat

EXCEL_TO_TERRAIN = {
    1: EXIT,
    2: OBSTACLE,
    3: OPEN_SPACE,
    4: SEAT
}

# Treatment of empty Excel cells.
#
# Empty cells inside the detected map range
# are treated as obstacles by default.
#
# Change this to OPEN_SPACE if empty cells
# represent open areas in your Excel drawing.

EMPTY_AS = OBSTACLE


# ------------------------------------------------------------
# 1.4 Output files
# ------------------------------------------------------------

OUT_CSV = Path(__file__).with_name("map_easy_terrain.csv")
OUT_PNG = Path(__file__).with_name("map_easy_terrain.png")


# ============================================================
# 2. READ EXCEL FILE
# ============================================================

def read_excel_to_terrain(xlsx_path, sheet_name):
    """
    Read numerical values from Excel and convert them
    into a NumPy terrain matrix.

    Returns:
        terrain:
            Converted NumPy 2D array.

        source_counts:
            Number of cells containing each Excel value.
    """

    # Open Excel workbook.
    wb = openpyxl.load_workbook(
        xlsx_path,
        data_only=True
    )

    # Check whether the requested worksheet exists.
    if sheet_name not in wb.sheetnames:
        raise ValueError(
            f"Worksheet '{sheet_name}' not found. "
            f"Available sheets: {wb.sheetnames}"
        )

    ws = wb[sheet_name]


    # --------------------------------------------------------
    # 2.1 Determine the map boundaries
    # --------------------------------------------------------

    if MAP_RANGE is not None:

        # Read manually specified Excel range.
        from openpyxl.utils.cell import range_boundaries

        min_col, min_row, max_col, max_row = (
            range_boundaries(MAP_RANGE)
        )

    else:

        # Automatically locate all non-empty cells.
        positions = []

        for row in ws.iter_rows():

            for cell in row:

                if cell.value is not None:
                    positions.append(
                        (cell.row, cell.column)
                    )

        if not positions:
            raise ValueError(
                "No data found in the Excel worksheet."
            )

        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]

        min_row = min(rows)
        max_row = max(rows)
        min_col = min(cols)
        max_col = max(cols)


    # --------------------------------------------------------
    # 2.2 Initialize terrain matrix
    # --------------------------------------------------------

    ROWS = max_row - min_row + 1
    COLS = max_col - min_col + 1

    terrain = np.full(
        (ROWS, COLS),
        EMPTY_AS,
        dtype=int
    )

    # Track the number of cells corresponding to
    # each original Excel numerical value.
    source_counts = {}

    unknown_values = {}


    # --------------------------------------------------------
    # 2.3 Convert Excel cells into terrain values
    # --------------------------------------------------------

    for r in range(min_row, max_row + 1):

        for c in range(min_col, max_col + 1):

            cell = ws.cell(
                row=r,
                column=c
            )

            value = cell.value

            # Convert Excel coordinates to NumPy coordinates.
            i = r - min_row
            j = c - min_col

            # Empty cells.
            if value is None:
                terrain[i, j] = EMPTY_AS
                continue

            # Convert numerical values and numerical strings.
            try:
                numeric_value = float(value)

                if not numeric_value.is_integer():
                    raise ValueError

                value = int(numeric_value)

            except (ValueError, TypeError):
                unknown_values[value] = (
                    unknown_values.get(value, 0) + 1
                )
                continue

            # Count original Excel values.
            source_counts[value] = (
                source_counts.get(value, 0) + 1
            )

            # Convert Excel values into terrain types.
            if value in EXCEL_TO_TERRAIN:

                terrain[i, j] = EXCEL_TO_TERRAIN[value]

            else:

                unknown_values[value] = (
                    unknown_values.get(value, 0) + 1
                )


    wb.close()

    # --------------------------------------------------------
    # 2.4 Check for invalid input
    # --------------------------------------------------------

    if unknown_values:

        raise ValueError(
            "Unknown Excel values detected:\n"
            f"{unknown_values}\n"
            "Please check that all non-empty map cells "
            "contain only 1, 2, 3, or 4."
        )

    return terrain, source_counts


# ============================================================
# 3. VALIDATE TERRAIN MATRIX
# ============================================================

def validate_terrain(terrain):
    """
    Check whether the converted terrain matrix
    contains valid terrain classifications.
    """

    valid_values = {
        OBSTACLE,
        OPEN_SPACE,
        SEAT,
        EXIT
    }

    actual_values = set(np.unique(terrain))

    # Check for invalid terrain values.
    if not actual_values.issubset(valid_values):

        raise ValueError(
            f"Invalid terrain values: {actual_values}"
        )

    # Check whether the entire map is obstacles.
    if np.all(terrain == OBSTACLE):

        raise ValueError(
            "The entire terrain matrix contains only obstacles. "
            "Please check the Excel input."
        )

    # Check whether exits exist.
    if not np.any(terrain == EXIT):

        print("WARNING: No exits detected in the map.")

    # Check whether seats exist.
    if not np.any(terrain == SEAT):

        print("WARNING: No seating areas detected in the map.")

    print("\nTerrain validation passed.")


# ============================================================
# 4. VISUALIZATION
# ============================================================

def plot_terrain(terrain, out_png=None):
    """
    Visualize the terrain matrix.

    Color classification:
        0 = Obstacle (Black)
        1 = Open Space (Blue)
        2 = Seat (Orange)
        3 = Exit (Green)
    """

    # Color order MUST match the Python terrain values.
    colors = [
        "#222222",  # 0: Obstacle
        "#4472C4",  # 1: Open Space
        "#F4B183",  # 2: Seat
        "#00B050"   # 3: Exit
    ]

    cmap = ListedColormap(colors)

    norm = BoundaryNorm(
        np.arange(-0.5, 4.5, 1),
        cmap.N
    )

    # Create figure.
    fig, ax = plt.subplots(figsize=(10, 7))

    # Display terrain matrix.
    ax.imshow(
        terrain,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper"
    )

    # --------------------------------------------------------
    # Color legend
    # --------------------------------------------------------

    legend_items = [
        Patch(facecolor=colors[0], label="Obstacle"),
        Patch(facecolor=colors[1], label="Open Space"),
        Patch(facecolor=colors[2], label="Seat"),
        Patch(facecolor=colors[3], label="Exit")
    ]

    ax.legend(
        handles=legend_items,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Terrain Type"
    )

    # Remove unnecessary visual elements.
    ax.set_axis_off()

    ax.set_title(
        "Bailey Hall — Initial Terrain Grid",
        fontsize=13,
        weight="bold"
    )

    ax.set_aspect("equal")

    plt.tight_layout(
        rect=[0, 0, 0.82, 1]
    )

    # Save image.
    if out_png is not None:

        plt.savefig(
            out_png,
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.1
        )

    plt.show()

    plt.close(fig)


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 5.1 Check Excel file
    # --------------------------------------------------------

    if not XLSX_PATH.exists():

        raise FileNotFoundError(
            f"Excel file not found: {XLSX_PATH}"
        )


    # --------------------------------------------------------
    # 5.2 Read and convert Excel
    # --------------------------------------------------------

    print("Reading Excel map...")

    terrain, source_counts = read_excel_to_terrain(
        XLSX_PATH,
        SHEET_NAME
    )


    # --------------------------------------------------------
    # 5.3 Display Excel input statistics
    # --------------------------------------------------------

    print("\nExcel input statistics:")

    excel_labels = {
        1: "Green / Exit",
        2: "Black / Obstacle",
        3: "Blue / Open Space",
        4: "Orange / Seat"
    }

    for value in [1, 2, 3, 4]:

        count = source_counts.get(value, 0)

        print(
            f"  {value} ({excel_labels[value]}): "
            f"{count} cells"
        )


    # --------------------------------------------------------
    # 5.4 Display matrix information
    # --------------------------------------------------------

    print("\nTerrain matrix shape:")
    print(terrain.shape)

    print("\nTerrain statistics:")

    terrain_labels = {
        OBSTACLE: "Obstacle",
        OPEN_SPACE: "Open Space",
        SEAT: "Seat",
        EXIT: "Exit"
    }

    for value, label in terrain_labels.items():

        count = np.sum(terrain == value)

        percentage = count / terrain.size * 100

        print(
            f"  {label}: {count} cells "
            f"({percentage:.2f}%)"
        )


    # --------------------------------------------------------
    # 5.5 Validate matrix
    # --------------------------------------------------------

    validate_terrain(terrain)


    # --------------------------------------------------------
    # 5.6 Print matrix
    # --------------------------------------------------------

    print("\nTerrain matrix:")
    print(terrain)


    # --------------------------------------------------------
    # 5.7 Export matrix to CSV
    # --------------------------------------------------------

    np.savetxt(
        OUT_CSV,
        terrain,
        fmt="%d",
        delimiter=","
    )

    print(f"\nCSV saved: {OUT_CSV}")


    # --------------------------------------------------------
    # 5.8 Visualize terrain
    # --------------------------------------------------------

    plot_terrain(
        terrain,
        OUT_PNG
    )

    print(f"PNG saved: {OUT_PNG}")

    print("\nAll operations completed successfully.")
