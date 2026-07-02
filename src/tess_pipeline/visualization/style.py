"""
visualization/style.py — Publication-ready style settings based on regmachado/templots.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

def set_publication_style() -> None:
    """
    Apply MNRAS/AAS publication plotting styles.
    Uses STIX fonts for LaTeX-like math rendering, inside ticks, and specific sizing.
    """
    # Font settings
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['legend.frameon'] = False
    
    # STIX fonts (LaTeX look-alike without needing system LaTeX installation)
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # Tick directions and styling (pointing inward)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    
    # Tick sizes and widths
    plt.rcParams['xtick.major.size'] = 4.0
    plt.rcParams['xtick.minor.size'] = 2.0
    plt.rcParams['ytick.major.size'] = 4.0
    plt.rcParams['ytick.minor.size'] = 2.0
    plt.rcParams['xtick.major.width'] = 0.75
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.75
    plt.rcParams['ytick.minor.width'] = 0.5
    
    # Spines/Borders width
    plt.rcParams['axes.linewidth'] = 0.75
    
    # High-quality figures
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['figure.dpi'] = 120
