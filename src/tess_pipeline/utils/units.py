"""
utils/units.py — Unit conversion helpers for transit and stellar parameters.
"""

from __future__ import annotations

import math

# Constants
R_SUN_CM = 6.957e10
R_EARTH_CM = 6.371e8
M_SUN_G = 1.989e33
AU_CM = 1.496e13
G_CGS = 6.674e-8


def r_sun_to_r_earth(r_sun: float) -> float:
    """Convert solar radii to Earth radii."""
    return r_sun * (R_SUN_CM / R_EARTH_CM)


def r_earth_to_r_sun(r_earth: float) -> float:
    """Convert Earth radii to solar radii."""
    return r_earth * (R_EARTH_CM / R_SUN_CM)


def semi_major_axis_au(period_days: float, m_star_solar: float) -> float:
    """
    Compute semi-major axis in AU using Kepler's third law.

    Parameters
    ----------
    period_days : float
    m_star_solar : float
        Stellar mass in solar masses.
    """
    period_s = period_days * 86400.0
    m_cgs = m_star_solar * M_SUN_G
    a_cm = (G_CGS * m_cgs * period_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)
    return a_cm / AU_CM


def stellar_density_cgs(m_star_solar: float, r_star_solar: float) -> float:
    """Compute stellar density in g/cm^3."""
    m = m_star_solar * M_SUN_G
    r = r_star_solar * R_SUN_CM
    return m / (4.0 / 3.0 * math.pi * r**3)
