"""Geometry helpers.

Trips are spread all over the world (longitudes from about -93 to +77), so a
single projected CRS is not appropriate. Each trip gets its own local
equirectangular (tangent-plane) projection centred on the trip's GPS
centroid. For the extents in this dataset (< ~50 km) the scale error is well
below 1 %, i.e. centimetres at the 10-50 m distances that matter for map
matching, and it keeps distance computations vectorised and cheap.
"""
from __future__ import annotations

import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius (m)


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self._kx = np.deg2rad(1.0) * EARTH_R * np.cos(np.deg2rad(self.lat0))
        self._ky = np.deg2rad(1.0) * EARTH_R

    @classmethod
    def from_points(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjection":
        return cls(float(np.mean(lon)), float(np.mean(lat)))

    def to_xy(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return (lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky

    def to_lonlat(self, x, y):
        return np.asarray(x) / self._kx + self.lon0, np.asarray(y) / self._ky + self.lat0


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments (broadcasting) and projection parameter t in [0,1]."""
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    t = ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1.0)
    t = np.clip(t, 0.0, 1.0)
    qx = ax + t * dx
    qy = ay + t * dy
    return np.hypot(px - qx, py - qy), t
