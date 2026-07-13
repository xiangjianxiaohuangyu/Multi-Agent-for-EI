"""Environment State model: weather, terrain, RF environment, GPS.

This state is critical for embodied-intelligence (具身智能) agents because
the same mission executed in different environments can require very
different flight / routing / MAC behavior.

Data sources
------------
- IMU (orientation, vibration)
- Weather API / onboard weather sensors
- Camera / LiDAR (obstacle, terrain)
- Digital elevation model (DEM) / map tiles
- Spectrum sensors (interference)
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import StateBase


@dataclass
class EnvironmentState(StateBase):
    """Physical and RF environment around the UAV.

    Field groups
    ------------
    Weather:    wind_speed, wind_direction, temperature, humidity,
                visibility, rain
    Terrain:    obstacle_density, terrain_type
    Sensing:    gps_quality
    RF:         interference_level
    """

    # --- weather ---
    wind_speed: float = 0.0          # m/s
    wind_direction: float = 0.0      # degrees
    temperature: float = 20.0        # Celsius
    humidity: float = 50.0           # percentage 0-100
    visibility: float = 10000.0      # meters
    rain: bool = False

    # --- terrain ---
    obstacle_density: float = 0.0    # 0-1, fraction of area blocked
    terrain_type: str = "open"       # 'open' | 'urban' | 'forest' | 'mountain' | ...

    # --- sensing ---
    gps_quality: float = 1.0         # 0-1

    # --- RF environment ---
    interference_level: float = 0.0  # 0-1, normalized

    # ------------------------------------------------------------------ helpers

    def is_harsh_weather(self,
                         wind_threshold: float = 12.0,
                         visibility_threshold: float = 1000.0) -> bool:
        """True when weather is likely to impact flight safety / comms."""
        return (self.wind_speed > wind_threshold
                or self.visibility < visibility_threshold
                or self.rain)

    def is_rf_hostile(self, interference_threshold: float = 0.6) -> bool:
        return self.interference_level > interference_threshold
