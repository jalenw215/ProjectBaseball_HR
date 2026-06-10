from __future__ import annotations


# Coordinates are approximate stadium locations. Park factors are conservative
# HR run environment placeholders centered around 1.00 for v1 modeling.
PARKS = {
    "Angel Stadium": {"lat": 33.8003, "lon": -117.8827, "hr_factor": 1.01},
    "Busch Stadium": {"lat": 38.6226, "lon": -90.1928, "hr_factor": 0.96},
    "Chase Field": {"lat": 33.4455, "lon": -112.0667, "hr_factor": 1.04},
    "Citi Field": {"lat": 40.7571, "lon": -73.8458, "hr_factor": 0.98},
    "Citizens Bank Park": {"lat": 39.9058, "lon": -75.1665, "hr_factor": 1.08},
    "Comerica Park": {"lat": 42.3390, "lon": -83.0485, "hr_factor": 0.95},
    "Coors Field": {"lat": 39.7559, "lon": -104.9942, "hr_factor": 1.25},
    "Dodger Stadium": {"lat": 34.0739, "lon": -118.2400, "hr_factor": 1.02},
    "Fenway Park": {"lat": 42.3467, "lon": -71.0972, "hr_factor": 1.03},
    "Globe Life Field": {"lat": 32.7473, "lon": -97.0842, "hr_factor": 1.00},
    "Great American Ball Park": {"lat": 39.0979, "lon": -84.5082, "hr_factor": 1.16},
    "Guaranteed Rate Field": {"lat": 41.8300, "lon": -87.6339, "hr_factor": 1.06},
    "Kauffman Stadium": {"lat": 39.0517, "lon": -94.4803, "hr_factor": 0.91},
    "loanDepot park": {"lat": 25.7781, "lon": -80.2197, "hr_factor": 0.93},
    "Minute Maid Park": {"lat": 29.7573, "lon": -95.3555, "hr_factor": 1.02},
    "Nationals Park": {"lat": 38.8730, "lon": -77.0074, "hr_factor": 1.00},
    "Oakland Coliseum": {"lat": 37.7516, "lon": -122.2005, "hr_factor": 0.91},
    "Oracle Park": {"lat": 37.7786, "lon": -122.3893, "hr_factor": 0.90},
    "Oriole Park at Camden Yards": {"lat": 39.2840, "lon": -76.6217, "hr_factor": 1.01},
    "Petco Park": {"lat": 32.7073, "lon": -117.1566, "hr_factor": 0.94},
    "PNC Park": {"lat": 40.4469, "lon": -80.0057, "hr_factor": 0.97},
    "Progressive Field": {"lat": 41.4962, "lon": -81.6852, "hr_factor": 0.98},
    "Rogers Centre": {"lat": 43.6414, "lon": -79.3894, "hr_factor": 1.05},
    "T-Mobile Park": {"lat": 47.5914, "lon": -122.3325, "hr_factor": 0.94},
    "Target Field": {"lat": 44.9817, "lon": -93.2776, "hr_factor": 0.99},
    "Tropicana Field": {"lat": 27.7682, "lon": -82.6534, "hr_factor": 0.96},
    "Truist Park": {"lat": 33.8908, "lon": -84.4678, "hr_factor": 1.04},
    "Wrigley Field": {"lat": 41.9484, "lon": -87.6553, "hr_factor": 1.03},
    "Yankee Stadium": {"lat": 40.8296, "lon": -73.9262, "hr_factor": 1.09},
}


def park_info(venue_name: str | None) -> dict[str, float]:
    if not venue_name:
        return {"lat": 39.8283, "lon": -98.5795, "hr_factor": 1.0}
    return PARKS.get(venue_name, {"lat": 39.8283, "lon": -98.5795, "hr_factor": 1.0})

