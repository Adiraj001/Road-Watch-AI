from __future__ import annotations

# =========================
# 📦 Imports
# =========================
from typing import Any, Dict, Optional

from geotagger import get_address, get_maps_link, google_maps_ready


# =========================
# 🌍 Google Maps Client Wrapper
# =========================
class GoogleMapsClient:
    """
    Lightweight compatibility wrapper for geospatial utilities.

    Provides:
    - Reverse geocoding (lat/lng → address)
    - Google Maps URL generation
    - Availability check for Maps integration
    """

    def __init__(self, api_key: Optional[str]) -> None:
        """
        Initialize client with optional API key.

        Note:
            API key is currently not directly used here but kept
            for future extensibility or external integrations.
        """
        self.api_key = api_key

    # =========================
    # 🔌 Feature Availability
    # =========================
    @property
    def enabled(self) -> bool:
        """
        Check if Google Maps/geotagging features are available.
        """
        return google_maps_ready()

    # =========================
    # 📍 Reverse Geocoding
    # =========================
    def reverse_geocode(
        self,
        latitude: Optional[float],
        longitude: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """
        Convert coordinates into a structured location payload.

        Args:
            latitude (float): Latitude value
            longitude (float): Longitude value

        Returns:
            dict | None: Geolocation metadata including address & map link
        """
        if latitude is None or longitude is None:
            return None

        address = get_address(latitude, longitude)

        return {
            "formatted_address": address,
            "place_id": None,  # Reserved for future Google API integration
            "types": [],       # Placeholder for location categories
            "google_maps_url": get_maps_link(latitude, longitude),
        }


# =========================
# 🔗 Utility Function
# =========================
def build_google_maps_url(
    latitude: Optional[float],
    longitude: Optional[float],
) -> Optional[str]:
    """
    Generate a browser-friendly Google Maps URL.

    Args:
        latitude (float): Latitude value
        longitude (float): Longitude value

    Returns:
        str | None: Google Maps link or None if invalid input
    """
    if latitude is None or longitude is None:
        return None

    return get_maps_link(latitude, longitude)