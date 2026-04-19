from __future__ import annotations


class MuseSparkError(Exception):
    """Base error for Muse Spark provider failures."""


class MissingAuthError(MuseSparkError):
    """No local Muse auth/session state is available."""


class ReauthRequiredError(MuseSparkError):
    """Stored Muse auth appears expired or invalid and must be refreshed."""


class ProviderTransportError(MuseSparkError):
    """Transport-level failure talking to Meta services."""


class ProviderProtocolError(MuseSparkError):
    """Meta transport succeeded but returned unusable or unexpected data."""
