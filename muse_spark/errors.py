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


class ProviderStallError(ProviderProtocolError):
    """Meta stream went idle mid-response after emitting some output.

    Distinguished from ``ProviderProtocolError`` so the SSE pipeline can
    surface partial content with ``finish_reason="length"`` (truncation) rather
    than ``"error"``. Clients that already received tokens get a graceful end
    of stream instead of having the request appear to fail outright.
    """


class ProviderEmptyResponseError(ProviderProtocolError):
    """Meta accepted the request and closed the stream without emitting text.

    Observed in production after a previous mid-response stall: the affected
    conversation enters a stuck state on Meta's backend and every subsequent
    turn sent through the same conversation id receives an empty stream. The
    SSE pipeline catches this, purges the stuck mapping, and retries once
    with a fresh meta conversation id so the client doesn't have to manually
    reset state.
    """
