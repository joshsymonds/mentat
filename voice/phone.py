"""Pure phone-control decisions and the injected LiveKit/HTTP boundary.

The Android participant remains the trust boundary for intent construction. This
module only validates closed command payloads, searches Places, and translates
phone-side failures into short results for the front voice.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from urllib.parse import urlparse
from typing import Any

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_FIELD_MASK = "places.id,places.displayName,places.formattedAddress,places.location"
PLACES_RADIUS_M = 20000.0
RPC_TIMEOUT_S = 10.0
EARTH_RADIUS_KM = 6371.0088

PHONE_UNREACHABLE = "PHONE_UNREACHABLE: no phone is connected to this conversation, or it did not answer."
PHONE_REFUSED = "PHONE_REFUSED: the phone rejected that action."
PHONE_NOT_IN_FRONT = "PHONE_NOT_IN_FRONT: the Mentat screen is covered by another app, so the phone will not launch anything until he taps the side button."
LOCATION_UNAVAILABLE = "LOCATION_UNAVAILABLE: the phone could not give a fresh location, so ask roughly where he is and search again with that locality."
PLACES_UNCONFIGURED = "PLACES_UNCONFIGURED: place search has no API key on the server."
PLACES_FAILED = "PLACES_FAILED: the place search request failed."
NO_CANDIDATES = "NO_CANDIDATES: there is no current place list to choose from; search first."
NO_RESULTS = "NO_RESULTS: no places matched."

DIAL_READY = "The dialer is ready."
TEXT_READY = "The message is ready."
ALARM_SET = "The alarm is set."
TIMER_SET = "The timer is set."
LINK_OPENED = "The link is open."
NAVIGATION_STARTED = "Navigation is starting."


class RpcFailure(Exception):
    """A phone RPC rejected by the Android participant."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Place:
    """The subset of a Places result needed by the phone navigation command."""

    name: str
    address: str
    place_id: str
    lat: float
    lng: float
    distance_km: float | None


def places_request(
    query: str,
    origin: tuple[float, float] | None,
    key: str,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build a Google Places Text Search request."""
    body: dict[str, Any] = {"textQuery": query, "pageSize": 5}
    if origin is not None:
        latitude, longitude = origin
        body["locationBias"] = {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": PLACES_RADIUS_M,
            }
        }
    return (
        PLACES_URL,
        {
            "X-Goog-Api-Key": key,
            "X-Goog-FieldMask": PLACES_FIELD_MASK,
        },
        body,
    )


def parse_places(payload: Any, origin: tuple[float, float] | None) -> list[Place]:
    """Validate and rank the relevant fields in a Places response."""
    if not isinstance(payload, Mapping) or not isinstance(payload.get("places"), list):
        raise ValueError("Places response has no places list")

    places: list[Place] = []
    for raw in payload["places"]:
        if not isinstance(raw, Mapping):
            raise ValueError("Place is not an object")
        name_data = raw.get("displayName")
        location = raw.get("location")
        name = name_data.get("text") if isinstance(name_data, Mapping) else None
        address = raw.get("formattedAddress")
        place_id = raw.get("id")
        latitude = location.get("latitude") if isinstance(location, Mapping) else None
        longitude = location.get("longitude") if isinstance(location, Mapping) else None
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(address, str)
            or not address
            or not isinstance(place_id, str)
            or not place_id
            or not _finite_number(latitude)
            or not _finite_number(longitude)
        ):
            raise ValueError("Place is missing required fields")
        lat = float(latitude)
        lng = float(longitude)
        distance = _haversine_km(origin, (lat, lng)) if origin is not None else None
        places.append(Place(name, address, place_id, lat, lng, distance))

    if origin is not None:
        places.sort(key=lambda place: place.distance_km or 0.0)
    return places


def describe_places(places: Sequence[Place], approximate: bool) -> str:
    """Turn up to three ranked places into short speech for disambiguation."""
    if not places:
        return NO_RESULTS
    shown = list(places[:3])
    if len(shown) == 1:
        return f"I found {_place_line(shown[0], approximate)}."
    ordinal_words = ("first", "second", "third")
    count_word = {2: "two", 3: "three"}[len(shown)]
    lines = [
        f"the {ordinal_words[index]} is {_place_line(place, approximate)}"
        for index, place in enumerate(shown)
    ]
    return f"I found {count_word}: " + "; ".join(lines) + ". Which one?"


def command_payload(kind: str, **fields: Any) -> str:
    """Validate one closed phone command and encode it as compact JSON."""
    if kind == "navigate":
        _exact_fields(fields, {"name", "address", "place_id", "lat", "lng"})
        _nonempty_text(fields, "name")
        _nonempty_text(fields, "address")
        _nonempty_text(fields, "place_id")
        _finite_required_number(fields, "lat")
        _finite_required_number(fields, "lng")
    elif kind == "dial":
        _exact_fields(fields, {"number"})
        _nonempty_text(fields, "number")
    elif kind == "sms":
        _exact_fields(fields, {"number", "body"})
        _nonempty_text(fields, "number")
        if not isinstance(fields["body"], str):
            raise ValueError("body must be text")
    elif kind == "alarm":
        _optional_label_fields(fields, {"hour", "minute"})
        _integer_range(fields, "hour", 0, 23)
        _integer_range(fields, "minute", 0, 59)
    elif kind == "timer":
        _optional_label_fields(fields, {"seconds"})
        _integer_range(fields, "seconds", 0, None)
    elif kind == "open":
        _exact_fields(fields, {"url"})
        if not isinstance(fields.get("url"), str) or not open_url_allowed(fields["url"]):
            raise ValueError("url must be an absolute http or https URL")
    else:
        raise ValueError(f"unknown command kind: {kind}")

    payload: dict[str, Any] = {"kind": kind}
    payload.update(fields)
    if "label" in payload and payload["label"] == "":
        del payload["label"]
    return json.dumps(payload, separators=(",", ":"))


def open_url_allowed(url: str) -> bool:
    """Whether a URL is an absolute HTTP(S) URL suitable for the phone."""
    if not isinstance(url, str):
        return False
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def phone_identity(identities: Sequence[str] | Mapping[str, Any]) -> str | None:
    """Return the first participant identity reserved for the Pixel phone."""
    candidates = identities.keys() if isinstance(identities, Mapping) else identities
    return next((identity for identity in candidates if identity.startswith("pixel-")), None)


class PhoneActions:
    """Phone and Places operations with all external calls injected."""

    def __init__(
        self,
        perform_rpc: Callable[[str, str, str, float], Any],
        post_json: Callable[[str, Mapping[str, str], Mapping[str, Any]], Any],
        identities: Callable[[], Sequence[str] | Mapping[str, Any]],
        key: str,
    ) -> None:
        self._perform_rpc = perform_rpc
        self._post_json = post_json
        self._identities = identities
        self._key = key
        self._candidates: list[Place] = []

    async def find_places(self, query: str, locality: str = "") -> str:
        """Search nearby places, retaining candidates only after valid parsing."""
        self._candidates.clear()
        identity = phone_identity(self._identities())
        if identity is None:
            return PHONE_UNREACHABLE
        if not self._key:
            return PLACES_UNCONFIGURED

        origin: tuple[float, float] | None
        approximate = False
        try:
            location_response = await self._perform_rpc(
                identity, "mentat.location", "{}", RPC_TIMEOUT_S
            )
            location = _decode_json_object(location_response)
            latitude = location.get("lat")
            longitude = location.get("lng")
            if not _finite_number(latitude) or not _finite_number(longitude):
                raise ValueError("location response has no coordinates")
            origin = (float(latitude), float(longitude))
            accuracy = location.get("accuracy_m")
            approximate = _finite_number(accuracy) and float(accuracy) > 100.0
        except RpcFailure as error:
            if error.code == 1602:
                if not locality.strip():
                    return LOCATION_UNAVAILABLE
                origin = None
            else:
                return _rpc_result(error)
        except asyncio.TimeoutError:
            return PHONE_UNREACHABLE
        except (TypeError, ValueError):
            if not locality.strip():
                return LOCATION_UNAVAILABLE
            origin = None

        search_query = query if origin is not None else f"{query} in {locality.strip()}"
        url, headers, body = places_request(search_query, origin, self._key)
        try:
            response = await self._post_json(url, headers, body)
            payload = _decode_json(response)
            parsed = parse_places(payload, origin)
        except Exception:
            return PLACES_FAILED
        if not parsed:
            return NO_RESULTS
        self._candidates = parsed
        return describe_places(parsed, approximate)

    async def navigate_to(self, choice: int) -> str:
        """Launch navigation for a retained one-based place choice."""
        identity = phone_identity(self._identities())
        if identity is None:
            return PHONE_UNREACHABLE
        if not isinstance(choice, int) or isinstance(choice, bool) or not 1 <= choice <= len(self._candidates):
            return NO_CANDIDATES
        place = self._candidates[choice - 1]
        try:
            payload = command_payload(
                "navigate",
                name=place.name,
                address=place.address,
                place_id=place.place_id,
                lat=place.lat,
                lng=place.lng,
            )
            await self._perform_rpc(identity, "mentat.command", payload, RPC_TIMEOUT_S)
        except RpcFailure as error:
            return _rpc_result(error)
        except asyncio.TimeoutError:
            return PHONE_UNREACHABLE
        except ValueError:
            return PHONE_REFUSED
        return NAVIGATION_STARTED

    async def dial(self, number: str) -> str:
        return await self._command("dial", {"number": number}, DIAL_READY)

    async def send_text(self, number: str, body: str) -> str:
        return await self._command("sms", {"number": number, "body": body}, TEXT_READY)

    async def set_alarm(self, hour: int, minute: int, label: str = "") -> str:
        return await self._command("alarm", {"hour": hour, "minute": minute, "label": label}, ALARM_SET)

    async def set_timer(self, minutes: int, seconds: int = 0, label: str = "") -> str:
        if (
            not isinstance(minutes, int)
            or isinstance(minutes, bool)
            or not isinstance(seconds, int)
            or isinstance(seconds, bool)
            or minutes < 0
            or seconds < 0
            or seconds > 59
        ):
            return PHONE_REFUSED
        return await self._command(
            "timer", {"seconds": minutes * 60 + seconds, "label": label}, TIMER_SET
        )

    async def open_link(self, url: str) -> str:
        return await self._command("open", {"url": url}, LINK_OPENED)

    async def _command(self, kind: str, fields: dict[str, Any], success: str) -> str:
        identity = phone_identity(self._identities())
        if identity is None:
            return PHONE_UNREACHABLE
        try:
            payload = command_payload(kind, **fields)
            await self._perform_rpc(identity, "mentat.command", payload, RPC_TIMEOUT_S)
        except RpcFailure as error:
            return _rpc_result(error)
        except asyncio.TimeoutError:
            return PHONE_UNREACHABLE
        except ValueError:
            return PHONE_REFUSED
        return success


def _decode_json(value: Any) -> Any:
    if isinstance(value, (str, bytes, bytearray)):
        return json.loads(value)
    return value


def _decode_json_object(value: Any) -> Mapping[str, Any]:
    decoded = _decode_json(value)
    if not isinstance(decoded, Mapping):
        raise ValueError("response is not an object")
    return decoded


def _rpc_result(error: RpcFailure) -> str:
    if error.code == 1603:
        return PHONE_NOT_IN_FRONT
    if 1001 <= error.code <= 1599:
        return PHONE_UNREACHABLE
    return PHONE_REFUSED


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _haversine_km(origin: tuple[float, float], point: tuple[float, float]) -> float:
    lat1, lng1 = map(math.radians, origin)
    lat2, lng2 = map(math.radians, point)
    delta_lat = lat2 - lat1
    delta_lng = lng2 - lng1
    value = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lng / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(value))


def _street(address: str) -> str:
    return address.split(",", 1)[0].strip()


def _place_line(place: Place, approximate: bool) -> str:
    line = f"{place.name} on {_street(place.address)}"
    if place.distance_km is not None:
        miles = place.distance_km * 0.621371
        about = "about " if approximate else ""
        line += f", {about}{miles:.1f} miles away"
    return line


def _exact_fields(fields: Mapping[str, Any], expected: set[str]) -> None:
    if set(fields) != expected:
        raise ValueError("invalid fields")


def _optional_label_fields(fields: Mapping[str, Any], required: set[str]) -> None:
    if set(fields) not in (required, required | {"label"}):
        raise ValueError("invalid fields")
    if "label" in fields and not isinstance(fields["label"], str):
        raise ValueError("label must be text")


def _nonempty_text(fields: Mapping[str, Any], name: str) -> None:
    if not isinstance(fields.get(name), str) or not fields[name]:
        raise ValueError(f"{name} must be non-empty text")


def _finite_required_number(fields: Mapping[str, Any], name: str) -> None:
    if not _finite_number(fields.get(name)):
        raise ValueError(f"{name} must be finite")


def _integer_range(fields: Mapping[str, Any], name: str, minimum: int, maximum: int | None) -> None:
    value = fields.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is out of range")
