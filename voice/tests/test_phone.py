"""Offline tests for the phone-control pure module."""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phone import (
    ALARM_SET,
    DIAL_READY,
    LINK_OPENED,
    LOCATION_UNAVAILABLE,
    NAVIGATION_STARTED,
    NO_CANDIDATES,
    NO_RESULTS,
    PHONE_NOT_IN_FRONT,
    PHONE_REFUSED,
    PHONE_UNREACHABLE,
    PLACES_FAILED,
    TEXT_READY,
    TIMER_SET,
    PLACES_UNCONFIGURED,
    Place,
    PhoneActions,
    RpcFailure,
    command_payload,
    describe_places,
    open_url_allowed,
    parse_places,
    phone_identity,
    places_request,
)


class FakePhone:
    def __init__(self, identities=("pixel-shrike",), response='{"lat":51.5,"lng":-0.1,"accuracy_m":5,"age_s":1}'):
        self._identities = identities
        self.response = response
        self.rpc_calls = []
        self.rpc_error = None
        self.rpc_timeout = False

    def identities(self):
        return self._identities

    async def perform_rpc(self, identity, method, payload, timeout):
        self.rpc_calls.append((identity, method, payload, timeout))
        if self.rpc_timeout:
            raise asyncio.TimeoutError()
        if self.rpc_error is not None:
            raise self.rpc_error
        return self.response


class FakeHttp:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.error = None

    async def post_json(self, url, headers, body):
        self.calls.append((url, headers, body))
        if self.error is not None:
            raise self.error
        return self.response


def run(coro):
    return asyncio.run(coro)


class ResultStringTest(unittest.TestCase):
    def test_status_lines_start_with_their_token_and_explain_next_action(self):
        self.assertEqual(
            PHONE_UNREACHABLE,
            "PHONE_UNREACHABLE: no phone is connected to this conversation, or it did not answer.",
        )
        self.assertEqual(
            PHONE_REFUSED,
            "PHONE_REFUSED: the phone rejected that action.",
        )
        self.assertEqual(
            PHONE_NOT_IN_FRONT,
            "PHONE_NOT_IN_FRONT: the Mentat screen is covered by another app, so the phone will not launch anything until he taps the side button.",
        )
        self.assertEqual(
            LOCATION_UNAVAILABLE,
            "LOCATION_UNAVAILABLE: the phone could not give a fresh location, so ask roughly where he is and search again with that locality.",
        )
        self.assertEqual(
            PLACES_UNCONFIGURED,
            "PLACES_UNCONFIGURED: place search has no API key on the server.",
        )
        self.assertEqual(
            PLACES_FAILED,
            "PLACES_FAILED: the place search request failed.",
        )
        self.assertEqual(
            NO_CANDIDATES,
            "NO_CANDIDATES: there is no current place list to choose from; search first.",
        )
        self.assertEqual(NO_RESULTS, "NO_RESULTS: no places matched.")


class PlacesRequestTest(unittest.TestCase):
    def test_request_with_origin_has_google_headers_and_circle_bias(self):
        url, headers, body = places_request("coffee", (51.5, -0.1), "secret")
        self.assertEqual(url, "https://places.googleapis.com/v1/places:searchText")
        self.assertEqual(headers["X-Goog-Api-Key"], "secret")
        self.assertEqual(
            headers["X-Goog-FieldMask"],
            "places.id,places.displayName,places.formattedAddress,places.location",
        )
        self.assertEqual(body["textQuery"], "coffee")
        self.assertEqual(body["pageSize"], 5)
        self.assertEqual(
            body["locationBias"],
            {
                "circle": {
                    "center": {"latitude": 51.5, "longitude": -0.1},
                    "radius": 20000.0,
                }
            },
        )

    def test_request_without_origin_omits_location_bias(self):
        _, _, body = places_request("coffee", None, "secret")
        self.assertNotIn("locationBias", body)


class ParsePlacesTest(unittest.TestCase):
    PAYLOAD = {
        "places": [
            {
                "id": "far",
                "displayName": {"text": "Far Cafe"},
                "formattedAddress": "20 High Street, Town",
                "location": {"latitude": 51.52, "longitude": -0.1},
            },
            {
                "id": "near",
                "displayName": {"text": "Near Cafe"},
                "formattedAddress": "2 Main Street, Town",
                "location": {"latitude": 51.5, "longitude": -0.1},
            },
        ]
    }

    def test_parses_and_sorts_by_haversine_distance(self):
        places = parse_places(self.PAYLOAD, (51.5, -0.1))
        self.assertEqual([place.place_id for place in places], ["near", "far"])
        self.assertIsNotNone(places[0].distance_km)
        self.assertAlmostEqual(places[0].distance_km or 0.0, 0.0, places=6)
        self.assertGreater(places[1].distance_km or 0.0, 0.0)

    def test_distance_is_none_without_origin(self):
        places = parse_places(self.PAYLOAD, None)
        self.assertTrue(all(place.distance_km is None for place in places))

    def test_malformed_payload_raises_value_error(self):
        with self.assertRaises(ValueError):
            parse_places({"places": [{"id": "missing"}]}, None)
        with self.assertRaises(ValueError):
            parse_places({"places": "nope"}, None)


class DescribePlacesTest(unittest.TestCase):
    def setUp(self):
        self.places = [
            Place("Alpha", "1 Alpha Road, Town", "a", 0.0, 0.0, 1.60934),
            Place("Bravo", "2 Bravo Road, Town", "b", 0.0, 0.01, 3.21868),
            Place("Charlie", "3 Charlie Road, Town", "c", 0.0, 0.02, 4.82802),
            Place("Delta", "4 Delta Road, Town", "d", 0.0, 0.03, 6.43736),
            Place("Echo", "5 Echo Road, Town", "e", 0.0, 0.04, 8.0467),
        ]

    def test_one_result_names_match_and_street(self):
        text = describe_places(self.places[:1], False)
        self.assertEqual(text, "I found Alpha on 1 Alpha Road, 1.0 miles away.")

    def test_three_results_use_spoken_ordinals_and_distances(self):
        text = describe_places(self.places[:3], False)
        self.assertEqual(
            text,
            "I found three: the first is Alpha on 1 Alpha Road, 1.0 miles away; "
            "the second is Bravo on 2 Bravo Road, 2.0 miles away; "
            "the third is Charlie on 3 Charlie Road, 3.0 miles away. Which one?",
        )
        self.assertNotIn("1. Alpha", text)
        self.assertNotIn("2. Bravo", text)
        self.assertNotIn("3. Charlie", text)

    def test_five_results_only_speak_three_and_approximate(self):
        text = describe_places(self.places, True)
        self.assertIn("I found three:", text)
        self.assertIn("about 1.0 miles away", text)
        self.assertNotIn("Delta", text)
        self.assertNotIn("Echo", text)

    def test_results_without_origin_have_no_distances(self):
        no_origin = [
            Place(place.name, place.address, place.place_id, place.lat, place.lng, None)
            for place in self.places[:3]
        ]
        text = describe_places(no_origin, False)
        self.assertIn("I found three: the first is Alpha on 1 Alpha Road", text)
        self.assertNotIn("at ", text)
        self.assertNotIn("miles", text)

    def test_empty_results_return_no_results(self):
        self.assertEqual(describe_places([], False), NO_RESULTS)


class CommandPayloadTest(unittest.TestCase):
    def assertPayload(self, kind, expected, **fields):
        self.assertEqual(json.loads(command_payload(kind, **fields)), expected)

    def test_all_six_command_kinds(self):
        self.assertPayload(
            "navigate",
            {"kind": "navigate", "name": "Cafe", "address": "1 Main", "place_id": "p", "lat": 1.0, "lng": 2.0},
            name="Cafe", address="1 Main", place_id="p", lat=1.0, lng=2.0,
        )
        self.assertPayload("dial", {"kind": "dial", "number": "+123"}, number="+123")
        self.assertPayload("sms", {"kind": "sms", "number": "+123", "body": "hi"}, number="+123", body="hi")
        self.assertPayload("alarm", {"kind": "alarm", "hour": 7, "minute": 30}, hour=7, minute=30)
        self.assertPayload("timer", {"kind": "timer", "seconds": 90}, seconds=90)
        self.assertPayload("open", {"kind": "open", "url": "https://example.com"}, url="https://example.com")

    def test_optional_labels_are_preserved(self):
        self.assertPayload("alarm", {"kind": "alarm", "hour": 7, "minute": 30, "label": "Wake"}, hour=7, minute=30, label="Wake")
        self.assertPayload("timer", {"kind": "timer", "seconds": 90, "label": "Tea"}, seconds=90, label="Tea")

    def test_missing_or_invalid_fields_raise_value_error(self):
        invalid = (
            ("dial", {}),
            ("sms", {"number": "+1"}),
            ("alarm", {"hour": 25, "minute": 0}),
            ("timer", {"seconds": -1}),
            ("open", {"url": "file:///tmp/x"}),
            ("navigate", {"name": "n", "address": "a", "place_id": "p", "lat": 0}),
            ("unknown", {}),
        )
        for kind, fields in invalid:
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    command_payload(kind, **fields)


class SmallPureHelpersTest(unittest.TestCase):
    def test_open_url_allows_only_absolute_http_and_https(self):
        self.assertTrue(open_url_allowed("https://example.com/a"))
        self.assertTrue(open_url_allowed("http://example.com"))
        self.assertFalse(open_url_allowed("/relative"))
        self.assertFalse(open_url_allowed("javascript:alert(1)"))
        self.assertFalse(open_url_allowed("file:///tmp/a"))

    def test_phone_identity_picks_pixel_identity(self):
        self.assertEqual(phone_identity(["agent", "pixel-shrike", "pixel-other"]), "pixel-shrike")
        self.assertIsNone(phone_identity(["agent", "browser"]))


class PhoneActionsTest(unittest.TestCase):
    def setUp(self):
        self.phone = FakePhone()
        self.http = FakeHttp({"places": []})
        self.actions = PhoneActions(self.phone.perform_rpc, self.http.post_json, self.phone.identities, "key")

    def test_missing_pixel_is_unreachable(self):
        phone = FakePhone(("browser",))
        actions = PhoneActions(phone.perform_rpc, self.http.post_json, phone.identities, "key")
        self.assertEqual(run(actions.find_places("coffee")), PHONE_UNREACHABLE)
        self.assertEqual(phone.rpc_calls, [])

    def test_rpc_1603_is_not_in_front_and_other_rpc_is_refused(self):
        self.phone.rpc_error = RpcFailure(1603, "not in front")
        self.assertEqual(run(self.actions.find_places("coffee")), PHONE_NOT_IN_FRONT)
        self.phone.rpc_error = RpcFailure(1602, "location unavailable")
        self.assertEqual(run(self.actions.find_places("coffee", "Cambridge")), NO_RESULTS)
        self.phone.rpc_error = RpcFailure(1600, "bad command")
        self.assertEqual(run(self.actions.dial("+1")), PHONE_REFUSED)

    def test_transport_rpc_codes_are_unreachable(self):
        for code in (1502, 1400):
            with self.subTest(code=code):
                self.phone.rpc_error = RpcFailure(code, "transport failure")
                self.assertEqual(run(self.actions.find_places("coffee")), PHONE_UNREACHABLE)
        self.phone.rpc_error = RpcFailure(1600, "invalid command")
        self.assertEqual(run(self.actions.find_places("coffee")), PHONE_REFUSED)

    def test_rpc_timeout_is_unreachable(self):
        self.phone.rpc_timeout = True
        self.assertEqual(run(self.actions.find_places("coffee")), PHONE_UNREACHABLE)

    def test_missing_key_is_unconfigured(self):
        actions = PhoneActions(self.phone.perform_rpc, self.http.post_json, self.phone.identities, "")
        self.assertEqual(run(actions.find_places("coffee")), PLACES_UNCONFIGURED)

    def test_location_1602_without_locality_does_not_call_http(self):
        self.phone.rpc_error = RpcFailure(1602, "location unavailable")
        self.assertEqual(run(self.actions.find_places("coffee")), LOCATION_UNAVAILABLE)
        self.assertEqual(self.http.calls, [])

    def test_location_1602_with_locality_searches_without_origin(self):
        self.phone.rpc_error = RpcFailure(1602, "location unavailable")
        self.http.response = {"places": []}
        self.assertEqual(run(self.actions.find_places("coffee", "Cambridge")), NO_RESULTS)
        self.assertEqual(self.http.calls[0][2]["textQuery"], "coffee in Cambridge")
        self.assertNotIn("locationBias", self.http.calls[0][2])
        self.assertEqual(
            self.phone.rpc_calls[0], ("pixel-shrike", "mentat.location", "{}", 10.0)
        )

    def test_malformed_location_with_locality_searches_without_origin(self):
        self.phone.response = "not json"
        self.http.response = {
            "places": [
                {
                    "id": "coffee",
                    "displayName": {"text": "Cambridge Coffee"},
                    "formattedAddress": "1 Main Street, Cambridge",
                    "location": {"latitude": 52.2, "longitude": 0.1},
                }
            ]
        }
        result = run(self.actions.find_places("coffee", "Cambridge"))
        self.assertEqual(result, "I found Cambridge Coffee on 1 Main Street.")
        self.assertEqual(len(self.http.calls), 1)
        self.assertEqual(self.http.calls[0][2]["textQuery"], "coffee in Cambridge")
        self.assertNotIn("locationBias", self.http.calls[0][2])
        self.assertNotIn("miles away", result)

    def test_malformed_location_without_locality_does_not_call_http(self):
        self.phone.response = "not json"
        self.assertEqual(run(self.actions.find_places("coffee")), LOCATION_UNAVAILABLE)
        self.assertEqual(self.http.calls, [])

    def test_http_failure_or_malformed_json_is_places_failed(self):
        self.http.error = RuntimeError("HTTP 500")
        self.assertEqual(run(self.actions.find_places("coffee")), PLACES_FAILED)
        self.http.error = None
        self.http.response = "not json"
        self.assertEqual(run(self.actions.find_places("coffee")), PLACES_FAILED)

    def test_empty_search_has_no_candidates(self):
        self.http.response = {"places": []}
        self.assertEqual(run(self.actions.find_places("coffee")), NO_RESULTS)
        self.assertEqual(run(self.actions.navigate_to(1)), NO_CANDIDATES)

    def test_failed_replacement_search_clears_previous_candidates(self):
        self.http.response = {
            "places": [
                {"id": "a", "displayName": {"text": "A"}, "formattedAddress": "1 A St, Town", "location": {"latitude": 0, "longitude": 0}},
            ]
        }
        self.assertNotEqual(run(self.actions.find_places("first")), NO_RESULTS)
        self.http.error = RuntimeError("network")
        self.assertEqual(run(self.actions.find_places("second")), PLACES_FAILED)
        self.assertEqual(run(self.actions.navigate_to(1)), NO_CANDIDATES)

    def test_out_of_range_choices_have_no_candidates(self):
        self.assertEqual(run(self.actions.navigate_to(0)), NO_CANDIDATES)
        self.assertEqual(run(self.actions.navigate_to(2)), NO_CANDIDATES)

    def test_navigate_to_second_sends_place_payload(self):
        self.http.response = {
            "places": [
                {"id": "a", "displayName": {"text": "A"}, "formattedAddress": "1 A St, Town", "location": {"latitude": 51.5, "longitude": -0.1}},
                {"id": "b", "displayName": {"text": "B"}, "formattedAddress": "2 B St, Town", "location": {"latitude": 51.51, "longitude": -0.1}},
            ]
        }
        run(self.actions.find_places("first"))
        result = run(self.actions.navigate_to(2))
        self.assertEqual(result, NAVIGATION_STARTED)
        identity, method, payload, timeout = self.phone.rpc_calls[-1]
        self.assertEqual((identity, method, timeout), ("pixel-shrike", "mentat.command", 10.0))
        self.assertEqual(json.loads(payload), {"kind": "navigate", "name": "B", "address": "2 B St, Town", "place_id": "b", "lat": 51.51, "lng": -0.1})

    def test_dial_text_alarm_timer_and_link_send_exact_payloads(self):
        self.assertEqual(run(self.actions.dial("+123")), DIAL_READY)
        self.assertEqual(run(self.actions.send_text("+123", "hello")), TEXT_READY)
        self.assertEqual(run(self.actions.set_alarm(7, 30, "Wake")), ALARM_SET)
        self.assertEqual(run(self.actions.set_timer(1, 30, "Tea")), TIMER_SET)
        self.assertEqual(run(self.actions.open_link("https://example.com")), LINK_OPENED)
        payloads = [json.loads(call[2]) for call in self.phone.rpc_calls]
        self.assertEqual(payloads, [
            {"kind": "dial", "number": "+123"},
            {"kind": "sms", "number": "+123", "body": "hello"},
            {"kind": "alarm", "hour": 7, "minute": 30, "label": "Wake"},
            {"kind": "timer", "seconds": 90, "label": "Tea"},
            {"kind": "open", "url": "https://example.com"},
        ])


if __name__ == "__main__":
    unittest.main()
