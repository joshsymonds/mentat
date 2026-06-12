"""Constants for the mentat integration."""

DOMAIN = "mentat"

CONF_BASE_URL = "base_url"
DEFAULT_BASE_URL = "http://127.0.0.1:8484"

# mentat session ids are namespaced so a daemon shared with other surfaces
# can never collide with HA's conversation ids.
SESSION_PREFIX = "ha-"

# Voice turns are latency-bound: low effort, identified surface. The user is
# part of the turn's authority context (mentat policy is per-turn).
TURN_META = {"surface": "voice", "user": "josh"}
TURN_EFFORT = "low"
