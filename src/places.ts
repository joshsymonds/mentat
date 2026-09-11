const PLACES_URL = 'https://places.googleapis.com/v1/places:searchText';
const PLACES_FIELD_MASK = 'places.id,places.displayName,places.formattedAddress,places.location';
const PLACES_RADIUS_M = 20_000;

export interface PlaceCandidate {
  name: string;
  address: string;
  place_id: string;
  lat: number;
  lng: number;
}

export interface PlacesLocation {
  lat: number;
  lng: number;
}

export interface PlacesDeps {
  apiKey?: string;
  fetch?: typeof globalThis.fetch;
}

export interface SearchPlacesOptions {
  query: string;
  locality?: string;
  location?: PlacesLocation;
  apiKey: string;
  fetch?: typeof globalThis.fetch;
}

/** Searches Google Places Text Search with an optional phone location bias. */
export async function searchPlaces({
  query,
  locality,
  location,
  apiKey,
  fetch: fetchFn = globalThis.fetch,
}: SearchPlacesOptions): Promise<PlaceCandidate[]> {
  const searchQuery = location
    ? query
    : locality !== undefined && locality.trim() !== ''
      ? `${query} in ${locality.trim()}`
      : query;
  const body: Record<string, unknown> = { textQuery: searchQuery, pageSize: 5 };
  if (location !== undefined) {
    body.locationBias = {
      circle: {
        center: { latitude: location.lat, longitude: location.lng },
        radius: PLACES_RADIUS_M,
      },
    };
  }

  const response = await fetchFn(PLACES_URL, {
    method: 'POST',
    headers: {
      'X-Goog-Api-Key': apiKey,
      'X-Goog-FieldMask': PLACES_FIELD_MASK,
      'content-type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`Places request failed with status ${String(response.status)}`);
  }
  const payload: unknown = await response.json();
  return parseCandidates(payload).slice(0, 3);
}

function parseCandidates(payload: unknown): PlaceCandidate[] {
  if (payload === null || typeof payload !== 'object') {
    throw new Error('Places response has no places list');
  }
  const places = (payload as { places?: unknown }).places;
  if (!Array.isArray(places)) {
    throw new Error('Places response has no places list');
  }
  return places.map((place) => parseCandidate(place));
}

function parseCandidate(value: unknown): PlaceCandidate {
  if (value === null || typeof value !== 'object') {
    throw new Error('Place is not an object');
  }
  const place = value as Record<string, unknown>;
  const displayName = place.displayName;
  const location = place.location;
  const name =
    displayName !== null && typeof displayName === 'object'
      ? (displayName as { text?: unknown }).text
      : undefined;
  const address = place.formattedAddress;
  const placeId = place.id;
  const lat =
    location !== null && typeof location === 'object'
      ? (location as { latitude?: unknown }).latitude
      : undefined;
  const lng =
    location !== null && typeof location === 'object'
      ? (location as { longitude?: unknown }).longitude
      : undefined;
  if (
    typeof name !== 'string' ||
    name === '' ||
    typeof address !== 'string' ||
    address === '' ||
    typeof placeId !== 'string' ||
    placeId === '' ||
    !finiteNumber(lat) ||
    !finiteNumber(lng)
  ) {
    throw new Error('Place is missing required fields');
  }
  return { name, address, place_id: placeId, lat, lng };
}

function finiteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}
