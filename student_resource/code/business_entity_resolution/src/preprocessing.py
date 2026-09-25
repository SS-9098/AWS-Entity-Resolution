"""
Preprocessing module for business entity resolution.

Handles text normalization, transliteration, and feature extraction for
business records with columns: entity_id, business_name, business_address, country.

Designed for large-scale datasets (2M+ rows per source). All regex patterns are
compiled at module level and expensive string functions are LRU-cached.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from tqdm import tqdm
from unidecode import unidecode

__all__ = [
    "transliterate_text",
    "standardize_name",
    "standardize_address",
    "extract_name_tokens",
    "parse_address_components",
    "preprocess_dataframe",
]

# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level for performance)
# ---------------------------------------------------------------------------

# Unicode script detection ranges
_RE_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_RE_TAMIL = re.compile(r"[\u0B80-\u0BFF]")
_RE_KANNADA = re.compile(r"[\u0C80-\u0CFF]")
_RE_NON_ASCII = re.compile(r"[^\x00-\x7F]")

# Name cleaning patterns
_RE_PARENTHESIZED = re.compile(r"[\(\[\{][^)\]\}]*[\)\]\}]")
_RE_URL = re.compile(
    r"(?:https?://)?(?:www\.)?[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?"
)
_RE_DBA_PREFIX = re.compile(r"^dba\s+")
_RE_THE_PREFIX = re.compile(r"^the\s+")
_RE_LEGAL_SUFFIX = re.compile(
    r"\b(?:private limited|pvt ltd|limited|ltd|corporation|corp"
    r"|incorporated|inc|llc|llp|company|co)\s*\.?\s*$"
)
_RE_MULTI_SPACE = re.compile(r"\s{2,}")
_RE_PUNCTUATION_STRIP = re.compile(r"[#\-\.',|@]")

# Address patterns
_RE_UNIT = re.compile(r"\bunit\s+\S+", re.IGNORECASE)
_RE_PO_BOX = re.compile(r"\bp\.?\s*o\.?\s*box\s+\S+", re.IGNORECASE)
_RE_HASH = re.compile(r"#")

# Address parsing patterns
_RE_US_ZIP = re.compile(r"\b(\d{5}(?:-\d{4})?)\b")
_RE_INDIA_PIN = re.compile(r"\b(\d{6})\b")
_RE_FRANCE_POSTAL = re.compile(r"\b(\d{5})\b")
_RE_GENERIC_POSTAL = re.compile(r"\b(\d{4,6}(?:-\d{4})?)\b")

# Common French regions / departments (partial; enough for soft matching)
_FRENCH_REGIONS: set[str] = {
    "ile de france", "île-de-france", "auvergne rhone alpes", "auvergne-rhône-alpes",
    "nouvelle aquitaine", "occitanie", "hauts de france", "provence alpes cote d azur",
    "grand est", "pays de la loire", "bretagne", "normandie", "bourgogne franche comte",
    "centre val de loire", "corse", "paris", "lyon", "marseille", "toulouse",
    "nice", "nantes", "montpellier", "strasbourg", "bordeaux", "lille", "rennes",
}
_FRENCH_REGIONS_SORTED = sorted(
    {unidecode(r).lower() for r in _FRENCH_REGIONS}, key=len, reverse=True
)
_RE_FRENCH_PLACE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _FRENCH_REGIONS_SORTED) + r")\b"
) if _FRENCH_REGIONS_SORTED else None

# ---------------------------------------------------------------------------
# Name abbreviation expansions
# ---------------------------------------------------------------------------

_NAME_ABBREVIATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bpvt\.?\b"), "private"),
    (re.compile(r"\bltd\.?\b"), "limited"),
    (re.compile(r"\bcorp\.?\b"), "corporation"),
    (re.compile(r"\binc\.?\b"), "incorporated"),
    # llc / llp kept as-is (no expansion)
    (re.compile(r"\bco\.?\b"), "company"),
    (re.compile(r"\bintl\.?\b"), "international"),
    (re.compile(r"\bassoc\.?\b"), "associates"),
    (re.compile(r"\bmfg\.?\b"), "manufacturing"),
    (re.compile(r"\btech\.?\b"), "technology"),
    (re.compile(r"\bgovt\.?\b"), "government"),
    (re.compile(r"\bnatl\.?\b"), "national"),
    (re.compile(r"\bsvcs\.?\b"), "services"),
    (re.compile(r"\bsvc\.?\b"), "service"),
    (re.compile(r"\bmgmt\.?\b"), "management"),
    (re.compile(r"\bgrp\.?\b"), "group"),
    (re.compile(r"\bdept\.?\b"), "department"),
    (re.compile(r"\bengg\.?\b"), "engineering"),
    (re.compile(r"\bedu\.?\b"), "education"),
    (re.compile(r"\bhosp\.?\b"), "hospital"),
]

# ---------------------------------------------------------------------------
# Address abbreviation expansions
# ---------------------------------------------------------------------------

_US_ADDRESS_ABBREVIATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bblvd\.?\b"), "boulevard"),
    (re.compile(r"\bpkwy\.?\b"), "parkway"),
    (re.compile(r"\bbldg\.?\b"), "building"),
    (re.compile(r"\bave\.?\b"), "avenue"),
    (re.compile(r"\bhwy\.?\b"), "highway"),
    (re.compile(r"\bapt\.?\b"), "apartment"),
    (re.compile(r"\bste\.?\b"), "suite"),
    (re.compile(r"\bcir\.?\b"), "circle"),
    (re.compile(r"\bst\.?\b"), "street"),
    (re.compile(r"\brd\.?\b"), "road"),
    (re.compile(r"\bdr\.?\b"), "drive"),
    (re.compile(r"\bln\.?\b"), "lane"),
    (re.compile(r"\bct\.?\b"), "court"),
    (re.compile(r"\bpl\.?\b"), "place"),
    (re.compile(r"\bfl\.?\b"), "floor"),
]

_INDIA_ADDRESS_ABBREVIATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bh\.?\s*no\.?\b"), "house number"),
    (re.compile(r"\bhno\.?\b"), "house number"),
    (re.compile(r"\bhn\.?\b"), "house number"),
    (re.compile(r"\bs\.?\s*no\.?\b"), "survey number"),
    (re.compile(r"\bsno\.?\b"), "survey number"),
    (re.compile(r"\bopp\.?\b"), "opposite"),
    (re.compile(r"\bnr\.?\b"), "near"),
    (re.compile(r"\bb/h\b"), "behind"),
]

# ---------------------------------------------------------------------------
# US state abbreviations (50 states + DC + common territories)
# ---------------------------------------------------------------------------

_US_STATES: dict[str, str] = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
    "pr": "puerto rico", "gu": "guam", "vi": "virgin islands", "as": "american samoa",
    "mp": "northern mariana islands",
}

# Build a regex that matches any 2-letter US state abbreviation as a whole word
_US_STATE_ABBREVS_SET = set(_US_STATES.keys())
_RE_US_CITY_STATE_ZIP = re.compile(
    r"(?P<city>[a-z][a-z ]+?)\s*,\s*(?P<state>[a-z]{2})\s*(?P<zip>\d{5}(?:-\d{4})?)?\s*$"
)

# ---------------------------------------------------------------------------
# Indian states / union territories
# ---------------------------------------------------------------------------

_INDIAN_STATES: set[str] = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
    # Union territories
    "andaman and nicobar islands", "chandigarh", "dadra and nagar haveli and daman and diu",
    "delhi", "jammu and kashmir", "ladakh", "lakshadweep", "puducherry",
    "new delhi",
}

# Pre-sort by length (longest first) so greedy matching works
_INDIAN_STATES_SORTED = sorted(_INDIAN_STATES, key=len, reverse=True)
_RE_INDIAN_STATE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _INDIAN_STATES_SORTED) + r")\b"
)

# Known major Indian cities (non-exhaustive, covers tier-1/2)
_INDIAN_CITIES: set[str] = {
    "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "ahmedabad",
    "chennai", "kolkata", "pune", "jaipur", "lucknow", "kanpur", "nagpur",
    "indore", "thane", "bhopal", "visakhapatnam", "pimpri chinchwad",
    "patna", "vadodara", "ghaziabad", "ludhiana", "agra", "nashik", "faridabad",
    "meerut", "rajkot", "kalyan dombivli", "vasai virar", "varanasi",
    "srinagar", "aurangabad", "dhanbad", "amritsar", "navi mumbai",
    "allahabad", "prayagraj", "ranchi", "howrah", "coimbatore", "jabalpur",
    "gwalior", "vijayawada", "jodhpur", "madurai", "raipur", "kota",
    "chandigarh", "guwahati", "solapur", "hubli dharwad", "mysore", "mysuru",
    "tiruchirappalli", "bareilly", "aligarh", "tiruppur", "moradabad",
    "jalandhar", "bhubaneswar", "salem", "warangal", "guntur", "bhiwandi",
    "saharanpur", "gorakhpur", "bikaner", "amravati", "noida", "jamshedpur",
    "bhilai", "cuttack", "firozabad", "kochi", "ernakulam", "nellore",
    "bhavnagar", "dehradun", "durgapur", "asansol", "rourkela", "nanded",
    "kolhapur", "ajmer", "gulbarga", "jamnagar", "ujjain", "loni", "siliguri",
    "jhansi", "ulhasnagar", "jammu", "sangli miraj kupwad", "mangalore",
    "erode", "belgaum", "ambattur", "tirunelveli", "malegaon", "gaya",
    "udaipur", "thiruvananthapuram", "trivandrum",
    "secunderabad", "gurugram", "gurgaon", "greater noida",
}

_INDIAN_CITIES_SORTED = sorted(_INDIAN_CITIES, key=len, reverse=True)
_RE_INDIAN_CITY = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _INDIAN_CITIES_SORTED) + r")\b"
)

# ---------------------------------------------------------------------------
# Business name stop words
# ---------------------------------------------------------------------------

_NAME_STOP_WORDS: set[str] = {"and", "of", "the", "for", "in", "a", "an"}

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1_000_000)
def transliterate_text(text: str) -> str:
    """Convert non-Latin scripts to Latin and normalize to ASCII lowercase.

    Handles:
    - Hindi (Devanagari) via ITRANS transliteration
    - Tamil via ITRANS transliteration
    - Kannada via ITRANS transliteration
    - Accented Latin characters (e.g. ``Béque`` → ``beque``)
    - Any remaining non-ASCII via ``unidecode``

    Parameters
    ----------
    text : str
        Input string, possibly containing Indic scripts or accented chars.

    Returns
    -------
    str
        Lowercased ASCII transliteration of the input.
    """
    if not text:
        return ""

    result = text

    # Transliterate Indic scripts to ITRANS (Latin representation)
    if _RE_DEVANAGARI.search(result):
        result = transliterate(result, sanscript.DEVANAGARI, sanscript.ITRANS)
    if _RE_TAMIL.search(result):
        result = transliterate(result, sanscript.TAMIL, sanscript.ITRANS)
    if _RE_KANNADA.search(result):
        result = transliterate(result, sanscript.KANNADA, sanscript.ITRANS)

    # Fallback: convert any remaining non-ASCII (accented chars, etc.)
    if _RE_NON_ASCII.search(result):
        result = unidecode(result)

    return result.lower()


@lru_cache(maxsize=1_000_000)
def standardize_name(name: str) -> str:
    """Normalize a business name for entity resolution.

    Pipeline:
    1. Transliterate to ASCII
    2. Expand common abbreviations (pvt → private, etc.)
    3. Remove parenthesized/bracketed content
    4. Remove URLs
    5. Remove ``dba`` / ``the`` prefixes and trailing legal suffixes
    6. Replace ``&`` with ``and``; strip punctuation
    7. Collapse whitespace

    Parameters
    ----------
    name : str
        Raw business name.

    Returns
    -------
    str
        Cleaned, lowercased business name.
    """
    if not name:
        return ""

    text = transliterate_text(name)

    # Expand abbreviations
    for pattern, replacement in _NAME_ABBREVIATIONS:
        text = pattern.sub(replacement, text)

    # Remove content inside parentheses / brackets / braces
    text = _RE_PARENTHESIZED.sub(" ", text)

    # Remove URLs
    text = _RE_URL.sub(" ", text)

    # Remove 'dba ' prefix
    text = _RE_DBA_PREFIX.sub("", text)

    # Remove 'the ' prefix
    text = _RE_THE_PREFIX.sub("", text)

    # Remove trailing legal suffixes (after abbreviation expansion)
    text = _RE_LEGAL_SUFFIX.sub("", text)

    # Replace '&' with 'and'
    text = text.replace("&", " and ")

    # Strip punctuation characters
    text = _RE_PUNCTUATION_STRIP.sub(" ", text)

    # Collapse multiple spaces and strip
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    return text


@lru_cache(maxsize=1_000_000)
def standardize_address(address: str, country: str) -> str:
    """Normalize a business address for entity resolution.

    Pipeline:
    1. Transliterate to ASCII
    2. Expand country-specific abbreviations (US road types, Indian prefixes)
    3. Normalize unit / PO Box references
    4. Strip ``#`` symbols
    5. Collapse whitespace

    Parameters
    ----------
    address : str
        Raw address string.
    country : str
        ISO-style country identifier (e.g. ``'us'``, ``'in'``, ``'india'``).

    Returns
    -------
    str
        Cleaned, lowercased address.
    """
    if not address:
        return ""

    text = transliterate_text(address)
    country_lower = country.lower().strip() if country else ""

    # Country-specific abbreviation expansion
    if country_lower in ("us", "usa", "united states", "united states of america"):
        for pattern, replacement in _US_ADDRESS_ABBREVIATIONS:
            text = pattern.sub(replacement, text)
    elif country_lower in ("in", "ind", "india"):
        for pattern, replacement in _INDIA_ADDRESS_ABBREVIATIONS:
            text = pattern.sub(replacement, text)

    # Normalize unit / PO box (remove for cleaner matching)
    text = _RE_UNIT.sub(" ", text)
    text = _RE_PO_BOX.sub(" ", text)

    # Remove hash symbols
    text = _RE_HASH.sub(" ", text)

    # Collapse multiple spaces and strip
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    return text


def extract_name_tokens(normalized_name: str) -> list[str]:
    """Extract meaningful tokens from an already-normalized business name.

    Splits on whitespace, removes very short tokens (< 2 chars unless they
    look like single-letter initials), and filters out common stop words.

    Parameters
    ----------
    normalized_name : str
        A name string that has already been through :func:`standardize_name`.

    Returns
    -------
    list[str]
        Sorted list of unique, meaningful tokens.
    """
    if not normalized_name:
        return []

    tokens: list[str] = []
    for tok in normalized_name.split():
        # Keep single-character tokens only if they are alphabetic (initials)
        if len(tok) < 2 and not tok.isalpha():
            continue
        if tok in _NAME_STOP_WORDS:
            continue
        tokens.append(tok)

    # Return sorted unique tokens for deterministic comparison
    return sorted(set(tokens))


def parse_address_components(address: str, country: str) -> dict[str, Any]:
    """Parse an address string into structured components.

    Attempts country-specific regex extraction of city, state, postal code,
    and street. Falls back gracefully to returning the raw address when
    parsing fails.

    Parameters
    ----------
    address : str
        Normalized address string (output of :func:`standardize_address`).
    country : str
        ISO-style country identifier.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``'city'``, ``'state'``, ``'pin_code'``,
        ``'street'``, and ``'raw'``.
    """
    result: dict[str, Any] = {
        "city": "",
        "state": "",
        "pin_code": "",
        "street": "",
        "raw": address if address else "",
    }

    if not address:
        return result

    addr_lower = address.lower().strip()
    country_lower = country.lower().strip() if country else ""

    # --- US parsing ---
    if country_lower in ("us", "usa", "united states", "united states of america"):
        _parse_us_address(addr_lower, result)

    # --- India parsing ---
    elif country_lower in ("in", "ind", "india"):
        _parse_india_address(addr_lower, result)

    # --- France / other countries: generic postal + place extraction ---
    else:
        _parse_generic_address(addr_lower, result, country_lower)

    return result


def _parse_generic_address(addr: str, result: dict[str, Any], country: str) -> None:
    """Best-effort parsing for France and other open-set countries."""
    if country in ("fr", "france"):
        postal = _RE_FRANCE_POSTAL.search(addr)
        if postal:
            result["pin_code"] = postal.group(1)
        if _RE_FRENCH_PLACE is not None:
            place = _RE_FRENCH_PLACE.search(addr)
            if place:
                result["city"] = place.group(1)
    else:
        postal = _RE_GENERIC_POSTAL.search(addr)
        if postal:
            result["pin_code"] = postal.group(1)

    # Street fallback: strip known components
    street = addr
    if result["pin_code"]:
        street = street.replace(result["pin_code"], "")
    if result["city"]:
        street = street.replace(result["city"], "")
    result["street"] = _RE_MULTI_SPACE.sub(" ", street).strip().strip(",").strip() or addr


def _parse_us_address(addr: str, result: dict[str, Any]) -> None:
    """Parse a US address into components (in-place update of *result*)."""
    # Try standard "city, ST ZIP" at end of string
    m = _RE_US_CITY_STATE_ZIP.search(addr)
    if m:
        state_abbr = m.group("state").lower()
        if state_abbr in _US_STATE_ABBREVS_SET:
            result["state"] = _US_STATES[state_abbr]
            result["city"] = m.group("city").strip()
            if m.group("zip"):
                result["pin_code"] = m.group("zip")
            # Everything before the city is the street
            street = addr[: m.start()].strip().rstrip(",").strip()
            result["street"] = street
            return

    # Fallback: extract ZIP code anywhere
    zip_match = _RE_US_ZIP.search(addr)
    if zip_match:
        result["pin_code"] = zip_match.group(1)

    # Fallback: look for any known state abbreviation in comma-separated parts
    parts = [p.strip() for p in addr.split(",")]
    for part in parts:
        token = part.strip()
        # Could be 'ST' or 'ST 12345'
        first_word = token.split()[0] if token else ""
        if first_word in _US_STATE_ABBREVS_SET:
            result["state"] = _US_STATES[first_word]
            break

    # Assign street as everything if nothing better found
    if not result["street"]:
        result["street"] = addr


def _parse_india_address(addr: str, result: dict[str, Any]) -> None:
    """Parse an Indian address into components (in-place update of *result*)."""
    # Extract PIN code (6-digit number)
    pin_match = _RE_INDIA_PIN.search(addr)
    if pin_match:
        result["pin_code"] = pin_match.group(1)

    # Extract state
    state_match = _RE_INDIAN_STATE.search(addr)
    if state_match:
        result["state"] = state_match.group(1)

    # Extract city
    city_match = _RE_INDIAN_CITY.search(addr)
    if city_match:
        result["city"] = city_match.group(1)

    # Street: everything that isn't state/city/pin — simplified as full raw
    street = addr
    if state_match:
        street = street.replace(state_match.group(1), "")
    if city_match:
        street = street.replace(city_match.group(1), "")
    if pin_match:
        street = street.replace(pin_match.group(1), "")
    result["street"] = _RE_MULTI_SPACE.sub(" ", street).strip().strip(",").strip()


# ---------------------------------------------------------------------------
# DataFrame-level preprocessing
# ---------------------------------------------------------------------------


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to a business-records DataFrame.

    Creates new columns:

    * ``norm_name`` – normalized business name
    * ``norm_address`` – normalized business address
    * ``name_tokens`` – sorted unique meaningful tokens from the name
    * ``addr_components`` – parsed address dict

    Uses :mod:`tqdm` progress bars and leverages ``lru_cache`` on the
    underlying functions for repeated-value deduplication.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``business_name``, ``business_address``, and
        ``country``.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with four new columns appended.
    """
    tqdm.pandas(desc="Normalizing names")
    df["norm_name"] = df["business_name"].fillna("").progress_apply(standardize_name)

    tqdm.pandas(desc="Normalizing addresses")
    df["norm_address"] = df.progress_apply(
        lambda row: standardize_address(
            str(row["business_address"]) if pd.notna(row["business_address"]) else "",
            str(row["country"]) if pd.notna(row["country"]) else "",
        ),
        axis=1,
    )

    tqdm.pandas(desc="Extracting name tokens")
    df["name_tokens"] = df["norm_name"].progress_apply(extract_name_tokens)

    tqdm.pandas(desc="Parsing address components")
    df["addr_components"] = df.progress_apply(
        lambda row: parse_address_components(
            row["norm_address"],
            str(row["country"]) if pd.notna(row["country"]) else "",
        ),
        axis=1,
    )

    return df
