from __future__ import annotations

import re


def map_ami_session(stem: str) -> str | None:
    match = re.match(r"^sdm_([A-Z]{2}\d+[a-z])-\d+$", stem)
    return match.group(1) if match else None


def map_nsf_session(stem: str) -> str | None:
    match = re.match(r"^sdm_(MTG_\d+)_(sc_[a-z]+)_([0-9]+)-[0-9]+$", stem)
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}_{match.group(3)}"


def map_l2m_session(stem: str) -> str:
    return stem


SESSION_MAPPERS = {
    "ami": map_ami_session,
    "nsf": map_nsf_session,
    "l2m": map_l2m_session,
}
