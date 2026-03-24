"""SoccerNet Label parser"""

import json
import os
from pathlib import Path
from typing import Optional
import yaml

# def load_config(path: str = None) -> dict:
#     if path is None:
#         path = Path(__file__).parent.parent / "config.yaml"
#     with open(path) as f:
#         return yaml.safe_load(f)
    
CLASS_NAMES = [
    "Kick-off",
    "Goal", 
    "Corner",
    "Cards",
    "Substitution",
    "Foul", 
    "Penalty (Kick)",
    "Direct free-kick",
    "Indirect free-kick",
    "Ball out of play",
    "Throw-in",
    "Clearance",
    "Shot",
    "Shot (Off)",
    "Keepersave",
    "Dangerous moment",
    "Starting"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Parse game time
def parse_gametime(game_time_str: str) -> tuple[int, float]:
    """
    Parse SoccerNet gameTime string "half - MM:SS" → (half: int, seconds: float)
    half is 1 or 2.
    """
    parts = game_time_str.split(" - ")
    half = int(parts[0])
    mm, ss = parts[1].split(":")
    seconds = int(mm) * 60 + float(ss)
    return half, seconds


def parse_labels_file(label_path: str, visibility_filter: bool = True) -> list[dict]:
    """
    Parse a single Labels-v2.json or Labels-ball.json file.

    Returns list of:
        {
            "half":        int (1 or 2),
            "timestamp":   float (seconds from half start),
            "position_ms": int,
            "label":       str (standard SoccerNet label),
            "class_idx":   int | None,
            "team":        str,
            "visibility":  str
        }
    """
    with open(label_path) as f:
        data = json.load(f)

    events = []
    for ann in data.get("annotations", []):
        label = ann.get("label", "")
        visibility = ann.get("visibility", "visible")

        # Optionally skip "not shown" events (off-camera actions)
        if visibility_filter and visibility == "not shown":
            continue

        half, ts = parse_gametime(ann["gameTime"])
        class_idx = CLASS_TO_IDX.get(label)

        events.append({
            "half":         half,
            "timestamp":    ts,
            "position_ms":  ann.get("position", 0),
            "label":        label,
            "class_idx":    class_idx,
            "team":         ann.get("team", ""),
            "visibility":   visibility,
        })

    return events

def load_game_labels(game_dir: str, visibility_filter: bool = True) -> dict[int, list[dict]]:
    """
    Load both halves for a game directory.
    Returns {1: [events...], 2: [events...]}
    """
    result = {}
    for half in (1, 2):
        label_file = os.path.join(game_dir, f"{half}_Labels-v2.json")
        # Fallback: single combined file
        if not os.path.exists(label_file):
            label_file = os.path.join(game_dir, "Labels-v2.json")
        if os.path.exists(label_file):
            events = [e for e in parse_labels_file(label_file, visibility_filter)
                     if e["half"] == half]
            result[half] = events
        else:
            result[half] = []
    return result
