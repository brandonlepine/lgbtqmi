"""BBQ JSONL parsing and standardization for all bias categories."""

import json
import random
from pathlib import Path
from typing import Any

from src.utils.logging import log

# Mapping from user-facing short names to BBQ JSONL filenames (without extension)
CATEGORY_MAP: dict[str, str] = {
    "so": "Sexual_orientation",
    "gi": "Gender_identity",
    "race": "Race_ethnicity",
    "religion": "Religion",
    "disability": "Disability_status",
    "physical_appearance": "Physical_appearance",
    "age": "Age",
}

# All valid short names
ALL_CATEGORIES: list[str] = list(CATEGORY_MAP.keys())

# Role tags in answer_info[ansN][1] that indicate the "unknown" answer
UNKNOWN_TAGS: set[str] = {"unknown"}


def resolve_category(name: str) -> str:
    """Resolve a category short name or BBQ filename to the canonical short name.

    Accepts: 'so', 'Sexual_orientation', 'sexual_orientation', etc.
    Returns: canonical short name like 'so'.
    """
    lower = name.lower().strip()
    # Direct short name match
    if lower in CATEGORY_MAP:
        return lower
    # Match against BBQ filenames (case-insensitive)
    for short, bbq_name in CATEGORY_MAP.items():
        if lower == bbq_name.lower():
            return short
    raise ValueError(
        f"Unknown category: {name!r}. "
        f"Valid names: {', '.join(ALL_CATEGORIES)} or BBQ filenames."
    )


def parse_categories(categories_str: str) -> list[str]:
    """Parse a comma-separated categories string into canonical short names.

    'all' returns all categories. Otherwise comma-separated short or long names.
    """
    if categories_str.strip().lower() == "all":
        return list(ALL_CATEGORIES)
    return [resolve_category(c) for c in categories_str.split(",")]


def find_bbq_file(category_short: str, data_dir: str | Path) -> Path:
    """Locate the BBQ JSONL file for a category.

    Searches in data_dir for the expected filename.
    """
    data_dir = Path(data_dir)
    bbq_name = CATEGORY_MAP[category_short]
    path = data_dir / f"{bbq_name}.jsonl"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"BBQ file not found: {path}\n"
        f"Expected BBQ data at {data_dir}/. "
        f"Clone BBQ: git clone https://github.com/nyu-mll/BBQ.git datasets/bbq"
    )


def load_bbq_items(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Load all items from a BBQ JSONL file."""
    items = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _classify_answer_role(
    role_tag: str,
    stereotyped_groups: list[str],
) -> str:
    """Classify an answer's role based on its tag and the item's stereotyped groups.

    Returns one of: 'stereotyped_target', 'non_stereotyped', 'unknown'.

    The role_tag comes from answer_info[ansN][1]. It's either:
    - 'unknown' -> the "can't be determined" answer
    - An identity group name that may or may not be in stereotyped_groups
    - A descriptive tag (e.g., 'nonDisabled', 'nonObese', 'nonOld', 'straightMan')

    For stereotyped group matching, we check if the role_tag matches any of the
    stereotyped_groups entries (case-insensitive). If not, it's 'non_stereotyped'.
    """
    tag_lower = role_tag.lower().strip()

    if tag_lower in UNKNOWN_TAGS:
        return "unknown"

    # Check if this role tag matches any stereotyped group
    stereo_lower = [g.lower().strip() for g in stereotyped_groups]

    # Exact match
    if tag_lower in stereo_lower:
        return "stereotyped_target"

    # Compound tag matching: BBQ Race_ethnicity uses tags like "F-Black", "M-Latino"
    # where the prefix is gender (F/M) and the suffix is the racial group.
    for sg in stereo_lower:
        # Check if stereotyped group is a suffix after a hyphen delimiter
        if "-" in tag_lower and tag_lower.split("-", 1)[1] == sg:
            return "stereotyped_target"

    # Multi-word group matching: handle cases like "African American" appearing
    # as a tag or stereotyped group. Only apply substring matching when the tag
    # or group contains a space (multi-word), to avoid false matches like
    # "old" in "nonOld" or "disabled" in "nonDisabled".
    for sg in stereo_lower:
        if " " in sg or " " in tag_lower:
            if sg in tag_lower or tag_lower in sg:
                return "stereotyped_target"

    return "non_stereotyped"


def _determine_alignment(
    context_condition: str,
    correct_role: str,
) -> str:
    """Determine alignment for an item.

    For disambig items:
    - 'aligned' if the correct answer is the stereotyped target
    - 'conflicting' if the correct answer is the non-stereotyped group
    - 'unknown_correct' if the correct answer is unknown (shouldn't happen for disambig)
    For ambig items: always 'ambiguous'.
    """
    if context_condition == "ambig":
        return "ambiguous"
    if correct_role == "stereotyped_target":
        return "aligned"
    if correct_role == "non_stereotyped":
        return "conflicting"
    return "unknown_correct"


def _shuffle_answers(
    ans_texts: list[str],
    ans_roles: list[str],
    correct_idx: int,
    rng: random.Random,
) -> tuple[dict[str, str], str, dict[str, str]]:
    """Shuffle answer positions and return (answers, correct_letter, answer_roles).

    Args:
        ans_texts: [ans0_text, ans1_text, ans2_text]
        ans_roles: [ans0_role, ans1_role, ans2_role]
        correct_idx: index of the correct answer in the original ordering
        rng: seeded Random instance

    Returns:
        answers: {"A": text, "B": text, "C": text}
        correct_letter: "A", "B", or "C"
        answer_roles: {"A": role, "B": role, "C": role}
    """
    indices = [0, 1, 2]
    rng.shuffle(indices)
    letters = ["A", "B", "C"]

    answers = {}
    answer_roles = {}
    correct_letter = ""

    for letter, idx in zip(letters, indices):
        answers[letter] = ans_texts[idx]
        answer_roles[letter] = ans_roles[idx]
        if idx == correct_idx:
            correct_letter = letter

    return answers, correct_letter, answer_roles


def standardize_item(
    raw: dict[str, Any],
    item_idx: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Convert a raw BBQ item into the standardized format.

    Shuffles answer positions, determines roles, computes alignment.
    """
    # Extract answer info
    answer_info = raw["answer_info"]
    stereotyped_groups = raw["additional_metadata"]["stereotyped_groups"]

    ans_texts = [raw["ans0"], raw["ans1"], raw["ans2"]]
    ans_role_tags = [
        answer_info["ans0"][1],
        answer_info["ans1"][1],
        answer_info["ans2"][1],
    ]
    ans_roles = [
        _classify_answer_role(tag, stereotyped_groups)
        for tag in ans_role_tags
    ]

    correct_idx = raw["label"]

    # Shuffle
    answers, correct_letter, answer_roles = _shuffle_answers(
        ans_texts, ans_roles, correct_idx, rng
    )

    # Determine alignment
    correct_role = ans_roles[correct_idx]
    alignment = _determine_alignment(raw["context_condition"], correct_role)

    # Extract all identity terms from answer_info role tags (excluding 'unknown')
    identity_role_tags = [
        answer_info[f"ans{i}"][1]
        for i in range(3)
        if answer_info[f"ans{i}"][1].lower() not in UNKNOWN_TAGS
    ]

    return {
        "item_idx": item_idx,
        "example_id": raw["example_id"],
        "category": raw["category"],
        "context": raw["context"],
        "question": raw["question"],
        "answers": answers,
        "correct_letter": correct_letter,
        "context_condition": raw["context_condition"],
        "question_polarity": raw["question_polarity"],
        "alignment": alignment,
        "stereotyped_groups": stereotyped_groups,
        "answer_roles": answer_roles,
        "identity_role_tags": identity_role_tags,
        "subcategory": raw["additional_metadata"].get("subcategory", "None"),
    }


def load_and_standardize(
    category_short: str,
    data_dir: str | Path,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load a BBQ category and return standardized items.

    Args:
        category_short: canonical short name (e.g., 'so', 'race')
        data_dir: directory containing BBQ JSONL files
        seed: random seed for answer shuffling

    Returns:
        List of standardized item dicts.
    """
    jsonl_path = find_bbq_file(category_short, data_dir)
    raw_items = load_bbq_items(jsonl_path)
    log(f"Loaded {len(raw_items)} raw items from {jsonl_path.name}")

    rng = random.Random(seed)
    standardized = []
    for i, raw in enumerate(raw_items):
        standardized.append(standardize_item(raw, i, rng))

    # Log role distribution
    role_counts: dict[str, int] = {"stereotyped_target": 0, "non_stereotyped": 0, "unknown": 0}
    for item in standardized:
        for role in item["answer_roles"].values():
            role_counts[role] = role_counts.get(role, 0) + 1
    log(f"  Answer role distribution: {role_counts}")

    # Log condition distribution
    conditions = {}
    for item in standardized:
        key = (item["context_condition"], item["question_polarity"])
        conditions[key] = conditions.get(key, 0) + 1
    log(f"  Condition distribution: {dict(conditions)}")

    return standardized
