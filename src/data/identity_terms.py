"""BPE-aware identity term detection in tokenized prompts."""

from typing import Any


def find_identity_positions(
    prompt: str,
    tokenizer: Any,
    identity_terms: list[str],
) -> dict[str, list[int]]:
    """Find token positions of identity terms in a tokenized prompt.

    Uses BPE-aware matching: for each identity term, finds all token positions
    whose decoded text contributes to that term in context.

    Args:
        prompt: the full text prompt
        tokenizer: HuggingFace tokenizer instance
        identity_terms: list of identity terms to locate (e.g., ["gay", "straight"])

    Returns:
        Dict mapping term -> sorted list of token indices where that term appears.
        Terms not found in the prompt are omitted.
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    n_tokens = len(input_ids)
    positions: dict[str, list[int]] = {}

    # Sort terms longest-first to prioritize longer matches
    sorted_terms = sorted(identity_terms, key=len, reverse=True)

    # Pre-decode each token for efficiency
    decoded_tokens = []
    for tid in input_ids:
        decoded_tokens.append(tokenizer.decode([tid]).lower())

    prompt_lower = prompt.lower()

    for term in sorted_terms:
        term_lower = term.lower()

        # First check if the term even appears in the prompt text
        if term_lower not in prompt_lower:
            continue

        term_positions: list[int] = []

        # Find all occurrences of the term in the prompt
        term_starts: list[int] = []
        search_start = 0
        while True:
            idx = prompt_lower.find(term_lower, search_start)
            if idx == -1:
                break
            term_starts.append(idx)
            search_start = idx + 1

        if not term_starts:
            continue

        # Map character positions to token positions
        # Build a char->token index mapping
        char_to_token = _build_char_to_token_map(prompt, tokenizer, input_ids)

        for char_start in term_starts:
            char_end = char_start + len(term_lower)
            # Collect all token indices that overlap with this character span
            tokens_in_span = set()
            for c in range(char_start, char_end):
                if c in char_to_token:
                    tokens_in_span.add(char_to_token[c])
            term_positions.extend(tokens_in_span)

        if term_positions:
            positions[term] = sorted(set(term_positions))

    return positions


def _build_char_to_token_map(
    prompt: str,
    tokenizer: Any,
    input_ids: list[int],
) -> dict[int, int]:
    """Build a mapping from character position in the prompt to token index.

    This handles BPE tokenization by reconstructing character offsets from
    the tokenizer's encoding.
    """
    # Try using the fast tokenizer's offset mapping if available
    try:
        encoding = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
        offsets = encoding.get("offset_mapping")
        if offsets is not None:
            char_to_token: dict[int, int] = {}
            for token_idx, (start, end) in enumerate(offsets):
                if start == end:
                    continue  # special tokens
                for c in range(start, end):
                    char_to_token[c] = token_idx
            return char_to_token
    except Exception:
        pass

    # Fallback: reconstruct by decoding token-by-token and matching
    char_to_token = {}
    current_char = 0
    prompt_lower = prompt.lower()

    for token_idx, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid])
        if not decoded:
            continue

        # Skip special tokens that don't appear in the prompt text
        # Find where this decoded text appears starting from current_char
        search_text = decoded
        # Handle leading space that some tokenizers add
        pos = prompt.find(search_text, max(0, current_char - 1))
        if pos == -1:
            # Try without leading/trailing whitespace
            stripped = decoded.strip()
            if stripped:
                pos = prompt.find(stripped, max(0, current_char - 1))
                if pos != -1:
                    search_text = stripped

        if pos != -1 and pos < current_char + len(decoded) + 5:
            for c in range(pos, pos + len(search_text)):
                char_to_token[c] = token_idx
            current_char = pos + len(search_text)

    return char_to_token


def extract_identity_terms_from_item(item: dict) -> list[str]:
    """Extract all identity terms from a standardized BBQ item.

    Pulls terms from identity_role_tags (the non-unknown answer_info tags).
    These are the actual identity descriptors used in the item's answers.
    """
    terms = []
    for tag in item.get("identity_role_tags", []):
        if tag.lower() not in ("unknown",):
            terms.append(tag)
    return terms
