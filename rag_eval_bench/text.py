"""Turn text into a list of lowercase word tokens."""

import re

STOPWORDS = frozenset(
    "a an and are as at be by do does for from how i in is it of on or "
    "that the to what when which why with you your".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase, split on anything that is not a letter or digit, drop stopwords."""
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]
