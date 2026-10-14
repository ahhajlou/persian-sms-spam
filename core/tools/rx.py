"""Rewrite Python `re` patterns for the Rust core's regex engine (fancy-regex).

The shorthand classes \\d \\w \\s \\b \\B mean different things in Python, Java, ICU
and Rust, so the rules files contain explicit classes only. `to_rust` expands them to
exactly what Python's str patterns match:

    \\w  ->  [\\p{L}\\p{N}_]   (checked against Python 3.13 for every code point)
    \\s  ->  the characters str.isspace() accepts (built from the running Python)
    \\b  ->  a lookaround pair over that \\w class
    $    ->  (?=\\n?\\z)       (Python's $ also matches before a final newline)
    \\d  ->  `digits` (the caller's choice: "0-9" or "\\p{Nd}")
    \\uXXXX -> \\x{XXXX}
"""

WORD = r"\p{L}\p{N}_"


def _space_body():
    chars = [i for i in range(0x30000) if chr(i).isspace()]
    ranges, start, prev = [], chars[0], chars[0]
    for c in chars[1:]:
        if c != prev + 1:
            ranges.append((start, prev))
            start = c
        prev = c
    ranges.append((start, prev))
    return "".join(
        rf"\x{{{a:x}}}" if a == b else rf"\x{{{a:x}}}-\x{{{b:x}}}" for a, b in ranges
    )


SPACE = _space_body()


def to_rust(pattern, digits="0-9", ignore_case=False):
    out = []
    i, n = 0, len(pattern)
    in_class = False
    while i < n:
        c = pattern[i]
        if c == "\\":
            e = pattern[i + 1]
            i += 2
            if e == "d":
                out.append(digits if in_class else f"[{digits}]")
            elif e == "s":
                out.append(SPACE if in_class else f"[{SPACE}]")
            elif e == "w":
                out.append(WORD if in_class else f"[{WORD}]")
            elif e in "bB":
                assert not in_class, f"\\{e} inside a class: {pattern!r}"
                w = f"[{WORD}]"
                if e == "b":
                    out.append(f"(?:(?<={w})(?!{w})|(?<!{w})(?={w}))")
                else:
                    out.append(f"(?:(?<={w})(?={w})|(?<!{w})(?!{w}))")
            elif e in "DSW":
                raise ValueError(f"\\{e} is not supported: {pattern!r}")
            elif e == "u":
                out.append(rf"\x{{{pattern[i:i + 4]}}}")
                i += 4
            elif e == "x":
                out.append(rf"\x{{{pattern[i:i + 2]}}}")
                i += 2
            else:
                out.append("\\" + e)
        elif in_class:
            if c == "]":
                in_class = False
                out.append(c)
            elif c in "[&~":
                out.append("\\" + c)  # nested class / set operators in Rust
            else:
                out.append(c)
            i += 1
        else:
            if c == "[":
                in_class = True
                out.append(c)
                i += 1
                if pattern[i] == "^":
                    out.append("^")
                    i += 1
                assert pattern[i] != "]", f"leading ] in a class: {pattern!r}"
            elif c == "$":
                out.append(r"(?=\n?\z)")
                i += 1
            else:
                out.append(c)
                i += 1
    assert not in_class, pattern
    text = "".join(out)
    assert "--" not in text, f"'--' is a set operator in Rust classes: {pattern!r}"
    return ("(?i)" if ignore_case else "") + text
