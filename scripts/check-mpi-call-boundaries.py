#!/usr/bin/env python3
"""Reject Hataori-owned MPI calls outside the canonical mpi_call!(...) boundary."""

from __future__ import annotations

import pathlib
import re
import sys

MACRO = "mpi_call!"
FORBIDDEN = re.compile(
    r"\b(?:comm|world)\s*\.\s*(?:rank|size|duplicate|abort|all_reduce_into|barrier)\s*\("
    r"|\.\s*(?:send_with_tag|receive_vec_with_tag|receive_into_with_tag|probe_with_tag"
    r"|broadcast_into|immediate_probe_with_tag|matched_probe_with_tag)\s*\("
    r"|\bstatus\s*\.\s*count\s*\("
    r"|\bthreading_support\s*\("
    r"|\bffi\s*::\s*MPI_"
)


def matching_paren(source: str, opening: int) -> int:
    depth = 0
    index = opening
    state = "code"
    block_depth = 0
    while index < len(source):
        char = source[index]
        nxt = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char == '"':
                state = "string"
            elif char == "/" and nxt == "/":
                state = "line"
                index += 1
            elif char == "/" and nxt == "*":
                state = "block"
                block_depth = 1
                index += 1
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index
        elif state == "string":
            if char == "\\":
                index += 1
            elif char == '"':
                state = "code"
        elif state == "line":
            if char == "\n":
                state = "code"
        elif state == "block":
            if char == "/" and nxt == "*":
                block_depth += 1
                index += 1
            elif char == "*" and nxt == "/":
                block_depth -= 1
                index += 1
                if block_depth == 0:
                    state = "code"
        index += 1
    raise ValueError("unterminated mpi_call! invocation")


def remove_wrapped_calls(source: str) -> str:
    output = list(source)
    position = 0
    while True:
        start = source.find(MACRO, position)
        if start < 0:
            break
        opening = start + len(MACRO)
        while opening < len(source) and source[opening].isspace():
            opening += 1
        if opening >= len(source) or source[opening] != "(":
            raise ValueError("mpi_call! must use canonical parenthesized form")
        closing = matching_paren(source, opening)
        output[start : closing + 1] = " " * (closing + 1 - start)
        position = closing + 1
    return "".join(output)


def strip_comments_and_strings(source: str) -> str:
    # Preserve line structure for useful diagnostics; Rust call tokens cannot
    # span a string/comment boundary.
    source = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), source, flags=re.S)
    source = re.sub(r"//[^\n]*", "", source)
    source = re.sub(r'"(?:\\.|[^"\\])*"', '""', source)
    return source


def violations(source: str) -> list[tuple[int, str]]:
    stripped = strip_comments_and_strings(remove_wrapped_calls(source))
    result = []
    for match in FORBIDDEN.finditer(stripped):
        line = stripped.count("\n", 0, match.start()) + 1
        result.append((line, match.group(0)))
    return result


def self_test() -> None:
    assert not violations("fn f(){ mpi_call!(comm.rank()); }")
    assert not violations('fn f(){ mpi_call!(comm.send_with_tag(b"(", 1)); }')
    assert violations("fn f(){ comm.rank(); }")
    assert violations("fn f(){ comm.process_at_rank(1).send_with_tag(&[1], 2); }")
    assert violations("fn f(){ comm.process_at_rank(1).receive_into_with_tag(&mut v, 3); }")
    assert violations("fn f(){ comm.process_at_rank(1).probe_with_tag(2); }")
    assert violations("fn f(){ status.count(T::equivalent_datatype()); }")
    assert not violations("fn f(){ mpi_call!(status.count(T::equivalent_datatype())); }")
    assert not violations("fn f(){ let n = values.iter().count(); }")
    assert not violations("// comm.rank()\nfn f(){}")
    try:
        remove_wrapped_calls("mpi_call!{comm.rank()}")
    except ValueError:
        pass
    else:
        raise AssertionError("noncanonical macro delimiter was accepted")


def main() -> int:
    self_test()
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        return 0
    root = pathlib.Path(__file__).resolve().parents[1]
    found = []
    for path in sorted((root / "src").rglob("*.rs")):
        if path.name == "mpi_check.rs":
            continue
        for line, token in violations(path.read_text(encoding="utf-8")):
            found.append(f"{path.relative_to(root)}:{line}: unwrapped MPI call: {token}")
    if found:
        print("\n".join(found), file=sys.stderr)
        return 1
    print("all Hataori-owned MPI call sites use mpi_call!(...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
