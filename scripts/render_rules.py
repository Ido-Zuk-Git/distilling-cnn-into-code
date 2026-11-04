#!/usr/bin/env python3
"""Render a rules template: keep <!-- IF LOOK --> ... <!-- ENDIF --> blocks only with --look 1, <!-- IF NOLOOK --> blocks only with --look 0.
Blocks may be whole lines (markers alone on a line, which are removed) or inline (markers inside a line).
Usage: render_rules.py IN OUT --look 0|1 [--set KEY=VALUE ...]  (replaces {{KEY}})"""
import re, sys
src, dst = sys.argv[1], sys.argv[2]
look = sys.argv[sys.argv.index("--look") + 1] == "1"
s = open(src).read()
def strip(s, tag, keep):
    pat = re.compile(r"<!-- IF %s -->\n?(.*?)<!-- ENDIF -->\n?" % tag, re.S)
    return pat.sub((lambda m: m.group(1)) if keep else "", s)
s = strip(s, "LOOK", look)
s = strip(s, "NOLOOK", not look)
assert "<!-- IF" not in s and "ENDIF" not in s, "unbalanced markers"
for i, a in enumerate(sys.argv):
    if a == "--set":
        k, v = sys.argv[i + 1].split("=", 1)
        s = s.replace("{{%s}}" % k, v)
open(dst, "w").write(s)
