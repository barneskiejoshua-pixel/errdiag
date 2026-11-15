"""Standalone check: prints got versus expected for each case and exits 1 on any mismatch."""
import sys

BAD = 0


def same(name, got, expected):
    global BAD
    ok = got == expected
    BAD += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r} expected {expected!r}")

import diagnose as D

py = ('Traceback (most recent call last):\n  File "app.py", line 12, in <module>\n    main()\n'
      '  File "app.py", line 7, in main\n    print(total / count)\nZeroDivisionError: division by zero\n')
p = D.primary(D.parse(py))
same("python file", p.file, "app.py")
same("python line is the innermost frame", p.line, 7)
same("python kind", p.kind, "division-by-zero")
c = D.primary(D.parse("main.cpp:14:5: error: use of undeclared identifier 'cout'\n"))
same("clang location", (c.file, c.line, c.col), ("main.cpp", 14, 5))
same("clang kind", c.kind, "undefined-name")
same("clean output has no diagnosis", D.parse("3 passed in 0.02s\n"), [])

sys.exit(1 if BAD else 0)
