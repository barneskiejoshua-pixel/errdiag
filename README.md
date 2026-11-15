# errdiag

One file, standard library only. Paste in the output of a compiler, interpreter, test runner or
crashed program; get back structured diagnoses: file, line, column, language, error code, a
normalised kind (`undefined-name`, `division-by-zero`, ...), and a one-line hint for the fix.

27 languages and tools are recognised (C, C++, C#, Go, Java, Python, Rust, cmd, HTTP errors and
more). For a traceback it reports the innermost frame in your code, not the top of the stack.

    python check.py

    import diagnose
    d = diagnose.primary(diagnose.parse(output_text))
    print(d.file, d.line, d.kind, d.hint)

Written so a 4B local model could be handed a diagnosis instead of 80 lines of stderr.
