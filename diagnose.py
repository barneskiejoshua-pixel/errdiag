"""Turn compiler, interpreter, test-runner and runtime error output into structured diagnoses.

DRock's debugging knowledge at the parsing layer: a 4B model is bad at reading a 300-line build log
or a stack trace that runs through twenty library frames. parse() finds every diagnosis, places it
precisely (file, line, column; the user's own frame, not the library frame), names the kind of
mistake and gives a short, language-specific fix hint. render() turns that into a few lines the model
can act on; primary() is the one to fix first.

    parse(text, cwd=None) -> list[Diag]
    primary(diags) -> Diag | None
    render(diags, limit=5) -> str
    classify(message, lang, code=None) -> (kind, hint)
    languages() -> list[str]

Stdlib only. parse() never raises.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from dataclasses import dataclass, field

__all__ = ["Diag", "parse", "primary", "render", "classify", "languages", "KINDS"]

MAX_DIAGS = 30
MAX_LINE = 4000

KINDS = (
    "syntax", "indentation", "undefined-name", "undefined-attribute", "import", "type-mismatch",
    "argument-count", "null-reference", "index-out-of-range", "key-missing", "division-by-zero",
    "overflow", "test-failure", "file-not-found", "permission", "network", "timeout", "stack-overflow",
    "out-of-memory", "memory-leak", "concurrency", "borrow", "lifetime", "linker", "unused", "deprecated",
    "encoding", "dependency-resolution", "cors", "http-status", "access-violation", "invalid-value",
    "unhandled-rejection", "disk-full", "port-in-use", "version-mismatch", "merge-conflict", "style", "unknown",
)

_LANGS = (
    "c", "cmd", "cpp", "csharp", "css", "dart", "elixir", "go", "haskell", "html", "http", "java",
    "javascript", "kotlin", "lua", "native", "perl", "php", "powershell", "python", "ruby", "rust",
    "scala", "shell", "sql", "swift", "typescript",
)


@dataclass
class Diag:
    file: str | None
    line: int | None
    col: int | None
    severity: str            # "error" | "warning" | "note"
    lang: str
    code: str | None
    message: str
    kind: str
    hint: str
    frame_kind: str          # "compile" | "runtime" | "test" | "lint"
    context: list[str] = field(default_factory=list)
    suggestion: str | None = None
    pos: int = field(default=0, repr=False, compare=False)
    rank: int = field(default=0, repr=False, compare=False)


# --------------------------------------------------------------------------------------------- hints

KIND_HINTS = {
    "syntax": "the code does not parse; READ_FILE {where} and the line before it (a missing bracket, quote, colon, comma or semicolon is the usual cause) and fix just that token",
    "indentation": "inconsistent indentation; READ_FILE {where} and indent the block exactly like its sibling lines (spaces only, no tabs)",
    "undefined-name": "a misspelt or undefined name; READ_FILE {where} and use the name defined earlier",
    "undefined-attribute": "the object has no such attribute/method; check the object's real type and the spelling, and use a member that exists",
    "import": "the module/package cannot be found; check the name's spelling, install the package, or fix the import path",
    "type-mismatch": "a value has the wrong type here; convert it explicitly or change the declared type so both sides agree",
    "argument-count": "wrong number of arguments; READ_FILE the function's definition and pass exactly the parameters it declares",
    "null-reference": "a value is null/None/undefined where an object was expected; find where it should be assigned (a missing return? a failed lookup?) and check it before use",
    "index-out-of-range": "an index is past the end of the sequence; check the length first and use `< len` (not `<= len`) in loop bounds",
    "key-missing": "the key is not in the map; check the spelling and the keys that really exist, or use a lookup with a default",
    "division-by-zero": "a divisor is zero; check it before dividing and handle the zero case",
    "overflow": "a number exceeded its type's range; use a wider type (long/int64/i64/BigInteger) or check the arithmetic before it wraps",
    "test-failure": "the test states the expected value; fix the code under test, not the assertion",
    "file-not-found": "the path does not exist relative to the working directory; check the spelling and the cwd, or create the file first",
    "permission": "access was denied; check the file/folder permissions, close programs holding the file open, or use a user-writable location",
    "network": "the connection failed; check the host and port are right and that the server is running and reachable",
    "timeout": "the operation took too long; make sure the other side responds, then raise the timeout or do less work per call",
    "stack-overflow": "unbounded recursion; make sure the recursive function reaches its base case on every path (or rewrite it as a loop)",
    "out-of-memory": "the program ran out of memory; look for a collection that grows without bound, or process the data in chunks",
    "memory-leak": "allocated memory is never freed; pair every new/malloc with delete/free, or use smart pointers / RAII containers",
    "concurrency": "threads are blocked or racing; make sure every lock is released, every channel receive has a matching send, and shared data is guarded",
    "borrow": "Rust ownership rules are broken; borrow with & instead of moving, .clone() the value, or end the other borrow before this use",
    "lifetime": "a reference may outlive its data; add a lifetime parameter tying output to input (fn f<'a>(x: &'a T) -> &'a U) or return an owned value",
    "linker": "the function is declared but its definition/source file or library isn't linked; add the .cpp to the command or the -l flag",
    "unused": "unused code; remove it, or use it (prefix the name with _ if it is intentionally unused)",
    "deprecated": "a deprecated API; switch to the replacement the message names",
    "encoding": "text was decoded with the wrong encoding; open the file with encoding='utf-8' explicitly (or errors='replace' for bad bytes)",
    "dependency-resolution": "package versions conflict or the package does not exist in the registry; check the name and version, then align or relax the version constraints",
    "cors": "the browser blocked a cross-origin request; the SERVER must send Access-Control-Allow-Origin for this origin (or serve page and API from one origin / via a proxy)",
    "http-status": "the server returned an HTTP error status; check the URL, method and credentials",
    "access-violation": "the program touched memory it doesn't own (0xC0000005 / SIGSEGV): a null or dangling pointer, or an out-of-bounds write; check pointers and array indexes just before the crash",
    "invalid-value": "a value could not be parsed/converted or broke a constraint; validate or strip the input before converting it, and print the offending value",
    "unhandled-rejection": "a promise rejected with no handler; await it inside try/catch or add .catch() to the promise chain",
    # CAPABILITY-STACK 9.6 / 9.18 / 9.16 / 9.1 (2026-10-03): the four situations the spec names that had no kind, so a
    # retry looked like the next step. Each hint names the DRock action that answers the real question.
    "disk-full": "the disk is full, so writing more fails the same way; CHECK_DISK for the free space and LARGEST_FILES for what to remove, then retry",
    "port-in-use": "another process already holds the port; LIST_PORTS port=<N> names it (STOP_SERVICE if it is one DRock started), or start on another port",
    "version-mismatch": "the installed version does not satisfy the requirement (both numbers are in the message); install the required version, or pin the package to one that supports the installed one",
    "merge-conflict": "the file holds unresolved merge conflict markers (<<<<<<<, =======, >>>>>>>); keep one side or combine them, delete the marker lines, then run it again",
    "style": "formatting/style only (whitespace, line length, naming, docstrings); it does not change behaviour",
    "unknown": "READ_FILE {where} and read the full message; fix the first error, later ones are often caused by it",
}

_SHELL_CMD = "the command is not installed or not on PATH; check the spelling, install it, or call it by its full path"
_JS_IMPORT = "npm install the package, or fix the relative path (./file) in the import/require"
_JS_NULL = "the object is undefined/null at this access; check it was assigned (awaited the promise? right property name?) or guard with obj?.prop"

LANG_HINTS = {
    ("undefined-name", "shell"): _SHELL_CMD,
    ("undefined-name", "powershell"): _SHELL_CMD,
    ("undefined-name", "cmd"): _SHELL_CMD,
    ("undefined-name", "sql"): "the table/column does not exist; check its spelling and the schema (.tables / \\d / SHOW TABLES) or create it first",
    ("undefined-name", "cpp"): "declare it before use, fix the spelling, #include the header that declares it, or add the std:: prefix",
    ("undefined-name", "c"): "declare it before use, fix the spelling, or #include the header that declares it",
    ("undefined-name", "java"): "the symbol is misspelt, not declared in this scope, or missing an import; declare it or add the import",
    ("syntax", "sql"): "the SQL is malformed near the quoted token; check the keyword spelling and order just before it (FROM, commas, quotes)",
    ("syntax", "python"): "the code does not parse; READ_FILE {where} and the line before (unclosed bracket or quote, missing ':' after if/for/def) and fix that token",
    ("indentation", "python"): "inconsistent indentation; READ_FILE {where} and indent with 4 spaces to match the surrounding block (no tabs)",
    ("import", "python"): "the module is not installed in this interpreter or the name is wrong; pip install it into the same python/venv that runs the code, or fix the import",
    ("import", "javascript"): _JS_IMPORT,
    ("import", "typescript"): _JS_IMPORT,
    ("import", "cpp"): "the header/library cannot be found: fix the #include name, or add -I<dir> for headers / -l<lib> for libraries",
    ("import", "c"): "the header/library cannot be found: fix the #include name, or add -I<dir> for headers / -l<lib> for libraries",
    ("import", "java"): "the class/package is not on the classpath: add the import statement or the dependency (pom.xml / build.gradle)",
    ("import", "rust"): "add the `use` path, or add the crate to Cargo.toml [dependencies]",
    ("import", "go"): "run `go get <module>` / `go mod tidy`, or fix the import path",
    ("null-reference", "python"): "a variable is None here (a function without `return`, or a failed lookup); find where it is assigned and handle the None case",
    ("null-reference", "javascript"): _JS_NULL,
    ("null-reference", "typescript"): _JS_NULL,
    ("null-reference", "java"): "the reference in the message was never initialized (new ...) or a method returned null; initialize it or check `!= null` before use",
    ("null-reference", "kotlin"): "a null reached a non-null use (!! or platform type); use ?. / ?: or initialize it",
    ("null-reference", "csharp"): "the object was never assigned (null); initialize it or guard with `?.` / `if (x != null)`",
    ("null-reference", "go"): "a nil map/pointer: create the map with make(map[K]V) (or the pointer with &T{}) before using it",
    ("null-reference", "rust"): "unwrap() on None: match on the Option, or use if let / ? / unwrap_or(default)",
    ("null-reference", "lua"): "the variable is nil: misspelt name, missing require, or not assigned yet",
    ("null-reference", "ruby"): "the receiver is nil: check where it is assigned (a failed lookup returns nil) or use &.",
    ("null-reference", "php"): "the variable is null: check where it is assigned before calling methods on it (or use ?->)",
    ("null-reference", "swift"): "a force-unwrapped Optional was nil: use if let / guard let / ?? instead of !",
    ("null-reference", "perl"): "the variable is undef here: assign it before use or check defined($x)",
    ("null-reference", "cpp"): "a null pointer was dereferenced; check the pointer is set (new / address-of) before using it",
    ("index-out-of-range", "python"): "the sequence is shorter than the index; check len() first and loop with range(len(x)), not range(len(x)+1)",
    ("index-out-of-range", "cpp"): "an index/pointer went past the end of the array; check i < size() (not <=) or use .at(i) for a checked access",
    ("index-out-of-range", "c"): "an index/pointer went past the end of the array; check i < length (not <=) before the access",
    ("index-out-of-range", "rust"): "check i < v.len() first, or use v.get(i) which returns an Option",
    ("index-out-of-range", "go"): "check i < len(s) before indexing the slice",
    ("index-out-of-range", "java"): "the index is >= length; loop with i < arr.length / size() (not <=) and check before indexing",
    ("index-out-of-range", "csharp"): "the index is >= Length/Count; loop with i < arr.Length (not <=) and check before indexing",
    ("key-missing", "python"): "the key is not in the dict; use d.get(key, default) or check `key in d`, and print d.keys() to see the real keys",
    ("unused", "go"): "Go refuses to compile unused variables/imports: delete it, or use it (`_ = x` to keep it deliberately)",
    ("type-mismatch", "python"): "the operand/argument has the wrong type; convert explicitly (int(x), str(x), float(x)) or fix what produces the value",
    ("type-mismatch", "typescript"): "the types don't match; convert the value (Number(x), String(x)) or fix the declared type",
    ("undefined-attribute", "javascript"): "the method/property doesn't exist on this value (misspelt, or the value isn't the type you think); log the value and use a member it has",
    ("undefined-attribute", "python"): "the object has no such attribute; check its real type (print(type(x))) and the spelling, and use an attribute it has",
}

_RUST_HINTS = {
    "E0382": "the value was moved (into a function or another variable) and then used; pass a reference &v, or .clone() it before the move",
    "E0502": "a mutable and an immutable borrow overlap; finish using the immutable borrow (copy the value out) before mutating, or split into separate scopes",
    "E0499": "two mutable borrows at once; do the mutations one after another, or split the data (split_at_mut / indices)",
    "E0505": "the value is moved while still borrowed; end the borrow first or clone the value",
    "E0384": "the variable is immutable; declare it with `let mut`",
    "E0596": "cannot borrow as mutable; declare the binding `let mut` (or take &mut)",
    "E0106": "missing lifetime specifier: add a lifetime parameter, e.g. fn f<'a>(x: &'a str) -> &'a str, or return an owned String",
    "E0597": "the borrowed value is dropped while still borrowed; declare it in an outer scope so it lives long enough",
    "E0308": "mismatched types: convert with .into() / as / .to_string() or change the annotation; check a missing or extra `;` on the last expression",
    "E0277": "the type doesn't implement the required trait: convert it, add #[derive(...)] or impl the trait, or use a type that does",
    "E0599": "no such method on this type: check the spelling, `use` the trait that provides it, or call it on the inner value (after unwrap / .iter())",
    "E0425": "unresolved name: misspelt or out of scope; declare it or bring it into scope",
    "E0433": "failed to resolve a path: add the `use` statement, or add the crate to Cargo.toml [dependencies]",
    "E0432": "unresolved import: fix the `use` path or add the crate to Cargo.toml",
    "E0061": "wrong number of arguments: match the function's parameter list",
}

_LNK_HINT = "the function is declared but its definition isn't compiled or linked; add the .cpp that defines it to the build (cl a.cpp b.cpp) or link its .lib"

CODE_HINTS = dict(_RUST_HINTS)
CODE_HINTS.update({
    "LNK2019": _LNK_HINT, "LNK2001": _LNK_HINT, "LNK1120": _LNK_HINT,
    "C2065": "undeclared identifier: declare it before use, fix the spelling, or #include the header that declares it (std:: for standard names)",
    "C1083": "the #include file cannot be opened: fix its name or add its folder with /I",
    "CS0103": "the name does not exist here: fix the spelling, declare it in this scope, or add the missing using directive",
    "CS0246": "type or namespace not found: add the `using` directive or the NuGet package / project reference",
    "CS1061": "the type has no such member: check the spelling and the object's real type",
    "TS2307": "cannot find module: npm install it (and @types/<name>), or fix the relative import path",
    "TS2345": "the argument's type doesn't match the parameter; convert the value (Number(x), String(x)) or fix the declared type",
    "TS2322": "the value's type doesn't match the declared type; convert the value or fix the annotation",
    "TS2532": "possibly undefined: narrow it first (if (x) ..., x?.y, x ?? default)",
    "TS18048": "possibly undefined: narrow it first (if (x) ..., x?.y, x ?? default)",
    "TS2531": "possibly null: narrow it first (if (x) ..., x?.y, x ?? default)",
    "TS18047": "possibly null: narrow it first (if (x) ..., x?.y, x ?? default)",
    "ERESOLVE": "peer dependency conflict: align the conflicting package versions, or retry with --legacy-peer-deps",
    "E404": "the package name/version doesn't exist on the registry: fix the spelling/version in package.json",
    "ENOENT": "a file npm needs is missing (usually package.json): run npm in the project folder, or npm init",
    "ELIFECYCLE": "an npm script failed: the real error is printed above this line; fix that",
    "MODULE_NOT_FOUND": _JS_IMPORT,
    "ERR_MODULE_NOT_FOUND": _JS_IMPORT,
})

HTTP_MEANING = {
    400: "400 Bad Request: the request body/parameters are malformed; check the JSON and required fields",
    401: "401 Unauthorized: credentials are missing or invalid; send a valid token/API key (Authorization header)",
    403: "403 Forbidden: authenticated but not allowed; check the key's permissions/scope or the resource's access rules",
    404: "404 Not Found: the URL/path is wrong or the resource doesn't exist; check the route, spelling and base URL",
    405: "405 Method Not Allowed: use the HTTP method the endpoint supports (GET vs POST)",
    408: "408 Request Timeout: the request was too slow; retry, or send less",
    409: "409 Conflict: the resource's state conflicts (already exists / version mismatch)",
    413: "413 Payload Too Large: send a smaller body or raise the server limit",
    415: "415 Unsupported Media Type: set the right Content-Type (e.g. application/json)",
    422: "422 Unprocessable: the body is well-formed but fails validation; read the error fields",
    429: "429 Too Many Requests: rate limited; wait (Retry-After) and back off",
    500: "500 Internal Server Error: the server's own code crashed; read the server log, not the client",
    501: "501 Not Implemented: the server doesn't support this method/feature",
    502: "502 Bad Gateway: the proxy couldn't reach the upstream app; the backend is down or crashed",
    503: "503 Service Unavailable: the server is overloaded or in maintenance; retry later",
    504: "504 Gateway Timeout: the upstream app was too slow to answer; check the backend",
}

# ------------------------------------------------------------------------------------------ rules

_Q = "['‘’\"`]"
_ESM_HINT = ("ES module vs CommonJS mixup: either use `import` with \"type\": \"module\" in package.json (or a .mjs file), "
             "or keep `require` and remove \"type\": \"module\" / rename to .cjs")


def _r(p):
    return re.compile(p, re.I)


# checked before any error code
_PRIORITY = [
    ("import", _r(r"cannot use import statement outside a module|require is not defined in es module scope|err_require_esm|require\(\) of es module|(?:exports|module|__dirname|__filename) is not defined in es module scope"), None, _ESM_HINT),
    ("invalid-value", _r(r"in json at position|is not valid json|unexpected end of json|unexpected token .{1,3} in json"), None,
     "the text is not JSON (often an HTML error page); log the raw response/file content before parsing it"),
    ("null-reference", _r(r"'nonetype' object|\bnonetype\b|item \"none\" of|cannot (?:read|set) propert(?:y|ies) of (?:undefined|null)|(?:undefined|null) is not an object|undefined method .+ for nil|nil:nilclass|attempt to (?:index|call|perform arithmetic on|concatenate|compare) (?:a )?nil|call to a member function .+ on null|nil pointer dereference|invalid memory address|unexpectedly found nil|on a `none` value|null ?pointer|nullreferenceexception|object reference not set|entry in nil map|nullable object must have a value|argumentnullexception|use of uninitialized value|because .+ is null|null not valid|construction from null"), None, None),
    ("concurrency", _r(r"concurrentmodification|collection was modified|changed size during iteration|modified during iteration"), None,
     "the collection was modified while being iterated; iterate over a copy, or collect the changes and apply them after the loop"),
    ("argument-count", _r(r"takes \d+ positional arguments? but \d+ (?:were|was) given|missing \d+ required (?:positional |keyword-only )?arguments?|unexpected keyword argument|wrong number of arguments|expected \d+ arguments?,? but got \d+|too (?:many|few) arguments|cannot be applied to given types|argumentcounterror|no overload for method .+ takes|does not take \d+ arguments|takes \d+ arguments? but \d+ arguments? (?:were|was) supplied|function takes \d+ arguments?"), None, None),
    ("index-out-of-range", _r(r"heap-buffer-overflow|stack-buffer-overflow|global-buffer-overflow|out of bounds|out-of-bounds|index out of range|index out of bounds|indexerror|arrayindexoutofbounds|stringindexoutofbounds|indexoutofbounds|indexoutofrange|argumentoutofrange|_m_range_check|out_of_range|subscript out of range|vector subscript|sequence contains no elements|nosuchelement"), None, None),
    ("stack-overflow", _r(r"maximum recursion depth|maximum call stack size|stack[ -]?overflow|stack level too deep|systemstackerror|goroutine stack exceeds|overflowed its stack|0xc00000fd"), None, None),
    ("concurrency", _r(r"all goroutines are asleep|deadlock|concurrent map (?:writes|read and map write|iteration)|data race"), None, None),
    ("linker", _r(r"multiple definition of|duplicate symbol|lnk2005|already defined in"), None,
     "the same function/variable is defined in two object files (often a body or global in a header); mark it inline/static, or keep one definition in a .cpp and only a declaration in the header"),
    ("linker", _r(r"undefined reference|unresolved external|undefined symbol|lnk20\d\d|lnk1120|ld returned|linker command failed|cannot find -l|library not found for|symbol\(s\) not found|linking with .+ failed"), None, None),
    ("import", _r(r"\.(?:h|hpp|hh|hxx)" + _Q + r"?: no such file|" + _Q + r"[^'\"]+\.(?:h|hpp)" + _Q + r" file not found|cannot open (?:source|include) file"), None,
     "the header is not on the include path: fix the #include name or add -I<dir> (/I on MSVC)"),
    ("memory-leak", _r(r"memory leak|leaksanitizer|detected memory leaks"), None, None),
    ("access-violation", _r(r"use-after-free|double[ -]free|free\(\): invalid|corrupted (?:size|double-linked|top)|malloc\(\): |munmap_chunk|segmentation fault|sigsegv|access violation|0xc0000005|stack smashing|bus error|heap corruption"), None, None),
    ("out-of-memory", _r(r"bad_alloc|out of memory|outofmemory|memoryerror|heap out of memory|cannot allocate memory|java heap space"), None, None),
    ("division-by-zero", _r(r"division by zero|divide by zero|divided by zero|dividebyzero|zerodivision|/ by zero|integer divide|floating point exception|sigfpe|0xc0000094"), None, None),
    ("overflow", _r(r"integer overflow|arithmetic overflow|overflowerror|overflowexception|with overflow|signed integer overflow|shift exponent|out of range for type|cannot be represented in type|number too large"), None, None),
    ("undefined-name", _r(r"not found"), ("scala",), None),
    ("undefined-name", _r(r"unbound variable"), None, "the variable is unset (set -u); assign it first or use ${VAR:-default}"),
    ("undefined-name", _r(r"before initialization|referenced before assignment"), None,
     "the variable is used before its declaration/assignment runs; move the use after the assignment (or assign it on every path)"),
    ("unhandled-rejection", _r(r"unhandledpromiserejection|unhandled promise rejection|err_unhandled_rejection"), None, None),
    ("invalid-value", _r(r"constraint failed|duplicate entry|duplicate key|violates .+ constraint|unique constraint|foreign key constraint"), None,
     "the row violates a UNIQUE/FOREIGN KEY/NOT NULL constraint; check for an existing row, or insert the parent row first"),
]

_GENERAL = [
    # The four situations the capability spec names (9.6, 9.18, 9.16, 9.1; 2026-10-03), ahead of the broader rows
    # below so "requires Python >=3.10" is a version mismatch before it is a dependency resolution.
    ("disk-full", _r(r"no space left on device|enospc|not enough space on the disk|there is not enough space|disk (?:is )?full|insufficient disk space|disk quota exceeded"), None, None),
    ("port-in-use", _r(r"eaddrinuse|address (?:is )?already in use|only one usage of each socket address|port \d+ is already in use|address in use"), None, None),
    ("version-mismatch", _r(r"requires? a different (?:python|node|version|\w+ version)|\d[\d.]* not in '?[><=~^!]|"
                            r"requires?\s+(?:python|node(?:\.js)?|java|go|rustc?|ruby|php)?\s*(?:version\s+)?[><=~^!]+\s*v?\d[\d.]*[^\n]{0,60}?(?:found|have|but|current(?:ly)?|installed|using|running|got)\D{0,24}v?\d[\d.]*|"
                            r"requires? \w+ v?\d[\d.]* or (?:newer|later|higher|above)|expected version [\"'][^\"']+[\"']\.? ?got [\"']|"
                            r"engine [\"'].{0,30}[\"'] is incompatible|incompatible (?:python|node|java|rust|ruby) version|unsupported (?:python|node|java) version|requires a newer version|"
                            r"you (?:have|are using) (?:python |node )?v?\d[\d.]*[^\n]{0,40}(?:required|requires|need)|requires? (?:python|node|java) \d[\d.]* or later; you have"), None, None),
    ("merge-conflict", _r(r"(?:^|\s)<<<<<<< |(?:^|\s)>>>>>>> |merge conflict|conflict marker|unmerged files|conflict \(content\)"), None, None),
    ("test-failure", _r(r"assert|expected:? ?<.*?> but was|expected .+ to (?:be|equal|have|contain)|\bgot\b.+\bwant\b|left == right|comparisonfailure|test failed"), None, None),
    ("indentation", _r(r"indentationerror|taberror|unexpected indent|unindent does not match|expected an indented block|inconsistent use of tabs|mixed spaces and tabs"), None, None),
    ("dependency-resolution", _r(r"eresolve|unable to resolve dependency|could not resolve dependenc|peer dep|conflicting dependencies|resolutionimpossible|no matching distribution|could not find a version that satisfies|version solving failed|failed to select a version|dependency conflict|404 not found - get https?://registry"), None, None),
    ("syntax", _r(r"syntaxerror|syntax error|parse error|parseerror|unexpected token|unexpected end of (?:input|file)|unexpected eof|eof while|expected expression|expected identifier|expected declaration|expected unqualified-id|expected primary-expression|expected " + _Q + r"[^\w'’]{1,2}" + _Q + r"|" + _Q + r"[^\w'’]{1,2}" + _Q + r" expected|invalid syntax|unterminated|unclosed|unmatched|was never closed|unexpected character|invalid character|incorrect syntax|error in your sql syntax|unexpected keyword|unexpected symbol|missing terminating|stray " + _Q + r"|unexpected end-of-input"), None, None),
    ("import", _r(r"no module named|modulenotfound|importerror|cannot import name|cannot find module|module not found|could not resolve (?!host)|can't resolve " + _Q + r"|package .+ does not exist|cannot find package|no required module provides|unresolved import|failed to resolve: use of undeclared|can't find crate|loaderror|cannot load such file|classnotfound|noclassdeffound|unable to load dll|0xc0000135|type or namespace name .+ could not be found|are you missing (?:a using|an assembly)|err_module_not_found|cannot find type definition"), None, None),
    ("key-missing", _r(r"keyerror|key not found|keynotfound|no such key|undefined index|undefined array key|missing key"), None, None),
    ("undefined-attribute", _r(r"has no attribute|has no member|no member named|is not a member of|no method named|undefined method|no field .+ on type|no such method|property .+ does not exist on type|does not contain a definition for|is not a function|isn't defined for the (?:class|type)|nomethoderror|attributeerror|cannot find method|has no field or method|undefined property"), None, None),
    ("undefined-name", _r(r"is not defined|undeclared identifier|" + _Q + r" undeclared|was not declared|cannot find symbol|cannot find name|cannot find value|cannot find " + _Q + r".+" + _Q + r" in scope|unresolved reference|not found in this scope|\bundefined: |undefined variable|undefined name|undefined function|undefined local variable|use of undeclared|nameerror|referenceerror|does not exist in the current context|unboundlocalerror|global symbol .+ requires explicit package|not in scope|not found: (?:value|type)|call to undefined function|command not found|is not recognized as|no such (?:table|column|function)|column .+ does not exist|relation .+ does not exist|table .+ doesn't exist|unknown column|unknown database|invalid object name|invalid column name|undefined identifier|identifier .+ is undefined|can't find variable|use of unassigned"), None, None),
    ("type-mismatch", _r(r"typeerror|type mismatch|mismatched types|incompatible types?|cannot be converted|is not assignable to|cannot convert|invalid conversion|no matching function for call|no operator .+ matches|unsupported operand|can only concatenate|must be (?:an? )?\w+, not|expected .+, found|couldn't match (?:expected )?type|cannot use .+ as .+ value|mismatched types|classcast|invalidcast|cannot be cast to|not iterable|not subscriptable|not callable|trait bound|argument of type|no implicit conversion|has incompatible type|invalid operands|cannot implicitly convert"), None, None),
    ("file-not-found", _r(r"no such file or directory|filenotfound|file not found|cannot find the (?:file|path) specified|enoent|could not find file|nosuchfile|directorynotfound|cannot open file|could not open file|failed to open|errno 2\b|os error 2\b|cannot find path|does not exist"), None, None),
    ("permission", _r(r"permission denied|permissionerror|access is denied|access denied|eacces|eperm\b|unauthorizedaccess|operation not permitted|read-only file system|errno 13\b|being used by another process"), None, None),
    ("cors", _r(r"cors policy|access-control-allow-origin|cross-origin request blocked"), None, None),
    ("cors", _r(r"mixed content"), None,
     "an HTTPS page requested an http:// resource and the browser blocked it; load that resource over https:// (or a relative URL)"),
    ("http-status", _r(r"status (?:code )?(?:of )?[45]\d\d|http/\d(?:\.\d)? [45]\d\d|[45]\d\d (?:client|server) error|returned error: [45]\d\d|\b(?:401 unauthorized|403 forbidden|404 not found|500 internal server error|502 bad gateway|503 service unavailable|504 gateway timeout)"), None, None),
    ("timeout", _r(r"timed? ?out|timeout|deadline exceeded|err_timed_out"), None, None),
    ("network", _r(r"connection[ _]refused|econnrefused|connection[ _]reset|econnreset|getaddrinfo|enotfound|name or service not known|nodename nor servname|could not resolve host|no route to host|network is unreachable|connecterror|connectionerror|unknownhost|failed to establish a new connection|\bssl|certificate|socket hang up|max retries exceeded|can't connect to|could not connect|unable to connect|failed to connect|host is unreachable|broken pipe|err_name_not_resolved|err_internet_disconnected|err_connection"), None, None),
    ("encoding", _r(r"unicodedecodeerror|unicodeencodeerror|codec can't|invalid utf-?8|invalid byte sequence|malformed input|charmap|invalid continuation byte|did not contain valid utf-8|unmappable character"), None, None),
    ("unused", _r(r"unused|never used|not used|never read|imported but unused|dead_code|never constructed"), None, None),
    ("deprecated", _r(r"deprecat|obsolete"), None, None),
    ("invalid-value", _r(r"invalid literal|could not convert|numberformat|formatexception|input string was not in a correct format|valueerror|invalid digit|parseinterror|parsefloaterror|invalid value|jsondecodeerror|expecting value|invalid input syntax for|conversion failed|invalid_argument|incorrect .+ value|invalid argument"), None, None),
]

_EXC_KINDS = {
    # python
    "NameError": "undefined-name", "UnboundLocalError": "undefined-name", "AttributeError": "undefined-attribute",
    "ModuleNotFoundError": "import", "ImportError": "import", "IndexError": "index-out-of-range", "KeyError": "key-missing",
    "ZeroDivisionError": "division-by-zero", "OverflowError": "overflow", "RecursionError": "stack-overflow",
    "MemoryError": "out-of-memory", "FileNotFoundError": "file-not-found", "NotADirectoryError": "file-not-found",
    "IsADirectoryError": "file-not-found", "PermissionError": "permission", "ConnectionError": "network",
    "ConnectionRefusedError": "network", "ConnectionResetError": "network", "ConnectionAbortedError": "network",
    "BrokenPipeError": "network", "TimeoutError": "timeout", "UnicodeDecodeError": "encoding",
    "UnicodeEncodeError": "encoding", "UnicodeError": "encoding", "AssertionError": "test-failure",
    "SyntaxError": "syntax", "IndentationError": "indentation", "TabError": "indentation", "ValueError": "invalid-value",
    "JSONDecodeError": "invalid-value",
    # java / kotlin
    "NullPointerException": "null-reference", "KotlinNullPointerException": "null-reference",
    "ArrayIndexOutOfBoundsException": "index-out-of-range", "StringIndexOutOfBoundsException": "index-out-of-range",
    "IndexOutOfBoundsException": "index-out-of-range", "ClassCastException": "type-mismatch",
    "NumberFormatException": "invalid-value", "ArithmeticException": "division-by-zero",
    "StackOverflowError": "stack-overflow", "OutOfMemoryError": "out-of-memory",
    "ConcurrentModificationException": "concurrency", "FileNotFoundException": "file-not-found",
    "NoSuchFileException": "file-not-found", "ClassNotFoundException": "import", "NoClassDefFoundError": "import",
    "AssertionFailedError": "test-failure", "ComparisonFailure": "test-failure", "InputMismatchException": "invalid-value",
    "NoSuchElementException": "index-out-of-range", "UnknownHostException": "network", "ConnectException": "network",
    "SocketTimeoutException": "timeout", "AccessDeniedException": "permission",
    # c#
    "NullReferenceException": "null-reference", "ArgumentNullException": "null-reference",
    "IndexOutOfRangeException": "index-out-of-range", "ArgumentOutOfRangeException": "index-out-of-range",
    "FormatException": "invalid-value", "InvalidCastException": "type-mismatch", "KeyNotFoundException": "key-missing",
    "DivideByZeroException": "division-by-zero", "OverflowException": "overflow", "StackOverflowException": "stack-overflow",
    "OutOfMemoryException": "out-of-memory", "DirectoryNotFoundException": "file-not-found",
    "UnauthorizedAccessException": "permission", "HttpRequestException": "network", "SocketException": "network",
    "TaskCanceledException": "timeout", "TimeoutException": "timeout", "JsonException": "invalid-value",
    # js
    "ReferenceError": "undefined-name",
    # ruby / php / elixir
    "NoMethodError": "undefined-attribute", "LoadError": "import", "SystemStackError": "stack-overflow",
    "DivisionByZeroError": "division-by-zero", "ArgumentCountError": "argument-count", "ParseError": "syntax",
    "UndefinedFunctionError": "undefined-name", "FunctionClauseError": "type-mismatch",
    # c++
    "std::out_of_range": "index-out-of-range", "std::bad_alloc": "out-of-memory", "std::invalid_argument": "invalid-value",
    "std::length_error": "index-out-of-range", "std::overflow_error": "overflow",
}

_WEAK = {
    "TypeError": "type-mismatch", "RangeError": "invalid-value", "IllegalArgumentException": "invalid-value",
    "ArgumentException": "invalid-value", "ArgumentError": "invalid-value", "ArithmeticError": "type-mismatch",
}

_RUST_CODES = {
    "E0382": "borrow", "E0499": "borrow", "E0502": "borrow", "E0505": "borrow", "E0506": "borrow", "E0507": "borrow",
    "E0384": "borrow", "E0596": "borrow", "E0716": "borrow", "E0373": "borrow",
    "E0106": "lifetime", "E0597": "lifetime", "E0495": "lifetime", "E0621": "lifetime", "E0623": "lifetime", "E0515": "lifetime",
    "E0308": "type-mismatch", "E0277": "type-mismatch", "E0369": "type-mismatch", "E0614": "type-mismatch",
    "E0425": "undefined-name", "E0412": "undefined-name", "E0423": "undefined-name",
    "E0433": "import", "E0432": "import", "E0599": "undefined-attribute", "E0609": "undefined-attribute",
    "E0560": "undefined-attribute", "E0061": "argument-count", "E0063": "argument-count",
}
_CS_CODES = {
    "CS0103": "undefined-name", "CS0165": "undefined-name", "CS0246": "import", "CS0234": "import",
    "CS1061": "undefined-attribute", "CS0117": "undefined-attribute", "CS0029": "type-mismatch", "CS1503": "type-mismatch",
    "CS0266": "type-mismatch", "CS0019": "type-mismatch", "CS1501": "argument-count", "CS7036": "argument-count",
    "CS1002": "syntax", "CS1513": "syntax", "CS1525": "syntax", "CS1026": "syntax", "CS1003": "syntax",
    "CS0168": "unused", "CS0219": "unused", "CS8019": "unused", "CS0618": "deprecated", "CS0612": "deprecated",
    "CS8600": "null-reference", "CS8602": "null-reference", "CS8603": "null-reference", "CS8604": "null-reference",
    "CS8618": "null-reference", "CS0161": "syntax",
}
_TS_CODES = {
    "TS2304": "undefined-name", "TS2552": "undefined-name", "TS2339": "undefined-attribute", "TS2551": "undefined-attribute",
    "TS2345": "type-mismatch", "TS2322": "type-mismatch", "TS2741": "type-mismatch", "TS2769": "type-mismatch",
    "TS7006": "type-mismatch", "TS2362": "type-mismatch", "TS2365": "type-mismatch", "TS2307": "import", "TS2792": "import",
    "TS2554": "argument-count", "TS2555": "argument-count", "TS2531": "null-reference", "TS2532": "null-reference",
    "TS2533": "null-reference", "TS18047": "null-reference", "TS18048": "null-reference", "TS1005": "syntax",
    "TS1109": "syntax", "TS1128": "syntax", "TS1002": "syntax", "TS6133": "unused", "TS6192": "unused", "TS6196": "unused",
}
_CPP_CODES = {
    "C2065": "undefined-name", "C3861": "undefined-name", "C2039": "undefined-attribute", "C2440": "type-mismatch",
    "C2664": "type-mismatch", "C2446": "type-mismatch", "C4244": "type-mismatch", "C2660": "argument-count",
    "C2143": "syntax", "C2059": "syntax", "C1075": "syntax", "C2146": "syntax", "C1083": "import",
    "C4996": "deprecated", "C4101": "unused", "C4189": "unused",
}
_PY_CODES = {
    "name-defined": "undefined-name", "attr-defined": "undefined-attribute", "union-attr": "null-reference",
    "arg-type": "type-mismatch", "assignment": "type-mismatch", "return-value": "type-mismatch", "operator": "type-mismatch",
    "call-overload": "type-mismatch", "list-item": "type-mismatch", "dict-item": "type-mismatch", "index": "type-mismatch",
    "var-annotated": "type-mismatch", "valid-type": "type-mismatch", "override": "type-mismatch", "return": "type-mismatch",
    "call-arg": "argument-count", "import": "import", "import-not-found": "import", "import-untyped": "import",
    "syntax": "syntax", "unused-ignore": "unused",
    "F401": "unused", "F841": "unused", "F811": "unused", "F821": "undefined-name", "F822": "undefined-name",
    "F823": "undefined-name", "E999": "syntax", "E902": "file-not-found",
    "E0602": "undefined-name", "E0401": "import", "E1101": "undefined-attribute", "E1120": "argument-count",
    "E1121": "argument-count", "E0001": "syntax", "W0611": "unused", "W0612": "unused", "W0613": "unused",
    "E0102": "unused", "E1136": "type-mismatch", "W4902": "deprecated", "W0402": "deprecated",
}
_JS_CODES = {
    "no-undef": "undefined-name", "no-unused-vars": "unused", "@typescript-eslint/no-unused-vars": "unused",
    "no-unreachable": "unused", "import/no-unresolved": "import", "n/no-missing-import": "import",
    "semi": "style", "quotes": "style", "indent": "style", "comma-dangle": "style", "prefer-const": "style",
    "no-console": "style", "eol-last": "style", "no-trailing-spaces": "style", "@typescript-eslint/no-explicit-any": "style",
    "no-dupe-keys": "syntax", "no-redeclare": "syntax", "no-const-assign": "type-mismatch",
}
_SQL_CODES = {
    "1064": "syntax", "1146": "undefined-name", "1054": "undefined-name", "1049": "undefined-name", "1045": "permission",
    "1062": "invalid-value", "1452": "invalid-value", "2002": "network", "2003": "network", "1048": "invalid-value",
    "102": "syntax", "156": "syntax", "207": "undefined-name", "208": "undefined-name", "2627": "invalid-value",
    "2601": "invalid-value", "547": "invalid-value", "8134": "division-by-zero", "245": "invalid-value",
    "515": "invalid-value", "18456": "permission", "4060": "undefined-name", "2812": "undefined-name",
}
_GLOBAL_CODES = {
    "ERESOLVE": "dependency-resolution", "E404": "dependency-resolution", "ETARGET": "dependency-resolution",
    "ENOENT": "file-not-found", "EACCES": "permission", "EPERM": "permission", "ETIMEDOUT": "timeout",
    "ECONNREFUSED": "network", "ENOTFOUND": "network", "ECONNRESET": "network", "EINTEGRITY": "dependency-resolution",
    "MODULE_NOT_FOUND": "import", "ERR_MODULE_NOT_FOUND": "import", "ERR_REQUIRE_ESM": "import",
    "ERR_UNHANDLED_REJECTION": "unhandled-rejection", "LNK2019": "linker", "LNK2001": "linker", "LNK1120": "linker",
    "LNK2005": "linker", "LNK1104": "linker", "SIGSEGV": "access-violation", "SIGFPE": "division-by-zero",
    "0xC0000005": "access-violation", "0xC00000FD": "stack-overflow", "0xC0000094": "division-by-zero",
    "0xC0000409": "access-violation", "0xC0000374": "access-violation", "0xC0000135": "import",
}
_LANG_CODES = {"rust": _RUST_CODES, "csharp": _CS_CODES, "typescript": _TS_CODES, "cpp": _CPP_CODES, "c": _CPP_CODES,
               "python": _PY_CODES, "javascript": _JS_CODES, "sql": _SQL_CODES}


def _code_kind(code, lang):
    if not code:
        return None
    k = _LANG_CODES.get(lang, {}).get(code)
    if k is None and code.startswith("TS") and code[2:].isdigit():
        k = _TS_CODES.get(code)
    if k is None:
        k = _GLOBAL_CODES.get(code)
    if k is None:
        k = _EXC_KINDS.get(code) or _EXC_KINDS.get(code.rsplit(".", 1)[-1])
        if code == "TypeError" and lang != "python":
            k = None
    if k is None and lang == "python" and re.match(r"^[EWC]\d{3,4}$", code):
        k = "indentation" if re.match(r"^E1[01]\d$", code) else "style"
    return k


@lru_cache(maxsize=4096)
def _classify_raw(message, lang, code=None, extra=""):
    text = f"{code or ''} {extra} {message}"
    for kind, rx, langs, hint in _PRIORITY:
        if (langs is None or lang in langs) and rx.search(text):
            return kind, hint
    k = _code_kind(code, lang)
    if k:
        return k, None
    for kind, rx, langs, hint in _GENERAL:
        if (langs is None or lang in langs) and rx.search(text):
            return kind, hint
    for c in (code, extra):
        if c:
            w = _WEAK.get(c.rsplit(".", 1)[-1])
            if w:
                return w, None
    return "unknown", None


def _http_hint(code, message):
    m = re.search(r"\b([45]\d\d)\b", f"{code or ''} {message}")
    if m and int(m.group(1)) in HTTP_MEANING:
        return HTTP_MEANING[int(m.group(1))]
    return None


def _hint(kind, lang, code, rule_hint, line, message, suggestion=None, extra_hint=None):
    h = rule_hint
    if h is None and code:
        ck = _code_kind(code, lang)
        if ck == kind:
            h = CODE_HINTS.get(code)
    if h is None and kind == "http-status":
        h = _http_hint(code, message)
    if h is None:
        h = LANG_HINTS.get((kind, lang)) or KIND_HINTS.get(kind) or KIND_HINTS["unknown"]
    where = f"line {line}" if line else "the reported location"
    out = h.replace("{where}", where)
    if suggestion:
        out += f" (did you mean '{suggestion}'?)"
    elif h == KIND_HINTS["undefined-name"]:
        out += ", or define/import it before use"
    if extra_hint:
        out += f" (help: {extra_hint[:160]})"
    return out


_SUGGEST = re.compile(r"[\s.;,(]*\b[Dd]id you mean:?\s*(?:the \w+ )?" + _Q + r"(?P<s>[^'‘’\"`]{1,80})" + _Q + r"\s*\??\)?")


def _split_suggestion(message):
    m = _SUGGEST.search(message or "")
    if not m:
        return None, message
    rest = (message[:m.start()] + message[m.end():]).strip()
    return m.group("s").strip(), rest.rstrip(" ;,")


def classify(message: str, lang: str, code: str | None = None) -> tuple[str, str]:
    """(kind, hint) for one error message in a language (see KINDS)."""
    message = message or ""
    lang = (lang or "unknown").lower()
    sug, msg = _split_suggestion(message)
    kind, rh = _classify_raw(msg, lang, code)
    return kind, _hint(kind, lang, code, rh, None, msg, sug)


def languages() -> list[str]:
    return list(_LANGS)


# ------------------------------------------------------------------------------------ path helpers

_EXT = {
    "py": "python", "pyw": "python", "pyi": "python",
    "c": "c", "h": "cpp", "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "c++": "cpp", "hpp": "cpp", "hh": "cpp",
    "hxx": "cpp", "ipp": "cpp", "tpp": "cpp", "inl": "cpp",
    "java": "java", "groovy": "java", "kt": "kotlin", "kts": "kotlin", "scala": "scala", "sc": "scala",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript", "jsx": "javascript", "vue": "javascript",
    "svelte": "javascript", "ts": "typescript", "tsx": "typescript", "mts": "typescript", "cts": "typescript",
    "cs": "csharp", "go": "go", "rs": "rust", "php": "php", "rb": "ruby", "pl": "perl", "pm": "perl", "t": "perl",
    "lua": "lua", "swift": "swift", "dart": "dart", "hs": "haskell", "lhs": "haskell", "ex": "elixir", "exs": "elixir",
    "sh": "shell", "bash": "shell", "zsh": "shell", "ps1": "powershell", "psm1": "powershell", "psd1": "powershell",
    "bat": "cmd", "cmd": "cmd", "sql": "sql", "html": "html", "htm": "html", "css": "css", "scss": "css", "less": "css",
}


def _lang_of(path, default="unknown"):
    if not path:
        return default
    base = re.split(r"[\\/]", path)[-1]
    if "." not in base:
        return default
    return _EXT.get(base.rsplit(".", 1)[-1].lower(), default)


_LIB_RE = re.compile(
    r"site-packages|dist-packages|[\\/]lib[\\/]python\d|[\\/]Lib[\\/]|<frozen|node_modules|^node:|^internal[\\/]"
    r"|[\\/]rustc[\\/][0-9a-f]{6,}|\.cargo[\\/]registry|[\\/]library[\\/](?:std|core|alloc)[\\/]"
    r"|[\\/]go[\\/]src[\\/]|[\\/]Go[\\/]src[\\/]|[\\/]pkg[\\/]mod[\\/]|^/usr/|^/opt/|^/lib/|Program Files"
    r"|[\\/]gems[\\/]|[\\/]\.rbenv[\\/]|[\\/]\.rvm[\\/]|[\\/]vendor[\\/]|[\\/]\.m2[\\/]|(?:^|[\\/])deps[\\/]"
    r"|Microsoft Visual Studio|[\\/]MSVC[\\/]|Windows Kits|libc\.so|libstdc\+\+|^dart:|^Swift/|sanitizer_common|compiler-rt"
)
_JAVA_LIB = re.compile(r"^(?:java|javax|jdk|sun|com\.sun|kotlin|kotlinx|scala|org\.junit|junit|org\.opentest4j|org\.apache"
                       r"|org\.springframework|org\.gradle|org\.hibernate|com\.google|io\.netty|org\.testng)\.")


def _is_lib(path):
    return bool(path) and bool(_LIB_RE.search(path))


def _norm_path(p):
    if not p:
        return p
    p = p.strip().strip("\"'")
    if p.startswith("file://"):
        p = p[7:]
    if re.match(r"^/[A-Za-z]:[/\\]", p):
        p = p[1:]
    return p


def _under_cwd(path, cwd):
    p = path.replace("\\", "/")
    if not (p.startswith("/") or re.match(r"^[A-Za-z]:/", p)):
        return True
    c = cwd.replace("\\", "/").rstrip("/")
    return p.lower().startswith(c.lower() + "/")


def _pick(frames, cwd):
    """frames innermost-first -> the frame to fix: user code under cwd, else any non-library frame."""
    user = [fr for fr in frames if fr.get("f") and not fr.get("lib")]
    if cwd:
        uc = [fr for fr in user if _under_cwd(fr["f"], cwd)]
        if uc:
            return uc[0]
    if user:
        return user[0]
    return frames[0] if frames else None


def _fmt_frame(fr):
    s = f"{fr.get('f')}:{fr.get('l')}"
    if fr.get("fn"):
        s += f" in {fr['fn']}"
    return s


def _frame_ctx(frames, chosen, first=None):
    ctx = [first] if first else []
    for fr in frames:
        if len(ctx) >= 3:
            break
        if fr is chosen and first:
            continue
        ctx.append("at " + _fmt_frame(fr))
    return ctx


def _read_line(path, line, cwd):
    try:
        p = path
        if cwd and not os.path.isabs(p):
            p = os.path.join(cwd, p)
        if not os.path.isfile(p) or os.path.getsize(p) > 5_000_000:
            return None
        with open(p, encoding="utf-8", errors="replace") as fh:
            for k, s in enumerate(fh, 1):
                if k == line:
                    return s.rstrip("\n")
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------------------- state

class _St:
    def __init__(self, lines, cwd, text):
        self.L = lines
        self.n = len(lines)
        self.cwd = cwd
        self.out = []
        self.consumed = bytearray(self.n)
        self.last = None
        self.last_i = -99
        self.required = None
        self.rust = ("error[E" in text) or ("cargo" in text) or (".rs:" in text and "-->" in text)


def _emit(st, pos, *, file=None, line=None, col=None, sev="error", lang="unknown", code=None, message="",
          fk="runtime", context=None, rank=0, suggestion=None, kind=None, hint=None, extra="", extra_hint=None):
    message = re.sub(r"\s+", " ", (message or "")).strip()
    sug, message = _split_suggestion(message)
    if suggestion:
        sug = suggestion
    if file:
        file = _norm_path(file)
    if lang == "unknown" and file:
        lang = _lang_of(file)
    if kind is None:
        kind, rh = _classify_raw(message, lang, code, extra)
        hint = hint or rh
    h = _hint(kind, lang, code, hint, line, message, sug, extra_hint)
    ctx = [c.rstrip()[:200] for c in (context or []) if c and c.strip()][:3]
    d = Diag(file or None, line, col, sev, lang, code, message[:500] or (code or "error"), kind, h, fk, ctx, sug, pos, rank)
    st.out.append(d)
    return d


# ---------------------------------------------------------------------------------------- python

_PY_FRAME = re.compile(r'^\s*File "(?P<f>[^"]+)", line (?P<l>\d+)(?:, in (?P<fn>.+?))?\s*$')
_PY_EXC = re.compile(r'^(?P<t>[A-Za-z_][\w.]*)(?::\s?(?P<m>.*))?$')
_PY_EXC_NAME = re.compile(r"(?:Error|Exception|Exit|Interrupt|Iteration|Warning|Failure|Timeout|Group)$|\.")
_PY_SYNTAX = ("SyntaxError", "IndentationError", "TabError")
_UNITTEST = re.compile(r"^(FAIL|ERROR): (\S+) \((.+?)\)")


def _py_block(L, j, n):
    if L[j].lstrip().startswith("Traceback"):
        j += 1
    frames = []
    while j < n:
        m = _PY_FRAME.match(L[j])
        if not m:
            if frames and L[j].startswith((" ", "\t")) and L[j].strip():
                frames[-1]["src"].append(L[j])
                j += 1
                continue
            break
        fr = {"f": m.group("f"), "l": int(m.group("l")), "fn": m.group("fn"), "src": []}
        fr["lib"] = _is_lib(fr["f"])
        frames.append(fr)
        j += 1
        while j < n and L[j].startswith((" ", "\t")) and L[j].strip() and not _PY_FRAME.match(L[j]):
            fr["src"].append(L[j])
            j += 1
    t = msg = None
    if j < n:
        m = _PY_EXC.match(L[j].rstrip())
        if m and (_PY_EXC_NAME.search(m.group("t")) or m.group("m") is not None):
            t, msg = m.group("t"), (m.group("m") or "")
            j += 1
    return frames, t, msg, j


def _py_col(fr, st):
    src = fr.get("src") or []
    if len(src) < 2:
        return None
    code, car = src[0], src[1].rstrip()
    if not car.strip() or set(car.strip()) - set("^~"):
        return None
    disp = (len(car) - len(car.lstrip())) - (len(code) - len(code.lstrip()))
    if disp < 0:
        return None
    orig = _read_line(fr["f"], fr["l"], st.cwd)
    if orig is None:
        return None
    return len(orig) - len(orig.lstrip()) + disp + 1


def _p_python(st, i):
    L, n = st.L, st.n
    s = L[i].lstrip()
    if s.startswith("Traceback (most recent call last)"):
        pass
    elif s.startswith('File "'):
        m = _PY_FRAME.match(L[i])
        if not m or m.group("fn"):
            return None
        if not any(L[k].startswith(_PY_SYNTAX) for k in range(i + 1, min(i + 6, n))):
            return None
    else:
        return None
    blocks = []
    j = i
    while j < n:
        frames, t, msg, j = _py_block(L, j, n)
        if t is None and not frames:
            break
        blocks.append((frames, t, msg))
        k = j
        while k < n and not L[k].strip():
            k += 1
        if k < n and L[k].startswith(("During handling of the above exception", "The above exception was the direct cause")):
            k += 1
            while k < n and not L[k].strip():
                k += 1
            if k < n and L[k].lstrip().startswith("Traceback"):
                j = k
                continue
        break
    if not blocks or blocks[-1][1] is None:
        return j if j > i else None
    ut = None
    for k in range(i - 1, max(-1, i - 4), -1):
        ut = _UNITTEST.match(L[k])
        if ut:
            break
    for bi, (frames, t, msg) in enumerate(blocks):
        if t is None:
            continue
        final = bi == len(blocks) - 1
        simple = t.rsplit(".", 1)[-1]
        if simple in _PY_SYNTAX and frames:
            fr, fk = frames[-1], "compile"
            src = [x[4:] if x.startswith("    ") else x.strip() for x in fr["src"][:2]]
            ctx = src
        else:
            inner = list(reversed(frames))
            fr, fk = _pick(inner, st.cwd), "runtime"
            first = fr["src"][0].strip() if fr and fr["src"] else None
            ctx = _frame_ctx(inner, fr, first)
        if ut:
            fk = "test"
            ctx = [f"{ut.group(1)}: {ut.group(2)} ({ut.group(3)})"] + ctx
        _emit(st, i, file=fr["f"] if fr else None, line=fr["l"] if fr else None, col=_py_col(fr, st) if fr else None,
              sev="error" if final else "note", lang="python", code=simple, message=msg or t, fk=fk, context=ctx,
              rank=0 if final else 1, extra=t)
    return j


# ---------------------------------------------------------------------------------------- pytest

_PYTEST_LOC = re.compile(r"^(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?\.py):(?P<l>\d+): (?P<t>[A-Za-z_][\w.]*)\s*$")
_PYTEST_IN = re.compile(r"^(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?\.py):(?P<l>\d+): in (?P<fn>\S+)\s*$")
_PYTEST_SUM = re.compile(r"^(?:FAILED|ERROR) (?P<f>[^\s:]+\.py)::(?P<t>\S+?)(?: - (?P<m>.*))?$")
_PYTEST_SEC = re.compile(r"^_{3,} .* _{3,}$")


def _p_pytest(st, i):
    L = st.L
    m = _PYTEST_LOC.match(L[i])
    if not m:
        return None
    t = m.group("t")
    j = i - 1
    while j >= 0 and not L[j].strip():
        j -= 1
    elines = []
    while j >= 0 and (L[j].startswith("E ") or L[j] == "E"):
        elines.append(L[j][1:].strip())
        j -= 1
    elines.reverse()
    frames, src, test = [], None, None
    k = j
    while k >= 0 and i - k < 400:
        if _PYTEST_SEC.match(L[k]):
            test = L[k].strip("_ ")
            break
        fm = _PYTEST_IN.match(L[k])
        if fm:
            frames.append({"f": fm.group("f"), "l": int(fm.group("l")), "fn": fm.group("fn"), "lib": _is_lib(fm.group("f"))})
        if src is None and L[k].startswith(">"):
            src = L[k][1:].strip()
        k -= 1
    loc = {"f": m.group("f"), "l": int(m.group("l")), "lib": _is_lib(m.group("f"))}
    fr = _pick([loc] + frames, st.cwd)
    msg = elines[0] if elines else t
    if msg.startswith(t + ":"):
        msg = msg[len(t) + 1:].strip()
    ctx = ([src] if src else []) + elines[:2]
    if test:
        ctx = ctx[:2] + [f"test: {test}"]
    _emit(st, i, file=fr["f"], line=fr["l"], lang="python", code=t.rsplit(".", 1)[-1], message=msg, fk="test",
          context=ctx, extra=t)
    return i + 1


def _p_pytest_sum(st, i):
    m = _PYTEST_SUM.match(st.L[i])
    if not m:
        return None
    msg = m.group("m") or "test failed"
    code = None
    em = re.match(r"^([A-Za-z_][\w.]*(?:Error|Exception)): ?(.*)$", msg)
    if em:
        code, msg = em.group(1).rsplit(".", 1)[-1], em.group(2) or em.group(1)
    _emit(st, i, file=m.group("f"), lang="python", code=code or "AssertionError", message=msg, fk="test",
          context=[f"test: {m.group('t')}"], rank=2)
    return i + 1


# ------------------------------------------------------------------------------------ javascript

_V8_FRAME = re.compile(r"^\s+at (?:async )?(?:(?P<fn>.*?) \()?(?P<f>(?:file:///|node:)?[^()\s][^()]*?):(?P<l>\d+):(?P<c>\d+)\)?(?: \{)?\s*$")
_V8_HEAD = re.compile(r"^(?:(?P<src>\S+?\.[cm]?[jt]sx?):(?P<sl>\d+) )?(?:error: )?(?P<unc>Uncaught (?:\(in promise\) )?)?"
                      r"(?P<t>(?:[A-Z][\w$]*)?Error|[A-Z][\w$]*Exception)(?: \[(?P<code>[A-Z][A-Z0-9_]+)\])?(?::\s?(?P<m>.*))?$")
_V8_REJ = re.compile(r"^\[(?P<t>UnhandledPromiseRejection): (?P<m>.*?)\]?\s*\{?\s*$")
_NODE_LOC = re.compile(r"^(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?\.[cm]?[jt]sx?|node:internal/[\w/]+|file:///\S+?):(?P<l>\d+)\s*$")
_INLINE_AT = re.compile(r"\s+at (?:.*?\()?(?P<f>(?:https?://|file:///)?[^\s()]+?):(?P<l>\d+):(?P<c>\d+)\)?\s*$")
_CODEFRAME = re.compile(r"^\s*(?P<gt>>)?\s*(?P<n>\d+) \|(?P<code>.*)$")


def _v8_frame(m):
    f = _norm_path(m.group("f"))
    return {"f": f, "l": int(m.group("l")), "c": int(m.group("c")), "fn": m.group("fn"), "lib": _is_lib(f) or f.startswith("node:")}


def _v8_build(st, i, j, hm, loc=None, rej=False):
    """hm: header match at line j. Consumes the frames; returns the next index."""
    L, n = st.L, st.n
    t = hm.group("t")
    msg = (hm.group("m") or "").strip()
    code = None if rej else hm.group("code")
    frames = []
    k = j + 1
    gap = 0
    while k < n:
        fm = _V8_FRAME.match(L[k])
        if fm:
            frames.append(_v8_frame(fm))
            k += 1
            continue
        if L[k].lstrip().startswith("at ") and frames:
            k += 1
            continue
        if not frames and gap < 6 and L[k].strip() and not _V8_HEAD.match(L[k]):
            gap += 1
            k += 1
            continue
        break
    if k < n and frames and L[k].startswith("  ") and ":" in L[k]:
        while k < n and L[k].strip() != "}" and k - j < 200:
            pm = re.match(r"^\s+code: '([A-Z_0-9]+)'", L[k])
            if pm and not code:
                code = pm.group(1)
            k += 1
        k += 1
    while k < n and not L[k].strip():
        k += 1
    if k < n and re.match(r"^Node\.js v\d", L[k]):
        k += 1
    if not frames and loc is None and not hm.groupdict().get("unc") and not _INLINE_AT.search(msg) and not rej \
            and not hm.groupdict().get("src"):
        return None
    im = _INLINE_AT.search(msg)
    if im:
        frames.insert(0, {"f": _norm_path(im.group("f")), "l": int(im.group("l")), "c": int(im.group("c")), "fn": None,
                          "lib": _is_lib(im.group("f"))})
        msg = msg[:im.start()].strip()
    if hm.groupdict().get("src") and not frames:
        frames.append({"f": hm.group("src"), "l": int(hm.group("sl")), "c": None, "fn": None, "lib": False})
    fr = _pick(frames, st.cwd) if frames else None
    if fr is not None and fr.get("lib") and loc and not loc.get("lib"):
        fr = None
    if fr is None and loc and not loc.get("lib"):
        fr = loc
    file = fr["f"] if fr else None
    line = fr["l"] if fr else None
    col = fr.get("c") if fr else None
    if loc and fr is not loc and file and loc.get("f") == file and loc.get("l") == line and col is None:
        col = loc.get("c")
    lang = _lang_of(file, "javascript") if file else "javascript"
    if lang not in ("javascript", "typescript"):
        lang = "javascript"
    ctx = []
    if loc and loc.get("src"):
        ctx.append(loc["src"].strip())
    ctx += ["at " + _fmt_frame(x) for x in frames[:3]]
    if rej:
        code = "ERR_UNHANDLED_REJECTION"
    _emit(st, i, file=file, line=line, col=col, lang=lang, code=code or t, message=msg or t, fk="runtime",
          context=ctx, extra=t, kind="unhandled-rejection" if rej else None)
    return max(k, j + 1)


def _p_node_caret(st, i):
    L, n = st.L, st.n
    m = _NODE_LOC.match(L[i])
    if not m:
        return None
    caret = None
    j = i + 1
    src = None
    while j < n and j <= i + 4:
        if re.match(r"^\s*\^+\s*$", L[j]):
            caret = L[j].index("^")
            j += 1
            break
        if src is None:
            src = L[j]
        j += 1
    if caret is None:
        return None
    while j < n and not L[j].strip():
        j += 1
    if j >= n:
        return None
    f = _norm_path(m.group("f"))
    loc = {"f": f, "l": int(m.group("l")), "c": caret + 1, "fn": None, "lib": _is_lib(f) or f.startswith("node:"), "src": src or ""}
    hm = _V8_HEAD.match(L[j])
    if hm:
        return _v8_build(st, i, j, hm, loc)
    rm = _V8_REJ.match(L[j])
    if rm:
        return _v8_build(st, i, j, rm, loc, rej=True)
    return None


def _p_v8(st, i):
    L = st.L
    hm = _V8_HEAD.match(L[i])
    if hm:
        return _v8_build(st, i, i, hm)
    rm = _V8_REJ.match(L[i])
    if rm:
        return _v8_build(st, i, i, rm, rej=True)
    return None


_NODE_WARN = re.compile(r"^\(node:\d+\) (?P<t>\w+Warning): (?P<m>.*)$")


def _p_node_warn(st, i):
    m = _NODE_WARN.match(st.L[i])
    if not m:
        return None
    _emit(st, i, lang="javascript", code=m.group("t"), message=m.group("m"), sev="warning", fk="runtime", extra=m.group("t"))
    return i + 1


_DENO = re.compile(r"^(?:error: )?(?P<code>TS\d+) \[ERROR\]: (?P<m>.*)$")


def _p_deno(st, i):
    L, n = st.L, st.n
    m = _DENO.match(L[i])
    if not m:
        return None
    ctx, f = [], None
    j = i + 1
    while j < n and j <= i + 8:
        am = re.match(r"^\s*at (?P<f>\S+?):(?P<l>\d+):(?P<c>\d+)\s*$", L[j])
        if am:
            f = am
            j += 1
            break
        if L[j].strip():
            ctx.append(L[j])
        j += 1
    file = _norm_path(f.group("f")) if f else None
    _emit(st, i, file=file, line=int(f.group("l")) if f else None, col=int(f.group("c")) if f else None,
          lang=_lang_of(file, "typescript") if file else "typescript", code=m.group("code"), message=m.group("m"),
          fk="compile", context=ctx[:2])
    return j


# ----------------------------------------------------------------------------------- jest / vitest

_JEST_HDR = re.compile(r"^\s*● (?P<name>.+?)\s*$")
_JEST_STOP = re.compile(r"^\s*●|^(?:Test Suites|Tests|Snapshots|Time):|^\s*(?:PASS|FAIL)\s{1,2}\S")


def _p_jest(st, i):
    L, n = st.L, st.n
    m = _JEST_HDR.match(L[i])
    if not m:
        return None
    j = i + 1
    body = []
    while j < n and j - i < 150 and not _JEST_STOP.match(L[j]):
        body.append(L[j])
        j += 1
    msgs, err, exp, rec, line, col, frames, seen_code, srcl = [], None, None, None, None, None, [], False, None
    for k, b in enumerate(body):
        cm = _CODEFRAME.match(b)
        if cm:
            seen_code = True
            if cm.group("gt"):
                line = int(cm.group("n"))
                srcl = cm.group("code").strip()
                if k + 1 < len(body):
                    car = body[k + 1]
                    bar, c = car.find("|"), car.find("^")
                    if 0 <= bar < c:
                        col = c - bar - 1
            continue
        fm = _V8_FRAME.match(b)
        if fm:
            frames.append(_v8_frame(fm))
            continue
        s = b.strip()
        if not s or seen_code:
            continue
        em = re.match(r"^Expected(?: value)?(?: length)?:?\s+(.*)$", s)
        rm = re.match(r"^Received(?: value)?(?: length)?:?\s+(.*)$", s)
        if em:
            exp = em.group(1)
        elif rm:
            rec = rm.group(1)
        else:
            tm = re.match(r"^((?:[A-Z]\w*)?Error): (.*)$", s)
            if tm and err is None:
                err = tm
            msgs.append(s)
    fr = _pick(frames, st.cwd) if frames else None
    file = fr["f"] if fr else None
    if fr:
        line, col = fr["l"], fr.get("c")
    if file is None:
        for k in range(i - 1, max(-1, i - 300), -1):
            fm2 = re.match(r"^\s*FAIL\s+(\S+)", L[k])
            if fm2:
                file = fm2.group(1)
                break
    first = re.sub(r"\s*//.*$", "", msgs[0]) if msgs else m.group("name")
    code = err.group(1) if err else None
    kind = None
    if exp is not None or rec is not None or first.startswith("expect("):
        first = f"{first}; Expected: {exp}; Received: {rec}"
        kind = "test-failure"
    if err:
        first = err.group(2)
    _emit(st, i, file=file, line=line, col=col, lang=_lang_of(file, "javascript") if file else "javascript", code=code,
          message=first, fk="test", context=[f"test: {m.group('name')}"] + ([srcl] if srcl else []), kind=kind)
    return j


_VITEST_HDR = re.compile(r"^\s*(?:FAIL|×|✗)\s+(?P<f>\S+\.[cm]?[jt]sx?)\s+>\s+(?P<name>.+?)\s*$")


def _p_vitest(st, i):
    L, n = st.L, st.n
    m = _VITEST_HDR.match(L[i])
    if not m:
        return None
    err, loc = None, None
    j = i + 1
    while j < n and j - i < 80 and not _VITEST_HDR.match(L[j]):
        s = L[j].strip()
        if err is None:
            em = re.match(r"^((?:[A-Z]\w*)?Error): (.*)$", s)
            if em:
                err = em
        if loc is None:
            lm = re.match(r"^❯ (?P<f>.+?):(?P<l>\d+):(?P<c>\d+)$", s)
            if lm and not _is_lib(lm.group("f")):
                loc = lm
        if s.startswith("⎯⎯⎯") and err is not None:
            j += 1
            break
        j += 1
    file = loc.group("f") if loc else m.group("f")
    _emit(st, i, file=file, line=int(loc.group("l")) if loc else None, col=int(loc.group("c")) if loc else None,
          lang=_lang_of(file, "javascript"), code=err.group(1) if err else None,
          message=re.sub(r"\s*//.*$", "", err.group(2)) if err else "test failed", fk="test",
          context=[f"test: {m.group('name')}"], kind="test-failure" if (not err or "Assertion" in err.group(1)) else None)
    return j


# ----------------------------------------------------------------------------------- npm / eslint

_NPM_CODE = re.compile(r"^npm (?:ERR!|error) code (?P<code>\S+)\s*$")
_NPM_LINE = re.compile(r"^npm (?:ERR!|error)\s?(?P<t>.*)$")


def _p_npm(st, i):
    L, n = st.L, st.n
    m = _NPM_CODE.match(L[i])
    if not m:
        return None
    j = i + 1
    msg, path = None, None
    while j < n:
        lm = _NPM_LINE.match(L[j])
        if not lm:
            break
        t = lm.group("t").strip()
        pm = re.match(r"^path (.+)$", t)
        if pm and not path:
            path = pm.group(1)
        elif t and msg is None and not re.match(r"^(?:code|errno|syscall|dest|A complete log|Log files)\b", t):
            msg = t
        j += 1
    _emit(st, i, file=path, lang="javascript", code=m.group("code"), message=msg or m.group("code"), fk="compile")
    return j


_ESLINT_HDR = re.compile(r"^(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*\.(?:[cm]?[jt]sx?|vue|svelte))\s*$")
_ESLINT_ROW = re.compile(r"^\s+(?P<l>\d+):(?P<c>\d+)\s+(?P<s>error|warning)\s+(?P<m>.+?)(?:\s{2,}(?P<r>[@\w][\w/@.-]*))?\s*$")


def _p_eslint(st, i):
    L, n = st.L, st.n
    m = _ESLINT_HDR.match(L[i])
    if not m or i + 1 >= n or not _ESLINT_ROW.match(L[i + 1]):
        return None
    f = m.group("f")
    j = i + 1
    while j < n:
        r = _ESLINT_ROW.match(L[j])
        if not r:
            break
        _emit(st, j, file=f, line=int(r.group("l")), col=int(r.group("c")), sev=r.group("s"),
              lang=_lang_of(f, "javascript"), code=r.group("r"), message=r.group("m"), fk="lint")
        j += 1
    return j


# ------------------------------------------------------------------------------------------ java

_JAVA_HEAD1 = re.compile(r'^Exception in thread "(?P<th>[^"]*)" (?P<t>[\w$.]+)(?::\s?(?P<m>.*))?$')
_JAVA_HEAD2 = re.compile(r"^(?P<t>[a-z][\w$]*(?:\.[\w$]+)+)(?::\s?(?P<m>.*))?$")
_JAVA_FRAME = re.compile(r"^\s+at (?:[\w.$-]+(?:@[\w.-]+)?/)*(?P<cls>[\w$.]+)\.(?P<meth>[\w$<>-]+)\((?P<src>[^)]*)\)\s*$")
_JAVA_CAUSE = re.compile(r"^\s*Caused by: (?P<t>[\w$.]+)(?::\s?(?P<m>.*))?$")


def _java_frame(m):
    src = m.group("src")
    f = l = None
    sm = re.match(r"^(?P<f>[\w$.-]+\.(?:java|kt|scala|groovy|clj)):(?P<l>\d+)$", src)
    if sm:
        f, l = sm.group("f"), int(sm.group("l"))
    cls = m.group("cls")
    return {"f": f, "l": l, "fn": f"{cls}.{m.group('meth')}", "lib": bool(_JAVA_LIB.match(cls)) or f is None,
            "text": f"at {cls}.{m.group('meth')}({src})"}


def _p_java(st, i):
    L, n = st.L, st.n
    hm = _JAVA_HEAD1.match(L[i])
    if not hm:
        hm = _JAVA_HEAD2.match(L[i])
        if not hm or i + 1 >= n or not _JAVA_FRAME.match(L[i + 1]):
            return None
    excs = [{"t": hm.group("t"), "m": hm.group("m") or "", "frames": []}]
    j = i + 1
    cont = 0
    while j < n:
        fm = _JAVA_FRAME.match(L[j])
        if fm:
            excs[-1]["frames"].append(_java_frame(fm))
            j += 1
            continue
        cm = _JAVA_CAUSE.match(L[j])
        if cm:
            excs.append({"t": cm.group("t"), "m": cm.group("m") or "", "frames": []})
            j += 1
            continue
        if re.match(r"^\s+\.\.\. \d+ (?:more|common frames omitted)", L[j]) or re.match(r"^\s+Suppressed: ", L[j]):
            j += 1
            continue
        if not excs[-1]["frames"] and cont < 3 and L[j].strip():
            excs[-1]["m"] += " " + L[j].strip()
            cont += 1
            j += 1
            continue
        break
    root = excs[-1]
    fr = _pick(root["frames"], None)
    if fr is None or fr.get("lib"):
        for e in reversed(excs[:-1]):
            cand = _pick(e["frames"], None)
            if cand and not cand.get("lib"):
                fr = cand
                break
    all_frames = [f for e in excs for f in e["frames"]]
    test = any(re.match(r"^(?:org\.junit|junit\.|org\.testng)", f["fn"] or "") for f in all_frames) or \
        bool(re.search(r"Assertion|ComparisonFailure", root["t"]))
    ctx = [fr["text"]] if fr else []
    if len(excs) > 1:
        ctx.append(f"wrapped by: {excs[0]['t']}: {excs[0]['m']}"[:200])
    file = fr["f"] if fr else None
    lang = _lang_of(file, "java") if file else "java"
    _emit(st, i, file=file, line=fr["l"] if fr else None, lang=lang, code=root["t"].rsplit(".", 1)[-1],
          message=root["m"] or root["t"], fk="test" if test else "runtime", context=ctx, extra=root["t"])
    return j


_JAVAC = re.compile(r"^(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?\.java):(?P<l>\d+): (?P<s>error|warning): (?P<m>.*)$")


def _p_javac(st, i):
    L, n = st.L, st.n
    m = _JAVAC.match(L[i])
    if not m:
        return None
    msg = m.group("m")
    col, src, sym, loc = None, None, None, None
    j = i + 1
    while j < n and j <= i + 6:
        s = L[j]
        if _JAVAC.match(s):
            break
        if re.match(r"^\s*\^\s*$", s):
            col = s.index("^") + 1
        elif re.match(r"^\s*symbol:\s+(.*)$", s):
            sym = s.split(":", 1)[1].strip()
        elif re.match(r"^\s*location:\s+(.*)$", s):
            loc = s.split(":", 1)[1].strip()
        elif src is None and s.strip() and not re.match(r"^\d+ (?:errors?|warnings?)$", s.strip()):
            src = s
        else:
            break
        j += 1
    kind = None
    if sym:
        msg = f"{msg}: {re.sub(chr(32) + '+', ' ', sym)}"
        if sym.startswith("class"):
            kind = "import"
    if loc:
        msg += f" (location: {re.sub(chr(32) + '+', ' ', loc)})"
    _emit(st, i, file=m.group("f"), line=int(m.group("l")), col=col, sev=m.group("s"), lang="java", message=msg,
          fk="compile", context=[src.strip()] if src else [], kind=kind)
    return j


_MVN = re.compile(r"^\[(?P<s>ERROR|WARNING)\] (?P<f>.+?\.(?:java|kt|scala|groovy)):\[(?P<l>\d+),(?P<c>\d+)\] (?P<m>.*)$")
_MVN_T = re.compile(r"^\[ERROR\]\s+(?P<cls>[A-Z][\w$]*)\.(?P<meth>[\w$]+):(?P<l>\d+)(?:->\S+)? (?P<m>.*)$")


def _p_maven(st, i):
    L, n = st.L, st.n
    m = _MVN.match(L[i])
    if m:
        msg = m.group("m")
        j = i + 1
        kind = None
        while j < n:
            sm = re.match(r"^\[(?:ERROR|WARNING)\]\s+(symbol|location):\s+(.*)$", L[j])
            if not sm:
                break
            v = re.sub(r"\s+", " ", sm.group(2))
            if sm.group(1) == "symbol":
                msg += f": {v}"
                if v.startswith("class"):
                    kind = "import"
            else:
                msg += f" (location: {v})"
            j += 1
        f = m.group("f")
        _emit(st, i, file=f, line=int(m.group("l")), col=int(m.group("c")), sev="error" if m.group("s") == "ERROR" else "warning",
              lang=_lang_of(f, "java"), message=msg, fk="compile", kind=kind)
        return j
    m = _MVN_T.match(L[i])
    if m:
        _emit(st, i, file=m.group("cls") + ".java", line=int(m.group("l")), lang="java", message=m.group("m"), fk="test",
              context=[f"test: {m.group('cls')}.{m.group('meth')}"], kind="test-failure")
        return i + 1
    return None


# -------------------------------------------------------------------------------------------- c#

_CS_HEAD = re.compile(r"^(?P<unh>Unhandled [Ee]xception[.:] ?)?(?P<t>(?:[A-Z]\w*\.)+[A-Z]\w*(?:Exception|Error)|[A-Z]\w*Exception)"
                      r"(?: \(0x[0-9A-Fa-f]+\))?: (?P<m>.*)$")
_CS_FRAME = re.compile(r"^\s+at (?P<fn>.+?)(?: in (?P<f>.+?):line (?P<l>\d+))?\s*$")


def _p_csharp(st, i):
    L, n = st.L, st.n
    m = _CS_HEAD.match(L[i])
    if not m:
        return None
    if not m.group("unh") and not (i + 1 < n and _CS_FRAME.match(L[i + 1])):
        return None
    t, msg = m.group("t"), m.group("m")
    if " ---> " in msg:
        inner = msg.split(" ---> ")[-1]
        im = re.match(r"^([\w.]+): (.*)$", inner)
        if im:
            t, msg = im.group(1), im.group(2)
    frames = []
    j = i + 1
    while j < n:
        fm = _CS_FRAME.match(L[j])
        if fm and not L[j].strip().startswith("---"):
            fn = fm.group("fn")
            f = fm.group("f")
            frames.append({"f": f, "l": int(fm.group("l")) if fm.group("l") else None, "fn": fn,
                           "lib": f is None or fn.startswith(("System.", "Microsoft."))})
            j += 1
            continue
        if L[j].strip().startswith("--- End of"):
            j += 1
            continue
        break
    fr = _pick(frames, st.cwd) if frames else None
    if fr and fr.get("lib"):
        fr = None
    ctx = ["at " + fr["fn"]] if fr else []
    _emit(st, i, file=fr["f"] if fr else None, line=fr["l"] if fr else None, lang="csharp", code=t.rsplit(".", 1)[-1],
          message=msg, fk="runtime", context=ctx, extra=t)
    return j


# ------------------------------------------------------------------------------------------ dart

_DART_FRAME = re.compile(r"^#\d+\s+(?P<fn>.+?) \((?P<f>.+?):(?P<l>\d+)(?::(?P<c>\d+))?\)\s*$")


def _p_dart_runtime(st, i):
    L, n = st.L, st.n
    if L[i].strip() != "Unhandled exception:" or i + 1 >= n:
        return None
    em = re.match(r"^(?P<t>[A-Z]\w*)(?::\s?(?P<m>.*))?$", L[i + 1].strip())
    t = em.group("t") if em else None
    msg = (em.group("m") if em else L[i + 1].strip()) or t
    frames = []
    j = i + 2
    while j < n:
        fm = _DART_FRAME.match(L[j])
        if not fm:
            if frames:
                break
            j += 1
            if j - i > 6:
                break
            continue
        f = _norm_path(fm.group("f"))
        frames.append({"f": f, "l": int(fm.group("l")), "c": int(fm.group("c")) if fm.group("c") else None,
                       "fn": fm.group("fn"), "lib": _is_lib(fm.group("f"))})
        j += 1
    fr = _pick(frames, st.cwd) if frames else None
    _emit(st, i, file=fr["f"] if fr else None, line=fr["l"] if fr else None, col=fr.get("c") if fr else None,
          lang="dart", code=t, message=msg, fk="runtime", extra=t or "",
          context=["at " + _fmt_frame(x) for x in frames[:3]])
    return j


_DART_AN = re.compile(r"^\s*(?P<s>error|warning|info) [•-] (?P<f>.+?):(?P<l>\d+):(?P<c>\d+) [•-] (?P<m>.+?) [•-] (?P<code>\w+)\s*$")


def _p_dart_an(st, i):
    m = _DART_AN.match(st.L[i])
    if not m:
        return None
    _emit(st, i, file=m.group("f"), line=int(m.group("l")), col=int(m.group("c")),
          sev={"info": "note"}.get(m.group("s"), m.group("s")), lang="dart", code=m.group("code"),
          message=m.group("m"), fk="lint", extra=m.group("code").replace("_", " "))
    return i + 1


# -------------------------------------------------------------------------------------------- go

_GO_PANIC = re.compile(r"^(?:panic|fatal error): (?P<m>.*)$")
_GO_LOC = re.compile(r"^\t(?P<f>.+?\.(?:go|s)):(?P<l>\d+)(?: \+0x[0-9a-f]+)?\s*$")


def _p_go_panic(st, i):
    L, n = st.L, st.n
    m = _GO_PANIC.match(L[i])
    if not m:
        return None
    msg = m.group("m")
    frames, fn = [], None
    j = i + 1
    started = False
    while j < n and j - i < 400:
        s = L[j]
        if s.startswith("goroutine ") and s.rstrip().endswith(":"):
            if started:
                break
            started = True
            j += 1
            continue
        lm = _GO_LOC.match(s)
        if lm:
            f = lm.group("f")
            frames.append({"f": f, "l": int(lm.group("l")), "fn": fn, "lib": _is_lib(f) or "/src/runtime/" in f})
            j += 1
            continue
        if started and s.strip() and not s.startswith("\t"):
            fn = s.strip()
            j += 1
            continue
        if not started and (s.startswith(("[signal", "\tpanic:", "panic:")) or not s.strip()):
            j += 1
            continue
        if started and not s.strip():
            if frames:
                j += 1
                break
            j += 1
            continue
        break
    if not frames and not re.search(r"runtime error|goroutine|deadlock|nil map|concurrent map", msg):
        return None
    while j < n and re.match(r"^(?:exit status \d+|FAIL\s|\s*$)", L[j]) and j - i < 500:
        if L[j].startswith("FAIL"):
            j += 1
            break
        j += 1
    fr = _pick(frames, st.cwd) if frames else None
    test = any("testing.tRunner" in (x.get("fn") or "") for x in frames)
    msg = re.sub(r"\s*\[recovered\]\s*$", "", msg)
    _emit(st, i, file=fr["f"] if fr else None, line=fr["l"] if fr else None, lang="go", message=msg,
          fk="test" if test else "runtime", context=["at " + _fmt_frame(x) for x in frames if not x["lib"]][:3])
    return j


_GO_FAIL = re.compile(r"^\s*--- FAIL: (?P<t>\S+) \(")
_GO_TLINE = re.compile(r"^\s+(?P<f>\S+\.go):(?P<l>\d+): (?P<m>.*)$")


def _p_go_test(st, i):
    L, n = st.L, st.n
    m = _GO_FAIL.match(L[i])
    if not m:
        return None
    j = i + 1
    while j < n:
        tm = _GO_TLINE.match(L[j])
        if not tm:
            break
        _emit(st, j, file=tm.group("f"), line=int(tm.group("l")), lang="go", message=tm.group("m"), fk="test",
              context=[f"test: {m.group('t')}"], kind="test-failure")
        j += 1
    return j


_GO_C = re.compile(r"^(?:vet: )?(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?\.go):(?P<l>\d+):(?P<c>\d+): (?P<m>.+)$")


def _p_go_compile(st, i):
    m = _GO_C.match(st.L[i])
    if not m:
        return None
    _emit(st, i, file=m.group("f"), line=int(m.group("l")), col=int(m.group("c")), lang="go", message=m.group("m"),
          fk="compile")
    return i + 1


# ------------------------------------------------------------------------------------------ rust

_RP = re.compile(r"^thread '(?P<th>[^']*)' (?:\(\d+\) )?panicked at (?:'(?P<om>.*)', )?(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?):(?P<l>\d+):(?P<c>\d+):?\s*$")


def _p_rust_panic(st, i):
    L, n = st.L, st.n
    m = _RP.match(L[i])
    if not m:
        return None
    msgs = [m.group("om")] if m.group("om") else []
    j = i + 1
    while j < n and not m.group("om"):
        s = L[j].strip()
        if not s or s.startswith(("note: run with", "stack backtrace:", "note: Some details")):
            break
        msgs.append(s)
        j += 1
    frames = []
    while j < n and j - i < 200:
        s = L[j]
        if s.startswith("note: run with") or s.startswith("stack backtrace:"):
            j += 1
            continue
        fm = re.match(r"^\s+at (?P<f>.+?):(?P<l>\d+):(?P<c>\d+)\s*$", s)
        if fm:
            frames.append({"f": fm.group("f"), "l": int(fm.group("l")), "c": int(fm.group("c")), "lib": _is_lib(fm.group("f"))})
            j += 1
            continue
        if re.match(r"^\s+\d+: ", s):
            j += 1
            continue
        break
    loc = {"f": m.group("f"), "l": int(m.group("l")), "c": int(m.group("c")), "lib": _is_lib(m.group("f"))}
    fr = loc if not loc["lib"] else (_pick(frames, st.cwd) if frames else loc)
    test = m.group("th") != "main" or any(re.match(r"^---- .+ stdout ----", L[k]) for k in range(max(0, i - 3), i))
    msg = "; ".join(x for x in msgs if x)
    _emit(st, i, file=fr["f"], line=fr["l"], col=fr.get("c"), lang="rust", message=msg or "panicked",
          fk="test" if test else "runtime", context=[f"thread '{m.group('th')}' panicked"], extra="panic")
    return j


_RH = re.compile(r"^(?P<s>error|warning)(?:\[(?P<code>E\d{4})\])?: (?P<m>.+)$")
_RUST_NOISE = re.compile(r"aborting due to|could not compile|generated \d+ warnings?|build failed|previous errors?|"
                         r"warnings? emitted|Some errors have detailed", re.I)


def _p_rust(st, i):
    L, n = st.L, st.n
    m = _RH.match(L[i])
    if not m:
        return None
    msg, code = m.group("m"), m.group("code")
    loc = None
    j = i + 1
    for k in range(i + 1, min(i + 4, n)):
        lm = re.match(r"^\s*--> (?P<f>.+?):(?P<l>\d+):(?P<c>\d+)\s*$", L[k]) or \
            re.search(r"└─ (?P<f>\S+?):(?P<l>\d+)(?::(?P<c>\d+))?", L[k])
        if lm:
            loc = lm
            break
    if loc is None:
        if _RUST_NOISE.search(msg):
            return i + 1
        if not code and not st.rust:
            return None
    block = []
    while j < n and L[j].strip() and not _RH.match(L[j]):
        block.append(L[j])
        j += 1
    helps, sug, ctx = [], None, []
    for bi, b in enumerate(block):
        hm = re.search(r"(?:^|\s)(?:= )?help: (.+)$", b)
        if hm:
            h = hm.group(1).strip()
            bm = re.search(r"similar name exists[^`]*`([^`]+)`|with a similar name[^`]*`([^`]+)`", h)
            if bm and not sug:
                sug = bm.group(1) or bm.group(2)
            elif "similar name" in h and not sug:
                for bj in range(bi + 1, min(bi + 5, len(block)) - 1):
                    cm = re.match(r"^\s*\d+\s*\|", block[bj])
                    if cm:
                        code_t = block[bj].split("|", 1)[1][1:]
                        mark = block[bj + 1].split("|", 1)[1][1:] if "|" in block[bj + 1] else ""
                        mm = re.match(r"^(\s*)([~+^-]+)", mark)
                        if mm:
                            sug = code_t[len(mm.group(1)):len(mm.group(1)) + len(mm.group(2))].strip() or None
                        break
            else:
                helps.append(h)
        if loc is not None and re.match(r"^\s*" + loc.group("l") + r"\s*\|", b) and not ctx:
            ctx.append(b.split("|", 1)[1].strip())
            if bi + 1 < len(block) and "|" in block[bi + 1]:
                ctx.append(block[bi + 1].split("|", 1)[1].rstrip())
    file = loc.group("f") if loc else None
    lang = "rust" if (code or (file and file.endswith(".rs")) or (not file and st.rust)) else _lang_of(file)
    _emit(st, i, file=file, line=int(loc.group("l")) if loc else None,
          col=int(loc.group("c")) if loc and loc.group("c") else None, sev=m.group("s"), lang=lang, code=code,
          message=msg, fk="compile", context=ctx, suggestion=sug, extra_hint=helps[0] if helps and not sug else None)
    return j


# ------------------------------------------------------------------------------------ native crash

_ASAN = re.compile(r"^==\d+==\s*(?:ERROR|WARNING): (?P<tool>\w+Sanitizer): (?P<m>.*)$")
_ASAN_FRAME = re.compile(r"^\s+#\d+ 0x[0-9a-fA-F]+ in (?P<fn>.+?) (?P<f>(?:[A-Za-z]:)?[^\s:()][^:()]*?):(?P<l>\d+)(?::(?P<c>\d+))?\s*$")


def _p_asan(st, i):
    L, n = st.L, st.n
    m = _ASAN.match(L[i])
    if not m:
        return None
    msg = re.sub(r" on (?:unknown )?address 0x[0-9a-fA-F]+.*$", "", m.group("m"))
    frames, run_done = [], False
    j = i + 1
    extra = None
    end = None
    while j < n and j - i < 300:
        s = L[j]
        if s.startswith("SUMMARY:"):
            end = j + 1
            break
        am = re.match(r"^(READ|WRITE) of size (\d+)", s)
        if am and extra is None:
            extra = f"{am.group(1)} of size {am.group(2)}"
        fm = _ASAN_FRAME.match(s)
        if fm and not run_done:
            frames.append({"f": fm.group("f"), "l": int(fm.group("l")), "c": int(fm.group("c")) if fm.group("c") else None,
                           "fn": fm.group("fn"), "lib": _is_lib(fm.group("f"))})
        elif frames and not re.match(r"^\s+#\d+ ", s):
            run_done = True
        j += 1
    if end is None:
        end = j
    while end < n and re.match(r"^==\d+==", L[end]):
        end += 1
    fr = _pick(frames, st.cwd) if frames else None
    code = msg.split()[0] if msg else m.group("tool")
    if extra:
        msg = f"{msg} ({extra})"
    _emit(st, i, file=fr["f"] if fr else None, line=fr["l"] if fr else None, col=fr.get("c") if fr else None,
          lang=_lang_of(fr["f"], "cpp") if fr else "cpp", code=code, message=msg, fk="runtime",
          context=["at " + _fmt_frame(x) for x in frames[:3]])
    return end


_UBSAN = re.compile(r"^(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?):(?P<l>\d+):(?P<c>\d+): runtime error: (?P<m>.*)$")


def _p_ubsan(st, i):
    m = _UBSAN.match(st.L[i])
    if not m:
        return None
    _emit(st, i, file=m.group("f"), line=int(m.group("l")), col=int(m.group("c")),
          lang=_lang_of(m.group("f"), "cpp"), code="UBSan", message=m.group("m"), fk="runtime")
    return i + 1


_TERM = re.compile(r"^terminate called after throwing an instance of '(?P<t>.+?)'\s*$")
_LIBCXX = re.compile(r"^libc\+\+abi(?:\.dylib)?: terminating (?:due to|with) uncaught exception of type (?P<t>[\w:<>]+): (?P<m>.*)$")


def _p_terminate(st, i):
    L, n = st.L, st.n
    m = _TERM.match(L[i])
    if m:
        msg, j = m.group("t"), i + 1
        if j < n:
            wm = re.match(r"^\s*what\(\):\s*(.*)$", L[j])
            if wm:
                msg = wm.group(1)
                j += 1
        if j < n and re.search(r"Aborted|core dumped", L[j]):
            j += 1
        _emit(st, i, lang="cpp", code=m.group("t"), message=msg, fk="runtime", extra=m.group("t"),
              context=["uncaught C++ exception: wrap the call in try/catch or fix the failing operation"])
        return j
    m = _LIBCXX.match(L[i])
    if m:
        _emit(st, i, lang="cpp", code=m.group("t"), message=m.group("m"), fk="runtime", extra=m.group("t"))
        return i + 1
    if L[i].startswith("terminate called without an active exception"):
        _emit(st, i, lang="cpp", code="std::terminate", message=L[i].strip(), fk="runtime",
              hint="std::terminate: a joinable std::thread was destroyed (call join()) or `throw;` ran outside a catch")
        return i + 1
    return None


_CRASH = re.compile(r"(?P<m>Segmentation fault|Bus error|Floating point exception|Illegal instruction|Aborted"
                    r"|free\(\): .+|double free or corruption.*|munmap_chunk\(\): invalid pointer|malloc\(\): .+"
                    r"|corrupted (?:size|double-linked|top).*|\*\*\* stack smashing detected \*\*\*.*)(?: \(core dumped\))?")
_SIG = {"Segmentation fault": "SIGSEGV", "Bus error": "SIGBUS", "Floating point exception": "SIGFPE",
        "Illegal instruction": "SIGILL", "Aborted": "SIGABRT"}


def _p_crash(st, i):
    line = st.L[i]
    m = _CRASH.search(line)
    if not m:
        return None
    pre = line[:m.start()]
    if pre.strip() and not re.match(r"^(?:.*: line \d+: +\d+ |\S*\s*$|\[\d+\]\s+\d+ )", pre):
        return None
    msg = m.group("m").strip()
    code = _SIG.get(msg, "SIGABRT" if ("free" in msg or "corrupt" in msg or "smashing" in msg or "malloc" in msg) else None)
    if msg == "Aborted":
        _emit(st, i, lang="native", code="SIGABRT", message="Aborted (abort() was called)", fk="runtime", rank=5,
              kind="unknown", hint="abort(): read the lines above it (a failed assert or an uncaught exception) and fix that")
        return i + 1
    _emit(st, i, lang="native", code=code, message=msg, fk="runtime", extra=code or "")
    return i + 1


_LD_UNDEF = re.compile(r"undefined reference to [`'‘](?P<sym>.+?)['’]\s*$")
_LD_MULTI = re.compile(r"multiple definition of [`'‘](?P<sym>.+?)['’]")
_LD_SRC = re.compile(r"(?P<f>(?:[A-Za-z]:)?[\w.\\/-]+\.(?:c|cc|cpp|cxx|C)):(?:(?P<l>\d+)|\([^)]*\)): (?:undefined reference|multiple definition)")
_LNK = re.compile(r"^(?P<obj>.+?) : (?:fatal )?error (?P<code>LNK\d{4}): (?P<m>.*)$")
_LLD = re.compile(r"^(?:ld\.lld|lld-link|ld64\.lld|ld|ld\.exe)(?:\.exe)?: error: (?P<m>undefined symbol: (?P<sym>.+))$")


def _p_linker(st, i):
    L, n = st.L, st.n
    line = L[i]
    m = _LD_UNDEF.search(line) or _LD_MULTI.search(line)
    if m:
        sm = _LD_SRC.search(line)
        what = "undefined reference to" if "undefined reference" in line else "multiple definition of"
        _emit(st, i, file=sm.group("f") if sm else None,
              line=int(sm.group("l")) if sm and sm.group("l") else None, lang=_lang_of(sm.group("f"), "cpp") if sm else "cpp",
              message=f"{what} `{m.group('sym')}'", fk="compile", extra="linker")
        return i + 1
    m = _LNK.match(line)
    if m:
        _emit(st, i, lang="cpp", code=m.group("code"), message=m.group("m"), fk="compile",
              rank=5 if m.group("code") == "LNK1120" else 0, context=[m.group("obj")])
        return i + 1
    m = _LLD.match(line)
    if m:
        f = l = None
        j = i + 1
        while j < n and L[j].startswith(">>>"):
            rm = re.match(r"^>>> referenced by (?P<f>[^\s:()]+\.\w+):(?P<l>\d+)", L[j]) or \
                re.match(r"^>>> referenced by (?P<f>(?:[A-Za-z]:)?[^:()]+\.\w+):(?P<l>\d+)", L[j])
            if rm and f is None:
                f, l = rm.group("f"), int(rm.group("l"))
            j += 1
        _emit(st, i, file=f, line=l, lang=_lang_of(f, "cpp") if f else "cpp", message=m.group("m"), fk="compile", extra="linker")
        return j
    if line.startswith("Undefined symbols for architecture"):
        j = i + 1
        sym = None
        while j < n and (L[j].startswith(" ") or L[j].startswith("\t")):
            sm = re.match(r'^\s+"(?P<s>[^"]+)", referenced from:', L[j])
            if sm and sym is None:
                sym = sm.group("s")
            j += 1
        _emit(st, i, lang="cpp", message=f"undefined symbol {sym}" if sym else line.strip(), fk="compile", extra="linker")
        return j
    if re.search(r"collect2(?:\.exe)?: error: ld returned|linker command failed with exit code|ld: symbol\(s\) not found|"
                 r"cannot find -l\S+|error: linking with .+ failed", line):
        _emit(st, i, lang="cpp" if "linking with" not in line else "rust", message=line.strip(), fk="compile",
              rank=0 if "cannot find -l" in line else 5, kind="linker",
              hint="the library isn't found: install it or add -L<dir> for its folder, and check the -l name" if "cannot find -l" in line else None)
        return i + 1
    return None


# ------------------------------------------------------------------------ compiler line formats

_MSC = re.compile(r"^\s*(?P<f>[^\s(][^(]*?)\((?P<l>\d+)(?:,(?P<c>\d+))?(?:,\d+,\d+)?\)\s*:\s*(?P<s>fatal error|error|warning|note|info|message)"
                  r"\s+(?P<code>[A-Za-z]+\d+)\s*:\s*(?P<m>.*?)(?:\s+\[(?P<proj>[^\]]+)\])?\s*$")
_SEV = {"fatal error": "error", "error": "error", "Error": "error", "fatal": "error", "Fatal": "error",
        "warning": "warning", "Warning": "warning", "warn": "warning", "note": "note", "remark": "note",
        "Context": "note", "info": "note", "Info": "note", "information": "note", "message": "note"}


def _p_msvc(st, i):
    m = _MSC.match(st.L[i])
    if not m:
        return None
    f, code = m.group("f").strip(), m.group("code")
    if code.startswith("CS"):
        lang = "csharp"
    elif code.startswith("TS"):
        lang = _lang_of(f, "typescript")
        lang = lang if lang in ("javascript", "typescript") else "typescript"
    else:
        lang = _lang_of(f, "cpp")
    _emit(st, i, file=f, line=int(m.group("l")), col=int(m.group("c")) if m.group("c") else None,
          sev=_SEV.get(m.group("s"), "error"), lang=lang, code=code, message=m.group("m"),
          fk="compile", context=[m.group("proj")] if m.group("proj") else [])
    return i + 1


_SBT = re.compile(r"^\[(?P<s>error|warn)\] (?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?\.(?:scala|sc|java|kt)):(?P<l>\d+):(?P<c>\d+):? (?P<m>.*)$")
_KT1 = re.compile(r"^(?P<s>[ew]): (?P<f>(?:file:///?)?(?:[A-Za-z]:)?[^\s:][^:]*?\.kts?):(?P<l>\d+):(?P<c>\d+) (?P<m>.*)$")
_KT2 = re.compile(r"^(?P<s>[ew]): (?P<f>(?:[A-Za-z]:)?[^:]+?\.kts?): \((?P<l>\d+), (?P<c>\d+)\): (?P<m>.*)$")
_SC3 = re.compile(r"^-- (?:\[(?P<code>E\d+)\] )?(?P<title>.+?) (?P<s>Error|Warning): (?P<f>(?:[A-Za-z]:)?[^:]+?):(?P<l>\d+):(?P<c>\d+) -+\s*$")
_PRETTY = re.compile(r"^\s*(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?):(?P<l>\d+):(?P<c>\d+) - (?P<s>error|warning|information|info)"
                     r"(?: (?P<code>TS\d+))?: (?P<m>.*?)(?: \((?P<rule>report\w+)\))?\s*$")


def _p_jvm_misc(st, i):
    L, n = st.L, st.n
    line = L[i]
    m = _SBT.match(line)
    if m:
        _emit(st, i, file=m.group("f"), line=int(m.group("l")), col=int(m.group("c")), sev=_SEV[m.group("s")],
              message=m.group("m"), fk="compile")
        return i + 1
    m = _KT1.match(line) or _KT2.match(line)
    if m:
        _emit(st, i, file=m.group("f"), line=int(m.group("l")), col=int(m.group("c")),
              sev="error" if m.group("s") == "e" else "warning", lang="kotlin", message=m.group("m"), fk="compile")
        return i + 1
    m = _SC3.match(line)
    if m:
        msg, j = None, i + 1
        while j < n and j <= i + 12 and not _SC3.match(L[j]):
            bm = re.match(r"^\s*\|\s?(.*)$", L[j])
            if bm and msg is None:
                t = bm.group(1).strip()
                if t and set(t) - set("^~ "):
                    msg = t
            if not L[j].strip():
                break
            j += 1
        _emit(st, i, file=m.group("f"), line=int(m.group("l")), col=int(m.group("c")), sev=_SEV[m.group("s")],
              lang="scala", code=m.group("code"), message=msg or m.group("title"), fk="compile", extra=m.group("title"))
        return j
    m = _PRETTY.match(line)
    if m:
        f = m.group("f")
        lang = _lang_of(f)
        j = i + 1
        while j < n and j <= i + 4 and (re.match(r"^\d+\s", L[j]) or re.match(r"^\s+~+\s*$", L[j])):
            j += 1
        _emit(st, i, file=f, line=int(m.group("l")), col=int(m.group("c")), sev=_SEV.get(m.group("s"), "error"),
              lang=lang, code=m.group("code") or m.group("rule"), message=m.group("m"),
              fk="lint" if lang == "python" else "compile")
        return j
    return None


_FL = re.compile(r"^(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?\.pyi?):(?P<l>\d+):(?P<c>\d+): (?P<code>[A-Z]{1,3}\d{2,4}):? (?:\[\*\] )?(?P<m>.*)$")
_RUFF_FULL = re.compile(r"^(?P<code>[A-Z]{1,3}\d{3,4}) (?:\[\*\] )?(?P<m>.+)$")


def _lint_sev(code):
    if re.match(r"^(?:E9\d\d|F82\d|F63\d|F7\d\d|E\d{4}|F\d{4})$", code):
        return "error"
    return "warning"


def _p_pylint(st, i):
    L, n = st.L, st.n
    m = _FL.match(L[i])
    loc = None
    if m:
        f, l, c, code, msg = m.group("f"), m.group("l"), m.group("c"), m.group("code"), m.group("m")
        j = i + 1
    else:
        m = _RUFF_FULL.match(L[i])
        if not m or i + 1 >= n:
            return None
        loc = re.match(r"^\s*--> (?P<f>.+?\.pyi?):(?P<l>\d+):(?P<c>\d+)\s*$", L[i + 1])
        if not loc:
            return None
        f, l, c, code, msg = loc.group("f"), loc.group("l"), loc.group("c"), m.group("code"), m.group("m")
        j = i + 2
        while j < n and L[j].strip():
            j += 1
    msg = re.sub(r"\s\([a-z][a-z0-9-]+\)$", "", msg)
    _emit(st, i, file=f, line=int(l), col=int(c), sev=_lint_sev(code), lang="python", code=code, message=msg, fk="lint")
    return j


_REQ = re.compile(r"^(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?):(?P<l>\d+):(?P<c>\d+):\s+(?:required from|required by|recursively required|in instantiation)")
_UNI = re.compile(r"^(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?):(?P<l>\d+)(?::(?P<c>\d+))?(?:-\d+)?: "
                  r"(?P<s>fatal error|error|warning|note|remark|Error|Warning|Context|Info):?\s*(?:\[(?P<code>[A-Z]+-\d+)\]:?)?\s*(?P<m>.*)$")


def _p_required(st, i):
    m = _REQ.match(st.L[i])
    if not m:
        return None
    if not _is_lib(m.group("f")):
        st.required = (m.group("f"), int(m.group("l")), int(m.group("c")), i)
    return i + 1


def _p_unified(st, i):
    L, n = st.L, st.n
    m = _UNI.match(L[i])
    if not m:
        return None
    f, l = m.group("f"), int(m.group("l"))
    c = int(m.group("c")) if m.group("c") else None
    sev = _SEV.get(m.group("s"), "error")
    msg, code = m.group("m").strip(), m.group("code")
    lang = _lang_of(f)
    if lang == "python":
        cm = re.search(r"\s{2}\[([a-z][\w-]*)\]\s*$", msg)
        if cm:
            code, msg = cm.group(1), msg[:cm.start()].strip()
    j = i + 1
    ctx = []
    if not msg:
        parts = []
        while j < n and (L[j].startswith((" ", "\t")) or not L[j].strip()) and j - i < 20:
            t = L[j].strip().lstrip("•").strip()
            if not L[j].strip() and parts:
                break
            if t and len(parts) < 2 and not re.match(r"^\d*\s*\|", t):
                parts.append(t)
            j += 1
        msg = "; ".join(parts)
    else:
        while j < n and j - i <= 6 and re.match(r"^\s*\d*\s*\|", L[j]):
            if len(ctx) < 2:
                ctx.append(L[j].split("|", 1)[1].rstrip())
            j += 1
        if j == i + 1 and j + 1 < n and L[j].startswith((" ", "\t")) and re.match(r"^\s*[\^~]+\s*$", L[j + 1] or ""):
            ctx = [L[j].rstrip(), L[j + 1].rstrip()]
            j += 2
    if sev == "note" and st.last is not None and i - st.last_i <= 15:
        d = st.last
        if len(d.context) < 3:
            d.context.append(f"note: {f}:{l}: {msg}"[:200])
        sug, _ = _split_suggestion(msg)
        if sug and not d.suggestion:
            d.suggestion = sug
            d.hint += f" (did you mean '{sug}'?)"
        if d.file and _is_lib(d.file) and not _is_lib(f):
            d.context.insert(0, f"reported in {d.file}:{d.line}")
            d.context = d.context[:3]
            d.file, d.line, d.col = f, l, c
        return j
    if sev != "note" and _is_lib(f) and st.required and i - st.required[3] <= 30:
        ctx = [f"reported in {f}:{l}"] + ctx
        f, l, c = st.required[0], st.required[1], st.required[2]
        st.required = None
    fk = "lint" if lang == "python" else "compile"
    d = _emit(st, i, file=f, line=l, col=c, sev=sev, lang=lang, code=code, message=msg, fk=fk, context=ctx)
    st.last, st.last_i = d, i
    return j


# ---------------------------------------------------------------------------- scripting languages

_PHP = re.compile(r"^(?:PHP )?(?P<s>Fatal error|Parse error|Warning|Notice|Deprecated|Recoverable fatal error|Catchable fatal error)\s*:\s+"
                  r"(?P<m>.*?) in (?P<f>\S.*?)(?::(?P<l>\d+)| on line (?P<l2>\d+))\s*$")


def _p_php(st, i):
    L, n = st.L, st.n
    line = L[i]
    if "<b>" in line:
        line = re.sub(r"<[^>]+>", "", line)
    m = _PHP.match(line)
    if not m:
        return None
    msg, code = m.group("m"), None
    um = re.match(r"^Uncaught (?P<t>[\w\\]+): (?P<m>.*)$", msg)
    if um:
        code, msg = um.group("t").rsplit("\\", 1)[-1], um.group("m")
    j = i + 1
    while j < n and (L[j].startswith(("Stack trace:", "#", "  thrown in", "PHP Stack trace", "PHP   "))):
        j += 1
    s = m.group("s")
    sev = "warning" if s in ("Warning", "Notice", "Deprecated") else "error"
    _emit(st, i, file=m.group("f"), line=int(m.group("l") or m.group("l2")), sev=sev, lang="php", code=code, message=msg,
          fk="compile" if s == "Parse error" else "runtime", extra=s if s == "Deprecated" else "")
    return j


_RB1 = re.compile(r"^(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?\.rb):(?P<l>\d+):in [`'](?P<fn>[^'`]+)'?: (?P<m>.*?)(?: \((?P<t>[A-Z]\w*(?:::\w+)*)\))?\s*$")
_RB2 = re.compile(r"^(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?\.rb):(?P<l>\d+): (?P<m>.*(?:syntax error|SyntaxError).*)$")


def _p_ruby(st, i):
    L, n = st.L, st.n
    m = _RB1.match(L[i])
    if m:
        t = m.group("t")
        msg = m.group("m")
        j = i + 1
        while t is None and j < n and j <= i + 4 and not L[j].lstrip().startswith("from "):
            tm = re.search(r"\((?P<t>[A-Z]\w*(?:::\w+)*)\)\s*$", L[j])
            if tm:
                t = tm.group("t")
            j += 1
        frames = [{"f": m.group("f"), "l": int(m.group("l")), "fn": m.group("fn"), "lib": _is_lib(m.group("f"))}]
        while j < n:
            fm = re.match(r"^\s+from (?P<f>.+?):(?P<l>\d+):in [`'](?P<fn>[^'`]+)", L[j])
            if not fm:
                if not L[j].strip() or L[j].startswith((" ", "\t")):
                    j += 1
                    if j - i > 50:
                        break
                    continue
                break
            frames.append({"f": fm.group("f"), "l": int(fm.group("l")), "fn": fm.group("fn"), "lib": _is_lib(fm.group("f"))})
            j += 1
        fr = _pick(frames, st.cwd)
        _emit(st, i, file=fr["f"], line=fr["l"], lang="ruby", code=t, message=msg, fk="runtime", extra=t or "",
              context=["at " + _fmt_frame(x) for x in frames[:3]])
        return j
    m = _RB2.match(L[i])
    if m:
        _emit(st, i, file=m.group("f"), line=int(m.group("l")), lang="ruby", code="SyntaxError", message=m.group("m"),
              fk="compile", kind="syntax")
        return i + 1
    return None


_PL = re.compile(r"^(?P<m>.+?) at (?P<f>\S+?\.(?:pl|pm|t|cgi)|-e) line (?P<l>\d+)(?P<rest>[,.].*)?$")


def _p_perl(st, i):
    m = _PL.match(st.L[i])
    if not m:
        return None
    msg = m.group("m")
    near = re.search(r'near "(.*?)"?$', m.group("rest") or "")
    if near:
        msg += f' (near "{near.group(1)}")'
    warn = re.match(r"^(?:Use of uninitialized|Useless use|Name .+ used only once|Scalar value|Possible attempt)", msg)
    fk = "compile" if re.search(r"syntax error|Global symbol|Can't locate", msg) else "runtime"
    _emit(st, i, file=m.group("f"), line=int(m.group("l")), sev="warning" if warn else "error", lang="perl", message=msg, fk=fk)
    return i + 1


_LUA = re.compile(r"^(?:\S*lua(?:jit)?[\d.]*(?:\.exe)?: )?(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?\.lua):(?P<l>\d+): (?P<m>.+)$")


def _p_lua(st, i):
    L, n = st.L, st.n
    m = _LUA.match(L[i])
    if not m or m.group("m").startswith("in "):
        return None
    j = i + 1
    if j < n and L[j].startswith("stack traceback:"):
        j += 1
        while j < n and (L[j].startswith("\t") or L[j].startswith("[C]")):
            j += 1
    _emit(st, i, file=m.group("f"), line=int(m.group("l")), lang="lua", message=m.group("m"), fk="runtime")
    return j


_SWF = re.compile(r"^(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?\.swift):(?P<l>\d+): Fatal error: (?P<m>.*)$")
_SWF2 = re.compile(r"^Fatal error: (?P<m>.*?): file (?P<f>\S+), line (?P<l>\d+)\s*$")


def _p_swift(st, i):
    m = _SWF.match(st.L[i]) or _SWF2.match(st.L[i])
    if not m:
        return None
    f = m.group("f")
    _emit(st, i, file=None if _is_lib(f) else f, line=None if _is_lib(f) else int(m.group("l")), lang="swift",
          code="Fatal error", message=m.group("m"), fk="runtime", context=[f"{f}:{m.group('l')}"])
    return i + 1


_EX = re.compile(r"^\*\* \((?P<t>[\w.]+)\) (?P<m>.*)$")
_EX_FRAME = re.compile(r"^\s+(?:\((?P<app>[\w .-]+)\) )?(?P<f>(?:[A-Za-z]:)?[^\s:][^:]*?\.exs?):(?P<l>\d+): (?P<fn>.*)$")


def _p_elixir(st, i):
    L, n = st.L, st.n
    m = _EX.match(L[i])
    if not m:
        return None
    t, msg = m.group("t"), m.group("m")
    frames = []
    lm = re.match(r"^(?P<f>\S+\.exs?):(?P<l>\d+):(?:(?P<c>\d+):)? (?P<m>.*)$", msg)
    if lm:
        frames.append({"f": lm.group("f"), "l": int(lm.group("l")), "fn": None, "lib": False})
        msg = lm.group("m")
    j = i + 1
    while j < n:
        fm = _EX_FRAME.match(L[j])
        if not fm:
            break
        app = (fm.group("app") or "").split(" ")[0]
        frames.append({"f": fm.group("f"), "l": int(fm.group("l")), "fn": fm.group("fn"),
                       "lib": app in ("elixir", "stdlib", "kernel", "ex_unit", "mix", "iex") or _is_lib(fm.group("f"))})
        j += 1
    fr = _pick(frames, st.cwd) if frames else None
    _emit(st, i, file=fr["f"] if fr else None, line=fr["l"] if fr else None, lang="elixir", code=t, message=msg,
          fk="compile" if t == "CompileError" else "runtime", extra=t)
    return j


# ------------------------------------------------------------------------------------------ shells

_BASH = re.compile(r"^(?P<f>[^\s:][^:]*?): line (?P<l>\d+): (?P<m>.+)$")
_BASH2 = re.compile(r"^(?P<sh>-?(?:ba|z|k|da)?sh): (?:(?P<n>\d+): )?(?P<m>.+)$")


def _p_bash(st, i):
    L, n = st.L, st.n
    m = _BASH.match(L[i])
    if m:
        f = m.group("f")
        if re.match(r"^-?(?:/\S*/)?(?:ba|z|k|da)?sh$", f):
            f = None
        j = i + 1
        ctx = []
        if j < n:
            em = re.match(r"^[^:]+: line \d+: [`'](?P<src>.*)'$", L[j])
            if em:
                ctx.append(em.group("src"))
                j += 1
        msg = m.group("m")
        _emit(st, i, file=f, line=int(m.group("l")), lang="shell", message=msg,
              fk="compile" if "syntax error" in msg else "runtime", context=ctx)
        return j
    m = _BASH2.match(L[i])
    if m:
        msg = m.group("m")
        kind = "undefined-name" if re.search(r"command not found|: not found$", msg) else None
        _emit(st, i, line=int(m.group("n")) if m.group("n") else None, lang="shell", message=msg, fk="runtime", kind=kind)
        return i + 1
    return None


_PS_AT = re.compile(r"^At (?P<f>.+?):(?P<l>\d+) char:(?P<c>\d+)\s*$")
_PS7 = re.compile(r"^(?:(?P<cmd>[^\s:][^:]*?): )?(?P<f>(?:[A-Za-z]:)?[^:]*?\.ps[dm]?1):(?P<l>\d+)\s*$")
_PS_TERM = re.compile(r"The term '(?P<n>.+?)' is not recognized as (?:the|a) name of a cmdlet")


def _p_powershell(st, i):
    L, n = st.L, st.n
    line = L[i]
    m = _PS_AT.match(line)
    if m:
        k = i - 1
        msg_lines = []
        while k >= 0 and L[k].strip() and not st.consumed[k] and len(msg_lines) < 4:
            msg_lines.append(L[k].strip())
            k -= 1
        msg_lines.reverse()
        msg = re.sub(r"^[\w.-]+ : ", "", " ".join(msg_lines))
        j = i + 1
        code, ctx = None, []
        while j < n and L[j].lstrip().startswith("+"):
            s = L[j].strip()
            qm = re.match(r"^\+ FullyQualifiedErrorId : (\S+)", s)
            if qm:
                code = qm.group(1).split(",")[0]
            elif not s.startswith("+ CategoryInfo") and len(ctx) < 2:
                ctx.append(s[1:].rstrip())
            j += 1
        _emit(st, k + 1, file=m.group("f"), line=int(m.group("l")), col=int(m.group("c")), lang="powershell", code=code,
              message=msg or "error", fk="runtime", context=ctx)
        return j
    m = _PS7.match(line)
    if m and i + 1 < n and L[i + 1].strip() == "Line |":
        j = i + 2
        col, msg, src = None, [], None
        while j < n and "|" in L[j] and j - i < 20:
            left, right = L[j].split("|", 1)
            body = right[1:] if right.startswith(" ") else right
            if left.strip().isdigit():
                src = body.strip()
            elif re.match(r"^\s*~+", body) and col is None:
                col = len(body) - len(body.lstrip()) + 1
            elif body.strip():
                msg.append(body.strip())
            j += 1
        _emit(st, i, file=m.group("f"), line=int(m.group("l")), col=col, lang="powershell", message=" ".join(msg) or "error",
              fk="runtime", context=[src] if src else [])
        return j
    m = _PS_TERM.search(line)
    if m:
        for k in range(i + 1, min(i + 6, n)):
            if _PS_AT.match(L[k]) or (k + 1 < n and _PS7.match(L[k]) and L[k + 1].strip() == "Line |"):
                return None
        _emit(st, i, lang="powershell", code="CommandNotFoundException", message=re.sub(r"^[\w.-]+ : ", "", line.strip()),
              fk="runtime", kind="undefined-name")
        return i + 1
    return None


_CMD = re.compile(r"^'(?P<n>.+?)' is not recognized as an internal or external command")


def _p_cmd(st, i):
    L, n = st.L, st.n
    m = _CMD.match(L[i])
    if not m:
        return None
    j = i + 1
    if j < n and L[j].startswith("operable program or batch file"):
        j += 1
    _emit(st, i, lang="cmd", message=f"'{m.group('n')}' is not recognized as an internal or external command",
          fk="runtime", kind="undefined-name")
    return j


# --------------------------------------------------------------------------------------------- sql

_SQ1 = re.compile(r"^(?P<s>Error|Parse error|Runtime error)(?::| near) ?(?:near )?line (?P<l>\d+): (?P<m>.*)$")
_SQ2 = re.compile(r'^(?:Error|Parse error|Runtime error): (?P<m>(?:in prepare, )?(?:no such (?:table|column|function)|near ".*?": syntax error|.*constraint failed|table .+ already exists|incomplete input|unrecognized token).*)$')
_PG = re.compile(r"^(?:psql:(?P<f>.+?):(?P<l>\d+): )?(?P<s>ERROR|FATAL|WARNING|PANIC):(?P<sp>\s+)(?:(?P<code>[0-9A-Z]{5}): )?(?P<m>.*)$")
_PG_SQLISH = re.compile(r"column|relation|syntax error at or near|does not exist|violates|duplicate key|permission denied for|"
                        r"division by zero|invalid input syntax|function .+\(|operator does not exist|database|role", re.I)
_MY = re.compile(r"^ERROR (?P<code>\d{4}) \((?P<state>[0-9A-Z]{5})\)(?: at line (?P<l>\d+))?: (?P<m>.*)$")
_MS = re.compile(r"^Msg (?P<code>\d+), Level (?P<lvl>\d+), State \d+, (?:(?:Procedure|Server) (?P<proc>.+?), )?Line (?P<l>\d+)\s*$")
_PIP = re.compile(r"^ERROR: (?P<m>Could not find a version that satisfies the requirement .+|No matching distribution found for .+|"
                  r"ResolutionImpossible.*|Cannot install .+ because these package versions have conflicting dependencies\.?)$")


def _p_sql(st, i):
    L, n = st.L, st.n
    line = L[i]
    m = _SQ1.match(line)
    if m or _SQ2.match(line):
        m = m or _SQ2.match(line)
        msg = re.sub(r"^in prepare, |\s*\(\d+\)$", "", m.group("m"))
        j = i + 1
        ctx = []
        while j < n and j <= i + 2 and (re.search(r"\^--- error here", L[j]) or (j == i + 1 and L[j].startswith("  "))):
            ctx.append(L[j].rstrip())
            j += 1
        _emit(st, i, line=int(m.group("l")) if "l" in m.groupdict() and m.group("l") else None, lang="sql", message=msg,
              fk="runtime", context=ctx)
        return j
    m = _PIP.match(line)
    if m:
        _emit(st, i, lang="python", code="pip", message=m.group("m"), fk="compile", kind="dependency-resolution")
        return i + 1
    m = _PG.match(line)
    if m:
        nxt = L[i + 1] if i + 1 < n else ""
        if not (m.group("f") or len(m.group("sp")) >= 2 or nxt.startswith("LINE ") or _PG_SQLISH.search(m.group("m"))):
            return None
        j = i + 1
        line_no, col, sug, ctx = None, None, None, []
        while j < n and j <= i + 6:
            s = L[j]
            lm = re.match(r"^LINE (\d+): (.*)$", s)
            if lm:
                line_no = int(lm.group(1))
                ctx.append(lm.group(2))
                if j + 1 < n and re.match(r"^\s*\^\s*$", L[j + 1]):
                    col = L[j + 1].index("^") - len(f"LINE {lm.group(1)}: ") + 1
                    j += 1
            elif s.startswith(("HINT:", "DETAIL:", "CONTEXT:", "QUERY:")):
                hm = re.search(r'Perhaps you meant to reference the column "([^"]+)"', s)
                if hm:
                    sug = hm.group(1)
                elif len(ctx) < 3:
                    ctx.append(s.strip())
            else:
                break
            j += 1
        file = m.group("f")
        line = int(m.group("l")) if m.group("l") else line_no
        _emit(st, i, file=file, line=line, col=col, sev="warning" if m.group("s") == "WARNING" else "error", lang="sql",
              code=m.group("code"), message=m.group("m"), fk="runtime", context=ctx, suggestion=sug)
        return j
    m = _MY.match(line)
    if m:
        l = m.group("l")
        if not l:
            lm = re.search(r"at line (\d+)\s*$", m.group("m"))
            l = lm.group(1) if lm else None
        _emit(st, i, line=int(l) if l else None, lang="sql", code=m.group("code"), message=m.group("m"), fk="runtime")
        return i + 1
    m = _MS.match(line)
    if m:
        msg = L[i + 1].strip() if i + 1 < n else ""
        _emit(st, i, file=m.group("proc"), line=int(m.group("l")), sev="note" if int(m.group("lvl")) <= 10 else "error",
              lang="sql", code=m.group("code"), message=msg, fk="runtime")
        return i + 2
    return None


# ------------------------------------------------------------------------------ browser / http

_CSS = re.compile(r"(?P<m>.*?(?:Declaration dropped|Ruleset ignored|Error in parsing value|Unknown property|Expected .+ but found).*?)\.?"
                  r"(?:\s+(?P<f>\S+\.(?:css|scss|less|html?)):(?P<l>\d+)(?::(?P<c>\d+))?)?\s*$")
_HTTP_RES = [
    re.compile(r"(?:status of|status code:?|responded with a status of)\s+(?P<st>[45]\d\d)", re.I),
    re.compile(r"\bHTTP/\d(?:\.\d)?\s+(?P<st>[45]\d\d)\b"),
    re.compile(r"\b(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS) (?P<url>https?://\S+|/\S*) (?:net::ERR_\w+ )?(?P<st>[45]\d\d)\b"),
    re.compile(r"\b(?P<st>[45]\d\d) (?:Client|Server) Error"),
    re.compile(r"returned error: (?P<st>[45]\d\d)"),
]


def _p_web(st, i):
    line = st.L[i].strip()
    if re.search(r"blocked by CORS policy|Cross-Origin Request Blocked", line):
        um = re.search(r"at '([^']+)'", line)
        _emit(st, i, lang="javascript", code="CORS", message=line, fk="runtime", kind="cors",
              context=[um.group(1)] if um else [])
        return i + 1
    if line.startswith("Mixed Content:"):
        _emit(st, i, lang="javascript", code="MixedContent", message=line[len("Mixed Content:"):].strip(), fk="runtime",
              extra="mixed content")
        return i + 1
    m = re.match(r"^curl: \((?P<code>\d+)\) (?P<m>.*)$", line)
    if m:
        _emit(st, i, lang="http", code=f"curl {m.group('code')}", message=m.group("m"), fk="runtime")
        return i + 1
    if re.search(r"Declaration dropped|Ruleset ignored|Error in parsing value|Unknown property", line):
        m = _CSS.match(line)
        if m:
            _emit(st, i, file=m.group("f"), line=int(m.group("l")) if m.group("l") else None,
                  col=int(m.group("c")) if m.group("c") else None, sev="warning", lang="css", message=m.group("m"),
                  fk="lint", kind="syntax",
                  hint="the browser dropped this CSS declaration: check the property name and the value's spelling/units (color: red; width: 10px;)")
            return i + 1
    for rx in _HTTP_RES:
        m = rx.search(line)
        if m:
            st_code = m.group("st")
            _emit(st, i, lang="http", code=st_code, message=line[:300], fk="runtime", kind="http-status",
                  context=[m.group("url")] if "url" in m.groupdict() and m.group("url") else [])
            return i + 1
    m = re.search(r"net::(ERR_[A-Z_]+)", line)
    if m:
        _emit(st, i, lang="javascript", code=m.group(1), message=line[:300], fk="runtime")
        return i + 1
    return None


# ------------------------------------------------------------------------------------ generic

_GEN_LOC = re.compile(r"(?<![/\w])(?P<f>(?:[A-Za-z]:[\\/])?[\w.\\/-]*\w\.(?P<ext>[A-Za-z][A-Za-z0-9+]{0,5}))"
                      r"(?::(?P<l>\d+)(?::(?P<c>\d+))?|\((?P<l2>\d+)(?:,(?P<c2>\d+))?\)|,? line (?P<l3>\d+))")
_ERRISH = re.compile(r"\b(?:error|exception|fatal|failed|failure|panic|abort(?:ed)?|cannot|can't|unable|invalid|undefined|denied|not found|warning)\b", re.I)


def _p_generic(st, i):
    line = st.L[i]
    em = _ERRISH.search(line)
    if not em:
        return None
    m = _GEN_LOC.search(line)
    if not m or m.group("ext").lower() in ("com", "org", "net", "io", "exe", "dll", "so", "o", "obj"):
        lm = re.search(r"\bline (\d+)\b", line, re.I)
        if not lm:
            return None
        f, l, c = None, int(lm.group(1)), None
    else:
        f = m.group("f")
        l = int(m.group("l") or m.group("l2") or m.group("l3"))
        c = m.group("c") or m.group("c2")
        c = int(c) if c else None
    warn = em.group(0).lower() == "warning" and not re.search(r"\berror\b", line, re.I)
    msg = line.strip()
    if m and f:
        rest = line[m.end():].strip(" :-")
        if len(rest) > 8:
            msg = rest
    _emit(st, i, file=f, line=l, col=c, sev="warning" if warn else "error", lang=_lang_of(f), message=msg, fk="runtime")
    return i + 1


_PARSERS = [
    _p_python, _p_pytest, _p_pytest_sum, _p_jest, _p_vitest, _p_node_caret, _p_deno, _p_v8, _p_node_warn, _p_npm,
    _p_eslint, _p_java, _p_csharp, _p_dart_runtime, _p_go_panic, _p_go_test, _p_rust_panic, _p_asan, _p_ubsan,
    _p_terminate, _p_crash, _p_linker, _p_msvc, _p_maven, _p_jvm_misc, _p_javac, _p_dart_an, _p_pylint, _p_required,
    _p_unified, _p_go_compile, _p_rust, _p_php, _p_ruby, _p_perl, _p_lua, _p_swift, _p_elixir, _p_powershell, _p_cmd,
    _p_bash, _p_sql, _p_web, _p_generic,
]

# The prefilter runs on every line, so it is split for speed: literal words on the lower-cased line,
# a file-extension pattern, and a few line-start shapes (a single (?i) alternation cost ~30 us/line).
_TRIG_WORDS = re.compile(
    r"error|warn|exception|traceback|panic|fatal|fail|line \d|segmentation|abort|denied"
    r"|not found|recogni[sz]ed|undefined|cors|status (?:of|code)|http/\d|\*\* \(|-->|0xc0|3221\d|-1073\d|●|❯|×|✗"
    r"|npm |mixed content|dropped|no such|syntax|unbound|goroutine|sanitizer|terminat|thread '|refused|timed out|double free"
    r"|corrupt|smashing|illegal|bus error|floating point|unhandled|uncaught|lnk\d|collect2|ld:|core dumped"
    r"|unresolved|what\(\)|is not|cannot|can't|unable|invalid")
_TRIG_EXT = re.compile(r"\.[a-z][a-z0-9+]{0,5}(?:[:(]\d|\s*$)")
_TRIG_START = re.compile(r"\s*(?:at |file \"|File \")|[Mm]sg \d|[ew]: |-- |curl:|[A-Z]{1,3}\d{3,4} ")


def _trigger(line):
    low = line.lower()
    return bool(_TRIG_WORDS.search(low) or _TRIG_EXT.search(low) or _TRIG_START.match(line))

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")

_WIN = [
    ("0xC0000005", 3221225477, "access-violation", "access violation",
     "Windows access violation (0xC0000005): a null or dangling pointer, or an out-of-bounds write; check pointers and array indexes just before the crash, or run it under a debugger / AddressSanitizer"),
    ("0xC00000FD", 3221225725, "stack-overflow", "stack overflow",
     "Windows stack overflow (0xC00000FD): unbounded recursion or a huge local array; check the recursion's base case, or put big arrays on the heap (std::vector)"),
    ("0xC0000094", 3221225620, "division-by-zero", "integer divide by zero",
     "integer divide by zero (0xC0000094): check the divisor before dividing"),
    ("0xC0000409", 3221226505, "access-violation", "stack buffer overrun / fast fail",
     "fast-fail (0xC0000409): usually abort() from an uncaught C++ exception or failed assert, or a buffer overrun caught by /GS; wrap main in try/catch and print e.what(), and check array writes"),
    ("0xC0000374", 3221226356, "access-violation", "heap corruption",
     "heap corruption (0xC0000374): a write past the end of a heap buffer or a double delete; check new[]/delete[] pairs and buffer sizes"),
    ("0xC0000135", 3221225781, "import", "DLL not found",
     "a required DLL was not found (0xC0000135): put the runtime DLL next to the .exe or on PATH, or link statically"),
]
_WIN_RES = [(h, re.compile(rf"(?<![\w-]){u}(?!\d)|(?<!\d){u - 2 ** 32}(?!\d)|{h}(?![0-9A-Fa-f])", re.I), k, name, hint)
            for h, u, k, name, hint in _WIN]
_POSIX_EXIT = re.compile(r"(?:exit(?:ed)?(?: with)?(?: code| status)?|returned|status)\s*:?\s*(?P<c>139|134|136)\b", re.I)


def _exit_codes(st, text):
    if any(d.frame_kind in ("runtime", "test") and d.severity == "error" and d.rank < 5 for d in st.out):
        return
    for h, rx, kind, name, hint in _WIN_RES:
        m = rx.search(text)
        if m:
            _emit(st, text.count("\n", 0, m.start()), lang="native", code=h, message=f"process crashed with {h} ({name})",
                  fk="runtime", kind=kind, hint=hint)
    if not any(d.code and d.code.startswith("0xC") for d in st.out):
        m = _POSIX_EXIT.search(text)
        if m:
            sig = {"139": ("SIGSEGV", "access-violation"), "134": ("SIGABRT", "unknown"), "136": ("SIGFPE", "division-by-zero")}[m.group("c")]
            _emit(st, text.count("\n", 0, m.start()), lang="native", code=sig[0],
                  message=f"process killed by {sig[0]} (exit {m.group('c')})", fk="runtime", kind=sig[1])


_LONE = re.compile(r"^(?:Uncaught )?(?P<t>[A-Z][\w.]*(?:Error|Exception))(?::\s*(?P<m>.*))?$")
_LONE2 = re.compile(r"^(?:error|fatal)(?:\[\w+\])?: (?P<m>.+)$", re.I)
_PY_BUILTIN = {"NameError", "TypeError", "ValueError", "KeyError", "IndexError", "AttributeError", "ZeroDivisionError",
               "ModuleNotFoundError", "ImportError", "FileNotFoundError", "RecursionError", "UnboundLocalError"}


def _last_resort(st):
    for i, raw in enumerate(st.L):
        s = raw.strip()
        m = _LONE.match(s)
        if m:
            t = m.group("t")
            _emit(st, i, lang="python" if t in _PY_BUILTIN else "unknown", code=t.rsplit(".", 1)[-1],
                  message=m.group("m") or t, fk="runtime", extra=t)
        else:
            m = _LONE2.match(s)
            if m:
                _emit(st, i, message=m.group("m"), fk="compile")
        if len(st.out) >= 5:
            break


_SEV_ORDER = {"error": 0, "warning": 1, "note": 2}


def _norm_msg(m):
    return re.sub(r"\s+", " ", (m or "").lower()).strip(" .")


def _finalize(out):
    if any(d.severity == "error" and d.rank < 5 for d in out):
        out = [d for d in out if d.rank < 5]
    lined = {_norm_msg(d.message) for d in out if d.line}
    seen, res = set(), []
    for d in sorted(out, key=lambda d: (_SEV_ORDER.get(d.severity, 3), d.rank, d.pos)):
        nm = _norm_msg(d.message)
        base = re.split(r"[\\/]", d.file)[-1].lower() if d.file else None
        key = (base, d.line, nm[:120])
        if key in seen:
            continue
        if d.line is None and nm in lined:
            continue
        seen.add(key)
        res.append(d)
        if len(res) >= MAX_DIAGS:
            break
    return res


def _parse(text, cwd):
    text = _ANSI.sub("", text).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = []
    for raw in text.split("\n"):
        raw = raw[:MAX_LINE]
        if ">" in raw[:6]:
            raw = re.sub(r"^\s*\d{1,3}>(?=\S)", "", raw)
        lines.append(raw)
    st = _St(lines, cwd, text)
    i, n = 0, st.n
    while i < n:
        line = lines[i]
        if not line.strip() or not _trigger(line):
            i += 1
            continue
        nxt = None
        for p in _PARSERS:
            try:
                nxt = p(st, i)
            except Exception:
                nxt = None
            if nxt is not None:
                break
        if nxt is None or nxt <= i:
            i += 1
        else:
            nxt = min(nxt, n)
            st.consumed[i:nxt] = b"\x01" * (nxt - i)
            i = nxt
        if len(st.out) > 1000:      # 30 are returned; errors come first, so a longer tail changes nothing
            break
    try:
        _exit_codes(st, text)
        if not st.out:
            _last_resort(st)
    except Exception:
        pass
    return _finalize(st.out)


# ------------------------------------------------------------------------------------- public API

def parse(text: str, cwd: str | None = None) -> list[Diag]:
    """Every diagnosis in a tool's output, most important first (errors, then warnings, then notes; cap 30)."""
    if text is None:
        return []
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("utf-8", "replace")
    if not isinstance(text, str):
        text = str(text)
    try:
        return _parse(text, cwd)
    except Exception:
        return []


def primary(diags: list[Diag]) -> Diag | None:
    """The diagnosis to fix first: the first error (runtime ones already sit on the user's own frame)."""
    if not diags:
        return None
    for d in diags:
        if d.severity == "error":
            return d
    return diags[0]


def render(diags: list[Diag], limit: int = 5) -> str:
    out = []
    for d in diags[:limit]:
        loc = d.file or "?"
        if d.line:
            loc += f":{d.line}"
            if d.col:
                loc += f":{d.col}"
        code = f" [{d.code}]" if d.code else ""
        out.append(f"{loc} {d.severity}{code} {d.message} — kind: {d.kind}. Fix: {d.hint}")
    if len(diags) > limit:
        out.append(f"... and {len(diags) - limit} more")
    return "\n".join(out)
