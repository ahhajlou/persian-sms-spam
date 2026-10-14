"""Differential test: the Rust core against the Python pipeline it replaces.

    make core-diff                       # uv run --project common --group reference python core/tools/core_diff.py

The reference is the Python implementation as it was before the flip (default: commit
REF_REV), read straight from git, so this keeps working after the Python rules are deleted:
  anonymize / urls / numbers   spam_common.anonymize and spam_common.pipeline
  normalize `full`             hazm's own Normalizer
  normalize `rules`            the export script's port of hazm run with empty word lists
  ngrams                       sklearn's char_wb analyzer

Each stage is compared on its own, fed the reference's output of the previous stage, so one
difference does not hide the stages after it. Corpora: the fixture battery, a seeded fuzz
generator, and every message in data/**/*.jsonl. Inputs are cut to MAX_CHARS first (part of
v5). The report is counts only; for local data it also lists the characters that differ
(code points and names, no surrounding text). Synthetic inputs can be shown with --show.

Known, intended differences (see docs/preprocessing-core-plan.md, "Phase 2 results"):
  - digits of other scripts (fullwidth, Devanagari, ...): the anonymizer's classes are
    [0-9], where Python's \\d matched every Nd digit;
  - U+0130 and U+0131: Python's re.I lets them match i, Rust's (?i) does not.
A difference counts as intended only if it disappears when those characters are replaced
(digits by ASCII digits, İ by I, ı by i). Everything else fails the run (exit 1).
"""
import argparse
import difflib
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

import spam_core
from sklearn.feature_extraction.text import TfidfVectorizer

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REV = "45fe347"  # Phase 1 commit: the last one with the Python pipeline
STAGES = ("anonymize", "counts", "urls", "numbers", "normalize-rules", "normalize-full", "ngrams")


# --- the reference, from git -------------------------------------------------

def load_reference(rev, workdir):
    def show(path):
        return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO, check=True, capture_output=True, text=True).stdout

    files = {
        "ref_anonymize": show("common/src/spam_common/anonymize.py"),
        "ref_pipeline": show("common/src/spam_common/pipeline.py").replace(
            "from spam_common.anonymize import", "from ref_anonymize import"),
        "ref_export": show("classic/export_android_assets.py").replace(
            "from spam_common.pipeline import", "from ref_pipeline import"),
    }
    modules = {}
    sys.path.insert(0, str(workdir))
    for name, source in files.items():
        (workdir / f"{name}.py").write_text(source, "utf-8")
        spec = importlib.util.spec_from_file_location(name, workdir / f"{name}.py")
        modules[name] = module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return modules["ref_anonymize"], modules["ref_pipeline"], modules["ref_export"]


# --- corpora -----------------------------------------------------------------

def battery():
    path = REPO / "core/spam-core/tests/cases/battery.json"
    return json.loads(path.read_text("utf-8"))


FUZZ_POOLS = [
    "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیيكىةۀأإؤئء",
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "،؛؟!?:.»«()[]{}%\"'/\\-_+=*#&~@$^|<>",
    "0123456789", "۰۱۲۳۴۵۶۷۸۹", "٠١٢٣٤٥٦٧٨٩", "０１２３４５६７८९०१२३", "०१२३४५६७८९",
    " " * 6 + "\t\n\r   　​\x1c\x85 ",
    "‌‍‎‏‪‫‬ـًٌٍَُِّْٰ﻿",
    "😀👍🏽İΣẞßǅﷲﷺ",
]
FUZZ_TOKENS = [
    "http://x.ir", "https://a.b.com/p?q=1", "www.t.com/a", "x.ir", "t.me/c", "a@b.com", "x@y", "IR820540102680020817909002",
    "6037997599999993", "6037-9975-1234-5678", "603799******1234", "0023191589", "09121234567", "+989121234567",
    "021-88776655", "123.4567.12345678.1", "vless://id@host:443",
    "رمز", "رمز عبور:", "کد", "Code:", "password", "نام کاربری", "حساب", "مانده:", "برداشت", "پیگیری", "@d.com #",
    "می", "نمی", "ها", "های", "ترین", "گری", "ام", "اش", "ند", "خانه", "کتاب", "سلام", "تلفن", "میز",
    "...", "،", " ", " ", "\n", "\n\n\n", "  ",
]


def fuzz(n, seed=20261014):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 14)):
            if rng.random() < 0.45:
                parts.append(rng.choice(FUZZ_TOKENS))
            else:
                pool = rng.choice(FUZZ_POOLS)
                parts.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 8))))
        out.append("".join(parts))
    return out


def local_data():
    seen, out = set(), []
    for path in sorted((REPO / "data").rglob("*.jsonl")):
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            text = json.loads(line).get("text")
            if isinstance(text, str) and text not in seen:
                seen.add(text)
                out.append(text)
    return out


# --- comparison --------------------------------------------------------------

def sanitize(text):
    """The text without the characters behind the known, intended differences."""
    out = []
    for c in text:
        if unicodedata.category(c) == "Nd" and not (c in "0123456789" or "\u06f0" <= c <= "\u06f9" or "\u0660" <= c <= "\u0669"):
            out.append(str(unicodedata.digit(c)))
        else:
            out.append({"İ": "I", "ı": "i"}.get(c, c))
    return "".join(out)


def name_of(c):
    return f"U+{ord(c):04X} {unicodedata.name(c, '?')}"


class Checker:
    def __init__(self, ref, rust, hazm_full, empty_res, analyzer):
        self.ref_anon, self.ref_pipe, self.ref_export = ref
        self.rules_norm, self.full_norm = rust
        self.hazm_full = hazm_full
        self.res, self.words = empty_res
        self.analyzer = analyzer

    def stages(self, text):
        """{stage: (python, rust)} for the stages that differ."""
        out = {}

        def add(stage, ref, rust):
            if ref != rust:
                out[stage] = (ref, rust)

        counts = Counter()
        a_ref = self.ref_anon.anonymize(text, counts)
        a_rs, rs_counts, _ = spam_core.anonymize(text)
        add("anonymize", a_ref, a_rs)
        add("counts", {k: v for k, v in counts.items() if v}, rs_counts)
        u_ref = self.ref_pipe.mask_urls(a_ref)
        add("urls", u_ref, spam_core.mask_urls(a_ref))
        n_ref = self.ref_pipe.mask_numbers(u_ref)
        add("numbers", n_ref, spam_core.mask_numbers(u_ref))
        r_ref = self.ref_export.normalize_ha(n_ref, self.res, self.words, self.words, self.words)
        add("normalize-rules", r_ref, self.rules_norm.normalize(n_ref))
        add("normalize-full", self.hazm_full.normalize(n_ref), self.full_norm.normalize(n_ref))
        add("ngrams", self.analyzer(r_ref), spam_core.ngrams(r_ref))
        return out


class Report:
    def __init__(self, secret):
        self.secret = secret
        self.n = 0
        self.intended = Counter()  # stage -> inputs
        self.unexpected = {s: [] for s in STAGES}  # stage -> [(input, python, rust)]
        self.chars = {s: Counter() for s in STAGES}

    def check(self, checker, raw):
        text = raw[: spam_core.MAX_CHARS]
        self.n += 1
        diffs = checker.stages(text)
        if not diffs:
            return
        clean = sanitize(text)
        if clean != text and not checker.stages(clean):
            self.intended.update(diffs.keys())
            return
        for stage, (ref, rust) in diffs.items():
            self.unexpected[stage].append((text, ref, rust))
            if self.secret and isinstance(ref, str):
                for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, rust, autojunk=False).get_opcodes():
                    if tag != "equal":
                        self.chars[stage].update(("python", c) for c in ref[i1:i2])
                        self.chars[stage].update(("rust", c) for c in rust[j1:j2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rev", default=DEFAULT_REV, help="git revision of the Python reference (default: %(default)s)")
    ap.add_argument("--fuzz", type=int, default=20000, help="number of fuzz inputs")
    ap.add_argument("--seed", type=int, default=20261014, help="fuzz seed")
    ap.add_argument("--no-data", action="store_true", help="skip data/**/*.jsonl")
    ap.add_argument("--show", type=int, default=0, metavar="N", help="print up to N differing synthetic inputs per stage")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        ref = load_reference(args.rev, Path(workdir))
        from hazm import Normalizer as Hazm
        hazm_full = Hazm()
        res = ref[2].export_resources(hazm_full)
        empty = (res, set())
        lexicons = REPO / "core/data/lexicons"
        rust = (spam_core.Normalizer("rules"), spam_core.Normalizer("full", str(lexicons)))
        analyzer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True).build_analyzer()

        corpora = [("battery", battery(), False), (f"fuzz ({args.fuzz})", fuzz(args.fuzz, args.seed), False)]
        if not args.no_data:
            corpora.append(("local data", local_data(), True))

        checker = Checker(ref, rust, hazm_full, empty, analyzer)
        failed = False
        for name, texts, secret in corpora:
            report = Report(secret)
            for text in texts:
                report.check(checker, text)
            bad = sum(len(d) for d in report.unexpected.values())
            known = sum(report.intended.values())
            print(f"{name}: {report.n} inputs, {bad} unexpected differences, {known} intended")
            for stage in STAGES:
                if report.intended[stage]:
                    print(f"  intended   {stage}: {report.intended[stage]}")
                diffs = report.unexpected[stage]
                if not diffs:
                    continue
                failed = True
                print(f"  UNEXPECTED {stage}: {len(diffs)}")
                if secret:
                    for (side, c), k in report.chars[stage].most_common(8):
                        print(f"    {side} only: {name_of(c)} x{k}")
                else:
                    for text, r, rs in diffs[: args.show]:
                        print(f"    input {text!r}\n      python {r!r}\n      rust   {rs!r}")
        print("UNEXPECTED DIFFERENCES" if failed else "no unexpected differences")
        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
