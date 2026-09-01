# Contributing

The unusual thing about this repository is the evidence rule, so start there.

## The rule

**A performance claim needs a pre-registration.** If you are proposing a change that is
supposed to make the model *better*, write the study before you run it:

1. A `VALIDATION_V*.md` with the arms, a numeric bar per hypothesis, what each outcome
   means, and a section naming the outcome you would least like.
2. Commit that file **before** the first cell runs.
3. A `report_v*.py` that applies the bars mechanically. No verdict is reached by reading a
   table.
4. If it misses the bar, it does not ship — and the failure gets written up rather than
   retried until it passes.

Ten studies have gone through this and all ten failed their bars. Five features were
deleted as a result. That is the system working, not a problem with it.

Bug fixes, speedups that are bit-identical, documentation and tests need none of this.

## Things worth knowing before you change something

**Exact leave-one-group-out is load-bearing.** `tests/test_dense.py` checks the block
hold-out against literally refitting without each group, to ~5e-15. It has caught two
defects that had already produced confident wrong answers. A change that breaks it does not
ship, however good the accuracy looks.

**Split by group, not by row.** This project has shipped the row-wise/group-wise confusion
three separate times. If rows share a subject and the statics are constant within a
subject, a random row split lets a model recover a static by recognising whose data it is.

**Watch the winner's curse.** Adding candidate types is tempting and mostly does not work:
a node takes the maximum gain over its pool, so a bigger pool raises the winner's expected
gain whether or not anything in it is informative. This is measured in
`validation/HEADROOM.md`, and four separate attempts to fix it are in V16 through V19.

**Do not tune on the benchmarks.** Optuna and friends are explicitly out of scope — they
industrialise exactly the validation optimism this project keeps catching in itself.

## Running the tests

```bash
pip install -e ".[test]"
python -m pytest tests -q
```

They take about 80 seconds. The suite is deliberately full of tests that assert *why* a
thing is the way it is, not just that it runs — if one fails, read its docstring first.
