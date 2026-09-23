# MedReason Example Docker I/O Schema

## Input

The container reads:

```text
/input/cases.json
/input/images/...
```

`cases.json` can be:

```json
{
  "cases": [ ... ]
}
```

or directly:

```json
[ ... ]
```

Each case requires:

- `case_id`
- `task_type`: `mcq` or `open`
- `question`
- `image_path` or `image_paths`

MCQ cases also require `options`.

## Output

The container writes:

```text
/output/results.json
```

with one prediction per case under the `answers` list.
