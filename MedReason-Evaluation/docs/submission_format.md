# MedReason Validation Submission Format

An example submission archive with the expected file structure is provided as:

```text
sample_submission.zip
```

Participants must upload one zip file containing a single `results.json` file for all validation cases.

## Zip file structure

Correct:

```text
submission.zip
└── results.json
```

Incorrect:

```text
submission.zip
└── submission/
    └── results.json
```

## `results.json` format

```json
{
  "name": "MedReason system submission",
  "type": "Medical visual reasoning",
  "answers": [
    {
      "case_id": "case_mcq_001",
      "task_type": "mcq",
      "answer": "A",
      "reasoning_trace": "Optional for MCQ."
    },
    {
      "case_id": "case_open_001",
      "task_type": "open",
      "reasoning_trace": "Image-grounded evidence used by the system.",
      "answer": "Concise final answer."
    }
  ],
  "version": {"major": 1, "minor": 0}
}
```

## Required fields

Each answer must contain:

- `case_id`
- `task_type`
- `answer`

For open-ended cases, `reasoning_trace` is also required and must be non-empty.

## Additional notes

- Submit one prediction for every validation case.
- Do not include additional files in the zip archive.
- For MCQ cases, `answer` must be exactly one of the official option labels.
- For open-ended cases, final answer alone is not sufficient; include the reasoning trace.
