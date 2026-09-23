# MedReason Evaluation

The challenge uses three official leaderboard metrics.

## MCQ Accuracy

For closed-ended multiple-choice cases, participants must return one official option label. MCQ Accuracy is the mean exact-match accuracy over all MCQ cases.

## Open-ended GT

For open-ended cases, `GT_final` is the organizer-side ground-truth correctness score assigned to the submitted final answer.

The public `scoring.py` aggregates case-level `GT_final` scores by taking the mean over open-ended cases.

## Open-ended VA

For open-ended cases, visual accuracy is computed using `VA_answer` and `RVF_trace`.

```python
if RVF_trace <= 1:
    VA_final = min(VA_answer, 1)
elif RVF_trace == 2:
    VA_final = min(VA_answer, 3)
else:
    VA_final = VA_answer
```

Open-ended VA is the mean of `VA_final` over open-ended cases.

## Public and official evaluation boundary

`scoring.py` defines metric aggregation. During official hidden-test evaluation, case-level open-ended judge scores are produced by the organizers and then passed into the aggregation script.
