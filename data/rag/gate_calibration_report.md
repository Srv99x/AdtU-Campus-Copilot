# Gate Calibration Report

## Overview

- Calibration queries       : 50 (25 answerable / 15 knowledge_gap / 10 boundary)
- OOS-blocked by classifier : 8 (short-circuited before retrieval)
- In-scope rows measured    : 42
- Primary gate feature      : `distinct_parent_top3_mean_distance`
- Distance convention       : cosine (lower = more similar to query)

## Distance Distributions (dp-top3-mean)

| Group | n | min | median | mean | max |
|---|---|---|---|---|---|
| `answerable` | 21 | 0.1476 | 0.2218 | 0.2336 | 0.3527 |
| `knowledge_gap` | 14 | 0.2216 | 0.3219 | 0.3135 | 0.4127 |
| `boundary` | 7 | 0.2696 | 0.3169 | 0.3184 | 0.3608 |

## Key Observations

- **Answerable** cluster at low distances: median 0.2218, max 0.3527.
- **Knowledge-gap** at higher distances: median 0.3219, min 0.2216.
- **Boundary** in the middle zone: median 0.3169.
- The distributions show a **natural separation** between answerable and knowledge-gap groups.

## Threshold Selection — Key Candidate Rows

| theta | A-acc | A-rej | KG-acc | KG-rej | B-acc | B-rej | FAR | FRR | EscRate |
|---|---|---|---|---|---|---|---|---|---|
| 0.220 | 10 | 11 | 0 | 14 | 0 | 7 | 0.000 | 0.524 | 0.762 |
| 0.240 | 12 | 9 | 2 | 12 | 0 | 7 | 0.143 | 0.429 | 0.667 |
| 0.260 | 13 | 8 | 3 | 11 | 0 | 7 | 0.214 | 0.381 | 0.619 |
| 0.280 | 18 | 3 | 4 | 10 | 1 | 6 | 0.286 | 0.143 | 0.452 |
| 0.300 | 18 | 3 | 4 | 10 | 2 | 5 | 0.286 | 0.143 | 0.429 |
| 0.320 | 19 | 2 | 7 | 7 | 4 | 3 | 0.500 | 0.095 | 0.286 |
| 0.340 | 20 | 1 | 11 | 3 | 5 | 2 | 0.786 | 0.048 | 0.143 |
| 0.360 | 21 | 0 | 12 | 2 | 6 | 1 | 0.857 | 0.000 | 0.071 |
| 0.380 | 21 | 0 | 12 | 2 | 7 | 0 | 0.857 | 0.000 | 0.048 |
| 0.400 | 21 | 0 | 13 | 1 | 7 | 0 | 0.929 | 0.000 | 0.024 |
| 0.420 | 21 | 0 | 14 | 0 | 7 | 0 | 1.000 | 0.000 | 0.000 |
| 0.440 | 21 | 0 | 14 | 0 | 7 | 0 | 1.000 | 0.000 | 0.000 |
| **0.220** | **10** | **11** | **0** | **14** | **0** | **7** | **0.000** | **0.524** | **0.762** | **<-- RECOMMENDED** |

## Calibration-Recommended Threshold

**theta_d = 0.220** (on `distinct_parent_top3_mean_distance`)

### Gate Performance at Recommended Threshold

| Metric | Value |
|---|---|
| Answerable accepted (correct high-conf) | 10 / 21 (48%) |
| Answerable rejected (over-escalation / FRR) | 11 / 21 (52%) |
| Knowledge-gap accepted (false-accept / FAR) | 0 / 14 (0%) |
| Knowledge-gap rejected (correct escalation) | 14 / 14 |
| Boundary accepted | 0 / 7 |
| Boundary rejected (expected escalation) | 7 / 7 |
| Overall escalation rate | 76.2% |

### Rationale

Selected as the **lowest threshold achieving FAR = 0** across the calibration set.
This means zero knowledge-gap queries are falsely accepted as high-confidence.
A moderate FRR is accepted because over-escalation to SQLite is preferable to hallucinated answers.
Boundary queries are intentionally expected to escalate.

### Runtime Reconciliation Note

The frozen runtime threshold is **`GATE_THETA_D = 0.275`** (`app/rag/pipeline.py`),
not the 0.220 calibration-recommended value above. 0.220 was this calibration
study's own FAR=0 recommendation, derived solely from the 50-query
calibration set described in this report; it was never applied to the
runtime. The runtime deliberately uses 0.275 as a frozen precision/recall
trade-off, accepting a non-zero false-accept rate in exchange for fewer
over-escalations than 0.220 would produce. The closest calibration row
actually measured near that trade-off, `theta=0.280`, shows **FAR=0.286**
in the table above — this document does not claim 0.275 itself was
empirically measured or optimal; it is recorded here only as the frozen
value currently in force.

Operators must **NOT** change `GATE_THETA_D` merely to reconcile it with
this report's 0.220 recommendation. The runtime value is frozen: see
`AGENTS.md` ("Preserve the locked application architecture") and the
"FROZEN — do not change" comment directly above `GATE_THETA_D` in
`app/rag/pipeline.py`. Any change to the gate threshold requires the same
explicit approval as any other locked-architecture change.

### Compound Gate Logic (NOT YET IMPLEMENTED in pipeline.py)

```python
HIGH confidence if:
    top1_category_match == True
    AND distinct_parent_top3_mean_distance < 0.220  # calibrated
    AND top3_category_match_rate >= 0.50
LOW / escalate otherwise
```

> This threshold was derived solely from the 50-query calibration set.
> It MUST NOT be tuned using final_heldout_queries.csv or classifier evaluation data.