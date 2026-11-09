# Validation-only static baseline

The independent validation-only search complements the historical fixed-weight baseline. It uses a separate 60-fit development grid search, locks one weight pair, and evaluates that choice across 20 held-out five-candidate cases (100 PINN fits). The stored selection is **(1,3)**. The main Bratu benchmark files remain separate from these outputs.

The exact grid, loss-based selection objective, separation of development and evaluation seeds, and cost accounting are specified in [`BRATU_VALIDATION_ONLY_PROTOCOL.md`](../BRATU_VALIDATION_ONLY_PROTOCOL.md).

## Source and data

- `experiments/bratu_static_validation_tune.py`: development-stage training and selection.
- `experiments/bratu_fixed_validation.py`: held-out best-of-five evaluation.
- `utils/static_validation_baseline.py`: grid configuration, validation criterion, and output integrity checks.
- `experiments/collect_bratu_fixed_validation.py`: evaluation output and summary collector.
- `tests/test_static_validation_protocol.py`: configuration and isolation tests.
- `results/validation_static_v1/production/tuning/`: 60 development JSON records.
- `results/validation_static_v1/production/selected_weights.json`: selected global pair and objective.
- `results/validation_static_v1/production/evaluation/`: 20 held-out run records, including per-candidate metrics.
- `results/validation_static_v1/production/summary/`: derived CSV tables.

## Running the training and tests

The PBS scripts are in `cluster/`. Configure paths to the target Python runtime and project root before submitting jobs. Run tuning, selection finalization, held-out evaluation, and collection in that order. The `--smoke` mode is separate from the production dataset and is not part of the reported results.

```bash
python -m unittest discover -s tests -p 'test_static_validation_protocol.py'
```

Training times refer to model-fit duration; scheduler queue time is excluded. The selection-stage cost must be counted separately in full-cost comparisons.
