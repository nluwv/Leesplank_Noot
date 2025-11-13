# Model Card Evaluation Results
**Date**: Tue Nov 11 22:57:05 UTC 2025
**Output Directory**: evaluation_results/model_card_20251111_095937

## Evaluation Configuration

### Primary Benchmark (Quality)
- **Method**: Beam Search (num_beams=5)
- **Samples**: 1000
- **Runs**: 3 (for confidence intervals)
- **Seed**: 42
- **Purpose**: Main quality metrics for model card

### Speed Benchmark
- **Method**: Greedy Decoding
- **Samples**: 100
- **Seed**: 42
- **Purpose**: Inference speed comparison

## Results

### Quality Metrics (Beam Search)
See `evaluation_results/model_card_20251111_095937/beam_search/` for detailed results including:
- SARI scores with 95% confidence intervals
- Per-model predictions and statistics
- Comparative analysis between models

### Speed Metrics (Greedy)
See `evaluation_results/model_card_20251111_095937/greedy_speed/` for:
- Generation speed (seconds/sample)
- SARI scores with greedy decoding
- Memory usage statistics

## Usage

To incorporate these results in your model card:

1. Use beam search SARI scores as primary quality metric
2. Report confidence intervals from multi-run evaluation
3. Include generation speed from greedy benchmark
4. Note model-specific recommendations (e.g., system prompt usage)

## Reproducibility

To reproduce these results:
```bash
./scripts/run_model_card_evaluation.sh
```

All evaluation code available at: https://github.com/UWV/Playhouse-leesplank-noot
