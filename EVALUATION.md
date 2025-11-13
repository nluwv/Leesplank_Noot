# Dutch Text Simplification Evaluation Guide

This document describes the comprehensive SARI benchmark evaluation system for Dutch text simplification models in the Leesplank project.

## Overview

The evaluation suite provides a comprehensive framework for evaluating Dutch text simplification models using the SARI metric as the primary evaluation criterion. It supports multiple models, statistical analysis, and comparative evaluation.

## Quick Start

### Basic Evaluation (Default: 1000 samples)
```bash
python scripts/run_comprehensive_sari_evaluation.py
```

### Small Test Run (10 samples)
```bash
python scripts/run_comprehensive_sari_evaluation.py --num_samples 10 --no_system_prompt
```

### Full Test Set Evaluation
```bash
python scripts/run_comprehensive_sari_evaluation.py --full_test
```

## Main Evaluation Script

### `scripts/run_comprehensive_sari_evaluation.py`

The main evaluation runner that:
- Loads models without quantization (full BFloat16 precision)
- Evaluates on the Wikipedia versimpeleingen test dataset
- Calculates SARI scores and other metrics
- Generates comparative analysis between models
- Saves detailed results and predictions

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--num_samples` | Number of samples to evaluate | 1000 |
| `--full_test` | Use entire test set (268,783 samples) | False |
| `--no_system_prompt` | Skip system prompt for fair comparison | False |
| `--save_predictions` | Save all model outputs | True |
| `--output_dir` | Directory for results | "evaluation_results" |
| `--verbose` | Enable detailed logging | False |
| `--seed` | Random seed for reproducibility | 42 |

## Configured Models

### Currently Evaluated Models

1. **UWV/leesplank-noot-granite-3.3-2b** (Granite)
   - Fine-tuned Granite 3.3B model for Dutch simplification
   - Available on HuggingFace Hub
   - Uses Granite chat template format

2. **yhavinga/eurollm-1.7B-leesplank-2epoch-fsdp-20250220** (EuroLLM)
   - Note: This model may not be available yet (check HuggingFace)
   - Alternative: Use other Dutch language models if needed

### Adding New Models

To add a new model for evaluation, edit the `MODELS_TO_EVALUATE` list in the script:

```python
MODELS_TO_EVALUATE = [
    {
        "name": "model-display-name",
        "path": "huggingface/model-path",
        "type": "model_type",  # granite, llama, mistral, eurollm, etc.
        "local": False  # Set to True for local models
    }
]
```

## Output Structure

The evaluation creates a timestamped directory with all results:

```
evaluation_results/
└── sari_benchmark_YYYYMMDD_HHMMSS/
    ├── granite-leesplank_results.csv       # Detailed per-sample results
    ├── granite-leesplank_predictions.jsonl # All predictions
    ├── eurollm-leesplank_results.csv
    ├── eurollm-leesplank_predictions.jsonl
    ├── comparative_analysis.json           # Model comparison
    ├── statistical_summary.yaml            # All statistics
    └── evaluation_config.yaml              # Run configuration
```

## Metrics and Analysis

### Primary Metric: SARI

SARI (System output Against References and against the Input sentence) measures the quality of text simplification by evaluating:
- Keep operations: Important words retained
- Addition operations: Simpler words added
- Deletion operations: Complex words removed

Score range: 0-100 (higher is better)

### Statistical Analysis

For each model, the system calculates:
- **Mean SARI**: Average score across all samples
- **Standard Deviation**: Variability in performance
- **Median**: Middle value (robust to outliers)
- **Min/Max**: Range of scores
- **Quartiles (Q25, Q75)**: Distribution information
- **Generation Speed**: Average seconds per sample

### Comparative Analysis

When multiple models are evaluated:
- Best performing model identification
- Performance differences (SARI points)
- Statistical significance testing
- Speed vs. quality trade-offs

## Utility Functions

### `scripts/evaluation_utils.py`

Provides helper utilities:

1. **ChatTemplateDetector**: Automatically detects and applies correct chat templates
2. **ModelMemoryTracker**: Monitors GPU memory usage during evaluation
3. **ResultAnalyzer**: Statistical analysis and comparison tools
4. **TextSimplificationMetrics**: Additional simplification metrics
5. **CheckpointManager**: Resume interrupted evaluations

## Performance Considerations

### Memory Requirements
- **Minimum**: 16GB GPU VRAM
- **Recommended**: 24GB+ for larger models
- **BFloat16 Precision**: Models loaded without quantization

### Speed Optimization
- Models cached after first download
- Batch processing where possible
- Flash Attention 2 support (when available)

### Dataset Size
- Test set: 268,783 samples
- Default evaluation: 1,000 samples (~2-3 hours)
- Quick test: 10-100 samples (~2-20 minutes)
- Full evaluation: All samples (24+ hours)

## Troubleshooting

### Common Issues

1. **Model not found**: Verify model exists on HuggingFace Hub
   ```bash
   # Check if model is accessible
   hf auth login  # If private model
   ```

2. **Out of Memory**: Reduce batch size or use quantization
   ```python
   # In the script, modify load_model_unquantized to use 8-bit
   load_in_8bit=True
   ```

3. **Slow generation**: Check GPU utilization
   ```bash
   nvidia-smi  # Monitor GPU usage
   ```

4. **Missing dependencies**:
   ```bash
   pip install evaluate sacremoses sacrebleu
   ```

## Example Workflow

### 1. Quick Validation (5 samples)
```bash
# Test that everything works
python scripts/run_comprehensive_sari_evaluation.py \
    --num_samples 5 \
    --no_system_prompt
```

### 2. Standard Evaluation (1000 samples)
```bash
# Regular benchmark run
python scripts/run_comprehensive_sari_evaluation.py \
    --num_samples 1000 \
    --no_system_prompt \
    --save_predictions
```

### 3. Full Evaluation (all samples)
```bash
# Complete evaluation (run overnight)
python scripts/run_comprehensive_sari_evaluation.py \
    --full_test \
    --no_system_prompt \
    --save_predictions
```

### 4. Analyze Results
```bash
# View results
cat evaluation_results/sari_benchmark_*/statistical_summary.yaml

# Compare models
python -c "
import json
with open('evaluation_results/sari_benchmark_*/comparative_analysis.json') as f:
    data = json.load(f)
    print(f\"Best model: {data['best_model']}\")
"
```

## Interpreting Results

### SARI Score Ranges
- **0-30**: Poor simplification
- **30-50**: Below average
- **50-70**: Average performance
- **70-85**: Good simplification
- **85-100**: Excellent simplification

### Example Output
```
granite-leesplank Results:
  SARI Score: 79.36 ± 12.57
  Median SARI: 79.36
  Range: [66.79, 91.92]
  Generation Speed: 7.804s/sample
```

This indicates:
- Good average performance (79.36)
- Moderate variability (±12.57)
- Consistent quality (median close to mean)
- Reasonable speed (~8s per sample)

## Best Practices

1. **Reproducibility**: Always use the same seed for comparable results
2. **Fair Comparison**: Use `--no_system_prompt` when comparing models
3. **Sample Size**: Start with 100-1000 samples for initial evaluation
4. **Multiple Runs**: Consider running multiple times with different seeds
5. **Manual Inspection**: Review predictions.jsonl for quality assessment

## Integration with Training Pipeline

The evaluation script integrates with the training pipeline:

1. Train model using `granite-train/scripts/train.py`
2. Save model to outputs directory
3. Update model path in evaluation script
4. Run evaluation to measure improvement

## Future Enhancements

Potential improvements to the evaluation system:

- [ ] Batch inference for faster evaluation
- [ ] Additional metrics (BLEU, ROUGE, readability)
- [ ] Web interface for result visualization
- [ ] Automatic model discovery from HuggingFace
- [ ] Human evaluation integration
- [ ] Cross-validation support
- [ ] Multi-reference evaluation

## Citation

When using this evaluation system, please cite:

```bibtex
@software{leesplank_evaluation,
  title = {Leesplank Dutch Text Simplification Evaluation Suite},
  author = {Leesplank Team},
  year = {2025},
  url = {https://github.com/nluwv/Playhouse-leesplank-noot}
}
```

## Support

For issues or questions:
- Check existing issues: https://github.com/nluwv/Playhouse-leesplank-noot/issues
- Contact the Leesplank team

## License

This evaluation suite is part of the Leesplank project and follows the EUPL-1.2 license.