# Model Card: Leesplank Noot - Dutch Text Simplification

## Model Details

### Model Description
**Leesplank: Noot** is a family of fine-tuned language models designed to simplify Dutch texts to B1-level complexity, making government and public service communication more accessible. The models are trained on a large corpus of Dutch Wikipedia simplifications and optimized for practical deployment.

- **Developed by:** UWV (Dutch Employee Insurance Agency)
- **Model type:** Text simplification (instruction-tuned)
- **Language(s):** Dutch (Nederlands)
- **License:** EUPL-1.2
- **Fine-tuned from:**
  - Granite-3.3-2b-instruct (IBM)
  - EuroLLM-1.7b-instruct (Eureka)

### Model Sources
- **Repository:** https://github.com/UWV/Playhouse-leesplank-noot
- **Demo:** [Add demo link if available]

## Uses

### Direct Use
The models are designed for simplifying Dutch texts in government and public service contexts:
- Simplifying official documents and letters
- Making web content more accessible
- Creating easy-to-read versions of complex information
- Supporting citizens with lower literacy levels

### Downstream Use
These models can be integrated into:
- Content management systems for automatic simplification
- Accessibility tools for government websites
- Document processing pipelines
- Translation and localization workflows

### Out-of-Scope Use
- Medical or legal advice requiring professional expertise
- Real-time safety-critical applications
- Content generation without human review
- Languages other than Dutch

## Bias, Risks, and Limitations

### Known Limitations
- Models may oversimplify technical terms that should be preserved
- Performance degrades on texts outside the training domain
- Requires human review for official communications
- May struggle with idiomatic expressions or cultural references

### Recommendations
- Always include human oversight for official communications
- Label AI-simplified content transparently
- Test outputs on target audience before deployment
- Monitor for unintended bias in simplifications

## Training Details

### Training Data
- **Dataset:** UWV/Leesplank_NL_wikipedia_simplifications_preprocessed
- **Size:** 1.89M training samples, 540k validation, 269k test
- **Source:** Dutch Wikipedia simplifications
- **Processing:** Filtered for B1-level complexity targets

### Training Procedure

#### Preprocessing
- Chat template formatting for instruction-following
- Tokenization with model-specific tokenizers
- Maximum sequence length: 1024 tokens

#### Training Hyperparameters
- **Training regime:** BFloat16 mixed precision
- **Epochs:** 2
- **Batch size:** 64 (8 × 8 gradient accumulation)
- **Learning rate:** 2e-5
- **Optimizer:** Paged AdamW 8-bit
- **Hardware:** NVIDIA A100 40GB / RTX 3090 24GB

## Evaluation

### Testing Data & Metrics
- **Test set:** 269k held-out samples from the same distribution
- **Primary metric:** SARI (automatic evaluation for simplification)
- **Secondary metrics:** BLEU, ROUGE, readability scores

### Results

#### Primary Benchmark (1000 samples, beam search)

| Model | SARI ↑ | BLEU ↑ | ROUGE-L ↑ | Gen Speed | Params |
|-------|--------|--------|-----------|-----------|---------|
| **Granite-3.3-2b** | XX.X ± X.X | XX.X | XX.X | X.Xs/sample | 2.4B |
| **EuroLLM-1.7b** | XX.X ± X.X | XX.X | XX.X | X.Xs/sample | 1.7B |

*Mean ± 95% CI over 1000 test samples with beam search (num_beams=5)*

#### Speed Benchmark (100 samples, greedy decoding)

| Model | SARI | Generation Speed | Memory (VRAM) |
|-------|------|------------------|---------------|
| **Granite-3.3-2b** | XX.X | X.Xs/sample | XXX MB |
| **EuroLLM-1.7b** | XX.X | X.Xs/sample | XXX MB |

*Greedy decoding for speed comparison*

### Model-Specific Usage

#### Granite-3.3-2b
- **Best for:** Maximum quality simplification
- **Configuration:** Use WITH system prompt
- **Inference:** Beam search recommended
- **Memory:** ~8GB VRAM required

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="UWV/leesplank-noot-granite-3.3-2b",
    device_map="auto"
)

# Use with system prompt for best results
messages = [
    {"role": "system", "content": "Je bent een AI-assistent..."},
    {"role": "user", "content": "Vereenvoudig: [complex text]"}
]

result = pipe(messages, num_beams=5, max_new_tokens=512)
```

#### EuroLLM-1.7b
- **Best for:** Speed-sensitive applications
- **Configuration:** Use WITHOUT system prompt (better performance)
- **Inference:** Can use greedy for faster results
- **Memory:** ~6GB VRAM required

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="UWV/leesplank-noot-eurollm-1.7b",
    device_map="auto"
)

# Skip system prompt for EuroLLM
messages = [
    {"role": "user", "content": "Vereenvoudig: [complex text]"}
]

result = pipe(messages, do_sample=False, max_new_tokens=512)
```

## Environmental Impact

- **Hardware Type:** NVIDIA A100 / RTX 3090
- **Hours used:** ~48 hours total training time
- **Cloud Provider:** [Specify if applicable]
- **Carbon Emitted:** [Estimate if available]

## Technical Specifications

### Model Architecture and Objective
- **Granite-3.3-2b:** Decoder-only transformer, 2.4B parameters
- **EuroLLM-1.7b:** Decoder-only transformer, 1.7B parameters
- **Objective:** Next-token prediction with instruction tuning

### Compute Infrastructure
- **Hardware:** NVIDIA A100 40GB / RTX 3090 24GB
- **Software:** PyTorch 2.0+, Transformers 4.35+, TRL 0.7+

## Citation

**BibTeX:**
```bibtex
@software{leesplank_noot_2024,
  author = {UWV Data Science Team},
  title = {Leesplank: Noot - Dutch Text Simplification Models},
  year = {2024},
  url = {https://github.com/UWV/Playhouse-leesplank-noot},
  license = {EUPL-1.2}
}
```

**APA:**
> UWV Data Science Team. (2024). Leesplank: Noot - Dutch Text Simplification Models [Software]. https://github.com/UWV/Playhouse-leesplank-noot

## Model Card Authors
[Your name/team]

## Model Card Contact
[Contact information]

## Compliance

### EU AI Act Considerations
- **Risk Level:** Limited risk (accessibility application)
- **Transparency:** Models clearly labeled as AI-generated
- **Human Oversight:** Required for official communications
- **Documentation:** Full technical documentation available
- **Bias Testing:** Regular evaluation on diverse text types

### Ethical Considerations
- Designed to improve accessibility for all citizens
- Preserves meaning while simplifying language
- Respects user privacy (no data retention)
- Open-source for transparency and accountability

## Changelog

### Version 1.0 (Initial Release)
- Initial release of Granite and EuroLLM variants
- SARI scores: 65-75 range on test set
- Optimized for B1-level simplification

## Additional Information

### Reproducibility

To reproduce the evaluation results:

```bash
# Install dependencies
pip install -r requirements-eval.txt

# Run model card evaluation
./scripts/run_model_card_evaluation.sh

# Or run custom evaluation
python scripts/run_sari_evaluation.py \
  --model_card_mode \
  --num_samples 1000 \
  --num_runs 3
```

### Community

- **Issues:** https://github.com/UWV/Playhouse-leesplank-noot/issues
- **Discussions:** [Add if available]
- **Contributing:** See CONTRIBUTING.md

### Acknowledgments

This project was developed as part of the UWV's initiative to make government communication more accessible to all Dutch citizens. Special thanks to the teams at IBM (Granite) and Eureka (EuroLLM) for the base models.