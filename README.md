# PlantInquiryVQA — Thinking Like a Botanist

**Benchmark and framework for multi-turn, intent-driven visual question answering in plant pathology.**

*Accepted at ACL 2026 Findings.* &nbsp;·&nbsp; [📄 Paper](./plantinquiryvqa.pdf) &nbsp;·&nbsp; 🤗 Dataset on HuggingFace *(link TBA)*

![Scale](https://img.shields.io/badge/images-24%2C950-green) ![QA pairs](https://img.shields.io/badge/QA%20pairs-138%2C068-blue) ![Code](https://img.shields.io/badge/license-MIT-lightgrey) ![Data](https://img.shields.io/badge/data-CC%20BY%204.0-orange)

---

## What's inside

PlantInquiryVQA formalises diagnostic reasoning in plant pathology as a **Chain-of-Inquiry (CoI)** — an ordered sequence of visually-grounded questions that adapts to the plant's severity and the expert's epistemic intent (Diagnosis / Prognosis / Management). The benchmark evaluates whether modern Multimodal Large Language Models can *reason like a botanist*, not just classify a leaf.

- **24,950 curated leaf images** across 34 crop species and 116 disease categories
- **138,068 QA pairs** annotated with visual grounding, severity labels, and reasoning templates
- **Three evaluation protocols** — Scaffolded (no history) / Guided (GT history) / Cascading (own history)
- **18 MLLMs benchmarked**, including Gemini 3, Claude (via OpenRouter), Qwen3-VL, Seed-1.6, Llama-3.2/4, Grok-4.1, Pixtral, Mistral, Phi-4, and open-weight variants.

## Repository layout

```
PlantInquiryVQA/
├── paper/plantinquiryvqa.pdf         # camera-ready paper
├── scripts/
│   ├── download_images.py            # pull 3.5 GB image corpus from HF
│   └── sanitize_secrets.py           # (dev tool) strip keys + abs paths
├── dataset/
│   ├── plantinquiryvqa_train.csv     # 80% split
│   ├── plantinquiryvqa_test.csv      # 20% split
│   
├── diseases_knowledge_base/          # per-crop expert disease cards
│   ├── all_cards.jsonl
│   └── <crop>/<disease>.json
├── visual_cues/visual_cues.json      # 24,950 × expert-verified visual cues
├── eval/                             # evaluation scripts + cached logs
│   ├── test_1_<model>.py             # Cumulative Context Test (Guided)
│   ├── test_2_<model>.py             # Scaffolded Test
│   ├── run_cascading_context.py      # Cascading-history ablation
│   ├── run_ablation_unconstrained_v2.py
│   ├── llm_as_judge.py               # GPT-5 / Gemini-3-Pro semantic judge
│   ├── vhelm_fairness_robustness.py  # VHELM-aligned fairness & robustness
│   ├── weight_ablation.py            # α/β/γ sensitivity analysis
│   ├── compute_*.py                  # post-hoc aggregation scripts
│   ├── llm_judge_results/            # cached 1–5 scale judgments (18 models)
│   ├── cascading_context_results/    # raw cascading checkpoints
│   └── vhelm_analysis/               # fairness + perturbation outputs
└── images/                           # NOT tracked by git; see HuggingFace
```

## Quick start

```bash
# 1. Clone + install
git clone https://github.com/<your-org>/PlantInquiryVQA.git
cd PlantInquiryVQA
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Get API keys and configure
cp .env.example .env           # fill in GEMINI_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY

# 3. Download images (large — 3.5 GB)
python scripts/download_images.py   # pulls from HuggingFace to ./images/

# 4. Run one evaluation
python eval/test_1_gemini3_flash.py        # Guided setting
python eval/test_2_gemini_3_flash.py       # Scaffolded setting

# 5. Aggregate results
python eval/compute_cascading_all_models.py
python eval/compute_fairness_all_models.py
```

All model-provider credentials are read from environment variables (see
[`.env.example`](./.env.example)). No keys are committed to the repo.

## Benchmark results at a glance

Best-in-class per metric (full table in paper Table 2):

| Dimension | Leader | Score |
|---|---|---:|
| Disease Accuracy | Gemini-3-Flash | 0.444 |
| Clinical Utility | Llama-3.2-90B-Vision | 0.185 |
| Safety | Llama-3.2-90B-Vision | 0.214 |
| Visual Grounding | Qwen-VL-Plus | 0.508 |
| Explainability Efficiency | Grok-4.1-Fast | 5.20 |

**Key finding.** All 18 MLLMs describe symptoms competently but fail at reliable clinical reasoning (top score for Clinical Utility = 0.188 / 1.0). Structured question-guided inquiry improves diagnostic accuracy by ~48% over direct diagnosis and reduces hallucination.

## Chain-of-Inquiry evaluation protocols

| Protocol | Conversation history | Purpose |
|---|---|---|
| **Scaffolded** (Test 2) | none | Lower bound — raw single-turn capability |
| **Guided** (Test 1, main) | ground-truth | Upper bound — isolates per-turn reasoning |
| **Cascading** (ablation) | model's own answers | Realistic — compounding-error deployment |
| **Unconstrained** (ablation) | no CoI templates | Worst case — natural dialogue |

See `eval/run_cascading_context.py` and `eval/run_ablation_unconstrained_v2.py` for reproduction.

## Domain-specific evaluation metrics

All defined in paper Appendix A.1:

- **S_dis** — Disease Identification Score (strict entity match)
- **S_safe** — Safety Score (false-reassurance penalty)
- **S_clin** = 0.5·S_dis + 0.3·S_act − 0.2·(1 − S_safe) — composite clinical utility
- **S_vg** — Visual Grounding recall of expert-verified cues
- **E** — Explainability Efficiency (verified cues per 100 words)
- **B** — Prevalence Bias (Eq. 7)
- **F** — Cross-Class Fairness (Eq. 8)

## Reproducing rebuttal tables

```bash
# Extended Cascading / Scaffolded / Guided accuracy + efficiency for all 18 models
python eval/compute_cascading_all_models.py

# Cross-Class Fairness (VHELM-aligned) across all 18 models
python eval/compute_fairness_all_models.py

# Stratified error analysis for Appendix A.4
python eval/compute_stratified_error_analysis.py
```

Outputs are written to `eval/cascading_context_results/`, `eval/vhelm_analysis/`, and `eval/stratified_error_analysis.json`.

## Dataset on HuggingFace

The image corpus (~3.5 GB) and annotated CSVs are mirrored at
**🤗 `<user>/PlantInquiryVQA`** *(link TBA).* Load with:

```python
from datasets import load_dataset
ds = load_dataset("<user>/PlantInquiryVQA", split="test")
```

## Licence

- **Code**: [MIT](./LICENSE)
- **Dataset & annotations**: [CC BY 4.0](./LICENSE)
- **Source images**: retain their upstream licences — see Appendix A.1 of the paper for the 39 component datasets and their individual licences (most are CC BY 4.0; a few are CC0 / CC BY-NC 3.0).

## Citation

```bibtex
@inproceedings{sakib2026plantinquiryvqa,
  title     = {Thinking Like a Botanist: Challenging Multimodal Language Models with Intent-Driven Chain-of-Inquiry},
  author    = {Sakib, Syed Nazmus and Haque, Nafiul and Amin, Shahrear Bin and
               Abdullah, Hasan Muhammad and Hasan, Md Mehedi and
               Hossain, Mohammad Zabed and Arman, Shifat E.},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026}
}
```

## Acknowledgements

We thank Ali Akbar for large-scale data collection and Abdullah Shahriar for creating the figures and diagrams.

## Contact

Open an issue on GitHub, or reach out to the corresponding author listed in the paper.
