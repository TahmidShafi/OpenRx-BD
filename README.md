# OpenRx-BD 🇧🇩💊

### Bangladeshi Prescription Safety, Data-Sanitization & Retail Cost-Optimization Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Pytest Tests](https://img.shields.io/badge/tests-65%2F65%20passed-success.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OpenRx-BD** is an open-source clinical safety, data-sanitization, and retail cost-optimization engine designed for the Bangladeshi pharmaceutical ecosystem. Instead of duplicating read-only directories like MedEx or DIMS, OpenRx-BD processes **21,714 commercial medicine records** to solve three real-world bottlenecks: chaotic pricing strings, polypharmacy drug-drug interactions (DDIs), and realistic retail dispensing economics.

---

## 🏛️ System Architecture & Engineering Pipeline

```text
[MedEx Raw Dataset (21,714 rows)]
         │
         ▼
[Phase 1-2: Template Profiler & Leakage Firewall] ──► (Immutable Dev / Blind Splits)
         │
         ▼
[Phase 3: Non-Silent AST Parser & Invariant Engine] ──► (|U * Q - P| <= 0.05 Check)
         │
         ▼
[Phase 4: Clinical Ontology & NTI Safety Firewall] ──► (Salt-Aware Equivalence & NTI Locks)
         │
         ▼
[Phase 5: Retail Dispensing Regimen Optimizer] ──► (Loose-Strip Pricing vs Knapsack DP)
         │
         ▼
[Phase 6: Production FastAPI Microservice & Tailwind Web UI Dashboard]
```

---

## 🚀 Core Engineering Innovations

### 1. Robust AST Grammar Parser & Cross-Column Reconciliation
Raw pricing strings in emerging markets are notoriously messy (e.g., double-comma SKU delimiters `,,` and mixed HTML artifacts). OpenRx-BD uses a strict AST parser that:
- Handles multi-SKU composite strings and single-tier liquid/vial containers without silent `try/except` fallbacks.
- Performs cross-column dimensional reconciliation—automatically mapping concentration strings (e.g., `25 mg/ml` in 5-Fluril) against total vial mass (`250 mg vial`) to compute accurate liquid volumes.

### 2. Property-Based Validation Invariants ($|U \times Q - P| \le 0.05$)
To eliminate manual data-entry errors, every multi-SKU record is validated against an invariant rule:
$$\lvert (\text{unit\_price} \times \text{pack\_qty}) - \text{pack\_price} \rvert \le 0.05$$
Violations are flagged as `INVARIANT_VIOLATION` rather than silently corrected or discarded.

### 3. Realistic Bangladeshi Retail Dispensing Economics
The regimen optimizer models how patients actually buy drugs in local pharmacies:
- **Divisible Solids (Tablets/Capsules):** Computes retail cut-strip pricing (`unit_price * exact_units_needed`) alongside bulk pack discounts.
- **Indivisible Containers (Syrups/Inhalers/Drops):** Employs Unbounded Knapsack Dynamic Programming to find the optimal combination of bottle sizes (e.g., combining 450 ml and 225 ml bottles) that covers prescribed volumes while minimizing financial cost and leftover liquid waste.

### 4. Clinical Ontology & NTI Safety Layer
- **Salt-Form & Release-Profile Locking:** Prevents dangerous cross-substitutions (e.g., ensuring Metoprolol Tartrate and Metoprolol Succinate SR are never treated as interchangeable).
- **Narrow Therapeutic Index (NTI) Locks:** Automatically locks critical drugs (Warfarin, Insulins, Digoxin, Levothyroxine) from automated brand substitution to protect patient safety.
- **WHO AWaRe Antibiotic Stewardship:** Audits multi-drug prescriptions for Access, Watch, and Reserve tiers.

---

## 📊 Empirical Benchmark Results

Evaluated across immutable dev and blind test splits with zero data leakage:

| Metric | Dev Split (70%) | Blind Test Split (30%) | Holdout (Tail Templates) |
| :--- | :--- | :--- | :--- |
| **Total Rows** | 14,867 | 6,319 | 486 |
| **Coverage % (Parsed Valid)** | 97.20% | 97.17% | 94.03% |
| **Invariant Check Rate %** | 62.95% | 63.55% | 27.32% |
| **Invariant Pass Rate %** | 95.84% | 95.78% | 98.21% |
| **Invariant Violation Rate %** | 2.76% | 2.80% | 0.21% |
| **Abstention Rate %** | 0.04% | 0.03% | 5.76% |

- **Test Suite Coverage:** 65 / 65 unit and integration tests passing (`pytest` suite executes in 4.34s).
- **Parser Generalization:** Over 97% valid extraction across unstructured container strings with zero drop on the blind test split.
- **Market Price Spread:** Across 1,908 multi-brand equivalence groups, **56.7%** exhibit > 25% price spread, with a median max-to-min ratio of 1.338x and 75th percentile of 2.000x.
- **API Latency:** Sub-millisecond prefix search and regimen optimization across 21,714 indexed records.

---

## 🛠️ Quickstart & Local Installation

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/TahmidShafi/OpenRx-BD.git
cd OpenRx-BD
python -m venv .venv

# Activate virtual environment:
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -e .
```

### 2. Run the Test Suite
Verify that all 65 tests pass successfully:
```bash
pytest -v
```

### 3. Launch the FastAPI Dashboard
Start the production server:
```bash
uvicorn src.openrx_bd.api:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser to access the interactive 3-tab glassmorphism web dashboard:
1. **Regimen Optimizer & Polypharmacy Audit:** Interactive prescription builder with clinical presets (DDI Overdose, NTI Locks, Syrup Knapsack DP).
2. **AST Grammar & Invariant Inspector:** Real-time container string parser sandbox with invariant check breakdown.
3. **Blind Holdout Benchmark Report:** Live audit metrics and market price dispersion analytics.

Interactive API Swagger documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 📁 Repository Structure

```text
OpenRx-BD/
├── data/
│   ├── raw/                  # Raw parquet datasets (git-ignored)
│   └── splits/               # Immutable dev and blind test splits (git-ignored)
├── reports/                  # Template profiles, split locks, and benchmark results
│   ├── benchmark_results.json
│   ├── split_lock.json
│   └── template_profile.json
├── src/
│   ├── openrx_bd/
│   │   ├── api.py            # FastAPI REST service & static file server
│   │   ├── engine.py         # Bioequivalence matcher & legacy optimizer
│   │   ├── ingest.py         # Dataset fetcher & schema validation
│   │   ├── parser.py         # Legacy regex parser
│   │   ├── schema.py         # Ground-truth schema enforcement
│   │   ├── split.py          # Anti-leakage partitioner
│   │   └── static/           # Tailwind CSS 3-tab interactive web dashboard
│   ├── etl/                  # Phase 1-2 dataset fetching & split guards
│   ├── evaluation/           # Phase 5 blind benchmark runner
│   ├── ontology/             # Phase 4 bioequivalence keys, salts, NTI, WHO AWaRe
│   ├── optimizer/            # Phase 5 knapsack DP regimen solver
│   ├── parser/               # Phase 3 grammar AST parser & invariants
│   └── profiler/             # Phase 1 structural template profiler
├── tests/                    # 65 comprehensive unit & integration tests
├── pyproject.toml
└── README.md
```

---

## 📜 License

Distributed under the MIT License. See LICENSE for more information.
