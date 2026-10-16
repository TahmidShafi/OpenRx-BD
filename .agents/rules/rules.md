# WORKSPACE RULES FOR ANTIGRAVITY AGENT (DO NOT VIOLATE)

## 1. Ground-Truth Dataset Schema (NEVER Guess Column Names)
The upstream dataset `Mahadih534/all-Bangladeshi-medicines` has EXACTLY these 10 columns (case and space sensitive):
- `brand id` (int/str)
- `brand name` (str)
- `type` (str: e.g., "allopathic", "herbal")
- `slug` (str)
- `dosage form` (str: e.g., "Tablet", "Syrup", "Injection", "Oral Suspension")
- `generic` (str: e.g., "Calcium Carbonate + Vitamin D3", "Paracetamol")
- `strength` (str or null: e.g., "500 mg", "25 mg/ml", or None for herbal)
- `manufacturer` (str)
- `package container` (str: messy price/pack text with `৳` symbol and `,,` delimiters)
- `Package Size` (str or null)

## 2. Anti-Hallucination & Engineering Constraints
1. **License-Safe Storage:** NEVER commit raw `medicine.csv` or full scraped dumps to git. Add `data/raw/` and `data/splits/` to `.gitignore`.
2. **Data Leakage Firewall:** Once `data/splits/blind_test.parquet` is generated in Step 2, you are STRICTLY FORBIDDEN from reading, inspecting, or sampling rows from `blind_test.parquet` while writing or debugging parser rules. Only `dev_split.parquet` may be inspected.
3. **No Silent Fallbacks:** NEVER write `except Exception: pass` or guess a price when a regex fails. Every row must return a strict Pydantic union: `ParsedSuccess` | `ParsedAbstain` | `InvariantViolation`.
4. **No Fake ML / No Paid APIs:** Use deterministic Python 3.11+, `pydantic>=2.0`, `polars` or `pandas`, `rapidfuzz`, and `pytest`. Zero OpenAI/LLM API calls.