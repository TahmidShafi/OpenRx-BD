"""Pharmaceutical entity resolver with deterministic equivalence keys and clinical safety locks."""

from enum import Enum
import re
from typing import Literal, NamedTuple, Optional
from pydantic import BaseModel, ConfigDict, Field
from src.ontology.reference_data import (
    check_nti_status,
    get_who_aware_category,
    strip_salt_form,
)


class PharmaceuticalEquivalenceKey(NamedTuple):
    """Immutable, deterministic key defining strict pharmaceutical bioequivalence.

    Fields:
      1. canonical_generic_tuple: Sorted tuple of INN/canonical active ingredients (e.g. ('metoprolol',))
      2. exact_salt_tuple: Sorted tuple of corresponding salts/esters (e.g. ('tartrate',) vs ('succinate',))
      3. normalized_strength_tuple: Sorted tuple of corresponding normalized strengths (e.g. ('50 mg',))
      4. canonical_dosage_form: Normalized dosage form (e.g. 'tablet', 'capsule', 'syrup')
      5. release_profile: Release modification ('IR', 'SR', 'ER', 'CR', 'MR', 'RETARD', 'DS', 'PEDIATRIC')
      6. medicine_type: 'allopathic' or 'herbal'
    """

    canonical_generic_tuple: tuple[str, ...]
    exact_salt_tuple: tuple[str | None, ...]
    normalized_strength_tuple: tuple[str | None, ...]
    canonical_dosage_form: str
    release_profile: str
    medicine_type: str


class ResolvedIngredient(BaseModel):
    """An isolated, normalized active pharmaceutical ingredient with its salt."""

    model_config = ConfigDict(extra="forbid")

    canonical_name: str
    salt_form: Optional[str] = None
    raw_token: str
    strength: Optional[str] = None


class SafetyAlert(BaseModel):
    """Clinical safety alert triggered during prescription evaluation."""

    model_config = ConfigDict(extra="forbid")

    alert_type: Literal["DUPLICATE_ACTIVE_INGREDIENT", "NTI_WARNING", "WHO_AWARE_WARNING"]
    severity: Literal["INFO", "WARNING", "HIGH", "CRITICAL"]
    message: str
    affected_brands: list[str]
    active_ingredient: Optional[str] = None
    citation: Optional[str] = None


class ResolvedEntity(BaseModel):
    """Comprehensively resolved pharmaceutical product with equivalence key and safety metadata."""

    model_config = ConfigDict(extra="forbid")

    brand_id: str
    brand_name: str
    generic_raw: str
    dosage_form_raw: str
    strength_raw: Optional[str] = None
    medicine_type: str = "allopathic"

    ingredients: list[ResolvedIngredient]
    canonical_dosage_form: str
    release_profile: str

    equivalence_key: Optional[PharmaceuticalEquivalenceKey] = None
    is_substitutable: bool = True
    substitution_block_reason: Optional[str] = None

    nti_locked: bool = False
    nti_citation: Optional[str] = None
    who_aware_category: Optional[str] = None
    who_atc_code: Optional[str] = None


RELEASE_PROFILE_PATTERNS = [
    (re.compile(r"\b(SUSTAINED RELEASE|SR)\b", re.IGNORECASE), "SR"),
    (re.compile(r"\b(EXTENDED RELEASE|XR|ER|XL)\b", re.IGNORECASE), "ER"),
    (re.compile(r"\b(CONTROLLED RELEASE|CR)\b", re.IGNORECASE), "CR"),
    (re.compile(r"\b(MODIFIED RELEASE|MR)\b", re.IGNORECASE), "MR"),
    (re.compile(r"\b(DELAYED RELEASE|DR)\b", re.IGNORECASE), "DR"),
    (re.compile(r"\bRETARD\b", re.IGNORECASE), "RETARD"),
    (re.compile(r"\b(DOUBLE STRENGTH|DS)\b", re.IGNORECASE), "DS"),
    (re.compile(r"\b(PEDIATRIC|PAEDIATRIC)\b", re.IGNORECASE), "PEDIATRIC"),
]


def extract_release_profile(dosage_form: str | None, brand_name: str | None) -> str:
    """Extract release profile modifier from dosage form and brand name."""
    combined = f"{dosage_form or ''} {brand_name or ''}"
    for pattern, code in RELEASE_PROFILE_PATTERNS:
        if pattern.search(combined):
            return code
    return "IR"


def normalize_canonical_dosage_form(dosage_form: str | None) -> str:
    """Normalize detailed dosage form to canonical root category."""
    if not dosage_form:
        return "unspecified"
    d = dosage_form.strip().lower()

    if "tablet" in d:
        return "tablet"
    if "capsule" in d:
        return "capsule"
    if "syrup" in d:
        return "syrup"
    if "suspension" in d:
        return "suspension"
    if "injection" in d or "infusion" in d:
        return "injection"
    if "drop" in d:
        return "drop"
    if "ointment" in d:
        return "ointment"
    if "cream" in d:
        return "cream"
    if "gel" in d:
        return "gel"
    if "inhaler" in d:
        return "inhaler"
    if "spray" in d:
        return "spray"
    if "lotion" in d:
        return "lotion"
    if "solution" in d:
        return "solution"
    if "sachet" in d:
        return "sachet"

    return d


def split_compound_generic(generic_str: str) -> list[str]:
    """Split compound generic strings on '+' and strip bracketed annotations."""
    if not generic_str:
        return []
    # Split on '+'
    raw_tokens = [t.strip() for t in generic_str.split("+") if t.strip()]
    cleaned_tokens: list[str] = []
    for token in raw_tokens:
        # Strip bracketed annotations like [Elemental source] or [USP]
        cleaned = re.sub(r"\[.*?\]|\(.*?\)", "", token).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned:
            cleaned_tokens.append(cleaned)
    return cleaned_tokens


def split_compound_strength(strength_str: str | None, num_ingredients: int) -> list[str | None]:
    """Split compound strength string on '+' and align with ingredients."""
    if not strength_str:
        return [None] * num_ingredients
    s = strength_str.strip()
    # If wrapped in (A+B)/Vol, keep as single or handle components
    if "+" in s and not s.startswith("("):
        parts = [p.strip() for p in s.split("+")]
        if len(parts) == num_ingredients:
            return parts
    return [s] * num_ingredients


def resolve_medicine_entity(
    brand_id: str | int,
    brand_name: str,
    generic: str,
    dosage_form: str,
    strength: str | None = None,
    medicine_type: str = "allopathic",
) -> ResolvedEntity:
    """Resolve a medicine record into canonical ingredients, equivalence key, and safety status."""
    bid = str(brand_id)
    raw_generic = str(generic or "").strip()
    raw_dosage = str(dosage_form or "").strip()
    raw_strength = str(strength).strip() if strength and str(strength).strip().lower() != "none" else None
    med_type = str(medicine_type or "allopathic").strip().lower()

    # 1. Split compound generic and strip bracketed annotations
    generic_tokens = split_compound_generic(raw_generic)
    strength_tokens = split_compound_strength(raw_strength, len(generic_tokens))

    # 2. Salt stripping and BAN-to-INN mapping
    ingredients: list[ResolvedIngredient] = []
    for g_tok, s_tok in zip(generic_tokens, strength_tokens):
        base_name, salt_form = strip_salt_form(g_tok)
        ingredients.append(
            ResolvedIngredient(
                canonical_name=base_name,
                salt_form=salt_form,
                raw_token=g_tok,
                strength=s_tok,
            )
        )

    # 3. Extract release profile & canonical dosage form
    rel_profile = extract_release_profile(raw_dosage, brand_name)
    canon_dosage = normalize_canonical_dosage_form(raw_dosage)

    # 4. Check NTI status
    is_nti, nti_citation = check_nti_status(raw_generic)

    # 5. Check WHO AWaRe antibiotic category
    who_aware = get_who_aware_category(raw_generic)
    aware_cat = who_aware.category.value if who_aware else None
    aware_atc = who_aware.atc_code if who_aware else None

    # 6. Build Pharmaceutical Equivalence Key with explicit safety locks
    is_substitutable = True
    block_reason: Optional[str] = None
    equiv_key: Optional[PharmaceuticalEquivalenceKey] = None

    # Explicit Safety Block: Herbal medicines with null/unstated strength
    if med_type == "herbal" and not raw_strength:
        is_substitutable = False
        block_reason = (
            "SAFETY LOCK: Herbal medicines with unstated/null strength cannot be safely substituted. "
            "Biochemical marker standardization is unverified."
        )
    elif is_nti:
        # NTI drugs are locked from automatic generic substitution
        is_substitutable = False
        block_reason = f"SAFETY LOCK: Narrow Therapeutic Index (NTI) drug. {nti_citation}"
    else:
        # Sort ingredients deterministically by canonical name
        sorted_ingredients = sorted(ingredients, key=lambda ing: ing.canonical_name)
        canonical_generic_tuple = tuple(ing.canonical_name for ing in sorted_ingredients)
        exact_salt_tuple = tuple(ing.salt_form for ing in sorted_ingredients)
        normalized_strength_tuple = tuple(ing.strength for ing in sorted_ingredients)

        equiv_key = PharmaceuticalEquivalenceKey(
            canonical_generic_tuple=canonical_generic_tuple,
            exact_salt_tuple=exact_salt_tuple,
            normalized_strength_tuple=normalized_strength_tuple,
            canonical_dosage_form=canon_dosage,
            release_profile=rel_profile,
            medicine_type=med_type,
        )

    return ResolvedEntity(
        brand_id=bid,
        brand_name=brand_name,
        generic_raw=raw_generic,
        dosage_form_raw=raw_dosage,
        strength_raw=raw_strength,
        medicine_type=med_type,
        ingredients=ingredients,
        canonical_dosage_form=canon_dosage,
        release_profile=rel_profile,
        equivalence_key=equiv_key,
        is_substitutable=is_substitutable,
        substitution_block_reason=block_reason,
        nti_locked=is_nti,
        nti_citation=nti_citation,
        who_aware_category=aware_cat,
        who_atc_code=aware_atc,
    )


def check_prescription_safety(entities: list[ResolvedEntity]) -> list[SafetyAlert]:
    """Audit a prescription for duplicate active ingredients, NTI locks, and antibiotic stewardship."""
    alerts: list[SafetyAlert] = []

    # 1. Duplicate Active Ingredient Detection
    # Map active ingredient -> list of brand names
    ingredient_to_brands: dict[str, list[str]] = {}
    for entity in entities:
        for ing in entity.ingredients:
            ingredient_to_brands.setdefault(ing.canonical_name, []).append(entity.brand_name)

    for ing_name, brand_list in ingredient_to_brands.items():
        unique_brands = list(dict.fromkeys(brand_list))
        if len(unique_brands) > 1:
            alerts.append(
                SafetyAlert(
                    alert_type="DUPLICATE_ACTIVE_INGREDIENT",
                    severity="HIGH",
                    message=(
                        f"DUPLICATE ACTIVE INGREDIENT ALERT: Active ingredient '{ing_name}' is concurrently "
                        f"prescribed across {len(unique_brands)} different products: {', '.join(unique_brands)}. "
                        f"Risk of cumulative toxicity, accidental overdose, or acute organ injury."
                    ),
                    affected_brands=unique_brands,
                    active_ingredient=ing_name,
                )
            )

    # 2. Narrow Therapeutic Index (NTI) Alerts
    for entity in entities:
        if entity.nti_locked:
            alerts.append(
                SafetyAlert(
                    alert_type="NTI_WARNING",
                    severity="CRITICAL",
                    message=(
                        f"NTI SAFETY WARNING: '{entity.brand_name}' contains Narrow Therapeutic Index active "
                        f"ingredient '{entity.generic_raw}'. Automated brand substitution is strictly locked. "
                        f"Clinical Citation: {entity.nti_citation}"
                    ),
                    affected_brands=[entity.brand_name],
                    active_ingredient=entity.generic_raw,
                    citation=entity.nti_citation,
                )
            )

    # 3. WHO AWaRe Antibiotic Alerts (Watch / Reserve awareness)
    for entity in entities:
        if entity.who_aware_category in ["WATCH", "RESERVE", "NOT_RECOMMENDED_FDC"]:
            severity_level = "CRITICAL" if entity.who_aware_category != "WATCH" else "WARNING"
            alerts.append(
                SafetyAlert(
                    alert_type="WHO_AWARE_WARNING",
                    severity=severity_level,
                    message=(
                        f"WHO AWaRe STEWARDSHIP [{entity.who_aware_category}]: '{entity.brand_name}' ({entity.generic_raw}) "
                        f"is classified under WHO AWaRe as {entity.who_aware_category} (ATC: {entity.who_atc_code}). "
                        f"Requires documented clinical indication and stewardship monitoring to limit resistance."
                    ),
                    affected_brands=[entity.brand_name],
                    active_ingredient=entity.generic_raw,
                )
            )

    return alerts
