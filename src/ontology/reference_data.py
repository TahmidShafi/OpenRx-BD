"""Curated deterministic reference tables for WHO AWaRe, NTI restrictions, and BAN-to-INN synonyms."""

from enum import Enum
import re
from typing import NamedTuple, Optional


class AWaReCategory(str, Enum):
    """WHO AWaRe Antibiotic Categorization."""

    ACCESS = "ACCESS"
    WATCH = "WATCH"
    RESERVE = "RESERVE"
    NOT_RECOMMENDED_FDC = "NOT_RECOMMENDED_FDC"


class AntibioticInfo(NamedTuple):
    category: AWaReCategory
    atc_code: str
    description: str


class NTIRecord(NamedTuple):
    generic_name: str
    is_nti: bool
    do_not_substitute: bool
    citation: str


# ==============================================================================
# 1. WHO AWaRe Antibiotic Classifications
# Core systemic antibiotics mapped to ACCESS, WATCH, RESERVE, and NOT_RECOMMENDED_FDC
# ==============================================================================
WHO_AWARE_DATABASE: dict[str, AntibioticInfo] = {
    # --- ACCESS: First and second-choice antibiotics with wider safety margins ---
    "amoxicillin": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01CA04",
        description="First-choice for respiratory, ENT, and dental infections",
    ),
    "amoxicillin + clavulanic acid": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01CR02",
        description="Broad-spectrum beta-lactamase inhibitor combination",
    ),
    "ampicillin": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01CA01",
        description="Aminopenicillin for systemic bacterial infections",
    ),
    "cefalexin": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01DB01",
        description="First-generation cephalosporin for skin and soft tissue",
    ),
    "cefazolin": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01DB04",
        description="Surgical prophylaxis and MSSA bacteremia",
    ),
    "chloramphenicol": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01BA01",
        description="Broad-spectrum bacteriostatic antibiotic",
    ),
    "cloxacillin": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01CF02",
        description="Antistaphylococcal penicillin for MSSA",
    ),
    "co-trimoxazole": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01EE01",
        description="Sulfamethoxazole + Trimethoprim combination",
    ),
    "sulfamethoxazole + trimethoprim": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01EE01",
        description="Sulfamethoxazole + Trimethoprim combination",
    ),
    "doxycycline": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01AA02",
        description="Tetracycline for atypical and vector-borne infections",
    ),
    "gentamicin": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01GB03",
        description="Aminoglycoside for severe gram-negative sepsis",
    ),
    "metronidazole": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01XD01",
        description="Nitroimidazole for anaerobic and protozoal infections",
    ),
    "nitrofurantoin": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01XE01",
        description="Urinary tract antiseptic for uncomplicated cystitis",
    ),
    "phenoxymethylpenicillin": AntibioticInfo(
        category=AWaReCategory.ACCESS,
        atc_code="J01CE02",
        description="Penicillin V for streptococcal pharyngitis",
    ),

    # --- WATCH: Highest priority critically important antimicrobials ---
    "azithromycin": AntibioticInfo(
        category=AWaReCategory.WATCH,
        atc_code="J01FA10",
        description="Macrolide with high risk of selection for resistance",
    ),
    "cefixime": AntibioticInfo(
        category=AWaReCategory.WATCH,
        atc_code="J01DD08",
        description="Oral 3rd-generation cephalosporin requiring strict stewardship",
    ),
    "ceftriaxone": AntibioticInfo(
        category=AWaReCategory.WATCH,
        atc_code="J01DD04",
        description="Parenteral 3rd-generation cephalosporin for severe infections",
    ),
    "cefotaxime": AntibioticInfo(
        category=AWaReCategory.WATCH,
        atc_code="J01DD01",
        description="Parenteral 3rd-generation cephalosporin",
    ),
    "cefuroxime": AntibioticInfo(
        category=AWaReCategory.WATCH,
        atc_code="J01DC02",
        description="2nd-generation cephalosporin with moderate resistance risk",
    ),
    "ciprofloxacin": AntibioticInfo(
        category=AWaReCategory.WATCH,
        atc_code="J01MA02",
        description="Fluoroquinolone with high collateral damage on microflora",
    ),
    "levofloxacin": AntibioticInfo(
        category=AWaReCategory.WATCH,
        atc_code="J01MA12",
        description="Respiratory fluoroquinolone with cardiac and tendon toxicity risk",
    ),
    "clarithromycin": AntibioticInfo(
        category=AWaReCategory.WATCH,
        atc_code="J01FA09",
        description="Macrolide for H. pylori and atypical pneumonia",
    ),
    "meropenem": AntibioticInfo(
        category=AWaReCategory.WATCH,
        atc_code="J01DH02",
        description="Carbapenem reserved for multidrug-resistant gram-negative bacteria",
    ),
    "piperacillin + tazobactam": AntibioticInfo(
        category=AWaReCategory.WATCH,
        atc_code="J01CR05",
        description="Antipseudomonal penicillin combination",
    ),
    "vancomycin": AntibioticInfo(
        category=AWaReCategory.WATCH,
        atc_code="J01XA01",
        description="Glycopeptide for MRSA and C. difficile",
    ),

    # --- RESERVE: Last-resort options for confirmed or suspected MDR pathogens ---
    "colistin": AntibioticInfo(
        category=AWaReCategory.RESERVE,
        atc_code="J01XB01",
        description="Polymyxin E for carbapenem-resistant gram-negative pathogens",
    ),
    "polymyxin b": AntibioticInfo(
        category=AWaReCategory.RESERVE,
        atc_code="J01XB02",
        description="Last-resort polymyxin for MDR Acinetobacter and Pseudomonas",
    ),
    "linezolid": AntibioticInfo(
        category=AWaReCategory.RESERVE,
        atc_code="J01XX08",
        description="Oxazolidinone for VRE and complicated MRSA infections",
    ),
    "tigecycline": AntibioticInfo(
        category=AWaReCategory.RESERVE,
        atc_code="J01AA12",
        description="Glycylcycline for complicated MDR intra-abdominal infections",
    ),
    "ceftazidime + avibactam": AntibioticInfo(
        category=AWaReCategory.RESERVE,
        atc_code="J01DD52",
        description="Beta-lactamase inhibitor combo for KPC and OXA-48 carbapenemases",
    ),

    # --- NOT_RECOMMENDED_FDC: Irrational fixed-dose antibiotic combinations ---
    "cefixime + azithromycin": AntibioticInfo(
        category=AWaReCategory.NOT_RECOMMENDED_FDC,
        atc_code="J01RA99",
        description="WHO AWaRe discounselled FDC: Accelerated dual-resistance risk",
    ),
    "ciprofloxacin + tinidazole": AntibioticInfo(
        category=AWaReCategory.NOT_RECOMMENDED_FDC,
        atc_code="J01RA98",
        description="WHO AWaRe discounselled FDC: Suboptimal dual empirical exposure",
    ),
}


# ==============================================================================
# 2. Narrow Therapeutic Index (NTI) & Do-Not-Substitute Generics List
# ==============================================================================
NTI_DATABASE: dict[str, NTIRecord] = {
    "warfarin": NTIRecord(
        generic_name="Warfarin",
        is_nti=True,
        do_not_substitute=True,
        citation=(
            "Warfarin NTI: FDA Narrow Therapeutic Index Guidance; Holford NHG. Clinical Pharmacokinetics 1986; "
            "Small variations in bioavailability cause catastrophic thrombosis or hemorrhage."
        ),
    ),
    "levothyroxine": NTIRecord(
        generic_name="Levothyroxine",
        is_nti=True,
        do_not_substitute=True,
        citation=(
            "Levothyroxine NTI: American Thyroid Association Clinical Practice Guidelines; FDA Orange Book Narrow "
            "Therapeutic Ratio; 12.5 mcg formulation shifts cause iatrogenic thyrotoxicosis or myxedema."
        ),
    ),
    "digoxin": NTIRecord(
        generic_name="Digoxin",
        is_nti=True,
        do_not_substitute=True,
        citation=(
            "Digoxin NTI: FDA NTI Classification; Bauman JL et al. Arch Intern Med 2006; "
            "Therapeutic window 0.5-0.9 ng/mL; minor bioavailability variance triggers fatal arrhythmias."
        ),
    ),
    "phenytoin": NTIRecord(
        generic_name="Phenytoin",
        is_nti=True,
        do_not_substitute=True,
        citation=(
            "Phenytoin NTI: FDA NTI Guidance; Browne TR. Clinical Neuropharmacology 1990; "
            "Michaelis-Menten non-linear saturable metabolism makes bioequivalent substitution dangerous."
        ),
    ),
    "carbamazepine": NTIRecord(
        generic_name="Carbamazepine",
        is_nti=True,
        do_not_substitute=True,
        citation=(
            "Carbamazepine NTI: FDA NTI Guidance; Bialer M. Clin Pharmacokinet 1993; "
            "Autoinduction and narrow therapeutic range (4-12 mcg/mL) forbid automated brand switching."
        ),
    ),
    "lithium": NTIRecord(
        generic_name="Lithium",
        is_nti=True,
        do_not_substitute=True,
        citation=(
            "Lithium NTI: APA Practice Guidelines for Bipolar Disorder; Grandjean EM et al. J Clin Psychopharmacol; "
            "Target serum range 0.6-1.0 mEq/L; toxic threshold begins at 1.5 mEq/L."
        ),
    ),
    "theophylline": NTIRecord(
        generic_name="Theophylline",
        is_nti=True,
        do_not_substitute=True,
        citation=(
            "Theophylline NTI: FDA NTI Classification; Weinberger M. N Engl J Med 1996; "
            "Therapeutic range 5-15 mcg/mL; minor absorption shifts precipitate seizures and tachycardia."
        ),
    ),
    "tacrolimus": NTIRecord(
        generic_name="Tacrolimus",
        is_nti=True,
        do_not_substitute=True,
        citation=(
            "Tacrolimus NTI: KDIGO Clinical Practice Guideline for Kidney Transplant Recipients; "
            "AUC variations lead to acute allograft rejection or irreversible nephrotoxicity."
        ),
    ),
    "cyclosporine": NTIRecord(
        generic_name="Cyclosporine",
        is_nti=True,
        do_not_substitute=True,
        citation=(
            "Cyclosporine NTI: FDA NTI Guidance; Kahan BD. Transplant Proc; "
            "Critical immunosuppressive index; modified microemulsion and non-modified forms non-interchangeable."
        ),
    ),
    "insulin": NTIRecord(
        generic_name="Insulin",
        is_nti=True,
        do_not_substitute=True,
        citation=(
            "Insulin NTI: ADA Standards of Medical Care in Diabetes; Biologic variability and pharmacokinetic profiles "
            "of human vs analog and rapid vs basal insulins prohibit generic interchange."
        ),
    ),
}

# Insulin active ingredient keywords that automatically trigger NTI lock
INSULIN_KEYWORDS: tuple[str, ...] = (
    "insulin",
    "glargine",
    "aspart",
    "lispro",
    "detemir",
    "degludec",
    "isophane",
)


# ==============================================================================
# 3. BAN-to-INN Synonym Mapping
# ==============================================================================
BAN_TO_INN_SYNONYMS: dict[str, str] = {
    "paracetamol": "acetaminophen",
    "frusemide": "furosemide",
    "adrenaline": "epinephrine",
    "salbutamol": "albuterol",
    "noradrenaline": "norepinephrine",
    "amphotericin": "amphotericin b",
    "cholecalciferol": "vitamin d3",
    "ergocalciferol": "vitamin d2",
    "dothiepin": "dosulepin",
    "trimeprazine": "alimemazine",
    "flupenthixol": "flupentixol",
    "lignocaine": "lidocaine",
    "tetracosactrin": "tetracosactide",
    "hydroxyurea": "hydroxycarbamide",
    "glibenclamide": "glyburide",
    "cyproterone": "cyproterone acetate",
    "dicyclomine": "dicycloverine",
}

# Reverse mapping INN -> BAN for canonical normalization
INN_TO_CANONICAL: dict[str, str] = {
    "acetaminophen": "paracetamol",
    "furosemide": "furosemide",
    "albuterol": "salbutamol",
    "epinephrine": "epinephrine",
    "norepinephrine": "norepinephrine",
}


# ==============================================================================
# 4. Pharmaceutical Salts & Esters Dictionary
# ==============================================================================
PHARMACEUTICAL_SALTS: tuple[str, ...] = (
    "hydrochloride",
    "hcl",
    "succinate",
    "tartrate",
    "maleate",
    "sodium",
    "potassium",
    "besylate",
    "besilate",
    "mesylate",
    "mesilate",
    "fumarate",
    "phosphate",
    "sulfate",
    "sulphate",
    "nitrate",
    "acetate",
    "calcium",
    "citrate",
    "gluconate",
    "propionate",
    "dipropionate",
    "valerate",
    "trihydrate",
    "dihydrate",
    "monohydrate",
    "hydrate",
    "estolate",
    "ethylsuccinate",
    "stearate",
    "benzoate",
    "palmitate",
    "bromide",
    "chloride",
    "iodide",
    "disodium",
    "lactate",
    "gluceptate",
    "aspartate",
    "carbonate",
    "oxide",
    "hydroxide",
)


def strip_salt_form(generic_token: str) -> tuple[str, Optional[str]]:
    """Strip pharmaceutical salt/ester from active ingredient while preserving salt_form.

    Example:
        'Metoprolol Tartrate' -> ('metoprolol', 'tartrate')
        'Metoprolol Succinate' -> ('metoprolol', 'succinate')
        'Paracetamol' -> ('paracetamol', None)
        'Amlodipine Besylate' -> ('amlodipine', 'besylate')
    """
    token = generic_token.strip().lower()
    # Remove bracketed and parenthesized details
    token = re.sub(r"\[.*?\]|\(.*?\)", "", token).strip()

    detected_salt: Optional[str] = None
    clean_base = token

    # Check for known salts at the end or as separate tokens
    for salt in PHARMACEUTICAL_SALTS:
        # Match salt as a separate trailing word or word preceded by space
        pattern = rf"\b{re.escape(salt)}\b"
        if re.search(pattern, clean_base, re.IGNORECASE):
            detected_salt = salt
            clean_base = re.sub(pattern, "", clean_base, flags=re.IGNORECASE).strip()
            # Clean up residual punctuation or whitespace
            clean_base = re.sub(r"\s+", " ", clean_base).strip()
            break

    # Normalize BAN/INN synonyms on base
    canonical_base = BAN_TO_INN_SYNONYMS.get(clean_base, clean_base)
    # Also resolve INN to preferred canonical naming
    canonical_base = INN_TO_CANONICAL.get(canonical_base, canonical_base)

    return canonical_base, detected_salt


def check_nti_status(generic_string: str) -> tuple[bool, Optional[str]]:
    """Check if a generic string contains any Narrow Therapeutic Index (NTI) drug."""
    s_lower = generic_string.strip().lower()

    # Check direct NTI drugs
    for drug_key, record in NTI_DATABASE.items():
        if re.search(rf"\b{re.escape(drug_key)}\b", s_lower):
            return True, record.citation

    # Check insulin keywords
    for kw in INSULIN_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", s_lower):
            return True, NTI_DATABASE["insulin"].citation

    return False, None


def get_who_aware_category(generic_string: str) -> Optional[AntibioticInfo]:
    """Retrieve WHO AWaRe antibiotic category for a generic string."""
    s_lower = generic_string.strip().lower()
    # Normalize synonyms
    s_normalized = re.sub(r"\s*\+\s*", " + ", s_lower)

    # Check exact match first
    if s_normalized in WHO_AWARE_DATABASE:
        return WHO_AWARE_DATABASE[s_normalized]

    # Check substring or individual component match
    for abx_name, info in WHO_AWARE_DATABASE.items():
        if abx_name in s_normalized:
            return info

    return None
