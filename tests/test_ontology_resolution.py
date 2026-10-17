"""Unit tests for pharmaceutical entity resolution, equivalence keys, and clinical safety locks."""

import pytest
from src.ontology.entity_resolver import (
    check_prescription_safety,
    resolve_medicine_entity,
)
from src.ontology.reference_data import (
    AWaReCategory,
    get_who_aware_category,
    strip_salt_form,
)


def test_metoprolol_salt_and_release_profile_separation():
    """Verify Metoprolol Tartrate Tablet and Metoprolol Succinate Tablet (SR) do NOT share equivalence key."""
    tartrate_entity = resolve_medicine_entity(
        brand_id="101",
        brand_name="Betaloc",
        generic="Metoprolol Tartrate",
        dosage_form="Tablet",
        strength="50 mg",
        medicine_type="allopathic",
    )

    succinate_entity = resolve_medicine_entity(
        brand_id="102",
        brand_name="Selomet-SR",
        generic="Metoprolol Succinate",
        dosage_form="Tablet (Sustained Release)",
        strength="50 mg",
        medicine_type="allopathic",
    )

    assert tartrate_entity.equivalence_key is not None
    assert succinate_entity.equivalence_key is not None

    # Equivalence keys MUST differ
    assert tartrate_entity.equivalence_key != succinate_entity.equivalence_key

    # Salt difference: tartrate vs succinate
    assert tartrate_entity.equivalence_key.exact_salt_tuple == ("tartrate",)
    assert succinate_entity.equivalence_key.exact_salt_tuple == ("succinate",)

    # Release profile difference: IR vs SR
    assert tartrate_entity.equivalence_key.release_profile == "IR"
    assert succinate_entity.equivalence_key.release_profile == "SR"


def test_warfarin_triggers_nti_locked():
    """Verify Warfarin 5 mg Tablet triggers nti_locked = True with citation."""
    warfarin_entity = resolve_medicine_entity(
        brand_id="201",
        brand_name="Warf-5",
        generic="Warfarin Sodium",
        dosage_form="Tablet",
        strength="5 mg",
        medicine_type="allopathic",
    )

    assert warfarin_entity.nti_locked is True
    assert warfarin_entity.is_substitutable is False
    assert warfarin_entity.nti_citation is not None
    assert "Narrow Therapeutic Index" in (warfarin_entity.substitution_block_reason or "")
    assert "Warfarin" in (warfarin_entity.nti_citation or "")


def test_duplicate_active_ingredient_alert():
    """Verify plain Paracetamol brand + Paracetamol + Caffeine brand trigger DUPLICATE_ACTIVE_INGREDIENT."""
    brand_plain = resolve_medicine_entity(
        brand_id="301",
        brand_name="Napa",
        generic="Paracetamol",
        dosage_form="Tablet",
        strength="500 mg",
    )

    brand_compound = resolve_medicine_entity(
        brand_id="302",
        brand_name="Napa Extra",
        generic="Paracetamol + Caffeine",
        dosage_form="Tablet",
        strength="500 mg+65 mg",
    )

    alerts = check_prescription_safety([brand_plain, brand_compound])

    dup_alerts = [a for a in alerts if a.alert_type == "DUPLICATE_ACTIVE_INGREDIENT"]
    assert len(dup_alerts) == 1
    alert = dup_alerts[0]
    assert alert.active_ingredient == "paracetamol"
    assert "Napa" in alert.affected_brands
    assert "Napa Extra" in alert.affected_brands
    assert alert.severity == "HIGH"


def test_who_aware_classifications():
    """Verify WHO AWaRe classification for core antibiotics."""
    amox = get_who_aware_category("Amoxicillin")
    assert amox is not None
    assert amox.category == AWaReCategory.ACCESS
    assert amox.atc_code == "J01CA04"

    cefix = get_who_aware_category("Cefixime")
    assert cefix is not None
    assert cefix.category == AWaReCategory.WATCH
    assert cefix.atc_code == "J01DD08"

    colistin = get_who_aware_category("Colistin")
    assert colistin is not None
    assert colistin.category == AWaReCategory.RESERVE
    assert colistin.atc_code == "J01XB01"


def test_ban_to_inn_synonyms():
    """Verify BAN-to-INN mapping and salt stripping."""
    # Frusemide -> furosemide
    base, salt = strip_salt_form("Frusemide")
    assert base == "furosemide"
    assert salt is None

    # Salbutamol Sulphate -> salbutamol, sulphate
    base, salt = strip_salt_form("Salbutamol Sulphate")
    assert base == "salbutamol"
    assert salt == "sulphate"

    # Amlodipine Besylate -> amlodipine, besylate
    base, salt = strip_salt_form("Amlodipine Besylate")
    assert base == "amlodipine"
    assert salt == "besylate"


def test_herbal_null_strength_block():
    """Verify herbal medicines with null/unstated strength are blocked from equivalence."""
    herbal_entity = resolve_medicine_entity(
        brand_id="401",
        brand_name="Cofsil Herbal",
        generic="Adhatoda vasica + Glycyrrhiza glabra",
        dosage_form="Syrup",
        strength=None,
        medicine_type="herbal",
    )

    assert herbal_entity.is_substitutable is False
    assert herbal_entity.equivalence_key is None
    assert "Herbal medicines with unstated/null strength cannot be safely substituted" in (
        herbal_entity.substitution_block_reason or ""
    )


def test_compound_generic_sorting_determinism():
    """Verify that A + B and B + A produce identical equivalence keys."""
    ent1 = resolve_medicine_entity(
        brand_id="501",
        brand_name="Brand-AB",
        generic="Amlodipine + Atenolol",
        dosage_form="Tablet",
        strength="5 mg+50 mg",
    )

    ent2 = resolve_medicine_entity(
        brand_id="502",
        brand_name="Brand-BA",
        generic="Atenolol + Amlodipine",
        dosage_form="Tablet",
        strength="50 mg+5 mg",
    )

    assert ent1.equivalence_key is not None
    assert ent2.equivalence_key is not None
    assert ent1.equivalence_key.canonical_generic_tuple == ("amlodipine", "atenolol")
    assert ent2.equivalence_key.canonical_generic_tuple == ("amlodipine", "atenolol")


def test_nti_insulin_locked():
    """Verify insulins trigger NTI locks."""
    insulin_entity = resolve_medicine_entity(
        brand_id="601",
        brand_name="Lantus",
        generic="Insulin Glargine",
        dosage_form="Injection",
        strength="100 IU/ml",
    )

    assert insulin_entity.nti_locked is True
    assert insulin_entity.is_substitutable is False
