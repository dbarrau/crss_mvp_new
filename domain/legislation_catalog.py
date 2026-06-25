# domain/legislation_catalog.py

"""
Legislation Catalog Module
==========================

Central metadata repository for all EU legislative acts (Regulations,
Directives, Decisions) supported by the CRSS infrastructure.

The catalog is used by the scraping and loading stages of the pipeline
to verify CELEX identifiers and to provide human-readable names for
generated reports and folder structures.

Attributes:
    LEGISLATION (dict): A nested dictionary where keys are CELEX IDs
        and values contain metadata including 'name', 'type', and
        'jurisdiction'.
"""

#: Canonical CELEX identifiers — the single source of truth for these codes.
#: Reference these named constants from application + test code instead of
#: re-hardcoding the literal CELEX string (which drifts and obscures intent).
#: They are also the keys of :data:`LEGISLATION` below, so there is exactly one
#: place each code is written.
MDR_CELEX = "32017R0745"
AI_ACT_CELEX = "32024R1689"
IVDR_CELEX = "32017R0746"
GDPR_CELEX = "32016R0679"
CIR_CELEX = "32026R0977"  # Commission Implementing Regulation (EU) 2026/977

#: Central metadata store for supported EU legislation.
#:
#: Each entry must follow this schema:
#:
#: * **name** (*str*): Human-readable title of the legal act.
#: * **type** (*str*): Category used for downstream processing logic.
#: * **jurisdiction** (*str*): The legal territory (e.g., 'EU').

LEGISLATION = {
    MDR_CELEX: {
        "name": "MDR 2017/745",
        "number": "2017/745",
        "type": "medical_device_regulation",
        "jurisdiction": "EU",
        "source_celex": "02017R0745-20260101",
    },
    AI_ACT_CELEX: {
        "name": "EU AI Act",
        "number": "2024/1689",
        "type": "ai_regulation",
        "jurisdiction": "EU",
    },
    IVDR_CELEX: {
        "name": "IVDR 2017/746",
        "number": "2017/746",
        "type": "in_vitro_diagnostic_regulation",
        "jurisdiction": "EU",
        "source_celex": "02017R0746-20250110",
    },
    GDPR_CELEX: {
        "name": "General Data Protection Regulation (GDPR) 2016/679",
        "number": "2016/679",
        "type": "data_protection_regulation",
        "jurisdiction": "EU",
        "source_celex": "02016R0679-20160504",
    },
    CIR_CELEX: {
        "name": "Commission Implementing Regulation (EU) 2026/977",
        "number": "2026/977",
        "type": "medical_device_implementing_regulation",
        "jurisdiction": "EU",
    }
}

# Backward-compatible alias
REGULATIONS = LEGISLATION
