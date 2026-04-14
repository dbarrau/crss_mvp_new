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

#: Central metadata store for supported EU legislation.
#:
#: Each entry must follow this schema:
#:
#: * **name** (*str*): Human-readable title of the legal act.
#: * **type** (*str*): Category used for downstream processing logic.
#: * **jurisdiction** (*str*): The legal territory (e.g., 'EU').

LEGISLATION = {
    "32017R0745": {
        "name": "MDR 2017/745",
        "number": "2017/745",
        "type": "medical_device_regulation",
        "jurisdiction": "EU",
        "source_celex": "02017R0745-20260101",
    },
    "32024R1689": {
        "name": "EU AI Act",
        "number": "2024/1689",
        "type": "ai_regulation",
        "jurisdiction": "EU",
    },
    "32017R0746": {
        "name": "IVDR 2017/746",
        "number": "2017/746",
        "type": "in_vitro_diagnostic_regulation",
        "jurisdiction": "EU",
        "source_celex": "02017R0746-20250110",
    },
}

# Backward-compatible alias
REGULATIONS = LEGISLATION
