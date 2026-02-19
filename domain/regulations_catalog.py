# domain/regulations_catalog.py

"""
Regulations Catalog Module
==========================

This module acts as the central metadata repository for all legal acts
supported by the CRSS infrastructure. It defines the identity, scope,
and jurisdiction of each regulation.

The catalog is used by the scraping and loading stages of the pipeline
to verify CELEX identifiers and to provide human-readable names for
generated reports and folder structures.

Attributes:
    REGULATIONS (dict): A nested dictionary where keys are CELEX IDs
        and values contain metadata including 'name', 'type', and
        'jurisdiction'.
"""

#: Central metadata store for supported EU regulations.
#:
#: Each entry must follow this schema:
#:
#: * **name** (*str*): Human-readable title of the regulation.
#: * **type** (*str*): Category used for downstream processing logic.
#: * **jurisdiction** (*str*): The legal territory (e.g., 'EU').

REGULATIONS = {
    "32017R0745": {
        "name": "MDR 2017/745",
        "type": "medical_device_regulation",
        "jurisdiction": "EU"
    },
    "32024R1689": {
        "name": "EU AI Act",
        "type": "ai_regulation",
        "jurisdiction": "EU"
    }
}
