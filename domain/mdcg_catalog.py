"""
MDCG Catalog Module
===================

Central metadata repository for MDCG (Medical Device Coordination Group)
guidance documents supported by the CRSS infrastructure.

Used by the ingestion pipeline to resolve document identifiers, locate
source PDFs, and route to the correct parser.

Attributes:
    MDCG_DOCUMENTS (dict): A dictionary where keys are document IDs
        (e.g., ``MDCG_2020_3``) and values contain metadata including
        ``name``, ``title``, ``pdf_filename``, and ``download_url``.

Note:
    Download URLs can be found at https://health.ec.europa.eu/ by searching
    for the MDCG document number. The URL format is:
    https://health.ec.europa.eu/document/download/{UUID}_en?filename={name}.pdf
"""

# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: Core clinical + AI/ML guidance (highest priority)
# ═══════════════════════════════════════════════════════════════════════════

MDCG_DOCUMENTS = {
    "MDCG_2020_3": {
        "name": "MDCG 2020-3 Rev.1",
        "title": (
            "Guidance on significant changes regarding the transitional "
            "provision under Article 120 of MDR with regard to devices "
            "covered by certificates according to MDD or AIMDD"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2020_3_rev1.pdf",
        "download_url": "https://health.ec.europa.eu/document/download/800e8e87-d4eb-4cc5-b5ad-07a9146d7c90_en?filename=mdcg_2020-3_en_1.pdf",
    },
    "MDCG_2019_11": {
        "name": "MDCG 2019-11",
        "title": (
            "Guidance on Qualification and Classification of Software "
            "in Regulation (EU) 2017/745 – MDR and Regulation (EU) "
            "2017/746 – IVDR"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2019_11_en.pdf",
        "download_url": "https://health.ec.europa.eu/document/download/b45335c5-1679-4c71-a91c-fc7a4d37f12b_en?filename=mdcg_2019_11_en.pdf"
    },
    "MDCG_2020_6": {
        "name": "MDCG 2020-6",
        "title": (
            "Regulation (EU) 2017/745: Clinical evidence needed formedical devices previously CE marked underDirectives 93/42/EEC"
            "or 90/385/EECA guide for manufacturers and notified bodies"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2020_6_en.pdf",
        "download_url": "https://health.ec.europa.eu/system/files/2020-09/md_mdcg_2020_6_guidance_sufficient_clinical_evidence_en_0.pdf"
    },
    "MDCG_2020_5": {
        "name": "MDCG 2020-5",
        "title": (
            "Clinical Evaluation - Equivalence"
            "A guide for manufacturers and notified bodies"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2020_5_en.pdf",
        "download_url": "https://health.ec.europa.eu/system/files/2020-09/md_mdcg_2020_5_guidance_clinical_evaluation_equivalence_en_0.pdf"
    },
    "MDCG_2020_13": {
        "name": "MDCG 2020-13",
        "title": (
            "Clinical evaluation assessment report template"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2020_13_en.pdf",
        "download_url": "https://health.ec.europa.eu/system/files/2020-07/mdcg_clinical_evaluationtemplate_en_0.pdf"
    },
    "MDCG_2023_3": {
        "name": "MDCG 2023-3 Rev.2",
        "title": (
            "Questions and Answers on vigilance termsand concepts as outlined in the Regulation(EU) 2017/745 and Regulation (EU) 2017/746"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2023_3_en.pdf",
        "download_url": "https://health.ec.europa.eu/document/download/af1433fd-ed64-4c53-abc7-612a7f16f976_en?filename=mdcg_2023-3_en.pdf"
    },
    "MDCG_2019_5": {
        "name": "MDCG 2019-5",
        "title": (
            "Guidance on Technical Documentation for medical devices "
            "covered by Regulation (EU) 2017/745"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2019_5_en.pdf",
        "download_url": None,  # TODO: Add download URL
    },
    "MDCG_2025_6": {
        "name": "MDCG 2025-6",
        "title": (
            "Interplay between the Medical DevicesRegulation (MDR) & In vitro Diagnostic"
            "Medical Devices Regulation (IVDR) andthe Artificial Intelligence Act (AIA)"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2025_6_en.pdf",
        "download_url": "https://health.ec.europa.eu/document/download/b78a17d7-e3cd-4943-851d-e02a2f22bbb4_en?filename=mdcg_2025-6_en.pdf",
    },

# ═══════════════════════════════════════════════════════════════════════════
# TIER 2: Post-market + classification guidance
# ═══════════════════════════════════════════════════════════════════════════

    "MDCG_2020_7": {
        "name": "MDCG 2020-7",
        "title": (
            "Post-Market Clinical Follow-up (PMCF) Plan Template - "
            "A guide for manufacturers and notified bodies"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2020_7_en.pdf",
        "download_url": None,  # TODO: Add download URL
    },
    "MDCG_2022_21": {
        "name": "MDCG 2022-21",
        "title": (
            "Guidance on Periodic Safety Update Report (PSUR) according to "
            "Regulation (EU) 2017/745"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2022_21_en.pdf",
        "download_url": None,  # TODO: Add download URL
    },
    "MDCG_2021_24": {
        "name": "MDCG 2021-24",
        "title": (
            "Guidance on classification of medical devices"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2021_24_en.pdf",
        "download_url": None,  # TODO: Add download URL
    },

# ═══════════════════════════════════════════════════════════════════════════
# TIER 3: QMS, standards, borderline cases (nice to have)
# ═══════════════════════════════════════════════════════════════════════════

    "MDCG_2021_6": {
        "name": "MDCG 2021-6",
        "title": (
            "Guidance on sufficient clinical evidence for legacy devices - "
            "Guidance for manufacturers and notified bodies"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2021_6_en.pdf",
        "download_url": None,  # TODO: Add download URL
    },
    "MDCG_2021_5": {
        "name": "MDCG 2021-5",
        "title": (
            "Guidance on Harmonised Standards (HS) - Common Specifications (CS) "
            "and scientific opinions"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2021_5_en.pdf",
        "download_url": None,  # TODO: Add download URL
    },
    "MDCG_2022_5": {
        "name": "MDCG 2022-5",
        "title": (
            "Guidance on borderline between Medical Devices Regulation (EU) "
            "2017/745 and Medicinal Products Directive 2001/83/EC"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "mdcg_2022_5_en.pdf",
        "download_url": None,  # TODO: Add download URL
    },
}
