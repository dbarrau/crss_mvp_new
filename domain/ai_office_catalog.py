"""
AI Office Catalog Module
========================

Central metadata repository for European AI Office guidance documents
(published under the AI Act) supported by the CRSS infrastructure.

These are *guidance* documents in exactly the same sense as the MDCG
guidance in :mod:`domain.mdcg_catalog` — non-binding interpretive texts
parsed from PDF via LlamaParse — so every entry shares the MDCG entry
shape. The only extra field is ``parse_profile``, which selects the
LlamaParse prompt + post-processing profile (AI Office docs have no
"Medical Devices" running headers and no MDCG decision-tree flowcharts,
so they use the ``"ai_office"`` profile instead of the ``"mdcg"`` one).

Consumers should import the *merged* view
(:data:`domain.guidance_catalog.GUIDANCE_DOCUMENTS`) rather than this
dict directly, so routing and the build DAG stay family-agnostic.

Note:
    Download URLs come from the AI Office / DAE newsroom. The redirect
    form ``https://ec.europa.eu/newsroom/dae/redirection/document/<id>``
    resolves to the PDF and is handled transparently by
    :func:`ingestion.scrape.download_guidance.download_guidance_pdf`
    (``requests`` follows the redirect), so no special handling is needed.
"""

# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: Core AI Act interpretive guidance (highest priority)
# ═══════════════════════════════════════════════════════════════════════════

AI_OFFICE_DOCUMENTS = {
    "AI_OFFICE_2025_TRANSPARENCY_ART50": {
        "tier": 1,
        "parse_profile": "ai_office",
        "name": "AI Office Guidelines on Transparency (Art. 50 AI Act)",
        "title": (
            "Guidelines on the implementation of the transparency "
            "obligations for certain AI systems under Article 50 of the "
            "AI Act"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "ai_office_transparency_art50_en.pdf",
        "download_url": "https://ec.europa.eu/newsroom/dae/redirection/document/131215",
    },
    "AI_OFFICE_2025_HIGHRISK_PRINCIPLES": {
        "tier": 1,
        "parse_profile": "ai_office",
        "name": "AI Office Draft Guidelines on High-Risk Classification: General Principles",
        "title": (
            "Draft Guidelines on the classification of high-risk AI: "
            "General principles"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "ai_office_highrisk_principles_en.pdf",
        "download_url": "https://ec.europa.eu/newsroom/dae/redirection/document/128559",
    },
    "AI_OFFICE_2025_HIGHRISK_ANNEX_I": {
        "tier": 1,
        "parse_profile": "ai_office",
        "name": "AI Office Draft Guidelines on High-Risk Classification: Annex I",
        "title": (
            "Draft Guidelines on the classification of high-risk AI systems: "
            "Annex I of AI Act"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "ai_office_highrisk_annex_i_en.pdf",
        "download_url": "https://ec.europa.eu/newsroom/dae/redirection/document/128560",
    },
    "AI_OFFICE_2025_HIGHRISK_ANNEX_III": {
        "tier": 1,
        "parse_profile": "ai_office",
        "name": "AI Office Draft Guidelines on High-Risk Classification: Annex III",
        "title": (
            "Draft Guidelines on the classification of high-risk AI systems: "
            "Annex III of AI Act"
        ),
        "type": "guidance",
        "jurisdiction": "EU",
        "pdf_filename": "ai_office_highrisk_annex_iii_en.pdf",
        "download_url": "https://ec.europa.eu/newsroom/dae/redirection/document/128561",
    },
}
