# domain/ontology/roles.py

"""Business-level role vocabularies and detectors for EU regulations."""
from __future__ import annotations

import re
from typing import Dict, List, Mapping, Sequence

RoleKeywordMap = Mapping[str, Mapping[str, Sequence[str]]]


EU_AI_ROLE_KEYWORDS: RoleKeywordMap = {
    "EN": {
        "provider": [
            r"provider",
            r"supplier",
            r"manufacturer",
            r"developer",
            r"downstream provider",
        ],
        "deployer": [
            r"deployer",
            r"operator",
            r"user in own authority",
            r"integrator",
        ],
        "authorised_representative": [r"authorised representative"],
        "importer": [r"importer"],
        "distributor": [r"distributor", r"reseller"],
        "operator": [
            r"operator",
            r"actor",
            r"provider",
            r"manufacturer",
            r"deployer",
            r"authorised representative",
            r"importer",
            r"distributor",
        ],
        "notified_body": [r"notified body", r"conformity assessment body"],
        "regulator": [
            r"AI Office",
            r"national competent authority",
            r"market surveillance authority",
            r"notifying authority",
            r"law enforcement authority",
        ],
        "sponsor": [r"sponsor", r"clinical sponsor"],
    },
    "FR": {
        "provider": [r"fournisseur", r"fabricant", r"prestataire", r"fournisseur en aval"],
        "deployer": [r"d\u00e9ployeur", r"op\u00e9rateur", r"utilisateur en propre", r"int\u00e9grateur"],
        "authorised_representative": [r"mandataire"],
        "importer": [r"importateur"],
        "distributor": [r"distributeur"],
        "operator": [
            r"op\u00e9rateur",
            r"acteur",
            r"fournisseur",
            r"fabricant",
            r"d\u00e9ployeur",
            r"mandataire",
            r"importateur",
            r"distributeur",
        ],
        "notified_body": [r"organisme notifi\u00e9", r"organisme d'\u00e9valuation de la conformit\u00e9"],
        "regulator": [
            r"Bureau de l'IA",
            r"autorite nationale competente",
            r"autorite de surveillance du marche",
            r"autorite notifiante",
            r"autorites repressives",
        ],
        "sponsor": [r"sponsor", r"sponsor clinique"],
    },
    "DE": {
        "provider": [r"anbieter", r"hersteller", r"lieferant", r"nachgelagerter anbieter"],
        "deployer": [r"betreiber", r"nutzer in eigener verantwortung", r"integrator"],
        "authorised_representative": [r"bevollmaechtigter"],
        "importer": [r"einfuehrer"],
        "distributor": [r"haendler"],
        "operator": [
            r"betreiber",
            r"akteur",
            r"anbieter",
            r"hersteller",
            r"bevollmaechtigter",
            r"einfuehrer",
            r"haendler",
        ],
        "notified_body": [r"notifizierte stelle", r"konformitaetsbewertungsstelle"],
        "regulator": [
            r"Buro fuer Kuenstliche Intelligenz",
            r"zustaendige nationale Behoerde",
            r"marktueberwachungsbehoerde",
            r"notifizierende Behoerde",
            r"Strafverfolgungsbehoerde",
        ],
        "sponsor": [r"sponsor", r"klinischer sponsor"],
    },
}


MDR_ROLE_KEYWORDS: RoleKeywordMap = {
    "EN": {
        "manufacturer": [r"manufacturer", r"legal manufacturer"],
        "authorised_representative": [r"authorised representative"],
        "importer": [r"importer"],
        "distributor": [r"distributor", r"economic operator"],
        "notified_body": [r"notified body"],
        "conformity_assessment_body": [r"conformity assessment body"],
        "sponsor": [r"sponsor", r"clinical sponsor"],
        "investigator": [r"investigator"],
        "ethics_committee": [r"ethics committee"],
        "health_institution": [r"health institution", r"hospital"],
        "user": [r"user", r"professional user", r"healthcare professional"],
        "lay_person": [r"lay person", r"lay user"],
        "market_surveillance_authority": [r"market surveillance authority"],
        "competent_authority": [r"competent authority"],
    },
    "FR": {
        "manufacturer": [r"fabricant"],
        "authorised_representative": [r"mandataire"],
        "importer": [r"importateur"],
        "distributor": [r"distributeur", r"operateur economique"],
        "notified_body": [r"organisme notifie"],
        "conformity_assessment_body": [r"organisme d'evaluation de la conformite"],
        "sponsor": [r"sponsor", r"promoteur"],
        "investigator": [r"investigateur"],
        "ethics_committee": [r"comite d'ethique"],
        "health_institution": [r"etablissement de sante"],
        "user": [r"utilisateur", r"professionnel de sante"],
        "lay_person": [r"profane"],
        "market_surveillance_authority": [r"autorite de surveillance du marche"],
        "competent_authority": [r"autorite competente"],
    },
    "DE": {
        "manufacturer": [r"hersteller"],
        "authorised_representative": [r"bevollmaechtigter"],
        "importer": [r"einfuehrer"],
        "distributor": [r"haendler", r"wirtschaftsakteur"],
        "notified_body": [r"benannte stelle"],
        "conformity_assessment_body": [r"konformitaetsbewertungsstelle"],
        "sponsor": [r"sponsor"],
        "investigator": [r"pruefer"],
        "ethics_committee": [r"ethik-kommission"],
        "health_institution": [r"gesundheitseinrichtung"],
        "user": [r"anwender", r"gesundheitsfachkraft"],
        "lay_person": [r"laie"],
        "market_surveillance_authority": [r"marktueberwachungsbehoerde"],
        "competent_authority": [r"zustaendige behoerde"],
    },
}


def detect_roles_from_keywords(text: str, lang: str, keyword_map: RoleKeywordMap) -> List[str]:
    """Return normalized role tags found in ``text`` for ``lang`` using the keyword map."""

    lang = (lang or "EN").upper()
    patterns = keyword_map.get(lang, keyword_map.get("EN", {}))
    detected: List[str] = []
    for role, regexes in patterns.items():
        for pat in regexes:
            suffix_pattern = r"(?:['’]s|\w+)?"
            flexible_pattern = rf"\b{pat}{suffix_pattern}\b"
            if re.search(flexible_pattern, text, re.I):
                detected.append(role)
                break
    return detected


def eu_ai_role_detector(text: str, lang: str) -> List[str]:
    """Detector specific to EU AI Act actors (EN/FR/DE)."""

    return detect_roles_from_keywords(text, lang, EU_AI_ROLE_KEYWORDS)


def mdr_role_detector(text: str, lang: str) -> List[str]:
    """Detector specific to MDR actors (EN/FR/DE)."""

    return detect_roles_from_keywords(text, lang, MDR_ROLE_KEYWORDS)


__all__ = [
    "RoleKeywordMap",
    "EU_AI_ROLE_KEYWORDS",
    "MDR_ROLE_KEYWORDS",
    "detect_roles_from_keywords",
    "eu_ai_role_detector",
    "mdr_role_detector",
]
