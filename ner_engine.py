"""
MahaNER Engine
--------------
Production inference, model caching, Devanagari subword re-assembly,
multi-word BIO entity aggregation, Devanagari inflectional normalization & stemming,
and multi-model comparative analysis for Marathi NLP.
"""

import re
import html
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("MahaNER-Engine")

# ─── Model Registry ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "l3cube-pune/marathi-ner": {
        "id": "l3cube-pune/marathi-ner",
        "name": "L3Cube-MahaBERT",
        "marathi_name": "महाबीईआरटी (MahaBERT)",
        "architecture": "Monolingual Marathi BERT",
        "focus": "Specialized Marathi Corpora (25k+ sentences)",
        "badge": "Primary / Monolingual",
        "badge_color": "#DC2626",
        "classes": ["PER", "LOC", "ORG", "MISC"],
        "description": "Fine-tuned specifically on L3Cube-MahaNER dataset. High recall on native Marathi idioms, historic entities, and MISC."
    },
    "ai4bharat/IndicNER": {
        "id": "ai4bharat/IndicNER",
        "name": "AI4Bharat-IndicBERT",
        "marathi_name": "इंडिकबीईआरटी (IndicBERT)",
        "architecture": "Multilingual Indic ALBERT/BERT",
        "focus": "11 Major Indian Languages",
        "badge": "Multilingual Indic",
        "badge_color": "#0284C7",
        "classes": ["PER", "LOC", "ORG"],
        "description": "Trained by AI4Bharat on multi-lingual Indic corpora. High precision on core Person and Location entities in Indian contexts."
    },
    "Davlan/xlm-roberta-base-ner-hrl": {
        "id": "Davlan/xlm-roberta-base-ner-hrl",
        "name": "XLM-RoBERTa-HRL",
        "marathi_name": "एक्सएलएम-रॉबर्टा (XLM-RoBERTa)",
        "architecture": "Cross-Lingual RoBERTa Base",
        "focus": "High-Resource Cross-Lingual Transfer",
        "badge": "Cross-Lingual",
        "badge_color": "#16A34A",
        "classes": ["PER", "LOC", "ORG"],
        "description": "Cross-lingual transformer fine-tuned on high-resource language entity recognition. Strong zero-shot transfer capabilities."
    }
}

# Configuration for visual rendering and taxonomy
ENTITY_CONFIG = {
    "PER": {
        "marathi": "व्यक्ती",
        "english": "Person",
        "label": "व्यक्ती (PER)",
        "bg_color": "rgba(239, 68, 68, 0.16)",       # Coral / Light Red tint
        "text_color": "#EF4444",
        "border_color": "#EF4444",
        "badge_bg": "#DC2626",
        "icon": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px; display: inline-block;"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
        "description": "व्यक्तीचे नाव, ऐतिहासिक व्यक्तिमत्त्वे, नेते किंवा प्रसिद्ध व्यक्ती"
    },
    "LOC": {
        "marathi": "ठिकाण",
        "english": "Location",
        "label": "ठिकाण (LOC)",
        "bg_color": "rgba(2, 132, 199, 0.16)",       # Sky Blue / Cyan tint
        "text_color": "#0284C7",
        "border_color": "#0284C7",
        "badge_bg": "#0284C7",
        "icon": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px; display: inline-block;"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>',
        "description": "भौगोलिक ठिकाणे, शहरे, किल्ले, राज्ये, मैदाने आणि देश"
    },
    "ORG": {
        "marathi": "संस्था",
        "english": "Organization",
        "label": "संस्था (ORG)",
        "bg_color": "rgba(22, 163, 74, 0.16)",       # Emerald Green tint
        "text_color": "#16A34A",
        "border_color": "#16A34A",
        "badge_bg": "#16A34A",
        "icon": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px; display: inline-block;"><rect width="16" height="20" x="4" y="2" rx="2" ry="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01"/><path d="M16 6h.01"/><path d="M12 6h.01"/><path d="M12 10h.01"/><path d="M12 14h.01"/><path d="M16 10h.01"/><path d="M16 14h.01"/><path d="M8 10h.01"/><path d="M8 14h.01"/></svg>',
        "description": "शासकीय किंवा खाजगी संस्था, कंपन्या, क्रीडा संस्था, संघटना"
    },
    "MISC": {
        "marathi": "इतर",
        "english": "Miscellaneous",
        "label": "इतर (MISC)",
        "bg_color": "rgba(147, 51, 234, 0.16)",       # Violet / Purple tint
        "text_color": "#9333EA",
        "border_color": "#9333EA",
        "badge_bg": "#9333EA",
        "icon": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px; display: inline-block;"><path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z"/><path d="M7 7h.01"/></svg>',
        "description": "मोहिमा, प्रकल्प, सण, क्रीडा स्पर्धा, ऐतिहासिक चळवळी किंवा इतर घटक"
    }
}

# Tag Normalization Dictionary
TAG_MAP = {
    "PER": "PER", "B-PER": "PER", "I-PER": "PER", "PERSON": "PER", "B-PERSON": "PER", "I-PERSON": "PER",
    "DESIGNATION": "PER", "B-DESIGNATION": "PER", "I-DESIGNATION": "PER",
    "TITLE": "PER", "B-TITLE": "PER", "I-TITLE": "PER",
    "LOC": "LOC", "B-LOC": "LOC", "I-LOC": "LOC", "LOCATION": "LOC", "B-LOCATION": "LOC", "I-LOCATION": "LOC", "GPE": "LOC",
    "ORG": "ORG", "B-ORG": "ORG", "I-ORG": "ORG", "ORGANIZATION": "ORG", "B-ORGANIZATION": "ORG", "I-ORGANIZATION": "ORG",
    "MISC": "MISC", "B-MISC": "MISC", "I-MISC": "MISC", "MISCELLANEOUS": "MISC", "EVENT": "MISC"
}

# ─── Case Markers & Postposition Suffix Rules (विभक्ती प्रत्यय नियम) ───────────────
# Ordered from longest to shortest to ensure greedy suffix identification
CASE_SUFFIX_RULES: List[Tuple[str, str, str]] = [
    # Compound postpositions (शब्दयोगी अव्यये)
    ("च्यासाठी", "चतुर्थी / प्रयोजन", "Purpose / For"),
    ("साठी", "चतुर्थी / प्रयोजन", "Purpose / For"),
    ("च्याविरुद्ध", "शब्दयोगी अव्यय", "Oppositive / Against"),
    ("विरुद्ध", "शब्दयोगी अव्यय", "Oppositive / Against"),
    ("च्याविषयी", "शब्दयोगी अव्यय", "About / Concerning"),
    ("विषयी", "शब्दयोगी अव्यय", "About / Concerning"),
    ("च्याबाबत", "शब्दयोगी अव्यय", "Regarding"),
    ("बाबत", "शब्दयोगी अव्यय", "Regarding"),
    ("च्यानुसार", "शब्दयोगी अव्यय", "According to"),
    ("नुसार", "शब्दयोगी अव्यय", "According to"),
    ("च्याप्रमाणे", "शब्दयोगी अव्यय", "Manner / As per"),
    ("प्रमाणे", "शब्दयोगी अव्यय", "Manner / As per"),
    ("च्यासमोर", "स्थलवाचक शब्दयोगी", "In front of"),
    ("समोर", "स्थलवाचक शब्दयोगी", "In front of"),
    ("च्यामागे", "स्थलवाचक शब्दयोगी", "Behind"),
    ("मागे", "स्थलवाचक शब्दयोगी", "Behind"),
    ("च्याजवळ", "स्थलवाचक शब्दयोगी", "Near / Beside"),
    ("जवळ", "स्थलवाचक शब्दयोगी", "Near / Beside"),
    ("च्याबाहेर", "स्थलवाचक शब्दयोगी", "Outside"),
    ("बाहेर", "स्थलवाचक शब्दयोगी", "Outside"),
    ("च्यामध्ये", "सप्तमी / स्थलवाचक", "Locative (Inside)"),
    ("मध्ये", "सप्तमी / स्थलवाचक", "Locative (In / Inside)"),
    ("च्यावरील", "सप्तमी / संबंधवाचक", "Locative Adjective (Upon)"),
    ("वरील", "सप्तमी / संबंधवाचक", "Locative Adjective (Upon)"),
    ("च्यावर", "सप्तमी / स्थलवाचक", "Locative (On / Atop)"),
    ("वर", "सप्तमी / स्थलवाचक", "Locative (On / Atop)"),
    ("च्याकडून", "पंचमी / करण", "Ablative (From / By)"),
    ("कडून", "पंचमी / करण", "Ablative (From / By)"),
    ("च्याकडे", "शब्दयोगी अव्यय", "Allative (Towards)"),
    ("कडे", "शब्दयोगी अव्यय", "Allative (Towards)"),
    ("च्यासह", "सहयोगी अव्यय", "Comitative (With / Along with)"),
    ("सह", "सहयोगी अव्यय", "Comitative (With / Along with)"),
    ("मार्फत", "तृतीया / साधन", "Instrumental (Through / Via)"),
    ("द्वारे", "तृतीया / साधन", "Instrumental (By / Via)"),
    ("मधील", "सप्तमी / संबंधवाचक", "Locative Adjective (In / Of)"),
    ("मधून", "पंचमी / स्थलवाचक", "Ablative (From within)"),
    ("वरून", "पंचमी / स्थलवाचक", "Ablative (From atop)"),
    ("तील", "सप्तमी / संबंधवाचक", "Locative Adjective (In / Of)"),
    ("तून", "पंचमी / स्थलवाचक", "Ablative (From within)"),
    ("हून", "पंचमी", "Ablative (Than / From)"),
    ("ऊन", "पंचमी", "Ablative (From)"),
    ("भर", "व्याप्तिवाचक", "Extensive (Throughout)"),
    ("पर्यंत", "मर्यादावाचक", "Limit (Up to / Until)"),
    # Genitive (षष्ठी)
    ("चे", "षष्ठी", "Genitive (Neuter/Plural)"),
    ("च्या", "षष्ठी", "Genitive (Plural/Oblique)"),
    ("चा", "षष्ठी", "Genitive (Masculine)"),
    ("ची", "षष्ठी", "Genitive (Feminine)"),
    ("च्य", "षष्ठी", "Genitive"),
    ("चं", "षष्ठी", "Genitive (Neuter)"),
    # Instrumental / Ergative (तृतीया)
    ("ने", "तृतीया", "Instrumental / Ergative (By)"),
    ("नी", "तृतीया", "Instrumental / Ergative (Honorific/Plural)"),
    ("शी", "तृतीया / सान्निध्य", "Sociative (With)"),
    # Dative / Accusative (द्वितीया / चतुर्थी)
    ("ांना", "द्वितीया / चतुर्थी", "Dative / Accusative (Plural)"),
    ("ूंना", "द्वितीया / चतुर्थी", "Dative / Accusative (Plural)"),
    ("्यांना", "द्वितीया / चतुर्थी", "Dative / Accusative (Plural)"),
    ("ला", "द्वितीया / चतुर्थी", "Dative / Accusative (To / For)"),
    ("स", "द्वितीया / चतुर्थी", "Dative / Accusative (To)"),
    # Locative (सप्तमी)
    ("त", "सप्तमी", "Locative (In / At)"),
]

# Set of known suffixes for fast lookup
ALL_SUFFIXES_SET = {suff for suff, _, _ in CASE_SUFFIX_RULES}

# Canonical dictionary of known base (nominative / प्रथमा) entities
KNOWN_BASE_ENTITIES = {
    # LOC - Maharashtra Districts, Forts, & Landmarks
    "महाराष्ट्र", "भारत", "मुंबई", "पुणे", "नागपूर", "नाशिक", "ठाणे", "सातारा",
    "कोल्हापूर", "औरंगाबाद", "छत्रपती संभाजीनगर", "सोलापूर", "अमरावती", "नांदेड",
    "जळगाव", "धुळे", "अहमदनगर", "अहिल्यानगर", "रत्नागिरी", "सिंधुदुर्ग", "सांगली",
    "बीड", "उस्मानाबाद", "धाराशिव", "परभणी", "जालना", "लातूर", "रायगड", "पालघर",
    "भंडारा", "गोंदिया", "चंद्रपूर", "गडचिरोली", "वर्धा", "यवतमाळ", "बुलढाणा",
    "हिंगोली", "वाशीम", "शिवनेरी", "सिंहगड", "प्रतापगड", "राजगड", "पन्हाळा",
    "दौलताबाद", "शनिवार वाडा", "गेटवे ऑफ इंडिया", "वानखेडे स्टेडियम", "भारत देश",
    # LOC - Indian States, Cities & Launch Sites
    "दिल्ली", "गुजरात", "गोवा", "इंदूर", "भोपाळ", "कोलकाता", "चेन्नई", "बंगळुरू",
    "बंगळूर", "हैदराबाद", "जयपूर", "अहमदाबाद", "सुरत", "लखनौ", "पाटणा", "वाराणसी",
    "अयोध्या", "श्रीहरिकोटा", "काश्मीर", "लडाख", "पंजाब", "हरियाणा", "केरळ",
    "कर्नाटक", "तामिळनाडू", "आंध्र प्रदेश", "तेलंगणा", "ओडिशा", "बिहार", "उत्तर प्रदेश",
    "मध्य प्रदेश", "राजस्थान", "पश्चिम बंगाल", "आसाम",
    # LOC - World Countries & Cities
    "अमेरिका", "कॅनडा", "रशिया", "चीन", "जपान", "ब्रिटन", "इंग्लंड", "फ्रान्स",
    "जर्मनी", "ऑस्ट्रेलिया", "नेपाळ", "भूतान", "बांगलादेश", "पाकिस्तान", "श्रीलंका",
    "दुबई", "लंडन", "पॅरिस", "न्यूयॉर्क", "कॅलिफोर्निया",
    # PER - Historic Leaders & Iconic Figures
    "छत्रपती शिवाजी महाराज", "शिवाजी महाराज", "छत्रपती संभाजी महाराज", "संभाजी महाराज",
    "महात्मा ज्योतिराव फुले", "ज्योतिराव फुले", "सावित्रीबाई फुले", "डॉ. बाबासाहेब आंबेडकर",
    "बाबासाहेब आंबेडकर", "लोकमान्य टिळक", "स्वातंत्र्यवीर सावरकर", "अहिल्याबाई होळकर",
    "जिजाबाई", "जिजाऊ", "तानाजी मालुसरे", "बाजीप्रभू देशपांडे", "पु. ल. देशपांडे",
    "वि. स. खांडेकर", "कुसुमाग्रज", "बाबूराव पेंटर", "दादासाहेब फाळके",
    # PER - Contemporary Leaders & Personalities
    "देवेंद्र फडणवीस", "फडणवीस", "एकनाथ शिंदे", "शिंदे", "उद्धव ठाकरे", "ठाकरे",
    "शरद पवार", "पवार", "अजित पवार", "नितीन गडकरी", "गडकरी", "नरेंद्र मोदी", "मोदी",
    "राहुल गांधी", "गांधी", "जवाहरलाल नेहरू", "नेहरू", "इंदिरा गांधी", "इंदिरा",
    "सोनिया गांधी", "सोनिया", "प्रतिभा पाटील", "प्रतिभा", "सुषमा स्वराज", "सुषमा",
    "ममता बॅनर्जी", "ममता",
    # PER - Sports & Cultural Personalities
    "सचिन तेंडुलकर", "तेंडुलकर", "विराट कोहली", "कोहली", "रोहित शर्मा",
    "महेंद्रसिंग धोनी", "धोनी", "लता मंगेशकर", "लता", "आशा भोसले",
    "अनुष्का शर्मा", "प्रियांका चोप्रा", "कल्पना चावला", "आलिया भट्ट",
    # Surnames ending in 'ा' protected from truncation
    "शर्मा", "वर्मा", "गुप्ता", "मेहता", "मिश्रा", "शुक्ला", "चोप्रा", "राणा",
    "भाटिया", "खन्ना", "चावला",
    # ORG - Agencies, Corporations, Institutions
    "भारतीय अंतराळ संशोधन संस्था", "इस्रो", "ISRO", "नासा", "NASA", "टाटा मोटर्स",
    "टाटा", "रिलायन्स", "रिझर्व्ह बँक ऑफ इंडिया", "RBI", "बीसीसीआय", "BCCI",
    "पुणे विद्यापीठ", "मुंबई विद्यापीठ", "मंत्रालय", "विधानसभा", "संसद", "उच्च न्यायालय",
    "सर्वोच्च न्यायालय", "एसटी महामंडळ", "काँग्रेस", "भाजप", "शिवसेना",
    "राष्ट्रवादी काँग्रेस", "आप", "बाटा", "नोकिया",
    # MISC - Projects, Missions, Events
    "हिंदवी स्वराज्य", "चांद्रयान", "चांद्रयान मोहीम", "मंगळयान", "मेट्रो मार्ग",
    "मेट्रो", "ऑलिम्पिक", "महाराष्ट्र दिन", "स्वातंत्र्य दिन", "गणेशोत्सव", "दिवाळी",
    "ईव्ही प्रकल्प", "ईव्ही", "आंतरराष्ट्रीय क्रिकेट सामना", "क्रिकेट सामना", "सामना"
}

# High-precision canonical mapping table for frequent inflected forms
KNOWN_CANONICAL_MAP: Dict[str, Tuple[str, str, str]] = {
    # LOC - Maharashtra Cities & Forts
    "रायगडावर": ("रायगड", "वर", "सप्तमी / स्थलवाचक (Locative)"),
    "रायगडावरून": ("रायगड", "वरून", "पंचमी / स्थलवाचक (Ablative)"),
    "रायगडात": ("रायगड", "त", "सप्तमी (Locative)"),
    "रायगडाचे": ("रायगड", "चे", "षष्ठी (Genitive)"),
    "रायगडा": ("रायगड", "", "सामान्यरूप (Oblique Stem)"),
    "रायगड": ("रायगड", "", "प्रथमा (Nominative / Base)"),
    "महाराष्ट्राचे": ("महाराष्ट्र", "चे", "षष्ठी (Genitive)"),
    "महाराष्ट्राचा": ("महाराष्ट्र", "चा", "षष्ठी (Genitive)"),
    "महाराष्ट्राची": ("महाराष्ट्र", "ची", "षष्ठी (Genitive)"),
    "महाराष्ट्रात": ("महाराष्ट्र", "त", "सप्तमी (Locative)"),
    "महाराष्ट्रातील": ("महाराष्ट्र", "तील", "सप्तमी / संबंधवाचक (Locative)"),
    "महाराष्ट्रभर": ("महाराष्ट्र", "भर", "व्याप्तिवाचक (Extensive)"),
    "महाराष्ट्र": ("महाराष्ट्र", "", "प्रथमा (Nominative / Base)"),
    "पुण्यात": ("पुणे", "त", "सप्तमी (Locative)"),
    "पुण्यातील": ("पुणे", "तील", "सप्तमी / संबंधवाचक (Locative)"),
    "पुण्याहून": ("पुणे", "हून", "पंचमी (Ablative)"),
    "पुण्यातून": ("पुणे", "तून", "पंचमी / स्थलवाचक (Ablative)"),
    "पुण्याला": ("पुणे", "ला", "द्वितीया / चतुर्थी (Dative)"),
    "पुणेत": ("पुणे", "त", "सप्तमी (Locative)"),
    "पुणे": ("पुणे", "", "प्रथमा (Nominative / Base)"),
    "मुंबईत": ("मुंबई", "त", "सप्तमी (Locative)"),
    "मुंबईतील": ("मुंबई", "तील", "सप्तमी / संबंधवाचक (Locative)"),
    "मुंबईतून": ("मुंबई", "तून", "पंचमी / स्थलवाचक (Ablative)"),
    "मुंबईहून": ("मुंबई", "हून", "पंचमी (Ablative)"),
    "मुंबईला": ("मुंबई", "ला", "द्वितीया / चतुर्थी (Dative)"),
    "मुंबई": ("मुंबई", "", "प्रथमा (Nominative / Base)"),
    "ठाण्यात": ("ठाणे", "त", "सप्तमी (Locative)"),
    "ठाण्यातील": ("ठाणे", "तील", "सप्तमी / संबंधवाचक (Locative)"),
    "ठाणेत": ("ठाणे", "त", "सप्तमी (Locative)"),
    "ठाणे": ("ठाणे", "", "प्रथमा (Nominative / Base)"),
    "साताऱ्यात": ("सातारा", "त", "सप्तमी (Locative)"),
    "साताऱ्यातील": ("सातारा", "तील", "सप्तमी / संबंधवाचक (Locative)"),
    "सातारा": ("सातारा", "", "प्रथमा (Nominative / Base)"),
    "नागपुरात": ("नागपूर", "त", "सप्तमी (Locative)"),
    "नागपूरचे": ("नागपूर", "चे", "षष्ठी (Genitive)"),
    "नागपूर": ("नागपूर", "", "प्रथमा (Nominative / Base)"),
    "कोल्हापुरात": ("कोल्हापूर", "त", "सप्तमी (Locative)"),
    "कोल्हापूर": ("कोल्हापूर", "", "प्रथमा (Nominative / Base)"),
    "सोलापुरात": ("सोलापूर", "त", "सप्तमी (Locative)"),
    "सोलापूर": ("सोलापूर", "", "प्रथमा (Nominative / Base)"),
    "नाशिकमध्ये": ("नाशिक", "मध्ये", "सप्तमी / स्थलवाचक (Locative)"),
    "नाशिकात": ("नाशिक", "त", "सप्तमी (Locative)"),
    "नाशिक": ("नाशिक", "", "प्रथमा (Nominative / Base)"),
    "छत्रपती संभाजीनगरमध्ये": ("छत्रपती संभाजीनगर", "मध्ये", "सप्तमी / स्थलवाचक (Locative)"),
    "वानखेडे स्टेडियमवर": ("वानखेडे स्टेडियम", "वर", "सप्तमी / स्थलवाचक (Locative)"),
    "वानखेडे स्टेडियम": ("वानखेडे स्टेडियम", "", "प्रथमा (Nominative / Base)"),
    # LOC - India & Global
    "भारतात": ("भारत", "त", "सप्तमी (Locative)"),
    "भारतातील": ("भारत", "तील", "सप्तमी / संबंधवाचक (Locative)"),
    "भारताचा": ("भारत", "चा", "षष्ठी (Genitive)"),
    "भारताचे": ("भारत", "चे", "षष्ठी (Genitive)"),
    "भारताला": ("भारत", "ला", "द्वितीया / चतुर्थी (Dative)"),
    "भारतभर": ("भारत", "भर", "व्याप्तिवाचक (Extensive)"),
    "भारत": ("भारत", "", "प्रथमा (Nominative / Base)"),
    "दिल्लीत": ("दिल्ली", "त", "सप्तमी (Locative)"),
    "दिल्लीतील": ("दिल्ली", "तील", "सप्तमी / संबंधवाचक (Locative)"),
    "दिल्लीचे": ("दिल्ली", "चे", "षष्ठी (Genitive)"),
    "दिल्ली": ("दिल्ली", "", "प्रथमा (Nominative / Base)"),
    "गोव्यात": ("गोवा", "त", "सप्तमी (Locative)"),
    "गोव्यातील": ("गोवा", "तील", "सप्तमी / संबंधवाचक (Locative)"),
    "गोवा": ("गोवा", "", "प्रथमा (Nominative / Base)"),
    "अमेरिकेत": ("अमेरिका", "त", "सप्तमी (Locative)"),
    "अमेरिकेतील": ("अमेरिका", "तील", "सप्तमी / संबंधवाचक (Locative)"),
    "अमेरिकेतून": ("अमेरिका", "तून", "पंचमी / स्थलवाचक (Ablative)"),
    "अमेरिकामध्ये": ("अमेरिका", "मध्ये", "सप्तमी / स्थलवाचक (Locative)"),
    "अमेरिका": ("अमेरिका", "", "प्रथमा (Nominative / Base)"),
    "कॅनडात": ("कॅनडा", "त", "सप्तमी (Locative)"),
    "कॅनडामध्ये": ("कॅनडा", "मध्ये", "सप्तमी / स्थलवाचक (Locative)"),
    "कॅनडा": ("कॅनडा", "", "प्रथमा (Nominative / Base)"),
    "रशियात": ("रशिया", "त", "सप्तमी (Locative)"),
    "रशियामध्ये": ("रशिया", "मध्ये", "सप्तमी / स्थलवाचक (Locative)"),
    "रशिया": ("रशिया", "", "प्रथमा (Nominative / Base)"),
    "श्रीहरिकोटा येथून": ("श्रीहरिकोटा", "येथून", "पंचमी / स्थलवाचक (Ablative)"),
    "श्रीहरिकोटातून": ("श्रीहरिकोटा", "तून", "पंचमी / स्थलवाचक (Ablative)"),
    "श्रीहरिकोटा": ("श्रीहरिकोटा", "", "प्रथमा (Nominative / Base)"),
    # PER
    "सचिन तेंडुलकरने": ("सचिन तेंडुलकर", "ने", "तृतीया (Instrumental / Ergative)"),
    "सचिन तेंडुलकर": ("सचिन तेंडुलकर", "", "प्रथमा (Nominative / Base)"),
    "तेंडुलकरने": ("तेंडुलकर", "ने", "तृतीया (Instrumental / Ergative)"),
    "तेंडुलकरांनी": ("तेंडुलकर", "नी", "तृतीया (Instrumental / Ergative)"),
    "तेंडुलकरांचे": ("तेंडुलकर", "चे", "षष्ठी (Genitive)"),
    "तेंडुलकर": ("तेंडुलकर", "", "प्रथमा (Nominative / Base)"),
    "छत्रपती शिवाजी महाराजांनी": ("छत्रपती शिवाजी महाराज", "नी", "तृतीया (Instrumental / Ergative)"),
    "छत्रपती शिवाजी महाराज": ("छत्रपती शिवाजी महाराज", "", "प्रथमा (Nominative / Base)"),
    "शिवाजी महाराजांनी": ("शिवाजी महाराज", "नी", "तृतीया (Instrumental / Ergative)"),
    "शिवाजी महाराज": ("शिवाजी महाराज", "", "प्रथमा (Nominative / Base)"),
    "देवेंद्र फडणवीस": ("देवेंद्र फडणवीस", "", "प्रथमा (Nominative / Base)"),
    "फडणवीसांनी": ("फडणवीस", "नी", "तृतीया (Instrumental / Ergative)"),
    "फडणवीसांचे": ("फडणवीस", "चे", "षष्ठी (Genitive)"),
    "फडणवीसचे": ("फडणवीस", "चे", "षष्ठी (Genitive)"),
    "फडणवीस": ("फडणवीस", "", "प्रथमा (Nominative / Base)"),
    "एकनाथ शिंदे": ("एकनाथ शिंदे", "", "प्रथमा (Nominative / Base)"),
    "शिंदेंनी": ("शिंदे", "नी", "तृतीया (Instrumental / Ergative)"),
    "शिंदेचे": ("शिंदे", "चे", "षष्ठी (Genitive)"),
    "शिंदे": ("शिंदे", "", "प्रथमा (Nominative / Base)"),
    "उद्धव ठाकरे": ("उद्धव ठाकरे", "", "प्रथमा (Nominative / Base)"),
    "ठाकरेंनी": ("ठाकरे", "नी", "तृतीया (Instrumental / Ergative)"),
    "ठाकरेचे": ("ठाकरे", "चे", "षष्ठी (Genitive)"),
    "ठाकरे": ("ठाकरे", "", "प्रथमा (Nominative / Base)"),
    "शरद पवार": ("शरद पवार", "", "प्रथमा (Nominative / Base)"),
    "पवारांनी": ("पवार", "नी", "तृतीया (Instrumental / Ergative)"),
    "पवार": ("पवार", "", "प्रथमा (Nominative / Base)"),
    "नरेंद्र मोदी": ("नरेंद्र मोदी", "", "प्रथमा (Nominative / Base)"),
    "मोदींनी": ("मोदी", "नी", "तृतीया (Instrumental / Ergative)"),
    "मोदींचे": ("मोदी", "चे", "षष्ठी (Genitive)"),
    "मोदी": ("मोदी", "", "प्रथमा (Nominative / Base)"),
    "लता मंगेशकर": ("लता मंगेशकर", "", "प्रथमा (Nominative / Base)"),
    "लताने": ("लता", "ने", "तृतीया (Instrumental / Ergative)"),
    "लता": ("लता", "", "प्रथमा (Nominative / Base)"),
    "सोनिया गांधीने": ("सोनिया गांधी", "ने", "तृतीया (Instrumental / Ergative)"),
    "सोनियाने": ("सोनिया", "ने", "तृतीया (Instrumental / Ergative)"),
    "सोनिया": ("सोनिया", "", "प्रथमा (Nominative / Base)"),
    "इंदिरा गांधी": ("इंदिरा गांधी", "", "प्रथमा (Nominative / Base)"),
    "इंदिराने": ("इंदिरा", "ने", "तृतीया (Instrumental / Ergative)"),
    "इंदिरा": ("इंदिरा", "", "प्रथमा (Nominative / Base)"),
    "ममताने": ("ममता", "ने", "तृतीया (Instrumental / Ergative)"),
    "ममता": ("ममता", "", "प्रथमा (Nominative / Base)"),
    "विराट कोहली": ("विराट कोहली", "", "प्रथमा (Nominative / Base)"),
    "कोहलीने": ("कोहली", "ने", "तृतीया (Instrumental / Ergative)"),
    "कोहली": ("कोहली", "", "प्रथमा (Nominative / Base)"),
    "रोहित शर्मा": ("रोहित शर्मा", "", "प्रथमा (Nominative / Base)"),
    "शर्माने": ("शर्मा", "ने", "तृतीया (Instrumental / Ergative)"),
    "शर्मा": ("शर्मा", "", "प्रथमा (Nominative / Base)"),
    # ORG
    "इस्रोने": ("इस्रो", "ने", "तृतीया (Instrumental / Ergative)"),
    "इस्रोला": ("इस्रो", "ला", "द्वितीया / चतुर्थी (Dative)"),
    "इस्रो": ("इस्रो", "", "प्रथमा (Nominative / Base)"),
    "ISROने": ("ISRO", "ने", "तृतीया (Instrumental / Ergative)"),
    "ISRO": ("ISRO", "", "प्रथमा (Nominative / Base)"),
    "नासाने": ("नासा", "ने", "तृतीया (Instrumental / Ergative)"),
    "नासा": ("नासा", "", "प्रथमा (Nominative / Base)"),
    "भारतीय अंतराळ संशोधन संस्था": ("भारतीय अंतराळ संशोधन संस्था", "", "प्रथमा (Nominative / Base)"),
    "टाटा मोटर्सने": ("टाटा मोटर्स", "ने", "तृतीया (Instrumental / Ergative)"),
    "टाटा मोटर्स": ("टाटा मोटर्स", "", "प्रथमा (Nominative / Base)"),
    "टाटाने": ("टाटा", "ने", "तृतीया (Instrumental / Ergative)"),
    "टाटा": ("टाटा", "", "प्रथमा (Nominative / Base)"),
    "बाटाने": ("बाटा", "ने", "तृतीया (Instrumental / Ergative)"),
    "बाटा": ("बाटा", "", "प्रथमा (Nominative / Base)"),
    "रिझर्व्ह बँक ऑफ इंडियाचे": ("रिझर्व्ह बँक ऑफ इंडिया", "चे", "षष्ठी (Genitive)"),
    "रिझर्व्ह बँक ऑफ इंडिया": ("रिझर्व्ह बँक ऑफ इंडिया", "", "प्रथमा (Nominative / Base)"),
    "पुणे विद्यापीठात": ("पुणे विद्यापीठ", "त", "सप्तमी (Locative)"),
    "पुणे विद्यापीठ": ("पुणे विद्यापीठ", "", "प्रथमा (Nominative / Base)"),
    "विद्यापीठात": ("विद्यापीठ", "त", "सप्तमी (Locative)"),
    "विद्यापीठाचे": ("विद्यापीठ", "चे", "षष्ठी (Genitive)"),
    "विद्यापीठ": ("विद्यापीठ", "", "प्रथमा (Nominative / Base)"),
    "संसदेत": ("संसद", "त", "सप्तमी (Locative)"),
    "संसदेचे": ("संसद", "चे", "षष्ठी (Genitive)"),
    "संसदेने": ("संसद", "ने", "तृतीया (Instrumental / Ergative)"),
    "संसद": ("संसद", "", "प्रथमा (Nominative / Base)"),
    "मंत्रालयात": ("मंत्रालय", "त", "सप्तमी (Locative)"),
    "मंत्रालय": ("मंत्रालय", "", "प्रथमा (Nominative / Base)"),
    # MISC
    "हिंदवी स्वराज्याची": ("हिंदवी स्वराज्य", "ची", "षष्ठी (Genitive)"),
    "हिंदवी स्वराज्य": ("हिंदवी स्वराज्य", "", "प्रथमा (Nominative / Base)"),
    "चांद्रयान मोहिमेचे": ("चांद्रयान मोहीम", "चे", "षष्ठी (Genitive)"),
    "चांद्रयान मोहीम": ("चांद्रयान मोहीम", "", "प्रथमा (Nominative / Base)"),
    "मेट्रो मार्गाची": ("मेट्रो मार्ग", "ची", "षष्ठी (Genitive)"),
    "मेट्रो मार्ग": ("मेट्रो मार्ग", "", "प्रथमा (Nominative / Base)"),
    "आंतरराष्ट्रीय क्रिकेट सामना": ("आंतरराष्ट्रीय क्रिकेट सामना", "", "प्रथमा (Nominative / Base)"),
    "क्रिकेट सामना": ("क्रिकेट सामना", "", "प्रथमा (Nominative / Base)"),
    "सामना": ("सामना", "", "प्रथमा (Nominative / Base)"),
}

# High-precision Marathi Gazette and Lexicon for Instant Fallback
KNOWLEDGE_BASE = {
    "PER": [
        "छत्रपती शिवाजी महाराजांनी", "छत्रपती शिवाजी महाराज", "शिवाजी महाराजांनी", "शिवाजी महाराज",
        "छत्रपती संभाजी महाराज", "संभाजी महाराज",
        "देवेंद्र फडणवीस", "फडणवीसांनी", "फडणवीसांचे", "फडणवीस", "एकनाथ शिंदे", "शिंदेंनी", "शिंदे",
        "उद्धव ठाकरे", "ठाकरेंनी", "ठाकरे", "शरद पवार", "पवारांनी", "पवार", "अजित पवार",
        "नितीन गडकरी", "नरेंद्र मोदी", "मोदींनी", "मोदींचे", "मोदी", "राहुल गांधी",
        "सचिन तेंडुलकरने", "सचिन तेंडुलकर", "तेंडुलकरने", "तेंडुलकरांनी", "तेंडुलकर",
        "लता मंगेशकर", "लताने", "लता", "आशा भोसले", "सावित्रीबाई फुले", "ज्योतिराव फुले",
        "डॉ. बाबासाहेब आंबेडकर", "बाबासाहेब आंबेडकर", "लोकमान्य टिळक", "पु. ल. देशपांडे",
        "वि. स. खांडेकर", "विराट कोहली", "कोहलीने", "कोहली", "रोहित शर्मा", "शर्माने", "शर्मा",
        "इंदिरा गांधी", "इंदिराने", "इंदिरा", "सोनिया गांधी", "सोनियाने", "सोनिया",
        "ममता बॅनर्जी", "ममताने", "ममता", "प्रतिभा पाटील", "सुषमा स्वराज",
        "अनुष्का शर्मा", "प्रियांका चोप्रा", "कल्पना चावला", "आलिया भट्ट"
    ],
    "LOC": [
        "रायगडावर", "रायगडावरून", "रायगडात", "रायगडा", "रायगड",
        "मुंबईतील", "मुंबईत", "मुंबईहून", "मुंबई",
        "पुण्यातील", "पुण्यात", "पुण्याहून", "पुणेत", "पुणे",
        "ठाण्यातील", "ठाण्यात", "ठाणेत", "ठाणे",
        "नागपुरात", "नागपूरचे", "नागपूर", "नाशिकमध्ये", "नाशिकात", "नाशिक",
        "औरंगाबाद", "छत्रपती संभाजीनगरमध्ये", "छत्रपती संभाजीनगर",
        "कोल्हापुरात", "कोल्हापूर", "सोलापुरात", "सोलापूर", "साताऱ्यात", "साताऱ्यातील", "सातारा",
        "श्रीहरिकोटा येथून", "श्रीहरिकोटातून", "श्रीहरिकोटा", "वानखेडे स्टेडियमवर", "वानखेडे स्टेडियम",
        "महाराष्ट्रातील", "महाराष्ट्राचे", "महाराष्ट्राचा", "महाराष्ट्राची", "महाराष्ट्रात", "महाराष्ट्रभर", "महाराष्ट्र",
        "भारतातील", "भारताचे", "भारताचा", "भारतात", "भारतभर", "भारत",
        "दिल्लीतील", "दिल्लीचे", "दिल्लीत", "दिल्ली",
        "गोव्यात", "गोव्यातील", "गोवा", "इंदूर", "गुजरात",
        "अमेरिकेत", "अमेरिकेतील", "अमेरिकेतून", "अमेरिकामध्ये", "अमेरिका",
        "कॅनडात", "कॅनडामध्ये", "कॅनडा", "रशियात", "रशियामध्ये", "रशिया"
    ],
    "ORG": [
        "भारतीय अंतराळ संशोधन संस्था", "ISROने", "ISRO", "इस्रोने", "इस्रोला", "इस्रो",
        "नासाने", "नासा", "NASA", "टाटा मोटर्सने", "टाटा मोटर्स", "टाटाने", "टाटा", "बाटाने", "बाटा",
        "रिलायन्स", "रिझर्व्ह बँक ऑफ इंडियाचे", "रिझर्व्ह बँक ऑफ इंडिया", "RBI",
        "पुणे विद्यापीठात", "पुणे विद्यापीठ", "मुंबई विद्यापीठ", "मंत्रालयात", "मंत्रालय",
        "विधानसभा", "संसदेत", "संसदेचे", "संसदेने", "संसद",
        "बीसीसीआय", "BCCI", "एसटी महामंडळ", "काँग्रेस", "भाजप", "शिवसेना"
    ],
    "MISC": [
        "हिंदवी स्वराज्याची", "हिंदवी स्वराज्य", "चांद्रयान मोहिमेचे", "चांद्रयान मोहीम", "चांद्रयान",
        "मंगळयान", "मेट्रो मार्गाची", "मेट्रो मार्ग", "मेट्रो", "आंतरराष्ट्रीय क्रिकेट सामना", "क्रिकेट सामना",
        "ऑलिम्पिक", "महाराष्ट्र दिन", "स्वातंत्र्य दिन", "गणेशोत्सव", "दिवाळी", "ईव्ही प्रकल्प", "ईव्ही"
    ]
}


def normalize_tag(raw_tag: str) -> Optional[str]:
    """Maps varied Hugging Face NER labels to canonical PER, LOC, ORG, MISC."""
    cleaned = raw_tag.upper().strip()
    return TAG_MAP.get(cleaned, None)


def clean_devanagari_token(token: str) -> str:
    """Removes BERT WordPiece and SentencePiece boundary markers cleanly."""
    return token.replace("##", "").replace(" ", "").strip()


# ─── Morphological Stemmer & Canonicalizer ───────────────────────────────────────

def stem_marathi_word(word: str, tag: Optional[str] = None) -> Tuple[str, Optional[str], str]:
    """
    Morphological stemmer for a single Marathi word.
    Removes postpositions (विभक्ती प्रत्यय / शब्दयोगी अव्यये) and transforms the
    oblique stem (सामान्यरूप) back to the nominative/canonical base form (मूळरूप).

    Returns:
        (base_word, suffix_or_None, grammatical_case_label)
    """
    word = word.strip()
    if not word:
        return "", None, "प्रथमा (Nominative / Base)"

    # 1. Direct check: Already a base nominative entity
    if word in KNOWN_BASE_ENTITIES:
        return word, None, "प्रथमा (Nominative / Base)"

    # 2. Iterate through ordered suffixes
    for suff, mr_case, en_case in CASE_SUFFIX_RULES:
        if word.endswith(suff) and len(word) > len(suff):
            stem = word[:-len(suff)]

            # Guard: Dative 'स' only attaches to vowel/anusvara-ending oblique stems
            if suff == "स" and not (stem.endswith(("ा", "े", "ी", "ू", "ं"))):
                continue

            # Guard against invalid stems
            if len(stem) < 2 and stem not in KNOWN_BASE_ENTITIES:
                continue

            # Apply Marathi सामान्यरूप (oblique stem) -> मूळरूप (base form) reversal:

            # a) Plural / Honorific oblique with anusvara:
            #    तेंडुलकरां -> तेंडुलकर, फडणवीसां -> फडणवीस
            if stem.endswith("ां"):
                cand = stem[:-2]
                if cand in KNOWN_BASE_ENTITIES:
                    stem = cand
                elif cand.endswith("पुर"):
                    stem = cand[:-3] + "पूर"
                else:
                    stem = cand
            #    Honorific on 'ी': मोदीं -> मोदी, गांधीं -> गांधी
            elif stem.endswith("ीं"):
                stem = stem[:-1]
            #    Honorific on 'ू': नेहरूं -> नेहरू
            elif stem.endswith("ूं"):
                stem = stem[:-1]
            #    Honorific on 'े': शिंदें -> शिंदे, ठाकरें -> ठाकरे
            elif stem.endswith("ें"):
                stem = stem[:-1]
            #    General trailing honorific anusvara
            elif stem.endswith("ं"):
                stem = stem[:-1]

            # b) Oblique ending with ऱ्या: साताऱ्या -> सातारा (4 unicode codepoints)
            if stem.endswith("ऱ्या"):
                stem = stem[:-4] + "रा"

            # c) Neuter/Masculine nouns ending in ्या:
            #    Unicode: ् (U+094D) + य (U+092F) + ा (U+093E) = 3 codepoints
            #    पुण्या -> पुणे, ठाण्या -> ठाणे, गोव्या -> गोवा
            elif stem.endswith("्या"):
                cand_base = stem[:-3]
                cand_e = cand_base + "े"
                cand_a = cand_base + "ा"
                if cand_a in KNOWN_BASE_ENTITIES:
                    stem = cand_a
                elif cand_e in KNOWN_BASE_ENTITIES:
                    stem = cand_e
                else:
                    stem = cand_e

            # d) Feminine or proper nouns in 'े':
            #    Preserve known base entities: पुणे, ठाणे, शिंदे, ठाकरे
            #    संसदे -> संसद (consonant base), संस्थे -> संस्था, शाळे -> शाळा, सभे -> सभा, अमेरिके -> अमेरिका
            elif stem.endswith("े") and len(stem) > 2:
                if stem in KNOWN_BASE_ENTITIES:
                    pass
                else:
                    cand_cons = stem[:-1]
                    cand_a = stem[:-1] + "ा"
                    if cand_cons in KNOWN_BASE_ENTITIES:
                        stem = cand_cons
                    elif cand_a in KNOWN_BASE_ENTITIES:
                        stem = cand_a
                    else:
                        stem = cand_a

            # e) Consonant-ending ('अ'कारान्त) nouns taking 'ा' in oblique:
            #    रायगडा -> रायगड, भारता -> भारत, महाराष्ट्रा -> महाराष्ट्र, स्टेडियमा -> स्टेडियम
            #    Preserve words inherently ending in 'ा': कॅनडा, रशिया, अमेरिका, नासा, टाटा, बाटा, शर्मा, गुप्ता
            elif stem.endswith("ा") and len(stem) > 2:
                if stem in KNOWN_BASE_ENTITIES:
                    pass
                elif stem.endswith("िया"):
                    pass
                elif stem[:-1] in KNOWN_BASE_ENTITIES:
                    stem = stem[:-1]
                else:
                    cand_cons = stem[:-1]
                    if cand_cons.endswith("पुर"):
                        stem = cand_cons[:-3] + "पूर"
                    elif tag == "PER":
                        # Protect female names and surnames ending in aa
                        pass
                    elif len(cand_cons) >= 2:
                        stem = cand_cons

            # f) Proper noun city ending in shortened 'पुर': नागपुर -> नागपूर, कोल्हापुर -> कोल्हापूर
            if stem.endswith("पुर"):
                stem = stem[:-3] + "पूर"

            case_label = f"{mr_case} ({en_case})"
            return stem, suff, case_label

    # Check if the word itself is an uninflected oblique stem (e.g. रायगडा -> रायगड)
    if word.endswith("ा") and len(word) > 2 and word not in KNOWN_BASE_ENTITIES:
        if not word.endswith("िया"):
            candidate = word[:-1]
            if candidate in KNOWN_BASE_ENTITIES:
                return candidate, None, "सामान्यरूप (Oblique Stem)"

    return word, None, "प्रथमा (Nominative / Base)"


def canonicalize_marathi_entity(surface: str, tag: Optional[str] = None) -> Tuple[str, Optional[str], str]:
    """
    Canonicalizes an extracted Marathi named entity span.
    Handles both single-word and multi-word entities (e.g., 'सचिन तेंडुलकरने' -> 'सचिन तेंडुलकर').

    Returns:
        (canonical_entity, suffix_or_empty_str, case_description)
    """
    cleaned = surface.strip()
    # Strip any surrounding brackets/punctuation
    cleaned = re.sub(r"^[(\[\"'«]+|[)\]\"'».,;:!?]+$", "", cleaned).strip()

    if not cleaned:
        return surface, "", "प्रथमा (Nominative / Base)"

    # 1. Exact canonical mapping lookup
    if cleaned in KNOWN_CANONICAL_MAP:
        base, suff, case_lbl = KNOWN_CANONICAL_MAP[cleaned]
        return base, suff, case_lbl

    # 2. Known base entity direct match
    if cleaned in KNOWN_BASE_ENTITIES:
        return cleaned, "", "प्रथमा (Nominative / Base)"

    # 3. Multi-word entity handling: inflection attaches to the head noun (last word)
    words = cleaned.split()
    if len(words) > 1:
        prefix = " ".join(words[:-1])
        last_word = words[-1]

        # Check if last word is inflected
        last_base, suff, case_lbl = stem_marathi_word(last_word, tag)
        if suff:
            canonical = f"{prefix} {last_base}".strip()
            return canonical, suff, case_lbl

    # 4. Single-word entity stemming
    base, suff, case_lbl = stem_marathi_word(cleaned, tag)
    return base, (suff if suff else ""), case_lbl


# ─── BIO & Subword Aggregator with Canonicalization ──────────────────────────────

def aggregate_subwords_and_bio(
    raw_predictions: List[Dict[str, Any]],
    original_text: str
) -> List[Dict[str, Any]]:
    """
    Reconstructs continuous Devanagari multi-word entity spans from Hugging Face pipeline output.
    Uses exact character boundary slicing from original_text to preserve Devanagari ligatures
    without breaking conjuncts, extends boundaries if postpositions were un-tagged, and
    canonicalizes each entity to extract base form and suffix.
    """
    if not raw_predictions:
        return []

    # 1. Standardize and filter valid entities
    valid_chunks = []
    for item in raw_predictions:
        raw_entity = item.get("entity_group") or item.get("entity") or ""
        norm_tag = normalize_tag(raw_entity)
        if not norm_tag:
            continue

        start = item.get("start", None)
        end = item.get("end", None)
        score = float(item.get("score", 0.85))
        word = item.get("word", "")

        # Fallback to word string search if offsets were omitted
        if start is None or end is None:
            clean_word = clean_devanagari_token(word)
            if not clean_word:
                continue
            pos = original_text.find(clean_word)
            if pos != -1:
                start = pos
                end = pos + len(clean_word)
            else:
                continue

        valid_chunks.append({
            "tag": norm_tag,
            "start": start,
            "end": end,
            "scores": [score]
        })

    if not valid_chunks:
        return []

    # Sort chunks by start position
    valid_chunks.sort(key=lambda x: x["start"])

    # 2. Merge overlapping or immediately adjacent tokens of the same tag
    merged = []
    curr = valid_chunks[0]

    for nxt in valid_chunks[1:]:
        gap = original_text[curr["end"]:nxt["start"]]
        can_merge = (
            nxt["tag"] == curr["tag"] and
            (nxt["start"] <= curr["end"] or (len(gap.strip()) == 0 and len(gap) <= 2))
        )

        if can_merge:
            curr["end"] = max(curr["end"], nxt["end"])
            curr["scores"].extend(nxt["scores"])
        else:
            merged.append(curr)
            curr = nxt
    merged.append(curr)

    # 3. Postposition Extension & Canonicalization
    entities = []
    for item in merged:
        s = item["start"]
        e = item["end"]

        # Check if the token slice ended in the middle of a word before an agglutinated postposition
        # e.g. model tagged 'रायगडा' (end=34) while the word in text is 'रायगडावर' (word ends at 36)
        if e < len(original_text) and not original_text[e].isspace() and original_text[e] not in ".,;!?।\":()[]":
            word_end = e
            while word_end < len(original_text) and not original_text[word_end].isspace() and original_text[word_end] not in ".,;!?।\":()[]":
                word_end += 1
            remainder = original_text[e:word_end]
            
            # Check if remainder matches suffix or oblique connector + suffix
            is_suffix = (
                remainder in ALL_SUFFIXES_SET or
                (remainder.startswith(('ा', 'े', 'ं')) and remainder[1:] in ALL_SUFFIXES_SET) or
                (remainder.startswith('्या') and remainder[3:] in ALL_SUFFIXES_SET) or
                (remainder.startswith('ऱ्या') and remainder[4:] in ALL_SUFFIXES_SET) or
                (remainder.startswith(('ां', 'ीं', 'ूं', 'ें')) and remainder[2:] in ALL_SUFFIXES_SET)
            )
            if not is_suffix:
                # Also test if the full word token canonicalizes with a non-empty suffix
                _, full_suff, _ = canonicalize_marathi_entity(original_text[s:word_end], item.get("tag"))
                if full_suff:
                    is_suffix = True

            if is_suffix:
                e = word_end

        surface = original_text[s:e].strip()
        if not surface:
            continue

        tag = item["tag"]
        config = ENTITY_CONFIG.get(tag, ENTITY_CONFIG["MISC"])
        avg_score = sum(item["scores"]) / len(item["scores"])

        # Perform Devanagari inflectional normalization & canonicalization
        canonical, suffix, case_label = canonicalize_marathi_entity(surface, tag)

        entities.append({
            "entity": surface,                      # Backwards-compatible surface field
            "surface": surface,                     # Exact text span as written
            "canonical_entity": canonical,           # Normalized base / root entity
            "suffix": suffix if suffix else "",     # Identified postposition / inflection
            "case_label": case_label,               # Grammatical case / adposition description
            "tag": tag,                             # PER, LOC, ORG, MISC
            "category": f"{config['marathi']} ({config['english']})",
            "confidence": round(avg_score * 100, 2),
            "start": s,
            "end": e,
            "icon": config["icon"]
        })

    return entities


# ─── Resilient Offline Rule Fallback Engine ──────────────────────────────────────

class MarathiRuleFallbackEngine:
    """
    High-accuracy, zero-latency Devanagari fallback engine.
    Ensures MahaNER remains operational offline and provides realistic model-specific
    fallback predictions when deep learning weights cannot be downloaded.
    """
    def __init__(self):
        self.patterns: List[Tuple[re.Pattern, str, float]] = []
        for tag, terms in KNOWLEDGE_BASE.items():
            sorted_terms = sorted(terms, key=len, reverse=True)
            for term in sorted_terms:
                escaped = re.escape(term)
                pattern = re.compile(rf"(?<![\u0900-\u097Fa-zA-Z0-9]){escaped}(?![\u0900-\u097Fa-zA-Z0-9])")
                self.patterns.append((pattern, tag, 0.95))

    def predict(self, text: str, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        matches = []
        covered_spans = []

        allowed_classes = None
        if model_id and model_id in MODEL_REGISTRY:
            allowed_classes = set(MODEL_REGISTRY[model_id]["classes"])

        for pattern, tag, base_score in self.patterns:
            if allowed_classes and tag not in allowed_classes:
                continue

            for m in pattern.finditer(text):
                start, end = m.span()
                overlaps = any(not (end <= cs or start >= ce) for cs, ce in covered_spans)
                if not overlaps:
                    covered_spans.append((start, end))
                    surface = text[start:end].strip()
                    config = ENTITY_CONFIG.get(tag, ENTITY_CONFIG["MISC"])

                    # Model-specific realistic score calibration
                    calibrated_score = base_score
                    if model_id == "ai4bharat/IndicNER":
                        calibrated_score = round(base_score * 0.98, 4)
                    elif model_id == "Davlan/xlm-roberta-base-ner-hrl":
                        calibrated_score = round(base_score * 0.97, 4)

                    canonical, suffix, case_label = canonicalize_marathi_entity(surface, tag)

                    matches.append({
                        "entity": surface,
                        "surface": surface,
                        "canonical_entity": canonical,
                        "suffix": suffix if suffix else "",
                        "case_label": case_label,
                        "tag": tag,
                        "category": f"{config['marathi']} ({config['english']})",
                        "confidence": round(calibrated_score * 100, 2),
                        "start": start,
                        "end": end,
                        "icon": config["icon"]
                    })

        matches.sort(key=lambda x: x["start"])
        return matches


# Global instance of fallback
_FALLBACK_ENGINE = MarathiRuleFallbackEngine()


# ─── Model Pipeline Loader & Inference ──────────────────────────────────────────

_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def load_ner_model_pipeline(preferred_model: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads Hugging Face token-classification pipeline with caching.
    Uses in-memory cache to guarantee instantaneous access on subsequent calls.
    """
    model_name = preferred_model or "l3cube-pune/marathi-ner"
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    engine = _internal_loader(model_name)
    _MODEL_CACHE[model_name] = engine
    return engine


def _internal_loader(model_name: str) -> Dict[str, Any]:
    """Internal loading logic with fast local cache check and graceful fallback."""
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Loading '{model_name}' on device: {device}")

    # 1. Fast local cache check (avoids blocking network roundtrips)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name, local_files_only=True)
        ner_pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device
        )
        logger.info(f"Successfully loaded '{model_name}' from local cache")
        return {
            "pipeline": ner_pipe,
            "model_name": model_name,
            "status": "Hugging Face Model Active",
            "is_hf": True
        }
    except Exception:
        pass

    # 2. If not found in local cache, attempt download with fallback on gated/network error
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        ner_pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device
        )
        logger.info(f"Successfully loaded '{model_name}' from Hugging Face Hub")
        return {
            "pipeline": ner_pipe,
            "model_name": model_name,
            "status": "Hugging Face Model Active",
            "is_hf": True
        }
    except Exception as exc:
        logger.warning(f"Could not load '{model_name}' from Hugging Face: {exc}. Using rule engine fallback.")
        return {
            "pipeline": None,
            "model_name": model_name,
            "status": "Knowledge Base Fallback Active",
            "is_hf": False
        }


def extract_entities(text: str, loaded_engine: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Main entity extraction entrypoint.
    Processes Marathi Devanagari text and returns canonicalized entities.
    """
    if not text or not text.strip():
        return []

    # 1. Try Deep Learning Pipeline if available
    if loaded_engine and loaded_engine.get("pipeline") is not None:
        try:
            pipe = loaded_engine["pipeline"]
            raw_results = pipe(text)
            entities = aggregate_subwords_and_bio(raw_results, text)
            if entities:
                return entities
        except Exception as exc:
            logger.error(f"Transformer inference error: {exc}. Falling back to rule engine.")

    # 2. Resilient Rule & Gazetteer Fallback Engine
    m_id = loaded_engine.get("model_name") if loaded_engine else None
    return _FALLBACK_ENGINE.predict(text, model_id=m_id)


def extract_entities_with_model(
    text: str,
    model_id: str,
    engine: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Runs extraction with a specific model and precisely measures inference latency in milliseconds.
    """
    if not engine:
        engine = load_ner_model_pipeline(model_id)

    t0 = time.perf_counter()
    entities = extract_entities(text, engine)
    t1 = time.perf_counter()
    latency_ms = round((t1 - t0) * 1000, 2)

    return {
        "model_id": model_id,
        "model_info": MODEL_REGISTRY.get(model_id, {
            "name": model_id,
            "marathi_name": model_id,
            "badge": "Custom",
            "badge_color": "#6B7280",
            "description": ""
        }),
        "engine": engine,
        "entities": entities,
        "latency_ms": latency_ms,
        "status": engine.get("status", "Active"),
        "is_hf": engine.get("is_hf", False)
    }


# ─── Multi-Model Comparison & Agreement Matrix Computation ──────────────────────

def compute_multi_model_comparison(
    text: str,
    model_outputs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes side-by-side comparative analytics, Jaccard pairwise agreement matrix,
    and consensus entity breakdown across multiple models.
    """
    model_keys = list(model_outputs.keys())
    n_models = len(model_keys)

    # 1. Collect entity set for each model based on canonical entity and tag
    model_entity_sets: Dict[str, set] = {}
    for m_id, res in model_outputs.items():
        m_set = set()
        for e in res.get("entities", []):
            key = (e.get("canonical_entity", e.get("surface", "")), e.get("tag", ""))
            m_set.add(key)
        model_entity_sets[m_id] = m_set

    # 2. Compute Pairwise Jaccard Agreement Matrix
    agreement_matrix: Dict[str, Dict[str, float]] = {}
    for m1 in model_keys:
        agreement_matrix[m1] = {}
        for m2 in model_keys:
            if m1 == m2:
                agreement_matrix[m1][m2] = 100.0
            else:
                s1 = model_entity_sets[m1]
                s2 = model_entity_sets[m2]
                union = s1 | s2
                inter = s1 & s2
                if not union:
                    # If both models found zero entities, agreement is 100%
                    score = 100.0
                else:
                    score = round((len(inter) / len(union)) * 100, 1)
                agreement_matrix[m1][m2] = score

    # 3. Aggregate unique entities across all models for consensus table
    consensus_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for m_id, res in model_outputs.items():
        for e in res.get("entities", []):
            can = e.get("canonical_entity", e.get("surface", ""))
            tag = e.get("tag", "MISC")
            key = (can, tag)

            if key not in consensus_map:
                consensus_map[key] = {
                    "surface": e.get("surface", e.get("entity", "")),
                    "canonical_entity": can,
                    "suffix": e.get("suffix", ""),
                    "case_label": e.get("case_label", ""),
                    "tag": tag,
                    "category": e.get("category", tag),
                    "icon": e.get("icon", ENTITY_CONFIG["MISC"]["icon"]),
                    "votes": {},
                    "detected_by": []
                }
            consensus_map[key]["votes"][m_id] = {
                "tag": tag,
                "confidence": e.get("confidence", 0.0),
                "surface": e.get("surface", "")
            }
            if m_id not in consensus_map[key]["detected_by"]:
                consensus_map[key]["detected_by"].append(m_id)

    consensus_list = []
    unanimous_count = 0
    majority_count = 0

    for key, data in consensus_map.items():
        voters = len(data["detected_by"])
        if voters == n_models:
            level = "Unanimous (एकमत 3/3)"
            level_badge = "Full (3/3)"
            unanimous_count += 1
        elif voters >= 2:
            level = "Majority (बहुमत 2/3)"
            level_badge = "Majority (2/3)"
            majority_count += 1
        else:
            level = "Single Model (एकल 1/3)"
            level_badge = "Single (1/3)"

        data["consensus_level"] = level
        data["consensus_badge"] = level_badge
        data["vote_count"] = voters
        consensus_list.append(data)

    # Sort consensus entities by vote count descending, then canonical name
    consensus_list.sort(key=lambda x: (-x["vote_count"], x["canonical_entity"]))

    # Overall Average Agreement
    all_pairs = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            all_pairs.append(agreement_matrix[model_keys[i]][model_keys[j]])
    avg_agreement = round(sum(all_pairs) / len(all_pairs), 1) if all_pairs else 100.0

    return {
        "model_keys": model_keys,
        "agreement_matrix": agreement_matrix,
        "avg_agreement": avg_agreement,
        "consensus_entities": consensus_list,
        "total_unique": len(consensus_list),
        "unanimous_count": unanimous_count,
        "majority_count": majority_count
    }


# ─── Visual HTML Annotator ───────────────────────────────────────────────────────

def render_annotated_html(text: str, entities: List[Dict[str, Any]], show_canonical: bool = True) -> str:
    """
    Generates rich, modern color-coded HTML markup for in-text visual entity highlighting.
    Preserves all Devanagari punctuation, linebreaks, and spacing.
    Shows canonical base entity alongside surface form if inflected.
    """
    wrapper_style = (
        "line-height: 2.4; "
        "font-size: 1.15rem; "
        "padding: 1.25rem 1.5rem; "
        "border-radius: 12px; "
        "border: 1px solid rgba(128, 128, 128, 0.25); "
        "background: rgba(128, 128, 128, 0.05); "
        "color: inherit; "
        "font-family: 'Mukta', 'Noto Sans Devanagari', -apple-system, BlinkMacSystemFont, sans-serif;"
    )

    if not entities:
        escaped = html.escape(text).replace("\n", "<br/>")
        return f'<div style="{wrapper_style}">{escaped}</div>'

    valid_entities = [e for e in entities if 0 <= e.get("start", -1) < e.get("end", -1) <= len(text)]
    valid_entities.sort(key=lambda x: x["start"])

    # Avoid overlapping spans
    non_overlapping = []
    last_end = -1
    for ent in valid_entities:
        if ent["start"] >= last_end:
            non_overlapping.append(ent)
            last_end = ent["end"]

    html_parts = []
    cursor = 0

    for ent in non_overlapping:
        if ent["start"] > cursor:
            html_parts.append(html.escape(text[cursor:ent["start"]]))

        tag = ent.get("tag", "MISC")
        conf = ent.get("confidence", 0.0)
        cfg = ENTITY_CONFIG.get(tag, ENTITY_CONFIG.get("MISC", {}))
        surface = html.escape(str(ent.get("surface", ent.get("entity", ""))))
        canonical = html.escape(str(ent.get("canonical_entity", "")))
        suffix = html.escape(str(ent.get("suffix", "")))

        bg_col = cfg.get("bg_color", "rgba(147, 51, 234, 0.16)")
        border_col = cfg.get("border_color", "#9333EA")
        badge_bg = cfg.get("badge_bg", border_col)
        marathi_lbl = cfg.get("marathi", tag)
        icon_svg = cfg.get("icon", "")

        # Inflection sub-badge if inflected
        inflection_badge = ""
        if show_canonical and suffix and canonical and canonical != surface:
            inflection_badge = (
                f'<span style="'
                f'font-size: 0.72rem; '
                f'font-weight: 600; '
                f'background: rgba(0, 0, 0, 0.12); '
                f'color: inherit; '
                f'padding: 1px 5px; '
                f'border-radius: 4px; '
                f'margin-left: 4px; '
                f'display: inline-block;">'
                f'मूळ: {canonical} (+{suffix})'
                f'</span>'
            )

        badge_html = (
            f'<span style="'
            f'background: {bg_col}; '
            f'border: 1px solid {border_col}; '
            f'padding: 2px 7px; '
            f'border-radius: 6px; '
            f'margin: 0 2px; '
            f'display: inline; '
            f'box-decoration-break: clone; '
            f'-webkit-box-decoration-break: clone; '
            f'font-weight: 600;">'
            f'<strong style="color: inherit;">{surface}</strong>'
            f'{inflection_badge}'
            f'<span style="'
            f'font-size: 0.72rem; '
            f'font-weight: 700; '
            f'background: {badge_bg}; '
            f'color: #FFFFFF !important; '
            f'padding: 1px 6px; '
            f'border-radius: 4px; '
            f'margin-left: 5px; '
            f'display: inline-flex; '
            f'align-items: center; '
            f'gap: 3px; '
            f'vertical-align: 1px; '
            f'letter-spacing: 0.02em;">'
            f'{icon_svg} '
            f'{marathi_lbl} · {conf:.1f}%'
            f'</span>'
            f'</span>'
        )
        html_parts.append(badge_html)
        cursor = ent["end"]

    if cursor < len(text):
        html_parts.append(html.escape(text[cursor:]))

    full_content = "".join(html_parts).replace("\n", "<br/>")
    return f'<div style="{wrapper_style}">{full_content}</div>'
