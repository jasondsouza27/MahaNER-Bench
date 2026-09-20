"""
Comprehensive MahaNER Test Suite
Testing Devanagari Inflectional Normalization, Canonicalization & Multi-Model Comparison
"""

import sys
import io
import unittest

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import ner_engine


class TestDevanagariCanonicalization(unittest.TestCase):
    """Test Feature 1: Devanagari Inflectional Normalization & Stemming"""

    def test_explicit_task_cases(self):
        cases = [
            # surface, expected_canonical, expected_suffix
            ("रायगडावर", "रायगड", "वर"),
            ("सचिन तेंडुलकरने", "सचिन तेंडुलकर", "ने"),
            ("महाराष्ट्राचे", "महाराष्ट्र", "चे"),
            ("पुण्यात", "पुणे", "त"),
            ("पुण्यातील", "पुणे", "तील"),
            ("मुंबईतील", "मुंबई", "तील"),
            ("इस्रोने", "इस्रो", "ने"),
            ("ISROने", "ISRO", "ने"),
        ]
        for surface, exp_can, exp_suff in cases:
            can, suff, case_lbl = ner_engine.canonicalize_marathi_entity(surface)
            self.assertEqual(can, exp_can, f"Failed canonical for '{surface}': got '{can}', expected '{exp_can}'")
            self.assertEqual(suff, exp_suff, f"Failed suffix for '{surface}': got '{suff}', expected '{exp_suff}'")
            self.assertTrue(len(case_lbl) > 0, f"Case label missing for '{surface}'")

    def test_uninflected_base_entities(self):
        base_cases = [
            "छत्रपती शिवाजी महाराज",
            "देवेंद्र फडणवीस",
            "भारत",
            "पुणे",
            "मुंबई",
            "महाराष्ट्र",
            "ISRO"
        ]
        for surface in base_cases:
            can, suff, case_lbl = ner_engine.canonicalize_marathi_entity(surface)
            self.assertEqual(can, surface, f"Uninflected '{surface}' changed to '{can}'")
            self.assertIn(suff, ["", None], f"Expected no suffix for base word '{surface}', got '{suff}'")
            self.assertIn("प्रथमा", case_lbl)

    def test_multi_word_complex_entities(self):
        cases = [
            ("वानखेडे स्टेडियमवर", "वानखेडे स्टेडियम", "वर"),
            ("हिंदवी स्वराज्याची", "हिंदवी स्वराज्य", "ची"),
            ("चांद्रयान मोहिमेचे", "चांद्रयान मोहीम", "चे"),
            ("टाटा मोटर्सने", "टाटा मोटर्स", "ने"),
            ("पुणे विद्यापीठात", "पुणे विद्यापीठ", "त"),
            ("रिझर्व्ह बँक ऑफ इंडियाचे", "रिझर्व्ह बँक ऑफ इंडिया", "चे")
        ]
        for surface, exp_can, exp_suff in cases:
            can, suff, _ = ner_engine.canonicalize_marathi_entity(surface)
            self.assertEqual(can, exp_can, f"Mismatch for '{surface}': got '{can}'")
            self.assertEqual(suff, exp_suff, f"Mismatch suffix for '{surface}': got '{suff}'")

    def test_general_morphological_rules_unseen(self):
        # Testing general rule-based stemming for words not in the gazetteer
        test_words = [
            ("अमेरिकेत", "अमेरिका", "त"),
            ("नांदेडमध्ये", "नांदेड", "मध्ये"),
            ("सभेत", "सभा", "त"),
            ("नियमांनुसार", "नियम", "नुसार"),
        ]
        for surf, exp_can, exp_suff in test_words:
            can, suff, case_lbl = ner_engine.canonicalize_marathi_entity(surf)
            self.assertEqual(suff, exp_suff, f"Failed suffix extraction for unseen '{surf}'")
            self.assertEqual(can, exp_can, f"Failed stem recovery for unseen '{surf}'")

    def test_edge_cases_empty_and_punct(self):
        self.assertEqual(ner_engine.canonicalize_marathi_entity("")[0], "")
        self.assertEqual(ner_engine.canonicalize_marathi_entity("   ")[0], "   ")
        # Leading / trailing brackets
        can, suff, _ = ner_engine.canonicalize_marathi_entity("(रायगडावर)")
        self.assertEqual(can, "रायगड")
        self.assertEqual(suff, "वर")

    def test_unicode_ligature_and_oblique_slicing(self):
        # Testing exact Unicode multi-byte slicing for ्या (3 bytes) and ऱ्या (4 bytes)
        cases = [
            ("ठाण्यात", "ठाणे", "त"),
            ("गोव्यात", "गोवा", "त"),
            ("साताऱ्यात", "सातारा", "त"),
            ("पुणेत", "पुणे", "त"),
            ("ठाणेत", "ठाणे", "त"),
        ]
        for surf, exp_can, exp_suff in cases:
            can, suff, _ = ner_engine.canonicalize_marathi_entity(surf)
            self.assertEqual(can, exp_can, f"Failed ligature oblique for '{surf}': got '{can}', expected '{exp_can}'")
            self.assertEqual(suff, exp_suff, f"Failed suffix for '{surf}'")

    def test_female_names_and_surnames_ending_in_aa(self):
        # Preventing false-positive aa-truncation on personal names & surnames
        cases = [
            ("लताने", "लता", "ने"),
            ("इंदिराने", "इंदिरा", "ने"),
            ("सोनियाने", "सोनिया", "ने"),
            ("ममताने", "ममता", "ने"),
            ("प्रतिभाने", "प्रतिभा", "ने"),
            ("अनुष्का शर्माने", "अनुष्का शर्मा", "ने"),
            ("प्रियांका चोप्राने", "प्रियांका चोप्रा", "ने"),
        ]
        for surf, exp_can, exp_suff in cases:
            can, suff, _ = ner_engine.canonicalize_marathi_entity(surf, tag="PER")
            self.assertEqual(can, exp_can, f"Truncated female name/surname '{surf}': got '{can}', expected '{exp_can}'")
            self.assertEqual(suff, exp_suff, f"Failed suffix for '{surf}'")

    def test_honorific_plurals_anusvara(self):
        # Testing honorific plural anusvara stripping (ीं, ूं, ें, ां)
        cases = [
            ("मोदींचे", "मोदी", "चे"),
            ("मोदींनी", "मोदी", "नी"),
            ("शिंदेंनी", "शिंदे", "नी"),
            ("ठाकरेंनी", "ठाकरे", "नी"),
            ("पवारांनी", "पवार", "नी"),
            ("फडणवीसांनी", "फडणवीस", "नी"),
            ("तेंडुलकरांनी", "तेंडुलकर", "नी"),
        ]
        for surf, exp_can, exp_suff in cases:
            can, suff, _ = ner_engine.canonicalize_marathi_entity(surf, tag="PER")
            self.assertEqual(can, exp_can, f"Failed honorific canonical for '{surf}': got '{can}', expected '{exp_can}'")
            self.assertEqual(suff, exp_suff, f"Failed suffix for '{surf}'")

    def test_vowel_shortening_and_institutions(self):
        # Testing penultimate vowel restoration (पूर) and institutions
        cases = [
            ("नागपुरात", "नागपूर", "त"),
            ("नागपूरचे", "नागपूर", "चे"),
            ("कोल्हापुरात", "कोल्हापूर", "त"),
            ("सोलापुरात", "सोलापूर", "त"),
            ("कॅनडात", "कॅनडा", "त"),
            ("कॅनडामध्ये", "कॅनडा", "मध्ये"),
            ("रशियात", "रशिया", "त"),
            ("रशियामध्ये", "रशिया", "मध्ये"),
            ("अमेरिकामध्ये", "अमेरिका", "मध्ये"),
            ("संसदेत", "संसद", "त"),
            ("संसदेचे", "संसद", "चे"),
            ("मंत्रालयात", "मंत्रालय", "त"),
            ("विद्यापीठात", "विद्यापीठ", "त"),
            ("बाटाने", "बाटा", "ने"),
        ]
        for surf, exp_can, exp_suff in cases:
            can, suff, _ = ner_engine.canonicalize_marathi_entity(surf)
            self.assertEqual(can, exp_can, f"Mismatch for '{surf}': got '{can}', expected '{exp_can}'")
            self.assertEqual(suff, exp_suff, f"Failed suffix for '{surf}'")


class TestEntityExtractionAndStructure(unittest.TestCase):
    """Test output dictionary structure and fields"""

    def test_entity_fields_presence(self):
        text = "छत्रपती शिवाजी महाराज यांनी रायगडावर हिंदवी स्वराज्याची स्थापना केली."
        ents = ner_engine.extract_entities(text)
        self.assertGreater(len(ents), 0)

        for e in ents:
            # Check backwards compatibility
            self.assertIn("entity", e)
            # Check Feature 1 fields
            self.assertIn("surface", e)
            self.assertIn("canonical_entity", e)
            self.assertIn("suffix", e)
            self.assertIn("case_label", e)
            # Check core metadata
            self.assertIn("tag", e)
            self.assertIn("category", e)
            self.assertIn("confidence", e)
            self.assertIn("start", e)
            self.assertIn("end", e)
            self.assertIn("icon", e)

            # Character boundary check
            self.assertEqual(text[e["start"]:e["end"]], e["surface"])

    def test_empty_input(self):
        self.assertEqual(ner_engine.extract_entities(""), [])
        self.assertEqual(ner_engine.extract_entities("    "), [])


class TestMultiModelComparison(unittest.TestCase):
    """Test Feature 2: Multi-Model Comparison Mode & Agreement Matrix"""

    def test_extract_entities_with_model(self):
        text = "छत्रपती शिवाजी महाराज यांनी रायगडावर हिंदवी स्वराज्याची स्थापना केली."
        res = ner_engine.extract_entities_with_model(text, "l3cube-pune/marathi-ner")
        self.assertIn("model_id", res)
        self.assertIn("latency_ms", res)
        self.assertIn("entities", res)
        self.assertGreater(res["latency_ms"], 0.0)
        self.assertGreater(len(res["entities"]), 0)

    def test_multi_model_agreement_math(self):
        text = "छत्रपती शिवाजी महाराज यांनी रायगडावर हिंदवी स्वराज्याची स्थापना केली."
        m1 = ner_engine.extract_entities_with_model(text, "l3cube-pune/marathi-ner")
        m2 = ner_engine.extract_entities_with_model(text, "ai4bharat/IndicNER")
        m3 = ner_engine.extract_entities_with_model(text, "Davlan/xlm-roberta-base-ner-hrl")

        comp = ner_engine.compute_multi_model_comparison(text, {
            "l3cube-pune/marathi-ner": m1,
            "ai4bharat/IndicNER": m2,
            "Davlan/xlm-roberta-base-ner-hrl": m3
        })

        # Self-agreement must be 100%
        for m in ["l3cube-pune/marathi-ner", "ai4bharat/IndicNER", "Davlan/xlm-roberta-base-ner-hrl"]:
            self.assertEqual(comp["agreement_matrix"][m][m], 100.0)

        # Symmetry: Jaccard(A, B) == Jaccard(B, A)
        m_keys = comp["model_keys"]
        for i in range(len(m_keys)):
            for j in range(len(m_keys)):
                self.assertEqual(
                    comp["agreement_matrix"][m_keys[i]][m_keys[j]],
                    comp["agreement_matrix"][m_keys[j]][m_keys[i]]
                )

        # Total unique entities
        self.assertGreater(comp["total_unique"], 0)
        self.assertGreaterEqual(comp["unanimous_count"], 1)

    def test_html_rendering(self):
        text = "सचिन तेंडुलकरने वानखेडे स्टेडियमवर सामना खेळला."
        ents = ner_engine.extract_entities(text)
        html_out = ner_engine.render_annotated_html(text, ents, show_canonical=True)
        self.assertIn("सचिन तेंडुलकर", html_out)
        self.assertIn("वानखेडे स्टेडियम", html_out)
        self.assertIn("मूळ:", html_out)


if __name__ == "__main__":
    unittest.main()
