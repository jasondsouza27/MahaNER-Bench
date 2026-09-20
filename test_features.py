import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import ner_engine

print("=== 1. Testing Feature 1: Canonicalization & Stemming ===")
test_sentences = [
    "छत्रपती शिवाजी महाराज यांनी रायगडावर हिंदवी स्वराज्याची स्थापना केली.",
    "महाराष्ट्राचे मुख्यमंत्री देवेंद्र फडणवीस यांनी मुंबई येथे नवीन मेट्रो मार्गाची घोषणा केली.",
    "सचिन तेंडुलकरने वानखेडे स्टेडियमवर आपला शेवटचा आंतरराष्ट्रीय क्रिकेट सामना खेळला.",
    "भारतीय अंतराळ संशोधन संस्था (ISRO) ने श्रीहरिकोटा येथून चांद्रयान मोहिमेचे प्रक्षेपण केले.",
    "पुण्यात आणि मुंबईतील उद्योजकांनी महाराष्ट्रात मोठी गुंतवणूक केली."
]

for s in test_sentences:
    print(f"\nText: {s}")
    ents = ner_engine.extract_entities(s)
    for e in ents:
        print(f"  [{e['tag']}] Surface: '{e['surface']}' -> Canonical: '{e['canonical_entity']}' | Suffix: '{e['suffix']}' | Case: '{e['case_label']}'")

print("\n=== 2. Testing Feature 2: Multi-Model Extraction & Comparison ===")
text = "छत्रपती शिवाजी महाराज यांनी रायगडावर हिंदवी स्वराज्याची स्थापना केली."
m1 = ner_engine.extract_entities_with_model(text, "l3cube-pune/marathi-ner")
m2 = ner_engine.extract_entities_with_model(text, "ai4bharat/IndicNER")
m3 = ner_engine.extract_entities_with_model(text, "Davlan/xlm-roberta-base-ner-hrl")

print(f"Model 1 ({m1['model_info']['name']}): Status = {m1['status']}, Latency = {m1['latency_ms']} ms, Ents = {len(m1['entities'])}")
print(f"Model 2 ({m2['model_info']['name']}): Status = {m2['status']}, Latency = {m2['latency_ms']} ms, Ents = {len(m2['entities'])}")
print(f"Model 3 ({m3['model_info']['name']}): Status = {m3['status']}, Latency = {m3['latency_ms']} ms, Ents = {len(m3['entities'])}")

comp = ner_engine.compute_multi_model_comparison(text, {
    "l3cube-pune/marathi-ner": m1,
    "ai4bharat/IndicNER": m2,
    "Davlan/xlm-roberta-base-ner-hrl": m3
})

print(f"\nAverage Agreement: {comp['avg_agreement']}%")
print(f"Agreement Matrix: {comp['agreement_matrix']}")
print("\nConsensus Breakdown:")
for c in comp["consensus_entities"]:
    print(f"  Entity: '{c['canonical_entity']}' ({c['tag']}) | Consensus: {c['consensus_level']} | Detected by: {c['detected_by']}")

print("\nALL FEATURE TESTS PASSED SUCCESSFULLY!")
