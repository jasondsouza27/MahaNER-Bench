"""
MahaNER: Named Entity Recognition and Information Extraction System for Marathi Text
=====================================================================================
Production-ready Streamlit web application providing interactive Devanagari entity
extraction, inflectional normalization & stemming (विभक्ती प्रत्यय पृथक्करण),
side-by-side multi-model comparison mode, visual highlights, analytics, and export.
"""

import json
import logging
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import ner_engine

from ner_engine import (
    MODEL_REGISTRY,
    ENTITY_CONFIG,
    load_ner_model_pipeline,
    extract_entities,
    extract_entities_with_model,
    compute_multi_model_comparison,
    render_annotated_html,
    compute_entity_frequency,
    compute_cooccurrence,
    lookup_wikidata_entities,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MahaNER-App")

# ─── Page Configuration ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MahaNER - मराठी NER व विभक्ती पृथक्करण प्रणाली",
    page_icon=":material/flag:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Minimal theme-safe CSS ─────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Tighten top padding */
    .block-container { padding-top: 1rem; }
    /* Hide branding */
    #MainMenu, footer { visibility: hidden; }
    /* Metric styling */
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─── Preset Marathi Test Cases ───────────────────────────────────────────────────

PRESET_OPTIONS = {
    "-- स्वतःचा मजकूर लिहा (Custom) --": "",
    "इतिहास: छत्रपती शिवाजी महाराज व रायगड": "छत्रपती शिवाजी महाराज यांनी रायगडावर हिंदवी स्वराज्याची स्थापना केली.",
    "राजकारण: फडणवीस, मुंबई व मेट्रो": "महाराष्ट्राचे मुख्यमंत्री देवेंद्र फडणवीस यांनी मुंबई येथे नवीन मेट्रो मार्गाची घोषणा केली.",
    "क्रीडा: सचिन तेंडुलकर व वानखेडे": "सचिन तेंडुलकरने वानखेडे स्टेडियमवर आपला शेवटचा आंतरराष्ट्रीय क्रिकेट सामना खेळला.",
    "विज्ञान: ISRO, श्रीहरिकोटा व चांद्रयान": "भारतीय अंतराळ संशोधन संस्था (ISRO) ने श्रीहरिकोटा येथून चांद्रयान मोहिमेचे प्रक्षेपण केले.",
    "विविध शहरे व विभक्ती प्रत्यय (Inflection Demo)": "पुण्यात आणि मुंबईतील उद्योजकांनी महाराष्ट्रात मोठी गुंतवणूक केली आणि नागपुरात नवीन प्रकल्प सुरू केला.",
}

# ─── Sidebar ─────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("MahaNER", icon=":material/flag:")
        st.caption("मराठी नामनिर्देशित घटक ओळख व विभक्ती पृथक्करण प्रणाली")

        st.divider()

        # Settings
        st.subheader("Settings (प्राधान्ये)", icon=":material/settings:")

        model_choice = st.selectbox(
            "Primary Model (प्राथमिक मॉडेल)",
            list(MODEL_REGISTRY.keys()),
            index=0,
            format_func=lambda m: f"{MODEL_REGISTRY[m]['name']} ({MODEL_REGISTRY[m]['badge']})",
            help="Select transformer pipeline for single-model mode."
        )

        conf_threshold = st.slider(
            "Min Confidence % (किमान विश्वास)", 0, 100, 35, 5,
            help="Hide entities below this confidence threshold."
        )

        show_canonical_inline = st.checkbox(
            "Show Root Base in Highlights (मूळ घटक दाखवा)",
            value=True,
            help="Inline display of canonical base noun next to inflected surface forms."
        )

        st.divider()

        # Architecture & Innovations
        st.subheader("Architecture & Features", icon=":material/account_tree:")
        st.markdown(
            "- **Backbone Models:**\n"
            "  - `L3Cube-MahaBERT` (Monolingual Marathi)\n"
            "  - `AI4Bharat-IndicNER` (11 Indic Languages)\n"
            "  - `XLM-RoBERTa-HRL` (Cross-Lingual)\n"
            "- **विभक्ती पृथक्करण (Canonicalizer):**\n"
            "  - Postposition & Suffix Stripper\n"
            "  - सामान्यरूप ➔ मूळरूप Reversal\n"
            "- **Post-proc:** Ligature-safe BIO Aggregator\n"
            "- **Evaluation:** 3-Way Model Agreement Matrix"
        )

    return model_choice, conf_threshold, show_canonical_inline


# ─── Render Single Model Mode ───────────────────────────────────────────────────

def render_single_model_mode(selected_model: str, conf_threshold: int, show_canonical_inline: bool):
    st.markdown("#### :material/target: एकल मॉडेल विश्लेषण (Single Model Extraction & Canonicalization)")
    st.caption("निवडलेल्या ट्रान्सफॉर्मर मॉडेलद्वारे नामनिर्देशित घटक ओळख व विभक्ती प्रत्यय पृथक्करण.")

    # ── Load model ──
    with st.spinner(f"Loading '{selected_model}'…"):
        engine = load_ner_model_pipeline(selected_model)

    m_info = MODEL_REGISTRY.get(selected_model, {})
    is_hf = engine.get("is_hf", False)
    if is_hf:
        st.success(f"**{m_info.get('name', selected_model)}** सक्रिय (Hugging Face Active — {m_info.get('focus', '')})", icon=":material/check_circle:")
    else:
        st.warning(f"**{m_info.get('name', selected_model)}** (Knowledge Base Offline Fallback Active)", icon=":material/warning:")

    # ── Input area ──
    default_single_text = PRESET_OPTIONS["इतिहास: छत्रपती शिवाजी महाराज व रायगड"]
    if "single_text_input" not in st.session_state:
        st.session_state["single_text_input"] = default_single_text

    def _sync_single_preset():
        chosen = st.session_state.get("single_preset_select", "")
        if chosen in PRESET_OPTIONS and PRESET_OPTIONS[chosen]:
            st.session_state["single_text_input"] = PRESET_OPTIONS[chosen]

    preset_choice = st.selectbox(
        "नमुना वाक्य निवडा (Preset Test Cases):",
        list(PRESET_OPTIONS.keys()),
        index=1,
        key="single_preset_select",
        on_change=_sync_single_preset
    )

    col_text, col_upload = st.columns([4, 1])
    with col_upload:
        uploaded = st.file_uploader("Upload .txt", type=["txt"], key="single_uploader")
        if uploaded:
            try:
                file_text = uploaded.read().decode("utf-8")
                st.session_state["single_text_input"] = file_text
                st.success(f"{len(file_text)} chars loaded", icon=":material/check:")
            except Exception as e:
                st.error(str(e))

    with col_text:
        user_text = st.text_area(
            "मराठी मजकूर टाईप करा किंवा पेस्ट करा (Enter Marathi Text):",
            height=110,
            placeholder="येथे मराठी मजकूर टाईप करा…",
            key="single_text_input"
        )

    clicked = st.button(
        "घटक ओळखा व विभक्ती पृथक्करण करा (Extract & Canonicalize)",
        icon=":material/search:",
        type="primary",
        use_container_width=True,
        key="btn_single_extract"
    )

    text = user_text.strip()

    if clicked and text:
        with st.spinner("Analyzing…"):
            raw_entities = extract_entities(text, engine)
        filtered = [e for e in raw_entities if e["confidence"] >= conf_threshold]

        st.divider()

        # Metrics row
        words = len(text.split())
        n_ent = len(filtered)
        density = (n_ent / words * 100) if words else 0
        avg_conf = (sum(e["confidence"] for e in filtered) / n_ent) if n_ent else 0
        inflected_count = sum(1 for e in filtered if e.get("suffix"))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("शब्द (Words)", words, icon=":material/notes:")
        c2.metric("घटक (Entities)", n_ent, icon=":material/push_pin:")
        c3.metric("विभक्ती प्रत्यययुक्त (Inflected)", inflected_count, icon=":material/spellcheck:")
        c4.metric("घटक घनता (Density)", f"{density:.1f}%", icon=":material/pie_chart:")
        c5.metric("सरासरी विश्वास (Confidence)", f"{avg_conf:.1f}%", icon=":material/verified:")

        # Sub-tabs
        tab_hl, tab_tbl, tab_chart, tab_freq, tab_network, tab_linking, tab_export = st.tabs([
            ":material/palette: दृश्य हायलाइट्स (Highlights)",
            ":material/table_chart: तपशीलवार तक्ता (Table)",
            ":material/bar_chart: आलेख व विश्लेषण (Analytics)",
            ":material/leaderboard: घटक वारंवारता (Entity Frequency)",
            ":material/hub: सह-उपस्थिती जाळे (Co-occurrence Network)",
            ":material/link: विकिडेटा लिंकिंग (Entity Linking)",
            ":material/download: डेटा निर्यात (Export)",
        ])

        # ── Tab 1: Highlights ──
        with tab_hl:
            # Color legend
            legend_cols = st.columns(4)
            for idx, (tag, cfg) in enumerate(ENTITY_CONFIG.items()):
                b_col = cfg.get("border_color", "#9333EA")
                bg_c = cfg.get("bg_color", "rgba(147, 51, 234, 0.16)")
                lbl = cfg.get("label", tag)
                icon = cfg.get("icon", "")
                legend_cols[idx].markdown(
                    f'<div style="background:{bg_c}; border:1px solid {b_col}; '
                    f'padding:6px 12px; border-radius:8px; text-align:center; font-weight:600; font-size:0.88rem; display:flex; align-items:center; justify-content:center; gap:6px;">'
                    f'{icon} <span style="color:{b_col};">{lbl}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.write("")
            annotated = render_annotated_html(text, filtered, show_canonical=show_canonical_inline)
            st.markdown(annotated, unsafe_allow_html=True)

            # Enriched Entity chips with canonical base & suffix
            if filtered:
                st.write("")
                st.markdown("##### :material/push_pin: ओळखलेले घटक व मूळरूप (Identified Entities & Base Canonical Forms):")
                chip_html = '<div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:6px;">'
                for e in filtered:
                    cfg = ENTITY_CONFIG.get(e.get("tag", "MISC"), ENTITY_CONFIG.get("MISC", {}))
                    b_col = cfg.get("border_color", "#9333EA")
                    bg_c = cfg.get("bg_color", "rgba(147, 51, 234, 0.16)")
                    badge_b = cfg.get("badge_bg", b_col)
                    m_lbl = cfg.get("marathi", e.get("tag", "घटक"))
                    icon = cfg.get("icon", "")

                    surface_text = e.get("surface", e.get("entity", ""))
                    canon_text = e.get("canonical_entity", surface_text)
                    suffix_text = e.get("suffix", "")
                    case_text = e.get("case_label", "")

                    if suffix_text and canon_text != surface_text:
                        label_content = (
                            f'<strong style="color:inherit;">{surface_text}</strong>'
                            f'<span style="opacity:0.85; margin:0 3px;">➔</span>'
                            f'<strong style="color:{b_col};">{canon_text}</strong> '
                            f'<span style="background:rgba(0,0,0,0.1); padding:1px 5px; border-radius:4px; font-size:0.75rem;">+{suffix_text} ({case_text.split()[0]})</span>'
                        )
                    else:
                        label_content = (
                            f'<strong style="color:inherit;">{surface_text}</strong> '
                            f'<span style="opacity:0.75; font-size:0.75rem;">(मूळ रूप)</span>'
                        )

                    chip_html += (
                        f'<div style="display:inline-flex; align-items:center; gap:8px; background:{bg_c}; '
                        f'border:1px solid {b_col}; border-radius:20px; padding:6px 14px; font-size:0.92rem;">'
                        f'<span>{icon}</span>'
                        f'{label_content}'
                        f'<span style="background:{badge_b}; color:#FFFFFF !important; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:10px; margin-left:4px;">'
                        f'{m_lbl} · {e.get("confidence", 0.0):.1f}%'
                        f'</span>'
                        f'</div>'
                    )
                chip_html += '</div>'
                st.markdown(chip_html, unsafe_allow_html=True)

        # ── Tab 2: Table ──
        with tab_tbl:
            if filtered:
                rows = []
                for e in filtered:
                    rows.append({
                        "मूळ मजकूर (Surface)": e.get("surface", e.get("entity", "")),
                        "प्रमाणित घटक (Canonical Base)": e.get("canonical_entity", ""),
                        "विभक्ती प्रत्यय (Suffix)": f"+{e['suffix']}" if e.get("suffix") else "—",
                        "विभक्ती / कारकार्थ (Case)": e.get("case_label", "प्रथमा"),
                        "टॅग (Tag)": e.get("tag", ""),
                        "प्रवर्ग (Category)": e.get("category", ""),
                        "विश्वास (Confidence)": f"{e['confidence']:.1f}%",
                        "स्थान (Span)": f"{e['start']}–{e['end']}",
                    })
                df_entities = pd.DataFrame(rows)
                st.dataframe(df_entities, use_container_width=True, hide_index=True)
            else:
                st.info("या विश्वास मर्यादा स्तरावर कोणतेही घटक आढळले नाहीत.")

        # ── Tab 3: Analytics ──
        with tab_chart:
            if filtered:
                col_chart1, col_chart2 = st.columns(2)
                
                # Tag Distribution
                with col_chart1:
                    st.markdown("##### :material/label: घटक प्रवर्ग वितरण (Category Distribution)")
                    tag_counts = {}
                    for e in filtered:
                        tag_counts[e["tag"]] = tag_counts.get(e["tag"], 0) + 1

                    chart_rows = []
                    color_map = {}
                    for tag, cfg in ENTITY_CONFIG.items():
                        label = f"{cfg['marathi']} ({tag})"
                        chart_rows.append({"Category": label, "Count": tag_counts.get(tag, 0)})
                        color_map[label] = cfg["border_color"]

                    df_chart = pd.DataFrame(chart_rows)
                    if HAS_PLOTLY:
                        fig_cat = px.bar(
                            df_chart, x="Category", y="Count",
                            color="Category", color_discrete_map=color_map,
                            text="Count"
                        )
                        fig_cat.update_layout(
                            showlegend=False,
                            margin=dict(t=20, b=30, l=30, r=20),
                            height=320,
                            xaxis_title="", yaxis_title="घटक संख्या"
                        )
                        fig_cat.update_traces(textposition="outside")
                        st.plotly_chart(fig_cat, use_container_width=True)
                    else:
                        st.bar_chart(df_chart.set_index("Category")["Count"])

                # Inflection Breakdown
                with col_chart2:
                    st.markdown("##### :material/spellcheck: विभक्ती प्रत्यय प्रमाण (Inflected vs Base Form)")
                    inflected_cnt = sum(1 for e in filtered if e.get("suffix"))
                    base_cnt = len(filtered) - inflected_cnt
                    df_infl = pd.DataFrame([
                        {"Type": "विभक्ती प्रत्यययुक्त (Inflected)", "Count": inflected_cnt, "Color": "#EF4444"},
                        {"Type": "मूळ रूप (Nominative / Base)", "Count": base_cnt, "Color": "#10B981"}
                    ])
                    if HAS_PLOTLY:
                        fig_infl = px.pie(
                            df_infl, names="Type", values="Count",
                            color="Type",
                            color_discrete_map={
                                "विभक्ती प्रत्यययुक्त (Inflected)": "#F59E0B",
                                "मूळ रूप (Nominative / Base)": "#10B981"
                            },
                            hole=0.45
                        )
                        fig_infl.update_layout(
                            margin=dict(t=20, b=30, l=30, r=20),
                            height=320,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
                        )
                        st.plotly_chart(fig_infl, use_container_width=True)
                    else:
                        st.bar_chart(df_infl.set_index("Type")["Count"])
            else:
                st.info("विश्लेषणासाठी डेटा उपलब्ध नाही.")

        # ── Tab 4: Entity Frequency Dashboard ──
        with tab_freq:
            if filtered:
                freq_data = compute_entity_frequency(filtered)

                # Summary metrics
                fc1, fc2 = st.columns(2)
                fc1.metric("एकूण घटक (Total Mentions)", freq_data["total_entities"], icon=":material/format_list_numbered:")
                fc2.metric("अनन्य घटक (Unique Entities)", freq_data["unique_entities"], icon=":material/fingerprint:")

                st.write("")

                col_leader, col_treemap = st.columns([1, 1])

                # Leaderboard bar chart
                with col_leader:
                    st.markdown("##### :material/leaderboard: घटक वारंवारता क्रमवारी (Entity Frequency Leaderboard)")
                    if freq_data["freq_table"]:
                        leader_rows = []
                        color_map_freq = {}
                        for item in freq_data["freq_table"][:15]:
                            label = item["canonical_entity"]
                            leader_rows.append({
                                "Entity": label,
                                "Mentions": item["count"],
                                "Tag": item["tag"],
                            })
                            cfg = ENTITY_CONFIG.get(item["tag"], ENTITY_CONFIG.get("MISC", {}))
                            color_map_freq[label] = cfg.get("border_color", "#9333EA")

                        df_leader = pd.DataFrame(leader_rows)
                        if HAS_PLOTLY:
                            fig_freq = px.bar(
                                df_leader, x="Mentions", y="Entity",
                                color="Entity", color_discrete_map=color_map_freq,
                                text="Mentions", orientation="h",
                            )
                            fig_freq.update_layout(
                                showlegend=False,
                                margin=dict(t=10, b=20, l=20, r=20),
                                height=max(280, len(leader_rows) * 38),
                                yaxis=dict(autorange="reversed"),
                                xaxis_title="उल्लेख संख्या (Mentions)",
                                yaxis_title="",
                            )
                            fig_freq.update_traces(textposition="outside")
                            st.plotly_chart(fig_freq, use_container_width=True)
                        else:
                            st.bar_chart(df_leader.set_index("Entity")["Mentions"])

                # Treemap by category
                with col_treemap:
                    st.markdown("##### :material/grid_view: प्रवर्गानुसार ट्रीमॅप (Category Treemap)")
                    if HAS_PLOTLY and freq_data["freq_table"]:
                        tree_rows = []
                        for item in freq_data["freq_table"]:
                            cfg = ENTITY_CONFIG.get(item["tag"], ENTITY_CONFIG.get("MISC", {}))
                            tree_rows.append({
                                "entity": item["canonical_entity"],
                                "category": f"{cfg.get('marathi', item['tag'])} ({item['tag']})",
                                "count": item["count"],
                            })
                        df_tree = pd.DataFrame(tree_rows)
                        fig_tree = px.treemap(
                            df_tree, path=["category", "entity"], values="count",
                            color="category",
                            color_discrete_map={
                                f"{ENTITY_CONFIG[t]['marathi']} ({t})": ENTITY_CONFIG[t]["border_color"]
                                for t in ENTITY_CONFIG
                            },
                        )
                        fig_tree.update_layout(
                            margin=dict(t=10, b=10, l=10, r=10),
                            height=max(280, len(freq_data["freq_table"]) * 28),
                        )
                        fig_tree.update_traces(
                            textinfo="label+value",
                            textfont_size=13,
                        )
                        st.plotly_chart(fig_tree, use_container_width=True)
                    elif freq_data["freq_table"]:
                        st.dataframe(
                            pd.DataFrame(freq_data["freq_table"])[
                                ["canonical_entity", "count", "tag", "category"]
                            ],
                            use_container_width=True, hide_index=True,
                        )

                # Surface form variants table
                st.markdown("##### :material/swap_horiz: मूळ रूप ↔ विविध रूपे (Canonical ↔ Surface Variants)")
                variant_rows = []
                for item in freq_data["freq_table"]:
                    variant_rows.append({
                        "प्रमाणित घटक (Canonical)": item["canonical_entity"],
                        "प्रवर्ग (Tag)": item["tag"],
                        "उल्लेख (Mentions)": item["count"],
                        "विविध रूपे (Surface Forms)": ", ".join(item["surface_forms"]),
                    })
                st.dataframe(pd.DataFrame(variant_rows), use_container_width=True, hide_index=True)
            else:
                st.info("वारंवारता विश्लेषणासाठी कोणतेही घटक उपलब्ध नाहीत.")

        # ── Tab 5: Co-occurrence Network ──
        with tab_network:
            if filtered and len(filtered) >= 2:
                cooccur = compute_cooccurrence(text, filtered)

                if cooccur["edges"]:
                    st.markdown("##### :material/hub: घटक सह-उपस्थिती जाळे (Entity Co-occurrence Network)")
                    st.caption("एकाच वाक्यात एकत्र आढळणारे घटक जोडलेले दिसतात. रेषेची जाडी सह-उपस्थितीच्या संख्येवर अवलंबून आहे.")

                    col_graph, col_matrix = st.columns([3, 2])

                    # Interactive network graph using Plotly
                    with col_graph:
                        import math

                        nodes = cooccur["nodes"]
                        edges = cooccur["edges"]
                        node_tags = cooccur["node_tags"]
                        n = len(nodes)

                        # Circular layout
                        positions = {}
                        for i, node in enumerate(nodes):
                            angle = 2 * math.pi * i / n
                            positions[node] = (math.cos(angle), math.sin(angle))

                        if HAS_PLOTLY:
                            # Edge traces
                            edge_traces = []
                            for edge in edges:
                                x0, y0 = positions[edge["source"]]
                                x1, y1 = positions[edge["target"]]
                                w = edge["weight"]
                                edge_traces.append(go.Scatter(
                                    x=[x0, x1, None], y=[y0, y1, None],
                                    mode="lines",
                                    line=dict(width=max(1.5, w * 2.5), color="rgba(150, 150, 150, 0.5)"),
                                    hoverinfo="none",
                                    showlegend=False,
                                ))

                            # Node trace
                            node_x = [positions[n][0] for n in nodes]
                            node_y = [positions[n][1] for n in nodes]
                            node_colors = [
                                ENTITY_CONFIG.get(node_tags.get(n, "MISC"), ENTITY_CONFIG.get("MISC", {})).get("border_color", "#9333EA")
                                for n in nodes
                            ]
                            node_trace = go.Scatter(
                                x=node_x, y=node_y,
                                mode="markers+text",
                                text=nodes,
                                textposition="top center",
                                textfont=dict(size=11, family="Mukta, Noto Sans Devanagari, sans-serif"),
                                marker=dict(
                                    size=22,
                                    color=node_colors,
                                    line=dict(width=2, color="white"),
                                ),
                                hovertext=[
                                    f"{n}<br>Tag: {node_tags.get(n, '?')}"
                                    for n in nodes
                                ],
                                hoverinfo="text",
                                showlegend=False,
                            )

                            fig_net = go.Figure(data=edge_traces + [node_trace])
                            fig_net.update_layout(
                                margin=dict(t=10, b=10, l=10, r=10),
                                height=420,
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig_net, use_container_width=True)
                        else:
                            st.info("Plotly आवश्यक आहे — हीटमॅप खाली उपलब्ध आहे.")

                    # Co-occurrence heatmap matrix
                    with col_matrix:
                        st.markdown("###### :material/grid_on: सह-उपस्थिती मॅट्रिक्स (Co-occurrence Matrix)")
                        if HAS_PLOTLY and len(nodes) >= 2:
                            fig_co = px.imshow(
                                cooccur["matrix"],
                                x=nodes, y=nodes,
                                text_auto=True,
                                color_continuous_scale="YlOrRd",
                                aspect="auto",
                            )
                            fig_co.update_layout(
                                margin=dict(t=10, b=20, l=20, r=10),
                                height=420,
                                coloraxis_showscale=False,
                            )
                            st.plotly_chart(fig_co, use_container_width=True)
                        else:
                            df_co = pd.DataFrame(cooccur["matrix"], index=nodes, columns=nodes)
                            st.dataframe(df_co, use_container_width=True)

                    # Edge list table
                    if edges:
                        st.markdown("###### :material/link: सह-उपस्थिती जोड्या (Co-occurrence Pairs)")
                        edge_rows = [{
                            "घटक १ (Entity 1)": e["source"],
                            "घटक २ (Entity 2)": e["target"],
                            "सह-उपस्थिती (Co-occurrences)": e["weight"],
                        } for e in edges]
                        st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)

                elif len(set(e.get("canonical_entity", "") for e in filtered)) < 2:
                    st.info("सह-उपस्थिती जाळ्यासाठी किमान २ वेगवेगळे घटक आवश्यक आहेत.")
                else:
                    st.info("या मजकुरात एकाच वाक्यात एकत्र आढळणारे घटक नाहीत.")
            else:
                st.info("सह-उपस्थिती विश्लेषणासाठी किमान २ घटक आवश्यक आहेत.")

        # ── Tab 6: Wikidata Entity Linking ──
        with tab_linking:
            if filtered:
                st.markdown("##### :material/link: विकिडेटा घटक लिंकिंग (Wikidata Entity Linking)")
                st.caption(
                    "प्रत्येक ओळखलेल्या घटकाला Wikidata QID आणि Wikipedia लेखाशी जोडले जाते. "
                    "प्रथम मराठी विकिपीडियात शोधले जाते, नंतर इंग्रजीत."
                )

                with st.spinner("Wikidata API शी जोडत आहे (Querying Wikidata)…"):
                    linked_entities = lookup_wikidata_entities(filtered)

                if linked_entities:
                    # Summary metrics
                    linked_count = sum(1 for le in linked_entities if le["wikidata_id"])
                    total_count = len(linked_entities)

                    lm1, lm2, lm3 = st.columns(3)
                    lm1.metric("एकूण अनन्य घटक (Unique Entities)", total_count, icon=":material/fingerprint:")
                    lm2.metric("यशस्वी लिंक (Linked)", linked_count, icon=":material/check_circle:")
                    lm3.metric("लिंकिंग दर (Link Rate)", f"{(linked_count / total_count * 100) if total_count else 0:.0f}%", icon=":material/percent:")

                    st.write("")

                    # Entity cards
                    for le in linked_entities:
                        tag = le.get("tag", "MISC")
                        cfg = ENTITY_CONFIG.get(tag, ENTITY_CONFIG.get("MISC", {}))
                        b_col = cfg.get("border_color", "#9333EA")
                        bg_c = cfg.get("bg_color", "rgba(147, 51, 234, 0.16)")
                        icon = cfg.get("icon", "")
                        m_lbl = cfg.get("marathi", tag)

                        qid = le.get("wikidata_id", "")
                        desc = le.get("wikidata_description", "") or "—"
                        wiki_url = le.get("wikipedia_url", "")
                        wd_url = le.get("wikidata_url", "")
                        status = le.get("match_status", "")

                        # Build links
                        links_html = ""
                        if wiki_url:
                            links_html += f'<a href="{wiki_url}" target="_blank" style="color:{b_col}; text-decoration:none; font-size:0.82rem; margin-right:12px;">📖 Wikipedia</a>'
                        if wd_url:
                            links_html += f'<a href="{wd_url}" target="_blank" style="color:{b_col}; text-decoration:none; font-size:0.82rem;">🔗 Wikidata ({qid})</a>'

                        card_html = (
                            f'<div style="background:{bg_c}; border:1px solid {b_col}; border-radius:10px; '
                            f'padding:12px 16px; margin-bottom:8px;">'
                            f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
                            f'<div style="display:flex; align-items:center; gap:8px;">'
                            f'<span>{icon}</span>'
                            f'<strong style="font-size:1.05rem;">{le["canonical_entity"]}</strong>'
                            f'<span style="background:{cfg.get("badge_bg", b_col)}; color:#FFF; font-size:0.72rem; '
                            f'font-weight:700; padding:2px 7px; border-radius:6px;">{m_lbl}</span>'
                            f'</div>'
                            f'<span style="font-size:0.8rem;">{status}</span>'
                            f'</div>'
                            f'<div style="font-size:0.88rem; margin-top:6px; opacity:0.9;">{desc}</div>'
                            f'<div style="margin-top:6px;">{links_html}</div>'
                            f'</div>'
                        )
                        st.markdown(card_html, unsafe_allow_html=True)

                    # Summary table
                    with st.expander("तपशीलवार तक्ता (Detailed Linking Table)"):
                        link_rows = []
                        for le in linked_entities:
                            link_rows.append({
                                "घटक (Entity)": le["canonical_entity"],
                                "टॅग (Tag)": le["tag"],
                                "QID": le.get("wikidata_id") or "—",
                                "वर्णन (Description)": le.get("wikidata_description") or "—",
                                "Wikipedia": le.get("wikipedia_url") or "—",
                                "स्थिती (Status)": le["match_status"],
                            })
                        st.dataframe(pd.DataFrame(link_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("लिंकिंगसाठी कोणतेही घटक उपलब्ध नाहीत.")
            else:
                st.info("विकिडेटा लिंकिंगसाठी कोणतेही घटक उपलब्ध नाहीत.")

        # ── Tab 7: Export ──
        with tab_export:
            if filtered:
                export_data = []
                for e in filtered:
                    export_data.append({
                        "surface": e.get("surface", e.get("entity", "")),
                        "canonical_entity": e.get("canonical_entity", ""),
                        "suffix": e.get("suffix", ""),
                        "case_label": e.get("case_label", ""),
                        "tag": e.get("tag", ""),
                        "category": e.get("category", ""),
                        "confidence": e.get("confidence", 0.0),
                        "start": e.get("start", 0),
                        "end": e.get("end", 0),
                    })

                csv_bytes = pd.DataFrame(export_data).to_csv(index=False).encode("utf-8-sig")
                json_bytes = json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8")

                dl1, dl2 = st.columns(2)
                dl1.download_button("Download CSV (Excel UTF-8)", csv_bytes, "mahaner_canonical_entities.csv", "text/csv", icon=":material/download:", use_container_width=True)
                dl2.download_button("Download JSON", json_bytes, "mahaner_canonical_entities.json", "application/json", icon=":material/download:", use_container_width=True)

                with st.expander("डेटा प्रिव्ह्यू (JSON Preview)"):
                    st.json(export_data)
            else:
                st.info("निर्यात करण्यासाठी कोणतेही घटक उपलब्ध नाहीत.")

    elif clicked and not text:
        st.warning("कृपया आधी मराठी मजकूर प्रविष्ट करा.")


# ─── Render Multi-Model Comparison Mode ──────────────────────────────────────────

def render_multi_model_mode(conf_threshold: int):
    st.markdown("#### :material/balance: तिहेरी मॉडेल तुलना (Side-by-Side Multi-Model Comparison)")
    st.caption(
        "तीन नामांकित ट्रान्सफॉर्मर आर्किटेक्चर्स — **L3Cube-MahaBERT**, **AI4Bharat-IndicBERT**, "
        "आणि **XLM-RoBERTa** — यांची एकाच इनपुटवर तुलना करा, लॅटन्सी मोजा आणि मॉडेल सहमती मॅट्रिक्स (Agreement Matrix) तपासा."
    )

    # Preset selection
    default_compare_text = PRESET_OPTIONS["इतिहास: छत्रपती शिवाजी महाराज व रायगड"]
    if "compare_text_input" not in st.session_state:
        st.session_state["compare_text_input"] = default_compare_text

    def _sync_compare_preset():
        chosen = st.session_state.get("compare_preset_select", "")
        if chosen in PRESET_OPTIONS and PRESET_OPTIONS[chosen]:
            st.session_state["compare_text_input"] = PRESET_OPTIONS[chosen]

    preset_choice = st.selectbox(
        "तुलनेसाठी नमुना वाक्य निवडा (Select Comparison Sentence):",
        list(PRESET_OPTIONS.keys()),
        index=1,
        key="compare_preset_select",
        on_change=_sync_compare_preset
    )

    user_text = st.text_area(
        "मराठी मजकूर (Input Marathi Sentence):",
        height=95,
        placeholder="येथे तुलना करण्यासाठी मराठी मजकूर प्रविष्ट करा…",
        key="compare_text_input"
    )

    compare_clicked = st.button(
        "तिन्ही मॉडेल्सची तुलना करा (Compare All 3 Models)",
        icon=":material/rocket_launch:",
        type="primary",
        use_container_width=True,
        key="btn_run_comparison"
    )

    text = user_text.strip()

    if compare_clicked and text:
        models_to_run = [
            "l3cube-pune/marathi-ner",
            "ai4bharat/IndicNER",
            "Davlan/xlm-roberta-base-ner-hrl"
        ]

        model_results = {}
        with st.spinner("तिन्ही मॉडेल्सवर इन्फरन्स सुरू आहे (Running inference across all models)…"):
            for m_id in models_to_run:
                res = extract_entities_with_model(text, m_id)
                # Filter by confidence threshold
                res["entities"] = [e for e in res["entities"] if e["confidence"] >= conf_threshold]
                model_results[m_id] = res

        # Compute consensus and agreement
        comparison = compute_multi_model_comparison(text, model_results)

        st.divider()

        # ── High Level Metrics Banner ──
        fastest_model = min(models_to_run, key=lambda m: model_results[m]["latency_ms"])
        fastest_info = MODEL_REGISTRY.get(fastest_model, {})

        bm1, bm2, bm3, bm4 = st.columns(4)
        bm1.metric("सरासरी मॉडेल सहमती (Avg Agreement)", f"{comparison['avg_agreement']}%", icon=":material/handshake:")
        bm2.metric("पूर्ण एकमत (Unanimous 3/3)", f"{comparison['unanimous_count']} घटक", icon=":material/check_circle:")
        bm3.metric("बहुमत सहमती (Majority 2/3)", f"{comparison['majority_count']} घटक", icon=":material/pie_chart:")
        bm4.metric("वेगवान मॉडेल (Fastest)", f"{fastest_info.get('name', fastest_model)} ({model_results[fastest_model]['latency_ms']} ms)", icon=":material/bolt:")

        st.write("")

        # ── 1. Side-by-Side Model Cards ──
        st.markdown("##### :material/hub: मॉडेल्सचे परिणाम (Side-by-Side Model Cards):")
        cols = st.columns(3)

        for idx, m_id in enumerate(models_to_run):
            res = model_results[m_id]
            m_info = MODEL_REGISTRY.get(m_id, {})
            ents = res.get("entities", [])
            avg_c = (sum(e["confidence"] for e in ents) / len(ents)) if ents else 0.0

            with cols[idx]:
                status_dot = "#16A34A" if res["is_hf"] else "#F59E0B"
                status_text = "Hugging Face Active" if res["is_hf"] else "Offline Fallback Active"
                status_html = (
                    f'<div style="font-size:0.78rem; margin-top:2px; display:flex; align-items:center; gap:5px;">'
                    f'<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{status_dot};"></span>'
                    f'<span>{status_text}</span>'
                    f'</div>'
                )

                # Styled Card Header
                st.markdown(
                    f'<div style="border: 1px solid rgba(128,128,128,0.25); border-radius:12px; padding:12px 14px; background:rgba(128,128,128,0.04); min-height:100px;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<h4 style="margin:0; font-size:1.05rem;">{m_info.get("name", m_id)}</h4>'
                    f'<span style="background:{m_info.get("badge_color", "#4B5563")}; color:#FFFFFF; padding:2px 7px; border-radius:6px; font-size:0.72rem; font-weight:700;">'
                    f'{m_info.get("badge", "Model")}'
                    f'</span>'
                    f'</div>'
                    f'<div style="font-size:0.8rem; opacity:0.8; margin-top:3px;">{m_info.get("architecture", "")}</div>'
                    f'{status_html}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Card metrics
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("वेळ", f"{res['latency_ms']} ms", icon=":material/schedule:")
                mc2.metric("घटक", len(ents), icon=":material/push_pin:")
                mc3.metric("विश्वास", f"{avg_c:.1f}%", icon=":material/verified:")

                st.markdown("###### ओळखलेले घटक (Detected Entities):")
                if ents:
                    card_chip_html = '<div style="display:flex; flex-direction:column; gap:6px;">'
                    for e in ents:
                        cfg = ENTITY_CONFIG.get(e.get("tag", "MISC"), ENTITY_CONFIG.get("MISC", {}))
                        b_col = cfg.get("border_color", "#9333EA")
                        bg_c = cfg.get("bg_color", "rgba(147, 51, 234, 0.16)")
                        icon = cfg.get("icon", "")
                        m_lbl = cfg.get("marathi", e.get("tag", ""))

                        surf = e.get("surface", e.get("entity", ""))
                        can = e.get("canonical_entity", surf)
                        suff = e.get("suffix", "")

                        if suff and can != surf:
                            ent_desc = f"<strong>{surf}</strong> ➔ {can} (+{suff})"
                        else:
                            ent_desc = f"<strong>{surf}</strong>"

                        card_chip_html += (
                            f'<div style="background:{bg_c}; border-left:3px solid {b_col}; padding:5px 9px; border-radius:4px; font-size:0.86rem; display:flex; justify-content:space-between; align-items:center;">'
                            f'<span style="display:flex; align-items:center; gap:6px;">{icon} <span>{ent_desc}</span></span>'
                            f'<span style="background:{b_col}; color:#FFF; font-size:0.7rem; font-weight:700; padding:1px 5px; border-radius:3px;">{m_lbl} · {e["confidence"]:.0f}%</span>'
                            f'</div>'
                        )
                    card_chip_html += '</div>'
                    st.markdown(card_chip_html, unsafe_allow_html=True)
                else:
                    st.info("या मॉडेलला कोणतेही घटक आढळले नाहीत.")

        st.divider()

        # ── 2. Agreement Matrix & Latency Chart ──
        st.markdown("##### :material/bar_chart: मॉडेल सहमती मॅट्रिक्स आणि कार्यक्षमता विश्लेषण (Model Agreement & Latency Analytics):")
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.markdown("###### :material/handshake: मॉडेल सहमती मॅट्रिक्स (Pairwise Agreement Matrix %):")
            st.caption("Jaccard Overlap Score (%) घटक व टॅग साम्यावर आधारित.")

            model_labels = [MODEL_REGISTRY.get(m, {}).get("name", m) for m in models_to_run]
            matrix_data = []
            for m1 in models_to_run:
                row = []
                for m2 in models_to_run:
                    score = comparison["agreement_matrix"][m1][m2]
                    row.append(score)
                matrix_data.append(row)

            if HAS_PLOTLY:
                fig_hm = px.imshow(
                    matrix_data,
                    x=model_labels,
                    y=model_labels,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    aspect="auto"
                )
                fig_hm.update_layout(
                    margin=dict(t=10, b=20, l=20, r=20),
                    height=280,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                df_matrix = pd.DataFrame(matrix_data, index=model_labels, columns=model_labels)
                st.dataframe(df_matrix.style.format("{:.1f}%"), use_container_width=True)

        with c_right:
            st.markdown("###### :material/timer: इन्फरन्स लॅटन्सी तुलना (Inference Latency in ms):")
            st.caption("कमी वेळ = जलद प्रतिसाद (Lower latency is faster).")

            latency_rows = []
            for m_id in models_to_run:
                latency_rows.append({
                    "Model": MODEL_REGISTRY.get(m_id, {}).get("name", m_id),
                    "Latency (ms)": model_results[m_id]["latency_ms"],
                    "Entities": len(model_results[m_id]["entities"])
                })
            df_perf = pd.DataFrame(latency_rows)

            if HAS_PLOTLY:
                fig_lat = px.bar(
                    df_perf, x="Model", y="Latency (ms)",
                    color="Model",
                    color_discrete_sequence=["#DC2626", "#0284C7", "#16A34A"],
                    text="Latency (ms)"
                )
                fig_lat.update_layout(
                    showlegend=False,
                    margin=dict(t=10, b=20, l=20, r=20),
                    height=280,
                    xaxis_title="", yaxis_title="लॅटन्सी (ms)"
                )
                fig_lat.update_traces(textposition="outside")
                st.plotly_chart(fig_lat, use_container_width=True)
            else:
                st.bar_chart(df_perf.set_index("Model")["Latency (ms)"])

        st.divider()

        # ── 3. Unified Entity Consensus Matrix Table ──
        st.markdown("##### :material/table_chart: सर्वसमावेशक घटक सहमती तक्ता (Unified Entity Consensus Matrix):")
        st.caption("प्रत्येक घटकावर प्रत्येक मॉडेलचे मत आणि एकूण सहमती पातळी (Consensus Level).")

        consensus_rows = []
        for c in comparison["consensus_entities"]:
            row = {
                "मूळ घटक (Canonical Entity)": f"[{c['tag']}] {c['canonical_entity']}",
                "मूळ मजकूर (Surface)": c["surface"],
                "प्रत्यय (Suffix)": f"+{c['suffix']}" if c["suffix"] else "—",
                "सहमती टॅग (Tag)": c["tag"],
                "सहमती पातळी (Consensus)": c["consensus_badge"],
            }
            # Fill vote for each model
            for m_id in models_to_run:
                m_name = MODEL_REGISTRY.get(m_id, {}).get("name", m_id)
                if m_id in c["votes"]:
                    vote = c["votes"][m_id]
                    row[m_name] = f"✓ {vote['tag']} ({vote['confidence']:.0f}%)"
                else:
                    row[m_name] = "—"
            consensus_rows.append(row)

        if consensus_rows:
            st.dataframe(pd.DataFrame(consensus_rows), use_container_width=True, hide_index=True)
        else:
            st.info("कोणत्याही मॉडेलला घटक आढळले नाहीत.")

    elif compare_clicked and not text:
        st.warning("कृपया आधी तुलना करण्यासाठी मराठी मजकूर प्रविष्ट करा.")


# ─── Main Orchestrator ──────────────────────────────────────────────────────────

def main():
    # Sidebar
    selected_model, conf_threshold, show_canonical_inline = render_sidebar()

    # Title
    st.title("MahaNER", icon=":material/flag:")
    st.markdown("**मराठी नामनिर्देशित घटक ओळख व विभक्ती पृथक्करण प्रणाली** — Advanced Marathi Named Entity Recognition")

    # Main operational mode tabs
    tab_single, tab_compare = st.tabs([
        ":material/target: एकल मॉडेल विश्लेषण (Single Model Analysis)",
        ":material/balance: तिहेरी मॉडेल तुलना (Multi-Model Comparison)"
    ])

    with tab_single:
        render_single_model_mode(selected_model, conf_threshold, show_canonical_inline)

    with tab_compare:
        render_multi_model_mode(conf_threshold)


if __name__ == "__main__":
    main()
