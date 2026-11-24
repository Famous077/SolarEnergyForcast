"""Small HTML building blocks: KPI tiles, alert cards, status pills."""
from html import escape
from pathlib import Path

import streamlit as st

SEV_STYLE = {"High": ("sev-high", "▲ HIGH"), "Medium": ("sev-med", "● MEDIUM"), "Caution": ("sev-caution", "◆ CAUTION")}


def load_css():
    css = (Path(__file__).resolve().parents[1] / "assets" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def tile(label: str, value: str, note: str = "", tone: str = ""):
    st.markdown(f"""<div class="tile {tone}"><div class="tile-label">{escape(label)}</div>
<div class="tile-value">{escape(value)}</div><div class="tile-note">{escape(note)}</div></div>""",
                unsafe_allow_html=True)


def alert_card(c: dict, compact: bool = False):
    cls, tag = SEV_STYLE[c["severity"]]
    body = f"<div class='ac-impact'>{escape(c['impact'])}</div>"
    if not compact:
        body += (f"<div class='ac-row'><span>Why</span>{escape(c['why'])}</div>"
                 f"<div class='ac-row'><span>Check</span>{escape(c['check'])}</div>")
        if c["record"]:
            body += f"<div class='ac-row'><span>Track record</span>{escape(c['record'])}</div>"
    st.markdown(f"""<div class="ac {cls}"><div class="ac-head"><b class="ac-tag">{tag}</b>
<span class="ac-title">{escape(c['title'])}</span><span class="ac-when">{escape(c['when'])}</span></div>{body}</div>""",
                unsafe_allow_html=True)


def pills(items: list[tuple[str, str]]):
    """Row of status pills. tone: ok | warn | bad | info"""
    html = "".join(f"<span class='pill pill-{tone}'>{escape(text)}</span>" for text, tone in items)
    st.markdown(f"<div style='margin-top:14px'>{html}</div>", unsafe_allow_html=True)


def note(text: str):
    st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)
