# =========================================================
# streamlit_app.py — Full Standalone UMich Events Radar
# =========================================================

import os
import json
import requests
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from litellm import completion
import streamlit as st

# Load environment variables
load_dotenv()

# UMich events endpoint (weekly)
EVENTS_URL = "https://events.umich.edu/week/json?v=2"

DEFAULT_MODEL = "gpt-4o-mini"


# =========================================================
# 1. Fetch events
# =========================================================
def fetch_umich_events():
    resp = requests.get(EVENTS_URL, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "events" in data:
        return data["events"]

    print("Unexpected UMich events JSON:", type(data))
    return []


# =========================================================
# 2. Filter to today + future
# =========================================================
def filter_future_events(events, from_date=None):
    if from_date is None:
        from_date = date.today()

    def is_future_event(ev):
        d = ev.get("date_start")
        if not d:
            return False
        try:
            event_date = datetime.strptime(d, "%Y-%m-%d").date()
            return event_date >= from_date
        except Exception:
            return False

    return [ev for ev in events if is_future_event(ev)]


# =========================================================
# 3. LLM helper
# =========================================================
def run_completion(model, messages, **kwargs):
    try:
        response = completion(
            model=model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    except Exception as e:
        print("LLM error:", e)
        return None


# =========================================================
# 4. LLM: Free Stuff Classifier
# =========================================================
def analyze_event_with_llm(event, model=DEFAULT_MODEL, **kwargs):
    title = event.get("combined_title") or event.get("event_title") or ""
    desc = event.get("description", "") or ""
    location = event.get("location_name") or event.get("building_name") or ""
    cost = event.get("cost", "") or ""
    tags = " ".join(event.get("tags", []))

    event_text = (
        f"Title: {title}\n"
        f"Location: {location}\n"
        f"Cost: {cost}\n"
        f"Tags: {tags}\n"
        f"Description: {desc}"
    )

    system_prompt = (
        "You are analyzing a university event to detect free items. "
        "Classification:\n"
        "- free_food → pizza, dinner, lunch, breakfast\n"
        "- free_snacks → cookies, donuts, cider, coffee, tea\n"
        "- other_free → free entry, free merch, free exhibit\n"
        "- none → no free offering.\n\n"
        "Return ONLY JSON:\n"
        "{\n"
        "  \"label\": \"free_food\" | \"free_snacks\" | \"other_free\" | \"none\",\n"
        "  \"free_item\": \"short phrase\",\n"
        "  \"free_details\": \"1–2 sentence explanation\"\n"
        "}"
    )

    user_prompt = f"Analyze this event:\n\n{event_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",  "content": user_prompt},
    ]

    content = run_completion(model, messages, **kwargs)
    if not content:
        return {"label": "none", "free_item": "", "free_details": ""}

    try:
        parsed = json.loads(content)
        label = parsed.get("label", "none")
        if label not in {"free_food", "free_snacks", "other_free", "none"}:
            label = "none"
        return {
            "label": label,
            "free_item": parsed.get("free_item", "").strip(),
            "free_details": parsed.get("free_details", "").strip(),
        }
    except Exception:
        text = content.lower()
        if "free_food" in text:
            label = "free_food"
        elif "free_snacks" in text:
            label = "free_snacks"
        elif "other_free" in text:
            label = "other_free"
        else:
            label = "none"
        return {"label": label, "free_item": "", "free_details": content}


# =========================================================
# 5. Concurrent free-stuff scanning
# =========================================================
def process_events_concurrent(events, model=DEFAULT_MODEL, max_workers=20, **kwargs):
    enriched = []
    futures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, ev in enumerate(events):
            fut = ex.submit(analyze_event_with_llm, ev, model, **kwargs)
            futures[fut] = idx

        for fut in as_completed(futures):
            ev = events[futures[fut]]
            try:
                result = fut.result()
            except Exception:
                result = {"label": "none", "free_item": "", "free_details": ""}

            if result["label"] == "none":
                continue

            sponsors = "; ".join(
                s.get("group_name", "") for s in (ev.get("sponsors") or []) if s.get("group_name")
            )

            enriched.append({
                "id": ev.get("id", ""),
                "label": result["label"],
                "title": ev.get("combined_title") or ev.get("event_title") or "",
                "free_item": result["free_item"],
                "free_details": result["free_details"],
                "date_start": ev.get("date_start", ""),
                "time_start": ev.get("time_start", ""),
                "time_zone": ev.get("time_zone", ""),
                "location_name": ev.get("location_name", "") or ev.get("building_name", ""),
                "organizers": sponsors,
                "permalink": ev.get("permalink", ""),
            })

    buckets = {"free_food": [], "free_snacks": [], "other_free": []}
    for row in enriched:
        buckets[row["label"]].append(row)

    return enriched, buckets


# =========================================================
# 6. Professional helpfulness classifier
# =========================================================
def analyze_professional_relevance(event, model=DEFAULT_MODEL, **kwargs):
    title = event.get("combined_title") or event.get("event_title") or ""
    desc = event.get("description", "") or ""
    tags = " ".join(event.get("tags", []))
    orgs = "; ".join(
        o.get("group_name", "")
        for o in (event.get("sponsors") or [])
        if o.get("group_name")
    )

    text = (
        f"Title: {title}\n"
        f"Description: {desc}\n"
        f"Tags: {tags}\n"
        f"Organizers: {orgs}\n"
    )

    system_prompt = (
        "Evaluate if the event is professionally helpful to a PM/tech/AI/design/"
        "entrepreneurship-oriented graduate student.\n\n"
        "Return ONLY JSON:\n"
        "{\n"
        "  \"is_helpful\": true/false,\n"
        "  \"category\": \"career\" | \"networking\" | \"PM/tech\" | \"research\" | \"academic\" | \"leadership\" | \"entrepreneurship\" | \"general\",\n"
        "  \"reason\": \"short explanation\"\n"
        "}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Evaluate this event:\n\n{text}"},
    ]

    content = run_completion(model, messages, **kwargs)
    if not content:
        return {"is_helpful": False, "category": "general", "reason": ""}

    try:
        parsed = json.loads(content)
        return {
            "is_helpful": parsed.get("is_helpful", False),
            "category": parsed.get("category", "general"),
            "reason": parsed.get("reason", "").strip(),
        }
    except Exception:
        return {"is_helpful": False, "category": "general", "reason": content}


def process_professional_events_concurrent(events, model=DEFAULT_MODEL, max_workers=20, **kwargs):
    helpful = []
    futures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, ev in enumerate(events):
            fut = ex.submit(analyze_professional_relevance, ev, model, **kwargs)
            futures[fut] = idx

        for fut in as_completed(futures):
            ev = events[futures[fut]]
            try:
                result = fut.result()
            except Exception:
                result = {"is_helpful": False, "category": "general", "reason": ""}

            if not result["is_helpful"]:
                continue

            sponsors = "; ".join(
                s.get("group_name", "") for s in (ev.get("sponsors") or []) if s.get("group_name")
            )

            helpful.append({
                "id": ev.get("id", ""),
                "title": ev.get("combined_title") or ev.get("event_title") or "",
                "date_start": ev.get("date_start", ""),
                "time_start": ev.get("time_start", ""),
                "location": ev.get("location_name", "") or ev.get("building_name", ""),
                "organizers": sponsors,
                "category": result["category"],
                "reason": result["reason"],
                "link": ev.get("permalink", ""),
            })

    categories = {}
    for r in helpful:
        categories.setdefault(r["category"], []).append(r)

    return helpful, categories


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="UMich Events Radar", page_icon="🔍", layout="wide")

st.title("UMich Events Radar 🔍")
st.caption("Free events + professionally helpful events for PM/Tech career growth.")

st.sidebar.header("Settings")

max_workers = st.sidebar.slider("Max concurrent LLM workers", 5, 80, 20, step=5)
temperature = st.sidebar.slider("Model temperature", 0.0, 1.0, 0.0, step=0.1)

show_free = st.sidebar.checkbox("Show free/free-ish events", value=True)
show_prof = st.sidebar.checkbox("Show professionally helpful events", value=True)

run_button = st.sidebar.button("Run Scan")

st.sidebar.markdown("---")
st.sidebar.write("Today:", date.today().isoformat())


if run_button:
    with st.spinner("Fetching UMich events..."):
        events_raw = fetch_umich_events()
        events = filter_future_events(events_raw)

    st.write(f"Fetched **{len(events_raw)}** events total.")
    st.write(f"Filtered to **{len(events)}** future events.\n")

    tabs = []
    if show_free:
        tabs.append("🍕 Free Events")
    if show_prof:
        tabs.append("💼 Professional Events")

    if not tabs:
        st.warning("Enable at least one option in the sidebar.")
    else:
        tab_objs = st.tabs(tabs)

        # ---------- FREE EVENTS TAB ----------
        if show_free:
            tab_idx = tabs.index("🍕 Free Events")
            with tab_objs[tab_idx]:
                st.subheader("🍕 Free / Free-ish Events")
                with st.spinner("Scanning for free items..."):
                    enriched, buckets = process_events_concurrent(
                        events,
                        max_workers=max_workers,
                        temperature=temperature,
                    )

                st.write(f"Found **{len(enriched)}** events with something free.")

                df_free = pd.DataFrame(enriched)
                st.dataframe(df_free, use_container_width=True)

                st.markdown("### Breakdown by category")
                c1, c2, c3 = st.columns(3)
                c1.metric("Free Food", len(buckets["free_food"]))
                c2.metric("Free Snacks", len(buckets["free_snacks"]))
                c3.metric("Other Free Stuff", len(buckets["other_free"]))

        # ---------- PROFESSIONAL EVENTS TAB ----------
        if show_prof:
            tab_idx = tabs.index("💼 Professional Events")
            with tab_objs[tab_idx]:
                st.subheader("💼 Professionally Helpful Events")
                with st.spinner("Evaluating professional value..."):
                    helpful, categories = process_professional_events_concurrent(
                        events,
                        max_workers=max_workers,
                        temperature=temperature,
                    )

                st.write(f"Found **{len(helpful)}** professionally helpful events.")

                df_prof = pd.DataFrame(helpful)
                st.dataframe(df_prof, use_container_width=True)

                st.markdown("### Grouped by Category")
                for cat, rows in categories.items():
                    st.markdown(f"**{cat.upper()} ({len(rows)})**")
                    for r in rows:
                        md = (
                            f"- **{r['title']}**  \n"
                            f"  {r['date_start']} {r['time_start']}  \n"
                            f"  {r['location']}  \n"
                            f"  *Why helpful:* {r['reason']}  \n"
                            f"  [Event Link]({r['link']})"
                        )
                        st.markdown(md)

else:
    st.info("Configure settings and click **Run Scan** to start.")
