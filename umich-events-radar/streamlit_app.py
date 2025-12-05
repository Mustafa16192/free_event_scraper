# =========================================================
# streamlit_app.py — Full Standalone UMich Events Radar
# =========================================================

import os
import re
import json
import requests
import pandas as pd
from datetime import datetime, date
from html import unescape
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from litellm import completion
import streamlit as st
from google_auth_oauthlib.flow import Flow

# Load environment variables
load_dotenv()

# UMich events endpoint (weekly)
EVENTS_URL = "https://events.umich.edu/week/json?v=2"

DEFAULT_MODEL = "gpt-4o-mini"
TAG_RE = re.compile(r"<[^>]+>")


# =========================================================
# 1. Fetch events
# =========================================================
def clean_text(value):
    if not value:
        return ""
    cleaned = TAG_RE.sub(" ", unescape(str(value)))
    return " ".join(cleaned.split())


def truncate(text, limit=1200):
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


@st.cache_data(show_spinner=False, ttl=900)
def _fetch_umich_events_cached():
    resp = requests.get(EVENTS_URL, timeout=12)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list):
        events = data
    elif isinstance(data, dict) and "events" in data:
        events = data["events"]
    else:
        raise ValueError(f"Unexpected UMich events JSON: {type(data)}")

    fetched_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return events, fetched_at


def fetch_umich_events():
    try:
        events, fetched_at = _fetch_umich_events_cached()
        return events, fetched_at, None
    except Exception as exc:
        return [], None, str(exc)


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


def day_of_week(date_str):
    """Return short day-of-week label (e.g., Mon) from YYYY-MM-DD."""
    if not date_str:
        return ""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%a")
    except Exception:
        return ""


# =========================================================
# 3. LLM helper
# =========================================================
def ensure_llm_ready():
    expected_keys = ["OPENAI_API_KEY", "LITELLM_API_KEY"]
    missing = [k for k in expected_keys if not os.getenv(k)]
    if len(missing) == len(expected_keys):
        st.error("Missing API key. Set OPENAI_API_KEY or LITELLM_API_KEY.")
        return False
    return True


@st.cache_resource(show_spinner=False)
def get_completion_client():
    return completion


def run_completion(model, messages, response_format=None, **kwargs):
    try:
        completion_kwargs = kwargs.copy()
        if response_format:
            completion_kwargs["response_format"] = response_format
        client = get_completion_client()
        response = client(
            model=model,
            messages=messages,
            **completion_kwargs
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
    desc = truncate(clean_text(event.get("description", "")))
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

    content = run_completion(
        model,
        messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
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
    worker_count = max(1, min(max_workers, os.cpu_count() * 4 if os.cpu_count() else max_workers, len(events) or 1))

    with ThreadPoolExecutor(max_workers=worker_count) as ex:
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
                "day": day_of_week(ev.get("date_start", "")),
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
    desc = truncate(clean_text(event.get("description", "")))
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

    content = run_completion(
        model,
        messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
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
    worker_count = max(1, min(max_workers, os.cpu_count() * 4 if os.cpu_count() else max_workers, len(events) or 1))

    with ThreadPoolExecutor(max_workers=worker_count) as ex:
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
                "day": day_of_week(ev.get("date_start", "")),
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


def load_local_events(uploaded_file=None, path=None):
    if uploaded_file:
        try:
            return json.load(uploaded_file), "uploaded file", None
        except Exception as exc:
            return [], None, f"Unable to read uploaded JSON: {exc}"
    if path:
        try:
            with open(path, "r") as f:
                return json.load(f), path, None
        except FileNotFoundError:
            return [], None, f"Local path not found: {path}"
        except Exception as exc:
            return [], None, f"Unable to read local JSON: {exc}"
    return [], None, "No local file provided."


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="UMich Events Radar", page_icon="🔍", layout="wide")

def check_authentication():
    """
    Enforce Google OAuth and restrict to @umich.edu emails.
    Returns True if authenticated, False (stops execution) otherwise.
    """
    # 1. Check if already authenticated
    if st.session_state.get("authenticated"):
        return True

    # 2. Load credentials
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

    # Handle multiple redirect URLs (e.g. "http://localhost:8501,https://myapp.streamlit.app")
    redirect_urls_str = os.getenv("GOOGLE_REDIRECT_URL") or os.getenv("GOOGLE_REDIRECT_URI")
    redirect_uri = "http://localhost:8501"  # Default fallback

    if redirect_urls_str:
        urls = [u.strip() for u in redirect_urls_str.split(',')]
        
        # Heuristic to detect if running on Streamlit Cloud
        is_cloud = os.getenv("STREAMLIT_SHARING_MODE") is not None

        localhost_url = next((u for u in urls if "localhost" in u or "127.0.0.1" in u), None)
        remote_url = next((u for u in urls if "localhost" not in u and "127.0.0.1" not in u), None)

        if is_cloud and remote_url:
            redirect_uri = remote_url
        elif localhost_url:
            redirect_uri = localhost_url
        elif remote_url:
            redirect_uri = remote_url
        elif urls:
            redirect_uri = urls[0]

    if not client_id or not client_secret:
        st.warning("⚠️ Google OAuth not configured.")
        st.info("To enable auth, set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in `.env`.")
        # Allow bypass if explicitly allowed (e.g. local dev without auth)
        if os.getenv("ALLOW_UNAUTHENTICATED", "false").lower() == "true":
            return True
        st.stop()

    # 3. Setup OAuth Flow
    try:
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=[
                "https://www.googleapis.com/auth/userinfo.email",
                "openid"
            ],
            redirect_uri=redirect_uri,
        )
    except Exception as e:
        st.error(f"OAuth configuration error: {e}")
        st.stop()

    # 4. Handle Auth Code
    code = st.query_params.get("code")
    if code:
        try:
            flow.fetch_token(code=code)
            credentials = flow.credentials
            
            user_info = requests.get(
                "https://www.googleapis.com/oauth2/v1/userinfo",
                headers={"Authorization": f"Bearer {credentials.token}"}
            ).json()
            
            email = user_info.get("email", "")
            if email.endswith("@umich.edu"):
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = email
                st.query_params.clear()
                st.rerun()
            else:
                st.error("🚫 Access denied. Please sign in with a **@umich.edu** email.")
                if st.button("Try Again"):
                    st.query_params.clear()
                    st.rerun()
                st.stop()
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            st.stop()

    # 5. Show Login Button
    else:
        auth_url, _ = flow.authorization_url(prompt="consent")
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.title("UMich Events Radar 🔒")
            st.write("### Please sign in to continue")
            st.info("Access is restricted to **@umich.edu** accounts.")
            st.link_button("Sign in with Google", auth_url, type="primary", use_container_width=True)
        st.stop()

# Run Auth Check
check_authentication()

# Show User Info in Sidebar
if st.session_state.get("authenticated"):
    st.sidebar.write(f"👤 **{st.session_state.get('user_email')}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

st.title("UMich Events Radar 🔍")
st.caption("Free events + professionally helpful events for PM/Tech career growth.")

st.sidebar.header("Settings")

max_workers = st.sidebar.slider(
    "Max concurrent LLM workers",
    5,
    80,
    st.session_state.get("max_workers_slider", 20),
    step=5,
    key="max_workers_slider",
)
temperature = st.sidebar.slider(
    "Model temperature",
    0.0,
    1.0,
    st.session_state.get("temperature_slider", 0.0),
    step=0.1,
    key="temperature_slider",
)

show_free = st.sidebar.checkbox("Show free/free-ish events", value=True, key="show_free")
show_prof = st.sidebar.checkbox("Show professionally helpful events", value=True, key="show_prof")

enable_llm = st.sidebar.checkbox(
    "Enable LLM calls (required for classifications)",
    value=not st.session_state.get("use_local_only", False),
    key="enable_llm",
)

st.sidebar.markdown("### Data source")
use_local = st.sidebar.checkbox("Use local JSON (offline/demo)", value=False, key="use_local_only")
local_path = st.sidebar.text_input(
    "Local JSON path",
    value="free_events_state.json",
    disabled=not use_local,
)
uploaded_file = st.sidebar.file_uploader(
    "Or upload events JSON",
    type=["json"],
    disabled=not use_local,
)

run_button = st.sidebar.button("Run Scan", type="primary")

st.sidebar.markdown("---")
st.sidebar.write("Today:", date.today().isoformat())


if run_button:
    with st.spinner("Fetching UMich events..."):
        fetched_at = None
        fetch_error = None
        data_source = "UMich API"

        if use_local:
            events_raw, data_source, fetch_error = load_local_events(uploaded_file, local_path)
            fetched_at = datetime.utcnow().isoformat(timespec="seconds") + "Z" if not fetch_error else None
        else:
            events_raw, fetched_at, fetch_error = fetch_umich_events()

    if fetch_error:
        st.error(fetch_error)
        st.stop()

    if not events_raw:
        st.warning(f"No events returned from {data_source}.")
        st.stop()

    events = filter_future_events(events_raw)

    st.write(f"Fetched **{len(events_raw)}** events from **{data_source}**.")
    st.write(f"Filtered to **{len(events)}** future events.")
    if fetched_at:
        st.caption(f"Last updated: {fetched_at}")

    if not enable_llm:
        st.info("LLM disabled; showing future events only. Enable LLM to classify free/professional events.")
        st.dataframe(pd.DataFrame(events), use_container_width=True)
        st.stop()

    if not ensure_llm_ready():
        st.stop()

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
                if not df_free.empty:
                    st.download_button(
                        "Download free events CSV",
                        df_free.to_csv(index=False).encode("utf-8"),
                        "free_events.csv",
                        "text/csv",
                    )

                st.markdown("### Breakdown by category")
                c1, c2, c3 = st.columns(3)
                c1.metric("Free Food", len(buckets["free_food"]))
                c2.metric("Free Snacks", len(buckets["free_snacks"]))
                c3.metric("Other Free Stuff", len(buckets["other_free"]))

                for cat, rows in buckets.items():
                    if not rows:
                        continue
                    with st.expander(f"{cat.replace('_', ' ').title()} ({len(rows)})"):
                        for r in rows:
                            md = (
                                f"- **{r['title']}**  \n"
                                f"  {r['day']} — {r['date_start']} {r['time_start']} {r['time_zone']}  \n"
                                f"  {r['location_name']}  \n"
                                f"  *Free item:* {r['free_item']} — {r['free_details']}  \n"
                                f"  {r['organizers']}  \n"
                                f"  [Event Link]({r['permalink']})"
                            )
                            st.markdown(md)

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
                if not df_prof.empty:
                    st.download_button(
                        "Download professional events CSV",
                        df_prof.to_csv(index=False).encode("utf-8"),
                        "professional_events.csv",
                        "text/csv",
                    )

                st.markdown("### Grouped by Category")
                for cat, rows in categories.items():
                    with st.expander(f"{cat.upper()} ({len(rows)})"):
                        for r in rows:
                            md = (
                                f"- **{r['title']}**  \n"
                                f"  {r['day']} — {r['date_start']} {r['time_start']}  \n"
                                f"  {r['location']}  \n"
                                f"  *Why helpful:* {r['reason']}  \n"
                                f"  [Event Link]({r['link']})"
                            )
                            st.markdown(md)

else:
    st.info("Configure settings and click **Run Scan** to start.")
