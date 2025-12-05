# UMich Events Radar 🔍

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://freescraper.streamlit.app)

**Live At:** [https://freescraper.streamlit.app](https://freescraper.streamlit.app)

A standalone Streamlit application that scrapes University of Michigan events and uses Large Language Models (LLMs) to identify:
1.  **🍕 Free Stuff:** Events offering free food, snacks, or other perks.
2.  **💼 Professional Growth:** Events helpful for careers in PM, Tech, AI, Design, and Entrepreneurship.

## Features

-   **Automated Scraping:** Fetches the latest events from the UMich weekly JSON feed.
-   **LLM Classification:** Uses GPT-4o-mini (via `litellm`) to analyze unstructured event descriptions.
-   **Interactive Dashboard:** Filter events by category and view details including time, location, and specific "free" items.
-   **Export:** Download filtered event lists as CSV files.

## Installation

1.  **Clone the repository** (or navigate to the project directory).

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the project root to store your API keys. This project uses `litellm`, so it supports various providers, but is configured for OpenAI by default.
    ```bash
    OPENAI_API_KEY=sk-...
    ```

## Usage

Run the Streamlit application locally:

```bash
streamlit run streamlit_app.py
```

Once running, the app will be available at `http://localhost:8501`.

### Settings
-   **Max Workers:** Adjust the concurrency for LLM scanning in the sidebar.
-   **Model Temperature:** Fine-tune the creativity of the classification model.
-   **Show/Hide Categories:** Toggle between "Free Events" and "Professional Events" tabs.

## Dependencies

-   `streamlit`
-   `pandas`
-   `requests`
-   `python-dotenv`
-   `litellm`
-   `openai`
