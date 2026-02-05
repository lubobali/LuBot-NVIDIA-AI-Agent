# LuBot Data Schema (Sample)

LuBot supports two data modes: **My Data** (file uploads) and **LuBot Data** (live connections).

---

## Live Website Analytics

**click_logs** — Raw traffic events
- `timestamp` — When the visit happened
- `page_name` — Which page was viewed
- `referrer` — Traffic source (Google, LinkedIn, direct, etc.)
- `session_id` — Unique visitor session
- `time_on_page` — How long they stayed (seconds)
- `user_agent` — Device/browser info

**daily_click_summary** — Aggregated daily metrics
- `date` — The day
- `project_name` — Which project/page
- `total_clicks` — Total visits that day
- `avg_time_on_page` — Average time spent
- `device_split` — Mobile vs Desktop breakdown (JSON)
- `top_referrers` — Traffic sources ranking (JSON)
- `repeat_visits` — Returning visitors count

---

## User Data Storage

**user_uploads** — Uploaded files metadata
- `filename` — Original file name
- `file_type` — CSV, Excel, etc.
- `row_count` — Number of rows
- `columns` — Column names (JSON)
- `uploaded_at` — Upload timestamp

---

## Full Schema

The complete LuBot database includes **34 tables** covering:
- Core infrastructure
- User data storage
- Memory & context
- Learning & preferences
- Analytics & tracking
- Profiles & segmentation

See the full architecture at [lubot.ai/architecture](https://lubot.ai/architecture.html).
