# Squadmetric

Evidence-led squad and transfer analytics for Fantasy Premier League (FPL).
Enter your FPL team ID and Squadmetric rates your squad and suggests a
lineup, captain and transfers. It also covers transfer plans, player news and a
model-built squad. It uses a points model trained on past seasons, blended
with current form and FPL's availability data. Optional AI explanations
describe each suggestion in plain English, and every number in them is
checked against the model's facts.

## Run it locally

Requires Python 3.13 (the tested version).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The repo ships with a snapshot of the data the app needs, so it starts
straight away. To update the snapshot to the current gameweek:

```bash
python refresh.py
```

Other options:

- `FPL_OFFLINE=1 streamlit run app.py` runs on the cached data only and never
  uses the network.
- Run the tests with `pip install -r requirements-dev.txt`, then
  `pytest -c form_lab/pytest.ini`.
- `python -m form_lab.app_check` checks every cached team with the network
  blocked.

## Bring your own OpenRouter key

AI explanations and news ratings are optional. To turn them on, paste an
[OpenRouter API key](https://openrouter.ai/keys) into **OpenRouter API key
(optional)** in the sidebar.

- The key is kept only in your browser session (Streamlit session state) and
  is passed straight to the OpenRouter request. It is never written to disk,
  to logs or to caches, and it is masked in error messages.
- With no key, the app makes no AI calls. It shows cached or template text,
  and each item is labelled as such.
- If OpenRouter rejects the key, the app tells you so and falls back to
  cached or template text.
- Each session makes at most 200 AI calls.
- The app ignores `.env` files and environment variables. Local scripts such
  as `warmup.py` and `pregenerate.py` can read `OPENROUTER_API_KEY` if you
  also set `ALLOW_ENV_KEY=1`. Never commit a `.env` file; it is gitignored.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub. It can be private.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
   and click **Create app**.
3. Choose the repository and the `main` branch, and set the main file path to
   `app.py`.
4. Under **Advanced settings**, choose Python 3.13. You don't need any secrets,
   because visitors bring their own keys.
5. Click **Deploy**. The app installs `requirements.txt` and starts from the
   committed data snapshot.

The snapshot goes stale as the season moves on. To update it, run
`python refresh.py` locally, then commit and push the updated `data/cache/`
and `form_lab/cache/`. The app redeploys automatically.

## Data sources

- **Fantasy Premier League public API** (`fantasy.premierleague.com/api`):
  players, prices, fixtures, teams and gameweek histories. The app fetches
  these on demand and falls back to the committed cache.
- **Historical seasons** from
  [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League):
  these are used to train the points model and for backtests. They are not
  committed. If you want to retrain the model (`refresh.py`) or run the
  season backtests, put the `merged_gw_*.csv` and `players_raw_*.csv` files in
  `data/`.
- **Google News RSS**: recent headlines for the news ratings.
- **OpenRouter**: the AI explanations and news ratings, using your own key.

## Player images

Player photos, club badges and shirts belong to the Premier League and are
not part of this repository. The app fetches them from the Premier League's
servers at runtime and caches them locally in `data/cache/img/`, which is
gitignored. If an image can't be fetched, a neutral placeholder is shown.

Squadmetric is an independent project. It is not affiliated with or endorsed
by the Premier League or Fantasy Premier League.
