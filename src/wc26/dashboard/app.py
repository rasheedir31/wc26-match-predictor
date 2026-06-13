# Streamlit dashboard for the WC26 Match Predictor.
#
# Reads the same bundled snapshot as the API (single data layer). A **model selector**
# at the top drives the whole page, so the live tracker, group fixtures, tournament
# odds, and fixture predictor can be viewed for any of the trained models (Elo,
# Dixon-Coles, logistic, XGBoost); the model-comparison table stays global.

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from wc26.snapshot import SnapshotStore

st.set_page_config(page_title="WC26 Match Predictor", page_icon="⚽", layout="wide")


@st.cache_resource
def _load_store() -> SnapshotStore:
    return SnapshotStore()


def main() -> None:
    st.title("⚽ WC26 Match Predictor")
    st.caption("Predicting 2026 FIFA World Cup outcomes and simulating the tournament.")

    with st.sidebar:
        st.markdown("### Controls")
        if st.button("🔄 Refresh data", help="Reload the snapshot after running `make monitor`."):
            st.cache_resource.clear()
            st.rerun()

    try:
        store = _load_store()
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"No snapshot found ({exc}).\n\n"
            "Build one first:\n\n```\nmake pipeline\nmake snapshot\n```"
        )
        return

    # Pick which model drives the odds / fixtures / live tracker / predictor.
    names = store.model_names()
    model = st.selectbox(
        "Model",
        names,
        index=0,
        format_func=lambda m: f"{m}  ★ champion" if m == store.champion_name else m,
        help="Switch the whole dashboard between the trained models.",
    )

    _headline(store, model)
    st.divider()
    _live_tracker(store, model)
    st.divider()
    _group_fixtures(store, model)
    st.divider()
    _model_comparison(store, model)
    st.divider()
    _tournament(store, model)
    st.divider()
    _fixture_predictor(store, model)


def _headline(store: SnapshotStore, model: str) -> None:
    live = store.live_summary(model)
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected model", f"{model}{' ★' if model == store.champion_name else ''}")
    c2.metric("WC26 matches scored", live["n_matches"])
    if live["n_matches"]:
        c3.metric("Live accuracy", f"{live['accuracy']:.1%}")
    else:
        c3.metric("Live accuracy", "-", help="Populated once WC26 results arrive.")


def _live_tracker(store: SnapshotStore, model: str) -> None:
    st.subheader("Live tracker - predictions vs reality")
    summ = store.live_summary(model)
    c1, c2, c3 = st.columns(3)
    c1.metric("WC26 matches scored", summ["n_matches"])
    c2.metric("Accuracy", f"{summ['accuracy']:.0%}" if summ["n_matches"] else "-")
    c3.metric("Log loss", f"{summ['log_loss']:.3f}" if summ["n_matches"] else "-")

    live = store.live_predictions(model)
    scored = live.dropna(subset=["actual"]) if not live.empty else live
    if scored.empty:
        st.info(
            "No WC26 results in yet - this table fills as matches are played. "
            "Refresh the data and run `make monitor` to update it."
        )
        return

    label = {"H": "home win", "D": "draw", "A": "away win"}
    pred_code = (
        scored[["p_home", "p_draw", "p_away"]]
        .idxmax(axis=1)
        .map({"p_home": "H", "p_draw": "D", "p_away": "A"})
    )
    table = pd.DataFrame(
        {
            "Match": scored["home_team"] + "  vs  " + scored["away_team"],
            "P(home)": scored["p_home"],
            "P(draw)": scored["p_draw"],
            "P(away)": scored["p_away"],
            "Predicted": pred_code.map(label).to_numpy(),
            "Actual": scored["actual"].map(label).to_numpy(),
            "Hit": (pred_code.to_numpy() == scored["actual"].to_numpy()),
        }
    )
    table["Hit"] = table["Hit"].map({True: "✅", False: "❌"})
    st.dataframe(
        table.style.format({"P(home)": "{:.0%}", "P(draw)": "{:.0%}", "P(away)": "{:.0%}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Each played WC26 match vs the pre-tournament prediction. ✅ = the model's top pick was right."
    )


def _group_fixtures(store: SnapshotStore, model: str) -> None:
    st.subheader("Group-stage fixtures - predictions vs results")
    preds = store.group_predictions(model)
    if preds.empty:
        st.info("No group fixtures available.")
        return

    # Index played matches by unordered team pair (carries the real scoreline + outcome).
    played: dict[frozenset, dict] = {}
    live = store.live_predictions(model)
    for r in live.itertuples(index=False):
        played[frozenset((r.home_team, r.away_team))] = {
            "home": r.home_team,
            "hs": getattr(r, "home_score", None),
            "as": getattr(r, "away_score", None),
            "actual": r.actual,
        }

    swap = {"H": "A", "A": "H", "D": "D"}
    pick_label = {"H": "home", "D": "draw", "A": "away"}
    rows = []
    for r in preds.itertuples(index=False):
        probs = {"H": r.p_home, "D": r.p_draw, "A": r.p_away}
        pick = max(probs, key=probs.get)
        rec = played.get(frozenset((r.home_team, r.away_team)))
        if rec and rec["hs"] is not None:
            if rec["home"] == r.home_team:  # same orientation as the fixture
                actual, score = rec["actual"], f"{int(rec['hs'])}-{int(rec['as'])}"
            else:  # fixture lists the teams the other way round
                actual, score = swap[rec["actual"]], f"{int(rec['as'])}-{int(rec['hs'])}"
            status = "✅" if pick == actual else "❌"
        else:
            score, status = "-", "⏳"
        rows.append(
            {
                "Group": r.group,
                "Match": f"{r.home_team} vs {r.away_team}",
                "P(home)": r.p_home,
                "P(draw)": r.p_draw,
                "P(away)": r.p_away,
                "Pick": pick_label[pick],
                "Result": score,
                "✓": status,
            }
        )

    df = pd.DataFrame(rows).sort_values(["Group", "Match"]).reset_index(drop=True)
    choices = ["All groups", *[f"Group {g}" for g in sorted(df["Group"].unique())]]
    sel = st.selectbox("Filter", choices, key="gf_filter")
    if sel != "All groups":
        df = df[df["Group"] == sel.removeprefix("Group ")]
    st.dataframe(
        df.style.format({"P(home)": "{:.0%}", "P(draw)": "{:.0%}", "P(away)": "{:.0%}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "All 72 group-stage matches with the model's prediction; results fill in as games "
        "are played - ✅ correct pick · ❌ wrong · ⏳ not played yet."
    )


def _model_comparison(store: SnapshotStore, model: str) -> None:
    st.subheader("Model comparison (time-based CV)")
    df = store.model_comparison()
    if df.empty:
        st.info("No model comparison available.")
        return
    df = df.assign(selected=df["model"].eq(model))
    show = df[
        ["model", "oof_log_loss", "oof_brier", "oof_rps", "oof_accuracy", "is_champion", "selected"]
    ]
    show = show.rename(
        columns={
            "oof_log_loss": "log loss",
            "oof_brier": "Brier",
            "oof_rps": "RPS",
            "oof_accuracy": "accuracy",
            "is_champion": "champion",
        }
    )
    st.dataframe(
        show.style.format(
            {"log loss": "{:.4f}", "Brier": "{:.4f}", "RPS": "{:.4f}", "accuracy": "{:.1%}"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Lower is better for log loss / Brier / RPS. RPS is the ordered (football-standard) score."
    )


def _tournament(store: SnapshotStore, model: str) -> None:
    st.subheader("Tournament outlook")
    odds = store.tournament_odds(model)
    if odds.empty:
        st.info("No simulation output available.")
        return

    left, right = st.columns([3, 2])
    with left:
        top = odds.head(15)
        fig = px.bar(
            top,
            x="champion",
            y="team",
            orientation="h",
            labels={"champion": "Championship probability", "team": ""},
            title="Title favourites",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Per-stage reach probabilities**")
        stage_cols = ["reach_r16", "reach_qf", "reach_sf", "reach_final", "champion"]
        table = odds.head(10)[["team", *stage_cols]].set_index("team")
        st.dataframe(
            table.style.format("{:.1%}").background_gradient(cmap="Greens", axis=None),
            use_container_width=True,
        )


def _fixture_predictor(store: SnapshotStore, model: str) -> None:
    st.subheader("Fixture predictor")
    teams = store.teams()
    if not teams:
        st.info("No teams available.")
        return

    mode = st.radio(
        "Mode",
        ["Any two teams", "Group-stage matchup"],
        horizontal=True,
        help=(
            "Any two teams: a hypothetical match between any pair. "
            "Group-stage matchup: the opponent is restricted to the team's own WC26 group."
        ),
    )
    if mode == "Any two teams":
        _predict_any(store, teams, model)
    else:
        _predict_within_group(store, model)


def _predict_any(store: SnapshotStore, teams: list[str], model: str) -> None:
    c1, c2, c3 = st.columns([2, 2, 1])
    home = c1.selectbox("Team A", teams, index=0, key="any_a")
    away = c2.selectbox("Team B", teams, index=1, key="any_b")
    neutral = c3.checkbox("Neutral venue", value=True, key="any_neutral")
    if home == away:
        st.warning("Pick two different teams.")
        return
    _render_prediction(store, home, away, neutral, model)


def _predict_within_group(store: SnapshotStore, model: str) -> None:
    groups = store.groups()
    if groups.empty:
        st.info("No group draw available.")
        return
    team_to_group = dict(zip(groups["team"], groups["group"], strict=True))
    teams = sorted(team_to_group)

    c1, c2 = st.columns(2)
    team = c1.selectbox("Team", teams, index=0, key="grp_team")
    grp = team_to_group[team]
    # Opponent is restricted to the same group (group games are at neutral venues).
    opponents = sorted(t for t in teams if team_to_group[t] == grp and t != team)
    opponent = c2.selectbox(f"Opponent (Group {grp})", opponents, key="grp_opp")
    st.caption(f"**{team}** is in **Group {grp}** with: {', '.join(o for o in opponents)}")
    _render_prediction(store, team, opponent, True, model)


def _render_prediction(
    store: SnapshotStore, home: str, away: str, neutral: bool, model: str
) -> None:
    pred = store.predict(home, away, neutral=neutral, model=model)
    probs = pd.DataFrame(
        {
            "outcome": [f"{home} win", "Draw", f"{away} win"],
            "probability": [pred["p_home"], pred["p_draw"], pred["p_away"]],
        }
    )
    fig = px.bar(probs, x="outcome", y="probability", title=f"{home} vs {away}", text_auto=".1%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Most likely: **{pred['most_likely']}** · model: {model}")


main()
