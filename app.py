"""
Third Base Send Model - Web Application

A professional dashboard for analyzing third base coach decisions
using Win Probability analysis.

Updated: Feb 2026
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json
import re
import urllib.request
import threading

app = Flask(__name__)

# 2025 Third Base Coaches data
# Source: MLB team websites and press releases
THIRD_BASE_COACHES = {
    "AZ": "Shaun Larkin",
    "ATL": "Matt Tuiasosopo",
    "BAL": "Tony Mansolino",
    "BOS": "Kyle Hudson",
    "CHC": "Quintin Berry",
    "CWS": "Justin Jirschele",
    "CIN": "J.R. House",
    "CLE": "Rouglas Odor",
    "COL": "Warren Schaeffer",
    "DET": "Joey Cora",
    "HOU": "Tony Perezchica",
    "KC": "Vance Wilson",
    "LAA": "Brian Butterfield",
    "LAD": "Dino Ebel",
    "MIA": "Blake Lalli",
    "MIL": "Jason Lane",
    "MIN": "Tommy Watkins",
    "NYM": "Mike Sarbaugh",
    "NYY": "Luis Rojas",
    "ATH": "Eric Martins",
    "PHI": "Dusty Wathan",
    "PIT": "Mike Rabelo",
    "SD": "Tim Leiper",
    "SF": "Matt Williams",
    "SEA": "Kristopher Negrón",
    "STL": "Ron Warner",
    "TB": "Brady Williams",
    "TEX": "Tony Beasley",
    "TOR": "Carlos Febles",
    "WSH": "Ricky Gutierrez",
}

# Load data - check local data folder first (for production), then parent output (for dev)
_local_data = os.path.join(os.path.dirname(__file__), "data")
_parent_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
DATA_DIR = _local_data if os.path.exists(_local_data) else _parent_data

def reclassify_decisions(df):
    """Reclassify coach_decision_quality using model recommendation (should_send)
    instead of outcome (sent_success).

    The original pipeline classified 'bad_send' based on whether the runner was
    thrown out. The correct classification uses the model: a send is only 'bad'
    if the model recommended holding (should_send=False). A runner thrown out on
    a model-recommended send is a correct decision with an unlucky outcome.

    For held plays: if the model says send (WP_send > WP_hold) but the coach
    held, it's a missed opportunity. We don't distinguish based on runner
    behavior because that creates confusing results where we show "Model: Send"
    but classify as "Correct: Hold".

    When WP_delta is negligible (below a dynamic threshold), the send/hold
    decision has no meaningful impact on win probability. These plays are
    classified as correct regardless of what the coach decided.

    IMPORTANT: This function recalculates should_send and WP_delta using
    p_conservative (the physics-blended scoring probability) instead of
    p_score_if_sent (the inflated ML estimate). This ensures consistency
    between the displayed probabilities and the model recommendations.
    """
    if "was_sent" not in df.columns:
        return df

    # Import WP calculation function
    from win_probability import get_wp_for_batting_team

    df = df.copy()

    # Recalculate WP values using p_conservative
    def recalculate_wp(row):
        """Recalculate WP_send and should_send using p_conservative."""
        try:
            p_score = row.get("p_conservative", 0.5)
            if pd.isna(p_score):
                p_score = 0.5

            inning = int(row.get("inning", 5)) if pd.notna(row.get("inning")) else 5
            inning_topbot = row.get("inning_topbot", "Top")
            is_home_batting = "Bot" in str(inning_topbot)
            outs = int(row.get("outs_when_up", 0)) if pd.notna(row.get("outs_when_up")) else 0

            # Score diff from batting team's perspective AFTER 3B runner scores
            has_3b_runner = pd.notna(row.get("on_3b"))
            base_score_diff = int(row.get("score_diff", 0)) if pd.notna(row.get("score_diff")) else 0
            score_diff_after = base_score_diff + 1 if has_3b_runner else base_score_diff

            # WP if 2B runner scores
            wp_if_score = get_wp_for_batting_team(
                score_diff_batting=score_diff_after + 1,
                inning=inning,
                is_home_batting=is_home_batting,
                outs=outs,
                base_state="1B"
            )

            # WP if 2B runner out at home
            wp_if_out = get_wp_for_batting_team(
                score_diff_batting=score_diff_after,
                inning=inning,
                is_home_batting=is_home_batting,
                outs=outs + 1,
                base_state="1B"
            )

            # WP if 2B runner held at 3B
            wp_if_hold = get_wp_for_batting_team(
                score_diff_batting=score_diff_after,
                inning=inning,
                is_home_batting=is_home_batting,
                outs=outs,
                base_state="1B-3B"
            )

            # Expected WP of sending
            wp_send_calc = p_score * wp_if_score + (1 - p_score) * wp_if_out
            wp_delta_calc = wp_send_calc - wp_if_hold
            should_send_calc = wp_send_calc > wp_if_hold

            # Calculate run value (value of one run in WP terms)
            # This is the difference between scoring and not scoring
            run_value_wp_calc = wp_if_score - wp_if_hold  # Value of the 2B runner scoring

            return pd.Series({
                "WP_send_recalc": wp_send_calc,
                "WP_hold_recalc": wp_if_hold,
                "WP_delta_recalc": wp_delta_calc,
                "should_send_recalc": should_send_calc,
                "run_value_wp_recalc": run_value_wp_calc
            })
        except Exception:
            # On error, fall back to original values
            return pd.Series({
                "WP_send_recalc": row.get("WP_send", 0.5),
                "WP_hold_recalc": row.get("WP_hold", 0.5),
                "WP_delta_recalc": row.get("WP_delta", 0),
                "should_send_recalc": row.get("should_send", False),
                "run_value_wp_recalc": row.get("run_value_wp", 0.08)
            })

    # Apply recalculation to all rows
    recalc_df = df.apply(recalculate_wp, axis=1)
    df["WP_send"] = recalc_df["WP_send_recalc"]
    df["WP_hold"] = recalc_df["WP_hold_recalc"]
    df["WP_delta"] = recalc_df["WP_delta_recalc"]
    df["should_send"] = recalc_df["should_send_recalc"]
    df["run_value_wp"] = recalc_df["run_value_wp_recalc"]

    # Threshold for "no preference" classification
    # If |WP_delta| < 1%, the difference is too small to matter
    def get_threshold(row):
        return 0.01  # 1 percentage point

    def classify(row):
        was_sent = row.get("was_sent", False)
        should_send = row.get("should_send", False)
        wp_delta = row.get("WP_delta", None)

        # Get dynamic threshold based on how contested the play was
        threshold = get_threshold(row)

        # When WP difference is below threshold, both decisions are equally valid
        if pd.notna(wp_delta) and abs(wp_delta) < threshold:
            return "correct_send" if was_sent else "correct_hold"

        if was_sent:
            return "correct_send" if should_send else "bad_send"
        else:
            # If model says send and coach held, it's a missed opportunity
            return "missed_opportunity" if should_send else "correct_hold"

    df["coach_decision_quality"] = df.apply(classify, axis=1)
    df["is_coach_mistake"] = df["coach_decision_quality"].isin(["bad_send", "missed_opportunity"])
    df["decision_correct"] = ~df["is_coach_mistake"]

    # Add negligible_delta flag for plays where WP difference is below threshold
    def is_negligible(row):
        wp_delta = row.get("WP_delta", None)
        if pd.isna(wp_delta):
            return False
        threshold = get_threshold(row)
        return abs(wp_delta) < threshold

    df["negligible_delta"] = df.apply(is_negligible, axis=1)

    return df


def load_data():
    """Load all analysis data."""
    data = {}

    # Team rankings
    team_file = os.path.join(DATA_DIR, "team_rankings.csv")
    if os.path.exists(team_file):
        data["teams"] = pd.read_csv(team_file)

    # Full analysis
    full_file = os.path.join(DATA_DIR, "full_analysis.parquet")
    if os.path.exists(full_file):
        plays = pd.read_parquet(full_file)

        # Recalculate throw_time with corrected transfer time formula
        # The original formula was too slow for running catches
        # New formula: faster transfer for running catches (high difficulty = long run)
        def recalc_throw_time(row):
            throw_distance = row.get("throw_distance_from_fielding")
            arm_strength = row.get("fielder_arm_strength", 88.0)
            fielding_difficulty = row.get("fielding_difficulty", 0.5)

            if pd.isna(throw_distance):
                return np.nan
            if pd.isna(arm_strength):
                arm_strength = 88.0
            if pd.isna(fielding_difficulty):
                fielding_difficulty = 0.5

            # Transfer time: running catches are FASTER (momentum toward home)
            # Range: 1.4s (routine) to 1.1s (long running catch)
            transfer_time = 1.4 - (fielding_difficulty * 0.3)

            # Flight time: distance / velocity * 1.2 for arc
            throw_speed_fps = arm_strength * 1.467
            flight_time = (throw_distance / throw_speed_fps) * 1.2

            return transfer_time + flight_time

        # Store old throw_time to calculate the adjustment
        old_throw_time = plays["throw_time"].copy()
        plays["throw_time"] = plays.apply(recalc_throw_time, axis=1)

        # Recalculate scoring_window with new throw_time
        # scoring_window = total_play_time - runner_time_from_2b
        # total_play_time = hang_time + 1.0 + throw_time
        # So new_scoring_window = old_scoring_window - (old_throw_time - new_throw_time)
        throw_time_diff = old_throw_time - plays["throw_time"]  # How much faster the throw is
        plays["scoring_window"] = plays["scoring_window"] - throw_time_diff

        # Recalculate p_conservative with new scoring_window
        # This blends calibrated p_score with raw physics based on scoring_window
        def get_conservative_p_score(calibrated_p, scoring_window):
            if pd.isna(scoring_window) or pd.isna(calibrated_p):
                return calibrated_p

            # Raw physics probability (sigmoid of scoring_window)
            raw_p = 1 / (1 + np.exp(-scoring_window))

            # Blend weight: how much to trust calibrated vs raw
            if scoring_window >= 2.0:
                blend = 1.0  # 100% calibrated - automatic score
            elif scoring_window >= 1.0:
                blend = 0.8  # 80% calibrated - very likely to score
            elif scoring_window >= 0.5:
                blend = 0.6  # 60% calibrated - likely to score
            elif scoring_window >= 0.0:
                blend = 0.4  # 40% calibrated - borderline
            elif scoring_window >= -0.3:
                blend = 0.2  # 20% calibrated - slightly negative
            else:
                blend = 0.0  # 100% raw physics - clearly should hold

            return blend * calibrated_p + (1 - blend) * raw_p

        plays["p_conservative"] = plays.apply(
            lambda r: get_conservative_p_score(
                r.get("p_score_if_sent", 0.5),
                r.get("scoring_window", 0)
            ),
            axis=1
        )

        plays = reclassify_decisions(plays)
        data["plays"] = plays

    # Held should send
    held_file = os.path.join(DATA_DIR, "held_should_send.csv")
    if os.path.exists(held_file):
        data["held_should_send"] = pd.read_csv(held_file)

    # Sent should hold
    sent_file = os.path.join(DATA_DIR, "sent_should_hold.csv")
    if os.path.exists(sent_file):
        data["sent_should_hold"] = pd.read_csv(sent_file)

    return data

DATA = load_data()

# Cache for MLB API playId lookups: (game_pk, at_bat_number) -> playId UUID
_play_id_cache = {}
_play_id_cache_lock = threading.Lock()
_PLAY_ID_CACHE_FILE = os.path.join(DATA_DIR, "play_id_cache.json")

def _load_play_id_cache():
    """Load cached playIds from disk."""
    if os.path.exists(_PLAY_ID_CACHE_FILE):
        try:
            with open(_PLAY_ID_CACHE_FILE, "r") as f:
                raw = json.load(f)
            # Convert string keys back to (int, int) tuples
            for k, v in raw.items():
                parts = k.split(",")
                _play_id_cache[(int(parts[0]), int(parts[1]))] = v
        except Exception:
            pass

def _save_play_id_cache():
    """Save playId cache to disk."""
    try:
        raw = {f"{k[0]},{k[1]}": v for k, v in _play_id_cache.items()}
        with open(_PLAY_ID_CACHE_FILE, "w") as f:
            json.dump(raw, f)
    except Exception:
        pass

_load_play_id_cache()


def _fetch_game_play_ids(game_pk):
    """Fetch all playIds for a game from the MLB API and cache them."""
    try:
        url = f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live"
        req = urllib.request.Request(url, headers={"User-Agent": "ThirdBaseSendModel/1.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())

        for play in data["liveData"]["plays"]["allPlays"]:
            ab_num = play["about"]["atBatIndex"] + 1  # convert 0-based to 1-based
            play_events = play.get("playEvents", [])
            for event in reversed(play_events):
                if "playId" in event:
                    _play_id_cache[(int(game_pk), ab_num)] = event["playId"]
                    break
    except Exception:
        pass


def get_video_link(game_pk, at_bat_number):
    """Get a play-specific Baseball Savant video URL."""
    if pd.isna(game_pk):
        return ""

    game_pk = int(game_pk)
    at_bat_number = int(at_bat_number) if pd.notna(at_bat_number) else None

    if at_bat_number is None:
        return f"https://baseballsavant.mlb.com/gamefeed?gamePk={game_pk}"

    key = (game_pk, at_bat_number)

    # Check cache
    if key not in _play_id_cache:
        with _play_id_cache_lock:
            if key not in _play_id_cache:
                _fetch_game_play_ids(game_pk)
                _save_play_id_cache()

    play_id = _play_id_cache.get(key)
    if play_id:
        return f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"

    # Fallback to game feed
    return f"https://baseballsavant.mlb.com/gamefeed?gamePk={game_pk}"


def format_percentage(val, decimals=1):
    """Format a decimal as a percentage string."""
    if pd.isna(val):
        return "—"
    return f"{val * 100:.{decimals}f}%"


def format_number(val, decimals=1):
    """Format a number with specified decimals."""
    if pd.isna(val):
        return "—"
    return f"{val:.{decimals}f}"


def get_decision_color(decision_quality):
    """Get color class for decision quality."""
    colors = {
        "correct_send": "success",
        "correct_hold": "success",
        "missed_opportunity": "danger",
        "bad_send": "danger",
        "send_unknown": "warning",
    }
    return colors.get(decision_quality, "secondary")


def prepare_play_data(row):
    """Prepare a single play's data for display."""
    play = {
        "id": int(row.name) if hasattr(row, 'name') else 0,
        # Game context
        "batting_team": row.get("batting_team", "—"),
        "inning": f"{'Top' if row.get('inning_topbot') == 'Top' else 'Bot'} {int(row.get('inning_num', 0))}",
        "score_diff": int(row.get("score_diff", 0)) if pd.notna(row.get("score_diff")) else 0,
        "outs": int(row.get("outs_when_up", 0)),
        "base_state": row.get("base_state_pre", "2B"),

        # Decision
        "was_sent": bool(row.get("was_sent", False)),
        "should_send": bool(row.get("should_send", False)),
        "decision_quality": row.get("coach_decision_quality", "unknown"),
        "negligible_delta": bool(row.get("negligible_delta", False)),

        # Probabilities and run values
        # Conversion: 1 win ≈ 10 runs
        "p_score": format_percentage(row.get("p_conservative"), 1),
        "p_score_raw": row.get("p_conservative", 0),
        "wp_send": format_percentage(row.get("WP_send"), 1),
        "wp_hold": format_percentage(row.get("WP_hold"), 1),
        # Decision impact: flip sign if held (so positive = good decision, negative = bad decision)
        # WP_delta = WP_send - WP_hold, so if held, impact = -WP_delta
        "wp_delta": format_percentage(row.get("WP_delta", 0) if row.get("was_sent", False) else -row.get("WP_delta", 0), 2),
        "wp_delta_raw": row.get("WP_delta", 0) if row.get("was_sent", False) else -row.get("WP_delta", 0),
        "runs_delta": f"{(row.get('WP_delta', 0) if row.get('was_sent', False) else -row.get('WP_delta', 0)) * 10:+.2f}" if pd.notna(row.get("WP_delta")) else "—",
        "runs_delta_raw": (row.get("WP_delta", 0) if row.get("was_sent", False) else -row.get("WP_delta", 0)) * 10 if pd.notna(row.get("WP_delta")) else 0,
        # WP cost (absolute value) - for displaying mistake costs as positive values
        "wp_cost": format_percentage(abs(row.get("WP_delta", 0)), 2),

        # Batted ball
        "event_type": row.get("event_type", "single"),
        "launch_speed": format_number(row.get("launch_speed"), 1),
        "launch_angle": format_number(row.get("launch_angle"), 1),
        "batted_ball_type": row.get("batted_ball_type", "—"),
        "hit_location": int(row.get("hit_location", 0)) if pd.notna(row.get("hit_location")) else "—",

        # Timing
        "hang_time": format_number(row.get("hang_time"), 2),
        "throw_time": format_number(row.get("throw_time"), 2),
        "scoring_window": format_number(row.get("scoring_window"), 2),
        "scoring_window_raw": row.get("scoring_window", 0),

        # Player attributes
        "runner_speed": format_number(row.get("runner_speed"), 1),
        "fielder_arm": format_number(row.get("fielder_arm_strength"), 1),
        "throw_distance": format_number(row.get("fielding_dist_from_home"), 0),

        # Runner behavior
        "runner_behavior": row.get("runner_behavior", "—"),
        "had_opportunity": bool(row.get("had_scoring_opportunity", False)),

        # Outcome
        "sent_success": row.get("sent_success", None),
        "description": row.get("des", "—"),
        "game_pk": int(row.get("game_pk", 0)) if pd.notna(row.get("game_pk")) else None,
        "video_link": get_video_link(row.get("game_pk"), row.get("at_bat_number")),
    }
    return play


@app.route("/")
def index():
    """Home page with overview."""
    # Calculate summary stats
    plays_df = DATA.get("plays", pd.DataFrame())
    teams_df = DATA.get("teams", pd.DataFrame())

    singles = plays_df[plays_df["event_type"] == "single"] if len(plays_df) > 0 else pd.DataFrame()

    missed = singles[singles["coach_decision_quality"] == "missed_opportunity"] if len(singles) > 0 else pd.DataFrame()
    bad_sends = singles[singles["coach_decision_quality"] == "bad_send"] if len(singles) > 0 else pd.DataFrame()
    correct = singles[singles["coach_decision_quality"].isin(["correct_send", "correct_hold"])] if len(singles) > 0 else pd.DataFrame()

    # Calculate runs metrics (1 win ≈ 10 runs)
    WP_TO_RUNS = 10

    # Runs from mistakes (more actionable metric)
    missed_runs = missed["WP_delta"].sum() * WP_TO_RUNS if len(missed) > 0 else 0  # Runs left on table
    bad_send_runs = abs(bad_sends["WP_delta"].sum()) * WP_TO_RUNS if len(bad_sends) > 0 else 0  # Runs lost
    total_mistake_runs = missed_runs + bad_send_runs

    # Average runs per mistake
    total_mistakes = len(missed) + len(bad_sends)
    avg_runs_per_mistake = total_mistake_runs / total_mistakes if total_mistakes > 0 else 0

    stats = {
        "total_plays": len(singles),
        "plays_sent": int(singles["was_sent"].sum()) if len(singles) > 0 else 0,
        "plays_held": int((~singles["was_sent"]).sum()) if len(singles) > 0 else 0,
        "success_rate": format_percentage(singles[singles["was_sent"]]["sent_success"].mean()) if len(singles) > 0 else "—",
        "correct_decisions_count": len(correct),
        "missed_opportunities": len(missed),
        "bad_sends": len(bad_sends),
        "correct_decisions": format_percentage(singles["decision_correct"].mean()) if len(singles) > 0 else "—",
        "runs_per_mistake": f"{avg_runs_per_mistake:.2f}",
        "total_runs_from_mistakes": f"{total_mistake_runs:.1f}",
    }

    # League-wide impact summary
    total_wp_missed = missed["WP_delta"].sum() if len(missed) > 0 else 0
    total_wp_bad_sends = bad_sends["WP_delta"].sum() if len(bad_sends) > 0 else 0

    # Late & close breakdown
    late_close_missed = missed[missed["late_and_close"] == True] if len(missed) > 0 else pd.DataFrame()
    # High leverage: LI > 1.5
    high_leverage_missed = missed[missed["leverage_index"] > 1.5] if len(missed) > 0 else pd.DataFrame()

    # Format aggregate WP as percentage points (val * 100)
    def format_wp_agg(val):
        pp = val * 100
        return f"{pp:.1f} pp"

    # Average WP per play in percentage points
    avg_wp_missed = (total_wp_missed / len(missed) * 100) if len(missed) > 0 else 0
    avg_wp_bad = (abs(total_wp_bad_sends) / len(bad_sends) * 100) if len(bad_sends) > 0 else 0

    impact = {
        "total_wp_missed": format_wp_agg(total_wp_missed),
        "total_wp_missed_raw": total_wp_missed,
        "avg_wp_missed": f"{avg_wp_missed:.1f} pp",
        "total_wp_bad_sends": format_wp_agg(abs(total_wp_bad_sends)),
        "total_wp_bad_sends_raw": total_wp_bad_sends,
        "avg_wp_bad": f"{avg_wp_bad:.1f} pp",
        "late_close_missed": len(late_close_missed),
        "late_close_wp": format_wp_agg(late_close_missed["WP_delta"].sum()) if len(late_close_missed) > 0 else "0.0 pp",
        "high_leverage_missed": len(high_leverage_missed),
        "high_leverage_wp": format_wp_agg(high_leverage_missed["WP_delta"].sum()) if len(high_leverage_missed) > 0 else "0.0 pp",
    }

    # Outs breakdown for missed opportunities
    outs_breakdown = {}
    if len(missed) > 0:
        for outs in [0, 1, 2]:
            outs_plays = missed[missed["outs_when_up"] == outs]
            outs_breakdown[outs] = {
                "count": len(outs_plays),
                "wp": format_wp_agg(outs_plays["WP_delta"].sum()) if len(outs_plays) > 0 else "0.0 pp",
            }
    else:
        for outs in [0, 1, 2]:
            outs_breakdown[outs] = {"count": 0, "wp": "0.0 pp"}

    return render_template("index.html", stats=stats, impact=impact, outs_breakdown=outs_breakdown)


@app.route("/methodology")
def methodology():
    """Methodology page explaining the model."""
    return render_template("methodology.html")


@app.route("/calculations")
def calculations():
    """Calculations page explaining how each factor is calculated."""
    return render_template("calculations.html")


@app.route("/ideal-state")
def ideal_state():
    """Ideal state page comparing current model to player tracking."""
    return render_template("ideal_state.html")


@app.route("/teams")
def leaderboards():
    """Team rankings page."""
    teams_df = DATA.get("teams", pd.DataFrame())
    plays_df = DATA.get("plays", pd.DataFrame())

    if len(teams_df) == 0:
        return render_template("teams.html", teams=[])

    # Pre-calculate stats from reclassified plays data (for consistency with plays page)
    singles = plays_df[plays_df["event_type"] == "single"] if len(plays_df) > 0 else pd.DataFrame()
    team_reclassified_stats = {}
    if len(singles) > 0:
        for team_code in singles["batting_team"].unique():
            team_plays = singles[singles["batting_team"] == team_code]

            # Calculate net runs: value from correct decisions minus value lost from mistakes
            # Using WP_delta converted to runs (non-context would use RE, but we use WP*10 as proxy)
            correct_plays = team_plays[team_plays["decision_correct"] == True]
            mistake_plays = team_plays[team_plays["decision_correct"] == False]

            # For correct decisions: they captured the available value (|WP_delta|)
            # For mistakes: they lost value (|WP_delta|)
            correct_value = correct_plays["WP_delta"].abs().sum() * 10 if len(correct_plays) > 0 else 0
            mistake_cost = mistake_plays["WP_delta"].abs().sum() * 10 if len(mistake_plays) > 0 else 0
            net_runs = correct_value - mistake_cost

            missed = len(team_plays[team_plays["coach_decision_quality"] == "missed_opportunity"])
            bad_sends = len(team_plays[team_plays["coach_decision_quality"] == "bad_send"])
            total_correct = int(team_plays["decision_correct"].sum()) if len(team_plays) > 0 else 0

            team_reclassified_stats[team_code] = {
                "missed": missed,
                "bad_sends": bad_sends,
                "total_mistakes": missed + bad_sends,
                "total_correct": total_correct,
                "pct_correct": team_plays["decision_correct"].mean() if len(team_plays) > 0 else 0,
                "net_runs": net_runs,
            }

    # Prepare team data
    # Conversion: 1 win ≈ 10 runs (standard sabermetric approximation)
    WP_TO_RUNS = 10
    teams_list = []
    for _, row in teams_df.iterrows():
        team_code = row.get("batting_team", "—")

        # Context-weighted (Win Probability based) - accounts for game leverage
        total_wp = row.get("total_decision_value_wp", 0)
        total_runs_wp = total_wp * WP_TO_RUNS
        avg_wp = row.get("avg_decision_value_wp", 0)
        avg_runs_wp = avg_wp * WP_TO_RUNS

        # Non-context-weighted (Run Expectancy based) - treats all runs equally
        # These values are already in runs, no conversion needed
        total_runs_re = row.get("total_decision_value_re", 0)
        avg_runs_re = row.get("avg_run_value", 0)

        # Get stats from reclassified data
        reclassified = team_reclassified_stats.get(team_code, {"missed": 0, "bad_sends": 0, "total_mistakes": 0, "total_correct": 0, "pct_correct": 0, "net_runs": 0})
        net_runs = reclassified["net_runs"]

        # Get third base coach name
        coach_name = THIRD_BASE_COACHES.get(team_code)

        team = {
            "name": team_code,
            "coach_name": coach_name,
            "total_plays": int(row.get("total_plays", 0)),
            "plays_sent": int(row.get("plays_sent", 0)),
            "plays_held": int(row.get("plays_held", 0)),
            "send_rate": format_percentage(row.get("send_rate", 0) / 100, 1),
            # Use reclassified pct_correct for consistency
            "pct_correct": format_percentage(reclassified["pct_correct"], 1),
            "total_wp_value": format_percentage(total_wp, 2),
            "total_wp_value_raw": total_wp,
            # Context-weighted runs (WP-based)
            "total_runs": f"{total_runs_wp:+.1f}" if total_runs_wp != 0 else "0.0",
            "total_runs_raw": total_runs_wp,
            "avg_runs": f"{avg_runs_wp:+.2f}" if avg_runs_wp != 0 else "0.00",
            "avg_runs_raw": avg_runs_wp,
            # Non-context-weighted runs (RE-based)
            "total_runs_re": f"{total_runs_re:+.1f}" if total_runs_re != 0 else "0.0",
            "total_runs_re_raw": total_runs_re,
            "avg_runs_re": f"{avg_runs_re:+.2f}" if avg_runs_re != 0 else "0.00",
            "avg_runs_re_raw": avg_runs_re,
            # Net runs: value from correct decisions minus value lost from mistakes
            "net_runs": f"{net_runs:+.1f}" if net_runs != 0 else "0.0",
            "net_runs_raw": net_runs,
            # Net runs per play
            "net_runs_per_play": f"{net_runs / int(row.get('total_plays', 1)):+.3f}" if int(row.get('total_plays', 0)) > 0 else "0.000",
            "net_runs_per_play_raw": net_runs / int(row.get('total_plays', 1)) if int(row.get('total_plays', 0)) > 0 else 0,
            "avg_wp_value": format_percentage(avg_wp, 3),
            # Use reclassified counts for consistency with plays page
            "total_correct": reclassified["total_correct"],
            "total_mistakes": reclassified["total_mistakes"],
            "missed_opportunities": reclassified["missed"],
            "bad_sends": reclassified["bad_sends"],
            "runner_hesitation": int(row.get("runner_hesitation", 0)),
        }
        teams_list.append(team)

    # Sort by net runs (descending) by default
    teams_list.sort(key=lambda x: x["net_runs_raw"], reverse=True)

    return render_template("teams.html", teams=teams_list)


@app.route("/team/<team_code>")
def team_detail(team_code):
    """Individual team detail page."""
    plays_df = DATA.get("plays", pd.DataFrame())
    teams_df = DATA.get("teams", pd.DataFrame())

    if len(plays_df) == 0 or len(teams_df) == 0:
        return render_template("404.html"), 404

    # Get team row
    team_row = teams_df[teams_df["batting_team"] == team_code]
    if len(team_row) == 0:
        return render_template("404.html"), 404

    team_row = team_row.iloc[0]

    # Get team plays
    team_plays = plays_df[
        (plays_df["batting_team"] == team_code) &
        (plays_df["event_type"] == "single")
    ]

    # Team full names (could be extended)
    team_names = {
        'AZ': 'Arizona Diamondbacks', 'ATL': 'Atlanta Braves', 'BAL': 'Baltimore Orioles',
        'BOS': 'Boston Red Sox', 'CHC': 'Chicago Cubs', 'CWS': 'Chicago White Sox',
        'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians', 'COL': 'Colorado Rockies',
        'DET': 'Detroit Tigers', 'HOU': 'Houston Astros', 'KC': 'Kansas City Royals',
        'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers', 'MIA': 'Miami Marlins',
        'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins', 'NYM': 'New York Mets',
        'NYY': 'New York Yankees', 'ATH': 'Oakland Athletics', 'PHI': 'Philadelphia Phillies',
        'PIT': 'Pittsburgh Pirates', 'SD': 'San Diego Padres', 'SF': 'San Francisco Giants',
        'SEA': 'Seattle Mariners', 'STL': 'St. Louis Cardinals', 'TB': 'Tampa Bay Rays',
        'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays', 'WSH': 'Washington Nationals'
    }

    # Get plays by decision quality (using reclassified data for consistency with plays page)
    missed_opps = team_plays[team_plays["coach_decision_quality"] == "missed_opportunity"]
    bad_sends_plays = team_plays[team_plays["coach_decision_quality"] == "bad_send"]

    # Get worst plays for display (top 10, sorted by biggest impact)
    worst_holds = missed_opps.nlargest(10, "WP_delta")
    worst_sends = bad_sends_plays.nsmallest(10, "WP_delta")

    total_wp_raw = team_row.get("total_decision_value_wp", 0)
    total_runs_re = team_row.get("total_decision_value_re", 0)
    # Calculate pct_correct from reclassified data
    pct_correct_reclassified = team_plays["decision_correct"].mean() if len(team_plays) > 0 else 0

    # Calculate net runs: value from correct decisions minus value lost from mistakes
    correct_plays = team_plays[team_plays["decision_correct"] == True]
    mistake_plays = team_plays[team_plays["decision_correct"] == False]
    correct_value = correct_plays["WP_delta"].abs().sum() * 10 if len(correct_plays) > 0 else 0
    mistake_cost = mistake_plays["WP_delta"].abs().sum() * 10 if len(mistake_plays) > 0 else 0
    net_runs = correct_value - mistake_cost

    team = {
        "name": team_code,
        "full_name": team_names.get(team_code, team_code),
        "total_plays": int(team_row.get("total_plays", 0)),
        "plays_sent": int(team_row.get("plays_sent", 0)),
        "plays_held": int(team_row.get("plays_held", 0)),
        "send_rate": format_percentage(team_row.get("send_rate", 0) / 100, 1),
        "send_rate_pct": team_row.get("send_rate", 0),
        # Use reclassified pct_correct for consistency
        "pct_correct": format_percentage(pct_correct_reclassified, 1),
        "total_wp_value": format_percentage(total_wp_raw, 2),
        "total_wp_value_raw": total_wp_raw,
        # Non-context-weighted runs (RE-based)
        "total_runs_re": f"{total_runs_re:+.1f}" if total_runs_re != 0 else "0.0",
        "total_runs_re_raw": total_runs_re,
        # Net runs: value from correct decisions minus value lost from mistakes
        "net_runs": f"{net_runs:+.1f}" if net_runs != 0 else "0.0",
        "net_runs_raw": net_runs,
        # Use actual counts from reclassified data (consistent with plays page)
        "missed_opportunities": len(missed_opps),
        "bad_sends": len(bad_sends_plays),
        "runner_hesitation": int(team_row.get("runner_hesitation", 0)),
        "total_mistakes": len(missed_opps) + len(bad_sends_plays),
        "worst_holds": [prepare_play_data(row) for idx, row in worst_holds.iterrows()],
        "worst_sends": [prepare_play_data(row) for idx, row in worst_sends.iterrows()],
    }

    # Add IDs to plays
    for i, (idx, _) in enumerate(worst_holds.iterrows()):
        team["worst_holds"][i]["id"] = idx
    for i, (idx, _) in enumerate(worst_sends.iterrows()):
        team["worst_sends"][i]["id"] = idx

    # League averages for comparison (calculate from reclassified plays data for consistency)
    league_avg = {}
    if len(teams_df) > 0:
        # Calculate league-wide stats from reclassified data
        all_singles = plays_df[plays_df["event_type"] == "single"]
        team_reclassified_counts = []
        for t in all_singles["batting_team"].unique():
            t_plays = all_singles[all_singles["batting_team"] == t]
            # Calculate net runs for this team
            t_correct = t_plays[t_plays["decision_correct"] == True]
            t_mistakes = t_plays[t_plays["decision_correct"] == False]
            t_correct_value = t_correct["WP_delta"].abs().sum() * 10 if len(t_correct) > 0 else 0
            t_mistake_cost = t_mistakes["WP_delta"].abs().sum() * 10 if len(t_mistakes) > 0 else 0
            t_net_runs = t_correct_value - t_mistake_cost

            team_reclassified_counts.append({
                "missed": len(t_plays[t_plays["coach_decision_quality"] == "missed_opportunity"]),
                "bad_sends": len(t_plays[t_plays["coach_decision_quality"] == "bad_send"]),
                "pct_correct": t_plays["decision_correct"].mean() * 100 if len(t_plays) > 0 else 0,
                "net_runs": t_net_runs,
            })

        avg_missed = sum(t["missed"] for t in team_reclassified_counts) / len(team_reclassified_counts) if team_reclassified_counts else 0
        avg_bad_sends = sum(t["bad_sends"] for t in team_reclassified_counts) / len(team_reclassified_counts) if team_reclassified_counts else 0
        avg_pct_correct = sum(t["pct_correct"] for t in team_reclassified_counts) / len(team_reclassified_counts) if team_reclassified_counts else 0
        avg_net_runs = sum(t["net_runs"] for t in team_reclassified_counts) / len(team_reclassified_counts) if team_reclassified_counts else 0

        league_avg = {
            "send_rate": teams_df["send_rate"].mean(),
            "pct_correct": avg_pct_correct,  # From reclassified data
            "total_wp_value": teams_df["total_decision_value_wp"].mean(),
            "total_runs_re": teams_df["total_decision_value_re"].mean(),
            "net_runs": avg_net_runs,
            "missed_opportunities": avg_missed,
            "bad_sends": avg_bad_sends,
            "total_plays": teams_df["total_plays"].mean(),
        }

    return render_template("team_detail.html", team=team, league_avg=league_avg)


@app.route("/plays")
def plays():
    """Play explorer page."""
    plays_df = DATA.get("plays", pd.DataFrame())

    # Filter to singles only
    if len(plays_df) > 0:
        plays_df = plays_df[plays_df["event_type"] == "single"].copy()

    # Get filter parameters
    team = request.args.get("team", "all")
    decision_type = request.args.get("decision", "all")
    situation = request.args.get("situation", "all")
    outcome = request.args.get("outcome", "all")
    sort_by = request.args.get("sort", "wp_delta")
    sort_order = request.args.get("order", "desc")
    page = int(request.args.get("page", 1))
    per_page = 25

    # Apply filters
    if team != "all" and len(plays_df) > 0:
        plays_df = plays_df[plays_df["batting_team"] == team]

    if decision_type != "all" and len(plays_df) > 0:
        if decision_type == "mistakes":
            plays_df = plays_df[plays_df["coach_decision_quality"].isin(["missed_opportunity", "bad_send"])]
            # For mistakes, default to ascending sort (biggest mistakes first = most negative)
            if sort_order == "desc":
                sort_order = "asc"
        elif decision_type == "missed":
            plays_df = plays_df[plays_df["coach_decision_quality"] == "missed_opportunity"]
            if sort_order == "desc":
                sort_order = "asc"
        elif decision_type == "bad_send":
            plays_df = plays_df[plays_df["coach_decision_quality"] == "bad_send"]
            if sort_order == "desc":
                sort_order = "asc"
        elif decision_type == "correct":
            plays_df = plays_df[plays_df["coach_decision_quality"].isin(["correct_send", "correct_hold"])]

    # Situation filter (Late & Close, High Leverage)
    if situation != "all" and len(plays_df) > 0:
        if situation == "late_close":
            plays_df = plays_df[plays_df["late_and_close"] == True]
        elif situation == "high_leverage":
            plays_df = plays_df[plays_df["leverage_index"] > 1.5]

    # Outcome filter
    if outcome != "all" and len(plays_df) > 0:
        if outcome == "scored":
            plays_df = plays_df[plays_df["runner_2b_scored"] == True]
        elif outcome == "out":
            plays_df = plays_df[plays_df["runner_out_at_home"] == True]
        elif outcome == "held":
            # Runner was held at 3rd (not sent, not out at home)
            plays_df = plays_df[(plays_df["was_sent"] == False) & (plays_df["runner_out_at_home"] == False)]

    # Compute decision-adjusted WP delta for sorting (positive = good decision, negative = bad)
    if len(plays_df) > 0:
        plays_df = plays_df.copy()
        plays_df["decision_wp_delta"] = plays_df.apply(
            lambda r: r["WP_delta"] if r["was_sent"] else -r["WP_delta"], axis=1
        )
        # For mistake filters, ensure we only show plays with negative decision value
        if decision_type in ["mistakes", "missed", "bad_send"]:
            plays_df = plays_df[plays_df["decision_wp_delta"] < 0]

    # Sort
    sort_map = {
        "wp_delta": "decision_wp_delta",
        "p_score": "p_conservative",
        "scoring_window": "scoring_window",
        "team": "batting_team",
    }
    sort_col = sort_map.get(sort_by, "decision_wp_delta")
    if sort_col in plays_df.columns:
        plays_df = plays_df.sort_values(sort_col, ascending=(sort_order == "asc"))

    # Pagination
    total_plays = len(plays_df)
    total_pages = (total_plays + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    # Prepare plays for display
    plays_list = []
    for idx, row in plays_df.iloc[start_idx:end_idx].iterrows():
        play = prepare_play_data(row)
        play["id"] = idx
        plays_list.append(play)

    # Get unique teams for filter
    all_teams = sorted(DATA.get("plays", pd.DataFrame())["batting_team"].unique()) if len(DATA.get("plays", pd.DataFrame())) > 0 else []

    return render_template(
        "plays.html",
        plays=plays_list,
        teams=all_teams,
        current_team=team,
        current_decision=decision_type,
        current_situation=situation,
        current_outcome=outcome,
        current_sort=sort_by,
        current_order=sort_order,
        current_page=page,
        total_pages=total_pages,
        total_plays=total_plays,
    )


@app.route("/play/<int:play_id>")
def play_detail(play_id):
    """Individual play detail page."""
    plays_df = DATA.get("plays", pd.DataFrame())

    if play_id not in plays_df.index:
        return render_template("404.html"), 404

    row = plays_df.loc[play_id]
    play = prepare_play_data(row)
    play["id"] = play_id

    # Add additional detail fields
    play["leverage_index"] = format_number(row.get("leverage_index"), 2)
    play["late_and_close"] = bool(row.get("late_and_close", False))
    play["run_value_wp"] = format_percentage(row.get("run_value_wp"), 1)
    play["delta_ev"] = format_number(row.get("delta_EV"), 3)
    play["opportunity_type"] = row.get("opportunity_type", "—")
    play["score_likelihood"] = row.get("score_likelihood", "—")
    play["fielder_sprint_speed"] = format_number(row.get("fielder_sprint_speed"), 1)
    play["catch_probability"] = format_percentage(row.get("catch_probability"), 1)
    play["distance_to_ball"] = format_number(row.get("distance_to_ball_feet"), 0)

    # Calculate WP scenario breakdown for the math display
    # We need to calculate: WP if scores, WP if out at home, WP if held
    try:
        from win_probability import get_wp_for_batting_team

        inning = int(row.get("inning", 5))
        inning_topbot = row.get("inning_topbot", "Top")
        is_home_batting = "Bot" in str(inning_topbot)
        outs = int(row.get("outs_when_up", 0))

        # Score diff from batting team's perspective AFTER the runner from 3B scored
        # (since there was a runner on 3B who definitely scores on a single)
        has_3b_runner = pd.notna(row.get("on_3b"))
        base_score_diff = int(row.get("score_diff", 0))
        if has_3b_runner:
            score_diff_after = base_score_diff + 1  # 3B runner scored
        else:
            score_diff_after = base_score_diff

        # Scenario 1: Runner from 2B scores (2 runs total if 3B runner existed)
        wp_if_score = get_wp_for_batting_team(
            score_diff_batting=score_diff_after + 1,  # +1 more for 2B runner scoring
            inning=inning,
            is_home_batting=is_home_batting,
            outs=outs,
            base_state="1B"  # Just batter on 1B after everyone scores
        )

        # Scenario 2: Runner from 2B out at home
        wp_if_out = get_wp_for_batting_team(
            score_diff_batting=score_diff_after,
            inning=inning,
            is_home_batting=is_home_batting,
            outs=outs + 1,  # +1 out
            base_state="1B"
        )

        # Scenario 3: Runner from 2B held at 3B
        wp_if_hold = get_wp_for_batting_team(
            score_diff_batting=score_diff_after,
            inning=inning,
            is_home_batting=is_home_batting,
            outs=outs,
            base_state="1B-3B"  # Batter on 1B, held runner on 3B
        )

        # Get the probability used
        p_score = row.get("p_conservative", 0.5)
        if pd.isna(p_score):
            p_score = 0.5

        # Calculate expected WP of sending
        wp_send_calc = p_score * wp_if_score + (1 - p_score) * wp_if_out

        # Calculate WP delta (send - hold)
        wp_delta_calc = wp_send_calc - wp_if_hold

        # Calculate the gain/loss
        upside = wp_if_score - wp_if_out  # Gain from scoring vs out
        downside = wp_if_hold - wp_if_out  # What you give up by getting out vs holding

        play["wp_scenario"] = {
            "wp_if_score": wp_if_score,
            "wp_if_score_pct": f"{wp_if_score*100:.1f}%",
            "wp_if_out": wp_if_out,
            "wp_if_out_pct": f"{wp_if_out*100:.1f}%",
            "wp_if_hold": wp_if_hold,
            "wp_if_hold_pct": f"{wp_if_hold*100:.1f}%",
            "p_score": p_score,
            "p_score_pct": f"{p_score*100:.1f}%",
            "wp_send_calc": wp_send_calc,
            "wp_send_calc_pct": f"{wp_send_calc*100:.1f}%",
            "wp_delta_calc": wp_delta_calc,
            "wp_delta_calc_pct": f"{wp_delta_calc*100:+.1f}%",
            "wp_delta_calc_raw": wp_delta_calc,
            "runs_delta_calc": wp_delta_calc * 10,
            "runs_delta_calc_str": f"{wp_delta_calc * 10:+.2f}",
            "upside": upside * 100,  # In percentage points
            "upside_str": f"+{upside*100:.1f}" if upside >= 0 else f"{upside*100:.1f}",
            "downside": downside * 100,
            "downside_str": f"+{downside*100:.1f}" if downside >= 0 else f"{downside*100:.1f}",
            "risk_str": f"{downside*100:.1f}",  # Risk of out vs hold (formatted)
            "score_diff_after": score_diff_after,
            "has_3b_runner": has_3b_runner,
        }

        # Update should_send based on calculated WP values
        # Send is better if expected WP from sending exceeds WP from holding
        play["should_send"] = wp_send_calc > wp_if_hold

        # Update decision_quality to be consistent with recalculated should_send
        was_sent = play.get("was_sent", False)
        should_send = play["should_send"]
        negligible = play.get("negligible_delta", False)

        if negligible:
            # If difference is negligible, either decision is correct
            play["decision_quality"] = "correct_send" if was_sent else "correct_hold"
        elif was_sent:
            play["decision_quality"] = "correct_send" if should_send else "bad_send"
        else:
            play["decision_quality"] = "missed_opportunity" if should_send else "correct_hold"

    except Exception as e:
        play["wp_scenario"] = None

    # Key factors analysis for this play
    # Compare each factor to league medians to determine if it favors send or hold
    singles = DATA.get("plays", pd.DataFrame())
    singles = singles[singles["event_type"] == "single"] if len(singles) > 0 else pd.DataFrame()

    def compute_percentile(series, value):
        """Compute percentile rank of value within series (0-100)."""
        valid = series.dropna()
        if len(valid) == 0:
            return None
        return float((valid < value).sum()) / len(valid) * 100

    def ordinal(n):
        """Return ordinal string for an integer (1st, 2nd, 3rd, 11th, 21st, etc.)."""
        n = int(round(n))
        if 11 <= n % 100 <= 13:
            return f"{n}th"
        return f"{n}{['th','st','nd','rd'][n % 10] if n % 10 < 4 else 'th'}"

    def percentile_label(pct):
        """Return a descriptive label for a percentile value."""
        if pct <= 10:
            return "Bottom 10%"
        elif pct <= 25:
            return "Below Avg"
        elif pct <= 40:
            return "Slightly Below Avg"
        elif pct <= 60:
            return "Average"
        elif pct <= 75:
            return "Slightly Above Avg"
        elif pct <= 90:
            return "Above Avg"
        else:
            return "Top 10%"

    factors_send = []
    factors_hold = []

    if len(singles) > 0:
        # Throw distance - longer = favors send (more time for runner)
        throw_dist = row.get("throw_distance_from_fielding", None)
        median_throw = singles["throw_distance_from_fielding"].median()
        if pd.notna(throw_dist) and pd.notna(median_throw):
            pct = compute_percentile(singles["throw_distance_from_fielding"], throw_dist)
            factor = {"name": "Throw Distance", "value": f"{throw_dist:.0f} ft",
                       "percentile": round(pct, 1), "pct_label": percentile_label(pct),
                       "pct_ordinal": ordinal(round(pct)),
                       "low_label": "Short", "high_label": "Long"}
            if throw_dist > median_throw * 1.05:
                factor["detail"] = "Longer throw gives runner more time"
                factors_send.append(factor)
            elif throw_dist < median_throw * 0.95:
                factor["detail"] = "Short throw reduces runner's window"
                factors_hold.append(factor)

        # Runner speed - faster = favors send
        speed = row.get("runner_speed", None)
        median_speed = singles["runner_speed"].median()
        if pd.notna(speed) and pd.notna(median_speed):
            pct = compute_percentile(singles["runner_speed"], speed)
            # Extract runner name from play description
            # Look for "Name to 3rd" - this is the 2B runner we're analyzing
            # NOT "Name scores" - that's the 3B runner
            runner_name = None
            desc = str(row.get("des", ""))
            # Match runner going to 3rd (the 2B runner)
            name_match = re.search(
                r'\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z\.\'\-]+)?)\s+to 3rd',
                desc
            )
            if name_match:
                runner_name = name_match.group(1)

            factor_name = f"Runner Speed ({runner_name})" if runner_name else "Runner Speed"
            factor = {"name": factor_name, "value": f"{speed:.1f} ft/s",
                       "percentile": round(pct, 1), "pct_label": percentile_label(pct),
                       "pct_ordinal": ordinal(round(pct)),
                       "low_label": "Slow", "high_label": "Fast"}
            if speed > median_speed * 1.02:
                factor["detail"] = "Above-average sprint speed"
                factors_send.append(factor)
            elif speed < median_speed * 0.98:
                factor["detail"] = "Below-average sprint speed"
                factors_hold.append(factor)

        # Fielder arm strength - weaker = favors send
        arm = row.get("fielder_arm_strength", None)
        median_arm = singles["fielder_arm_strength"].median()
        if pd.notna(arm) and pd.notna(median_arm):
            pct = compute_percentile(singles["fielder_arm_strength"], arm)
            # Extract fielder name from play description
            fielder_name = None
            desc = str(row.get("des", ""))
            name_match = re.search(
                r'(?:left|right|center) fielder ([A-Z][a-zA-Z\.\'\- ]+?)(?:\.|,|\s+scores|\s+to\s)',
                desc
            )
            if not name_match:
                name_match = re.search(
                    r'(?:shortstop|first baseman|second baseman|third baseman) ([A-Z][a-zA-Z\.\'\- ]+?)(?:\.|,|\s+scores|\s+to\s)',
                    desc
                )
            if name_match:
                fielder_name = name_match.group(1).strip()
            factor_name = f"Fielder Arm ({fielder_name})" if fielder_name else "Fielder Arm"
            factor = {"name": factor_name, "value": f"{arm:.1f} mph",
                       "percentile": round(pct, 1), "pct_label": percentile_label(pct),
                       "pct_ordinal": ordinal(round(pct)),
                       "low_label": "Weak", "high_label": "Strong"}
            if arm < median_arm * 0.97:
                factor["detail"] = "Weaker arm gives runner advantage"
                factors_send.append(factor)
            elif arm > median_arm * 1.03:
                factor["detail"] = "Strong arm reduces scoring chance"
                factors_hold.append(factor)

        # Scoring window - positive = favors send, negative = favors hold
        window = row.get("scoring_window", None)
        if pd.notna(window):
            pct = compute_percentile(singles["scoring_window"], window)
            factor = {"name": "Runner vs. Throw", "value": f"{window:+.2f}s",
                       "percentile": round(pct, 1) if pct is not None else None,
                       "pct_label": percentile_label(pct) if pct is not None else "",
                       "pct_ordinal": ordinal(round(pct)) if pct is not None else "",
                       "low_label": "Tight", "high_label": "Wide"}
            if window > 0.5:
                factor["detail"] = f"Estimated {window:.2f}s gap between the runner reaching home and the throw arriving. Based on runner speed, throw distance, and arm strength."
                factors_send.append(factor)
            elif window < 0:
                # Any negative window means throw arrives before runner
                factor["detail"] = f"Throw is predicted to arrive {abs(window):.2f}s before the runner. Based on runner speed, throw distance, and arm strength."
                factors_hold.append(factor)
            elif window < 0.3:
                # Close play - slight positive but very tight
                factor["detail"] = f"Very close timing - runner is predicted to beat the throw by only {window:.2f}s. Small margin for error."
                factors_hold.append(factor)

        # Scoring probability
        p_score = row.get("p_conservative", None)
        if pd.notna(p_score):
            pct = compute_percentile(singles["p_conservative"], p_score)
            factor = {"name": "Score Probability", "value": f"{p_score*100:.0f}%",
                       "percentile": round(pct, 1) if pct is not None else None,
                       "pct_label": percentile_label(pct) if pct is not None else "",
                       "pct_ordinal": ordinal(round(pct)) if pct is not None else "",
                       "low_label": "Low", "high_label": "High"}
            if p_score > 0.8:
                factor["detail"] = f"Model estimates a {p_score*100:.0f}% chance the runner scores if sent, using batted ball data, fielder positioning, runner speed, arm strength, and runner delay."
                factors_send.append(factor)
            elif p_score > 0.65:
                # Moderate probability - could go either way, add context
                factor["detail"] = f"Model estimates a {p_score*100:.0f}% chance the runner scores if sent. This is a contested play where outcome is uncertain."
                # Don't add to either - it's borderline
            elif p_score < 0.65:
                # Below 65% is lower than typical - add to hold factors
                factor["detail"] = f"Model estimates only a {p_score*100:.0f}% chance the runner scores if sent. This is below typical scoring probability on send plays."
                factors_hold.append(factor)

        # Game context - run value based on score differential
        run_value = row.get("run_value_wp", None)
        score_diff = row.get("score_diff", None)
        inning = row.get("inning", 5)
        median_rv = singles["run_value_wp"].median()
        if pd.notna(run_value) and pd.notna(score_diff) and pd.notna(median_rv):
            pct = compute_percentile(singles["run_value_wp"], run_value)
            diff_str = f"{'up' if score_diff > 0 else 'down'} {abs(int(score_diff))}" if score_diff != 0 else "tied"
            factor = {"name": "Run Value", "value": f"{run_value*100:.1f} pp",
                       "percentile": round(pct, 1), "pct_label": percentile_label(pct),
                       "pct_ordinal": ordinal(round(pct)),
                       "low_label": "Blowout", "high_label": "Close game"}
            if run_value > median_rv * 1.3:
                factor["detail"] = f"Score is {diff_str}. Each run is highly valuable at {run_value*100:.1f} pp of win probability, making the send worth the risk."
                factors_send.append(factor)
            elif run_value < median_rv * 0.4:
                # Low run value - but distinguish blowout from "already winning" situations
                # In late innings with close scores, low run value means excellent position
                is_close_game = abs(score_diff) <= 2
                is_late = inning >= 9
                if is_late and is_close_game:
                    # Don't add as hold factor - low run value here means good position
                    pass
                else:
                    factor["detail"] = f"Score is {diff_str}. Each run is worth only {run_value*100:.1f} pp of win probability, so the risk of an out outweighs the value of scoring."
                    factors_hold.append(factor)

        # Leverage index - high leverage doesn't favor either, but context matters
        li = row.get("leverage_index", None)
        if pd.notna(li) and li > 1.5:
            pct = compute_percentile(singles["leverage_index"], li)
            factor = {"name": "High Leverage", "value": f"LI {li:.2f}",
                       "percentile": round(pct, 1) if pct is not None else None,
                       "pct_label": percentile_label(pct) if pct is not None else "",
                       "pct_ordinal": ordinal(round(pct)) if pct is not None else "",
                       "low_label": "Low", "high_label": "High"}
            if row.get("should_send", False):
                factor["detail"] = "High-stakes moment favoring aggressive play"
                factors_send.append(factor)
            else:
                factor["detail"] = "High stakes — an out here is costly"
                factors_hold.append(factor)

    # Generate explanation summary based on factors and recommendation
    decision_explanation = None
    should_send = row.get("should_send", False)
    p_score = row.get("p_conservative", None)
    was_sent = row.get("was_sent", False)
    runner_out = row.get("runner_out_at_home", False)

    # Build list of strongest factors for the recommendation
    if should_send:
        # Factors favoring send - extract names and sort by importance
        strong_factors = []
        for f in factors_send:
            name = f["name"].split(" (")[0]  # Remove player name in parentheses
            if name == "Score Probability":
                strong_factors.insert(0, f"high scoring probability ({f['value']})")
            elif name == "Throw Distance":
                strong_factors.append(f"long throw distance ({f['value']})")
            elif name == "Runner Speed":
                strong_factors.append(f"above-average runner speed ({f['value']})")
            elif name == "Fielder Arm":
                strong_factors.append(f"below-average fielder arm ({f['value']})")
            elif name == "Runner vs. Throw":
                strong_factors.append(f"favorable timing window ({f['value']})")
            elif name == "Run Value":
                strong_factors.append(f"high run value in this game state")

        if strong_factors:
            factors_text = strong_factors[0] if len(strong_factors) == 1 else \
                          f"{', '.join(strong_factors[:-1])}, and {strong_factors[-1]}" if len(strong_factors) > 2 else \
                          f"{strong_factors[0]} and {strong_factors[1]}"

            if was_sent and runner_out:
                # Runner was sent and was out - explain why send was still reasonable
                decision_explanation = f"The model recommends sending due to {factors_text}. While the runner was thrown out on this play, historically 97% of runners score in similar situations. The send was statistically reasonable but had an unlucky outcome."
            elif not was_sent:
                # Runner was held - missed opportunity
                decision_explanation = f"The model recommends sending due to {factors_text}. The combination of these factors gives the runner a strong chance to score, making the aggressive play worthwhile."
            else:
                # Runner was sent and scored - correct send
                decision_explanation = f"The model recommends sending due to {factors_text}. These factors combine to give the runner a high probability of scoring safely."
        else:
            if pd.notna(p_score) and p_score > 0.9:
                decision_explanation = f"The model recommends sending based on the overall {p_score*100:.0f}% scoring probability, though no single factor stands out as exceptionally favorable."
    else:
        # Factors favoring hold
        strong_factors = []
        for f in factors_hold:
            name = f["name"].split(" (")[0]
            if name == "Score Probability":
                strong_factors.insert(0, f"low scoring probability ({f['value']})")
            elif name == "Throw Distance":
                strong_factors.append(f"short throw distance ({f['value']})")
            elif name == "Runner Speed":
                strong_factors.append(f"below-average runner speed ({f['value']})")
            elif name == "Fielder Arm":
                strong_factors.append(f"strong fielder arm ({f['value']})")
            elif name == "Runner vs. Throw":
                strong_factors.append(f"tight timing window ({f['value']})")
            elif name == "Run Value":
                strong_factors.append(f"low marginal run value ({f['value']})")

        if strong_factors:
            factors_text = strong_factors[0] if len(strong_factors) == 1 else \
                          f"{', '.join(strong_factors[:-1])}, and {strong_factors[-1]}" if len(strong_factors) > 2 else \
                          f"{strong_factors[0]} and {strong_factors[1]}"

            if was_sent and runner_out:
                decision_explanation = f"The model recommends holding due to {factors_text}. The runner was thrown out, confirming that the risk outweighed the potential reward."
            elif was_sent:
                decision_explanation = f"The model recommends holding due to {factors_text}. While the runner scored on this play, the decision carried more risk than was warranted by the win probability impact."
            else:
                decision_explanation = f"The model recommends holding due to {factors_text}. Keeping the runner at third preserves a scoring opportunity without risking an out."

    return render_template("play_detail.html", play=play, factors_send=factors_send, factors_hold=factors_hold, decision_explanation=decision_explanation)


@app.route("/play/<int:play_id>/similar")
def similar_plays(play_id):
    """Find plays with similar characteristics to the given play."""
    plays_df = DATA.get("plays", pd.DataFrame())

    if play_id not in plays_df.index:
        return render_template("404.html"), 404

    row = plays_df.loc[play_id]

    # Filter to singles only
    singles = plays_df[plays_df["event_type"] == "single"].copy()

    # Get the reference play's characteristics
    ref_launch_angle = row.get("launch_angle")
    ref_exit_velo = row.get("launch_speed")
    ref_arm_strength = row.get("fielder_arm_strength")
    ref_runner_speed = row.get("runner_speed")
    ref_hit_location = row.get("hit_location")  # 7=LF, 8=CF, 9=RF
    ref_outs = row.get("outs_when_up")
    ref_throw_dist = row.get("throw_distance_from_fielding")

    # Find similar plays using tolerance ranges
    similar_mask = pd.Series(True, index=singles.index)

    # Launch angle: ±5 degrees
    if pd.notna(ref_launch_angle):
        similar_mask &= (singles["launch_angle"] >= ref_launch_angle - 5) & \
                        (singles["launch_angle"] <= ref_launch_angle + 5)

    # Exit velocity: ±5 mph
    if pd.notna(ref_exit_velo):
        similar_mask &= (singles["launch_speed"] >= ref_exit_velo - 5) & \
                        (singles["launch_speed"] <= ref_exit_velo + 5)

    # Same outfield zone (LF/CF/RF)
    if pd.notna(ref_hit_location):
        similar_mask &= (singles["hit_location"] == ref_hit_location)

    # Fielder arm strength: ±5 mph
    if pd.notna(ref_arm_strength):
        similar_mask &= (singles["fielder_arm_strength"] >= ref_arm_strength - 5) & \
                        (singles["fielder_arm_strength"] <= ref_arm_strength + 5)

    # Runner speed: ±1.5 ft/s
    if pd.notna(ref_runner_speed):
        similar_mask &= (singles["runner_speed"] >= ref_runner_speed - 1.5) & \
                        (singles["runner_speed"] <= ref_runner_speed + 1.5)

    # Same number of outs
    if pd.notna(ref_outs):
        similar_mask &= (singles["outs_when_up"] == ref_outs)

    # Exclude the reference play itself
    similar_mask &= (singles.index != play_id)

    similar_df = singles[similar_mask].copy()

    # If too few results, relax constraints
    if len(similar_df) < 5:
        # Relax: remove outs constraint
        similar_mask = pd.Series(True, index=singles.index)
        if pd.notna(ref_launch_angle):
            similar_mask &= (singles["launch_angle"] >= ref_launch_angle - 7) & \
                            (singles["launch_angle"] <= ref_launch_angle + 7)
        if pd.notna(ref_exit_velo):
            similar_mask &= (singles["launch_speed"] >= ref_exit_velo - 7) & \
                            (singles["launch_speed"] <= ref_exit_velo + 7)
        if pd.notna(ref_hit_location):
            similar_mask &= (singles["hit_location"] == ref_hit_location)
        if pd.notna(ref_runner_speed):
            similar_mask &= (singles["runner_speed"] >= ref_runner_speed - 2) & \
                            (singles["runner_speed"] <= ref_runner_speed + 2)
        similar_mask &= (singles.index != play_id)
        similar_df = singles[similar_mask].copy()

    # Sort by similarity score (weighted distance)
    if len(similar_df) > 0:
        # Calculate similarity score
        scores = pd.Series(0.0, index=similar_df.index)
        if pd.notna(ref_launch_angle):
            scores += ((similar_df["launch_angle"] - ref_launch_angle) / 5) ** 2
        if pd.notna(ref_exit_velo):
            scores += ((similar_df["launch_speed"] - ref_exit_velo) / 5) ** 2
        if pd.notna(ref_arm_strength) and "fielder_arm_strength" in similar_df.columns:
            arm_diff = (similar_df["fielder_arm_strength"].fillna(ref_arm_strength) - ref_arm_strength) / 5
            scores += arm_diff ** 2
        if pd.notna(ref_runner_speed):
            scores += ((similar_df["runner_speed"] - ref_runner_speed) / 1.5) ** 2
        if pd.notna(ref_throw_dist) and "throw_distance_from_fielding" in similar_df.columns:
            throw_diff = (similar_df["throw_distance_from_fielding"].fillna(ref_throw_dist) - ref_throw_dist) / 20
            scores += throw_diff ** 2

        similar_df["similarity_score"] = scores
        similar_df = similar_df.sort_values("similarity_score").head(20)

    # Calculate outcome summary (before filtering)
    total_similar = len(similar_df)
    if total_similar > 0:
        sent_count = int(similar_df["was_sent"].sum())
        held_count = total_similar - sent_count
        scored_count = int(similar_df["runner_2b_scored"].sum())
        out_count = int(similar_df["runner_out_at_home"].sum())

        # Success rate when sent
        sent_plays = similar_df[similar_df["was_sent"] == True]
        success_rate = sent_plays["runner_2b_scored"].mean() * 100 if len(sent_plays) > 0 else 0
    else:
        sent_count = held_count = scored_count = out_count = 0
        success_rate = 0

    # Get outcome filter parameter
    outcome_filter = request.args.get("outcome", "all")
    outcome_labels = {
        "all": "All Plays",
        "scored": "Scored",
        "out": "Out at Home",
        "held": "Held at 3rd"
    }
    current_outcome_label = outcome_labels.get(outcome_filter, "All Plays")

    # Filter by outcome if specified
    if outcome_filter == "scored":
        similar_df = similar_df[similar_df["runner_2b_scored"] == True]
    elif outcome_filter == "out":
        similar_df = similar_df[similar_df["runner_out_at_home"] == True]
    elif outcome_filter == "held":
        similar_df = similar_df[(similar_df["was_sent"] == False) & (similar_df["runner_out_at_home"] == False)]

    # Prepare play data for display
    similar_plays_list = []
    for idx, srow in similar_df.iterrows():
        play_data = prepare_play_data(srow)
        play_data["id"] = idx
        # Add outcome indicator
        if srow.get("runner_out_at_home", False):
            play_data["outcome"] = "Out at Home"
            play_data["outcome_class"] = "danger"
        elif srow.get("runner_2b_scored", False):
            play_data["outcome"] = "Scored"
            play_data["outcome_class"] = "success"
        else:
            play_data["outcome"] = "Held at 3rd"
            play_data["outcome_class"] = "warning"
        similar_plays_list.append(play_data)

    # Prepare reference play data
    ref_play = prepare_play_data(row)
    ref_play["id"] = play_id

    # Field position name
    field_names = {7: "Left Field", 8: "Center Field", 9: "Right Field"}
    field_name = field_names.get(ref_hit_location, "Outfield")

    return render_template(
        "similar_plays.html",
        ref_play=ref_play,
        similar_plays=similar_plays_list,
        total_similar=total_similar,
        sent_count=sent_count,
        held_count=held_count,
        scored_count=scored_count,
        out_count=out_count,
        success_rate=success_rate,
        field_name=field_name,
        ref_launch_angle=ref_launch_angle,
        ref_exit_velo=ref_exit_velo,
        ref_arm_strength=ref_arm_strength,
        ref_runner_speed=ref_runner_speed,
        ref_outs=ref_outs,
        current_outcome=outcome_filter,
        current_outcome_label=current_outcome_label,
    )


@app.route("/api/plays")
def api_plays():
    """API endpoint for plays data (for AJAX loading)."""
    plays_df = DATA.get("plays", pd.DataFrame())

    if len(plays_df) == 0:
        return jsonify({"plays": [], "total": 0})

    # Filter to singles
    plays_df = plays_df[plays_df["event_type"] == "single"].copy()

    # Get parameters
    team = request.args.get("team", "all")
    decision = request.args.get("decision", "all")
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))

    # Apply filters
    if team != "all":
        plays_df = plays_df[plays_df["batting_team"] == team]

    if decision == "missed":
        plays_df = plays_df[plays_df["coach_decision_quality"] == "missed_opportunity"]
    elif decision == "bad_send":
        plays_df = plays_df[plays_df["coach_decision_quality"] == "bad_send"]

    # Sort by WP delta
    plays_df = plays_df.sort_values("WP_delta", ascending=False)

    total = len(plays_df)
    plays_df = plays_df.iloc[offset:offset+limit]

    plays_list = []
    for idx, row in plays_df.iterrows():
        play = prepare_play_data(row)
        play["id"] = int(idx)
        plays_list.append(play)

    return jsonify({"plays": plays_list, "total": total})


@app.route("/api/team/<team_code>")
def api_team(team_code):
    """API endpoint for team-specific data."""
    plays_df = DATA.get("plays", pd.DataFrame())
    teams_df = DATA.get("teams", pd.DataFrame())

    if len(plays_df) == 0:
        return jsonify({"error": "No data available"}), 404

    # Get team plays
    team_plays = plays_df[
        (plays_df["batting_team"] == team_code) &
        (plays_df["event_type"] == "single")
    ]

    if len(team_plays) == 0:
        return jsonify({"error": "Team not found"}), 404

    # Calculate stats
    stats = {
        "team": team_code,
        "total_plays": len(team_plays),
        "plays_sent": int(team_plays["was_sent"].sum()),
        "success_rate": float(team_plays[team_plays["was_sent"]]["sent_success"].mean()),
        "missed_opportunities": int((team_plays["coach_decision_quality"] == "missed_opportunity").sum()),
        "bad_sends": int((team_plays["coach_decision_quality"] == "bad_send").sum()),
        "avg_wp_delta": float(team_plays["WP_delta"].mean()),
    }

    # Get worst plays
    worst_holds = team_plays[
        team_plays["coach_decision_quality"] == "missed_opportunity"
    ].nlargest(5, "WP_delta")

    worst_sends = team_plays[
        team_plays["coach_decision_quality"] == "bad_send"
    ].nsmallest(5, "WP_delta")

    stats["worst_holds"] = [prepare_play_data(row) for _, row in worst_holds.iterrows()]
    stats["worst_sends"] = [prepare_play_data(row) for _, row in worst_sends.iterrows()]

    return jsonify(stats)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
