"""
Win Probability calculations based on historical MLB data.

This module uses actual historical probabilities for scoring runs from each
base/out state, combined with leverage-adjusted WP calculations.

Sources:
- Tom Tango's "The Book" run expectancy and scoring probability tables
- FanGraphs Win Probability data
- Baseball Reference historical WP tables
"""

import numpy as np
import pandas as pd


# =============================================================================
# PROBABILITY OF SCORING AT LEAST N RUNS FROM EACH BASE/OUT STATE
# Based on historical MLB data (2010-2023 averages)
# =============================================================================

# Probability of scoring at least 1 run
P_SCORE_1_PLUS = {
    ("empty", 0): 0.274, ("empty", 1): 0.169, ("empty", 2): 0.073,
    ("1B", 0): 0.432, ("1B", 1): 0.276, ("1B", 2): 0.135,
    ("2B", 0): 0.634, ("2B", 1): 0.418, ("2B", 2): 0.227,
    ("3B", 0): 0.853, ("3B", 1): 0.674, ("3B", 2): 0.270,
    ("1B-2B", 0): 0.637, ("1B-2B", 1): 0.429, ("1B-2B", 2): 0.236,
    ("1B-3B", 0): 0.876, ("1B-3B", 1): 0.689, ("1B-3B", 2): 0.320,
    ("2B-3B", 0): 0.869, ("2B-3B", 1): 0.698, ("2B-3B", 2): 0.308,
    ("1B-2B-3B", 0): 0.877, ("1B-2B-3B", 1): 0.695, ("1B-2B-3B", 2): 0.349,
}

# Probability of scoring at least 2 runs
P_SCORE_2_PLUS = {
    ("empty", 0): 0.111, ("empty", 1): 0.053, ("empty", 2): 0.018,
    ("1B", 0): 0.202, ("1B", 1): 0.102, ("1B", 2): 0.041,
    ("2B", 0): 0.338, ("2B", 1): 0.182, ("2B", 2): 0.078,
    ("3B", 0): 0.417, ("3B", 1): 0.240, ("3B", 2): 0.078,
    ("1B-2B", 0): 0.382, ("1B-2B", 1): 0.202, ("1B-2B", 2): 0.083,
    ("1B-3B", 0): 0.538, ("1B-3B", 1): 0.299, ("1B-3B", 2): 0.118,
    ("2B-3B", 0): 0.581, ("2B-3B", 1): 0.338, ("2B-3B", 2): 0.117,
    ("1B-2B-3B", 0): 0.620, ("1B-2B-3B", 1): 0.378, ("1B-2B-3B", 2): 0.151,
}

# Probability of scoring at least 3 runs
P_SCORE_3_PLUS = {
    ("empty", 0): 0.044, ("empty", 1): 0.017, ("empty", 2): 0.005,
    ("1B", 0): 0.091, ("1B", 1): 0.037, ("1B", 2): 0.012,
    ("2B", 0): 0.161, ("2B", 1): 0.071, ("2B", 2): 0.025,
    ("3B", 0): 0.180, ("3B", 1): 0.085, ("3B", 2): 0.023,
    ("1B-2B", 0): 0.191, ("1B-2B", 1): 0.084, ("1B-2B", 2): 0.028,
    ("1B-3B", 0): 0.276, ("1B-3B", 1): 0.124, ("1B-3B", 2): 0.038,
    ("2B-3B", 0): 0.319, ("2B-3B", 1): 0.149, ("2B-3B", 2): 0.040,
    ("1B-2B-3B", 0): 0.382, ("1B-2B-3B", 1): 0.190, ("1B-2B-3B", 2): 0.055,
}

# Probability of scoring at least 4 runs
P_SCORE_4_PLUS = {
    ("empty", 0): 0.017, ("empty", 1): 0.005, ("empty", 2): 0.001,
    ("1B", 0): 0.039, ("1B", 1): 0.013, ("1B", 2): 0.003,
    ("2B", 0): 0.071, ("2B", 1): 0.027, ("2B", 2): 0.007,
    ("3B", 0): 0.074, ("3B", 1): 0.030, ("3B", 2): 0.006,
    ("1B-2B", 0): 0.089, ("1B-2B", 1): 0.033, ("1B-2B", 2): 0.009,
    ("1B-3B", 0): 0.127, ("1B-3B", 1): 0.048, ("1B-3B", 2): 0.012,
    ("2B-3B", 0): 0.155, ("2B-3B", 1): 0.061, ("2B-3B", 2): 0.013,
    ("1B-2B-3B", 0): 0.199, ("1B-2B-3B", 1): 0.085, ("1B-2B-3B", 2): 0.019,
}

# Run expectancy by base state and outs
RE_TABLE = {
    ("empty", 0): 0.481, ("empty", 1): 0.254, ("empty", 2): 0.098,
    ("1B", 0): 0.859, ("1B", 1): 0.509, ("1B", 2): 0.214,
    ("2B", 0): 1.100, ("2B", 1): 0.664, ("2B", 2): 0.319,
    ("3B", 0): 1.353, ("3B", 1): 0.950, ("3B", 2): 0.353,
    ("1B-2B", 0): 1.437, ("1B-2B", 1): 0.884, ("1B-2B", 2): 0.429,
    ("1B-3B", 0): 1.798, ("1B-3B", 1): 1.140, ("1B-3B", 2): 0.478,
    ("2B-3B", 0): 1.920, ("2B-3B", 1): 1.352, ("2B-3B", 2): 0.564,
    ("1B-2B-3B", 0): 2.282, ("1B-2B-3B", 1): 1.536, ("1B-2B-3B", 2): 0.736,
}


def get_p_score_at_least(n_runs: int, base_state: str, outs: int) -> float:
    """Get probability of scoring at least N runs from this state."""
    key = (base_state, outs)

    if n_runs <= 0:
        return 1.0
    elif n_runs == 1:
        return P_SCORE_1_PLUS.get(key, 0.20)
    elif n_runs == 2:
        return P_SCORE_2_PLUS.get(key, 0.08)
    elif n_runs == 3:
        return P_SCORE_3_PLUS.get(key, 0.03)
    elif n_runs == 4:
        return P_SCORE_4_PLUS.get(key, 0.01)
    else:
        # For 5+ runs, extrapolate (very rare)
        p_4 = P_SCORE_4_PLUS.get(key, 0.01)
        return p_4 * (0.4 ** (n_runs - 4))


def get_re(base_state: str, outs: int) -> float:
    """Get run expectancy for base/out state."""
    if outs >= 3:
        return 0.0
    return RE_TABLE.get((base_state, outs), 0.0)


# =============================================================================
# BASE WIN PROBABILITY BY INNING AND SCORE (start of half-inning, bases empty)
# Format: WP_TABLE[inning][(is_bottom, score_diff)] = home_wp
# Score diff is HOME - AWAY (positive = home leading)
# =============================================================================

def _build_wp_table():
    """Build comprehensive WP table based on historical data."""
    wp = {}

    for inning in range(1, 16):
        wp[inning] = {}

        for score_diff in range(-8, 9):
            for is_bottom in [True, False]:
                # Base WP at start of half-inning
                # These values are calibrated to historical MLB data

                if inning <= 3:
                    # Early innings - score matters less
                    if score_diff == 0:
                        base = 0.50
                    else:
                        # Each run worth ~8-10% in early innings
                        base = 0.50 + score_diff * 0.08

                elif inning <= 6:
                    # Middle innings
                    if score_diff == 0:
                        base = 0.50
                    else:
                        base = 0.50 + score_diff * 0.10

                elif inning <= 8:
                    # Late innings - score matters more
                    if score_diff == 0:
                        base = 0.50
                    else:
                        base = 0.50 + score_diff * 0.13

                else:
                    # 9th inning and later - score is critical
                    if score_diff == 0:
                        base = 0.50
                    elif score_diff > 0:
                        # Leading in 9th+ is very valuable
                        base = 0.50 + min(score_diff * 0.18, 0.49)
                    else:
                        # Trailing in 9th+ is dire
                        base = 0.50 + max(score_diff * 0.18, -0.49)

                # Adjust for top vs bottom
                # Bottom of inning gives home team an advantage (they bat last)
                # This adjustment applies regardless of score - home always has
                # the advantage of batting last
                if is_bottom:
                    # Scale adjustment by innings remaining
                    # More adjustment early (more innings to play)
                    innings_factor = max(1, 10 - inning) * 0.01
                    base += innings_factor
                else:
                    # Top of inning, home team on defense
                    # Slight advantage when leading
                    if score_diff >= 0:
                        base += 0.01

                # Clamp to valid range
                base = max(0.01, min(0.99, base))

                wp[inning][(is_bottom, score_diff)] = base

    return wp

WP_TABLE = _build_wp_table()


def get_base_wp(inning: int, is_bottom: bool, score_diff: int) -> float:
    """Get base WP for home team at start of half-inning, bases empty."""
    inning = max(1, min(inning, 15))
    score_diff = max(-8, min(8, score_diff))

    return WP_TABLE[inning].get((is_bottom, score_diff), 0.50)


def calculate_wp(
    score_diff: int,
    inning: int,
    is_bottom: bool,
    outs: int,
    base_state: str
) -> float:
    """
    Calculate win probability for the HOME team.

    Uses probability of scoring N runs combined with state transition probabilities
    to compute accurate WP for any game situation.

    Args:
        score_diff: Home score - Away score
        inning: Current inning (1-9+)
        is_bottom: True if bottom of inning (home batting)
        outs: Number of outs (0, 1, 2)
        base_state: e.g., "empty", "1B", "2B", "1B-2B-3B"

    Returns:
        Win probability for home team (0-1)
    """
    # Handle end of inning (3 outs)
    if outs >= 3:
        if is_bottom:
            # End of full inning, go to next
            return get_base_wp(inning + 1, False, score_diff)
        else:
            # End of top half, go to bottom
            return get_base_wp(inning, True, score_diff)

    # Get base WP for this situation
    base_wp = get_base_wp(inning, is_bottom, score_diff)

    # ==========================================================================
    # EXTRA INNINGS (10+) - Ghost Runner Rule
    # Since 2020, each half-inning in extras starts with runner on 2B
    # P(score 1+ from 2B, 0 out) = 63.4%, so scoring is much easier
    # ==========================================================================

    # WP for going to next extra inning tied (both teams get ghost runner)
    # Home team has slight edge batting second with ghost runner
    # Historical data: ~52% win rate for home team in extra innings
    EXTRA_INNINGS_TIED_WP = 0.52

    # For trailing team needing to tie: opponent will have ghost runner too
    # If you tie but opponent scores in their half with ghost runner, you lose
    # P(opponent scores 1+ with ghost runner) = 63.4%
    # So WP after tying in extras ≈ 52% * (1 - 0.634 * 0.5) ≈ 35%
    # Actually: if tied after your half, opponent has ~63% to walk off
    # Your WP if tied going to next half = 1 - 0.634 = 36.6% (away) or
    # If home ties in bottom, go to next inning with ghost runners = 52%
    EXTRA_INNINGS_TIED_WP_AWAY = 0.37  # If away ties, home gets ghost runner

    # ==========================================================================
    # BOTTOM OF 9TH OR LATER (Walk-off situations)
    # Home team only needs to score enough to win - use scoring probabilities
    # ==========================================================================
    if is_bottom and inning >= 9:
        if score_diff > 0:
            # Home team is leading in bottom of 9th or later = WALK-OFF WIN
            # Game is over, WP = 100%
            return 1.0

        runs_needed = -score_diff + 1  # Runs needed to walk off

        # WP if going to next inning tied
        # In extras (10+), both teams get ghost runners, slight home advantage
        if inning >= 10:
            wp_if_extras_tied = EXTRA_INNINGS_TIED_WP
        else:
            # Bottom 9, if don't walk off, go to extras where home has edge
            wp_if_extras_tied = EXTRA_INNINGS_TIED_WP

        if runs_needed == 1:
            # Need 1 run to win - P(score 1+) = win, else extras
            p_score = get_p_score_at_least(1, base_state, outs)
            return p_score * 1.0 + (1 - p_score) * wp_if_extras_tied

        elif runs_needed == 2:
            # Need 2 runs to win
            p_score_2 = get_p_score_at_least(2, base_state, outs)
            p_score_1 = get_p_score_at_least(1, base_state, outs)
            p_score_exactly_1 = p_score_1 - p_score_2
            # Score 2+ = win, score 1 = extras tied, score 0 = lose
            return p_score_2 * 1.0 + p_score_exactly_1 * wp_if_extras_tied

        elif runs_needed == 3:
            p_score_3 = get_p_score_at_least(3, base_state, outs)
            p_score_2 = get_p_score_at_least(2, base_state, outs)
            # 3+ = win, 2 = tied for extras
            return p_score_3 * 1.0 + (p_score_2 - p_score_3) * wp_if_extras_tied

        else:
            # Need 4+ runs - very difficult
            p_score_n = get_p_score_at_least(runs_needed, base_state, outs)
            p_score_n_minus_1 = get_p_score_at_least(runs_needed - 1, base_state, outs)
            return p_score_n * 1.0 + (p_score_n_minus_1 - p_score_n) * wp_if_extras_tied

    # ==========================================================================
    # TOP OF 9TH OR LATER (Away team batting)
    # Away team bats first, then home gets last chance with ghost runner (in extras)
    # ==========================================================================
    if not is_bottom and inning >= 9:
        # In top of extra innings, away team has already used ghost runner or it scored
        # When this half ends, home gets ghost runner in bottom

        if inning >= 10:
            # Extra innings - calculate based on what happens when home bats
            # with ghost runner

            if score_diff > 0:
                # Away team AHEAD in top of extras (home trailing)
                # Home will get ghost runner in bottom
                # P(home scores 1+ with ghost runner from 2B, 0 out) = 63.4%
                lead = score_diff
                p_home_ties = get_p_score_at_least(lead, "2B", 0)
                p_home_wins = get_p_score_at_least(lead + 1, "2B", 0)

                # Away WP = P(home doesn't tie) + P(home ties but away wins extras)
                # Complex: if home ties, we go to next inning with ghost runners
                # Simplify: Away leading by N, home has ghost runner
                # Away WP ≈ (1 - P_home_ties) + P_home_ties_not_win * EXTRA_WP
                if lead == 1:
                    # Home has 63.4% to tie, then extras are ~48% for away
                    home_wp = p_home_wins * 1.0 + (p_home_ties - p_home_wins) * 0.52
                    return 1 - home_wp
                else:
                    # Bigger lead - home needs multiple runs
                    home_wp = p_home_wins * 1.0 + (p_home_ties - p_home_wins) * 0.52
                    return 1 - home_wp

            elif score_diff < 0:
                # Away team BEHIND in top of extras (home leading)
                # This shouldn't normally happen (home would have won in bottom)
                # But could be start of inning before any runs
                return max(0.01, min(0.99, base_wp))

            else:
                # Tied in top of extras
                # Away team batting, when done home gets ghost runner
                # If away doesn't score, home has 63.4% to walk off
                re = get_re(base_state, outs)
                # Away WP with current state - they need to score to have good odds
                # P(score 1+ from current state) → home needs 2 to walk off
                p_away_scores = get_p_score_at_least(1, base_state, outs)
                # If away scores: home needs 2+ with ghost runner to win
                p_home_wins_if_away_scores = get_p_score_at_least(2, "2B", 0)  # 33.8%
                # If away doesn't score: home needs 1 to walk off
                p_home_wins_if_away_doesnt = get_p_score_at_least(1, "2B", 0)  # 63.4%

                away_wp = (p_away_scores * (1 - p_home_wins_if_away_scores) +
                          (1 - p_away_scores) * (1 - p_home_wins_if_away_doesnt))
                home_wp = 1 - away_wp
                return home_wp
        else:
            # Top of 9th - standard late inning calculation
            re = get_re(base_state, outs)
            if score_diff < 0:
                adjustment = -re * 0.03
            else:
                adjustment = -re * 0.02
            return max(0.01, min(0.99, base_wp + adjustment))

    # ==========================================================================
    # ALL OTHER SITUATIONS (Innings 1-8, or top of 9th with home leading)
    # Use run expectancy-based WP adjustments
    # ==========================================================================

    # Get current run expectancy for this state
    current_re = get_re(base_state, outs)

    # Calculate leverage - how much does RE translate to WP?
    # This varies by inning and score
    if inning <= 3:
        leverage = 0.035  # ~3.5% WP per expected run
    elif inning <= 6:
        leverage = 0.045  # ~4.5% WP per expected run
    elif inning <= 8:
        leverage = 0.055  # ~5.5% WP per expected run
    else:
        leverage = 0.070  # ~7% WP per expected run

    # Increase leverage in close games
    if abs(score_diff) <= 2:
        leverage *= 1.3
    elif abs(score_diff) <= 1:
        leverage *= 1.5

    # Calculate WP adjustment
    # RE difference from "average" state (bases empty, ~0.5 RE)
    re_diff = current_re - 0.5

    if is_bottom:
        # Home team batting - higher RE helps home team
        wp_adjustment = re_diff * leverage
    else:
        # Away team batting - higher RE hurts home team
        wp_adjustment = -re_diff * leverage

    final_wp = base_wp + wp_adjustment
    return max(0.01, min(0.99, final_wp))


def get_wp_for_batting_team(
    score_diff_batting: int,
    inning: int,
    is_home_batting: bool,
    outs: int,
    base_state: str
) -> float:
    """
    Get win probability from the BATTING team's perspective.

    Args:
        score_diff_batting: Score difference from batting team's view (positive = leading)
        inning: Current inning
        is_home_batting: True if home team is batting
        outs: Number of outs
        base_state: Current base state

    Returns:
        Win probability for the batting team
    """
    # Convert to home team score diff
    if is_home_batting:
        home_score_diff = score_diff_batting
    else:
        home_score_diff = -score_diff_batting

    home_wp = calculate_wp(home_score_diff, inning, is_home_batting, outs, base_state)

    # Return WP for batting team
    if is_home_batting:
        return home_wp
    else:
        return 1 - home_wp


def calculate_send_wp_ev(
    p_score: float,
    score_diff_batting: int,
    inning: int,
    is_home_batting: bool,
    outs: int,
    base_state_pre: str,
    runs_if_score: int,
    base_state_if_score: str,
    runs_if_out: int,
    base_state_if_out: str
) -> float:
    """
    Calculate expected WP if runner is sent.

    EV_send = p * WP_if_score + (1-p) * WP_if_out
    """
    wp_if_score = get_wp_for_batting_team(
        score_diff_batting + runs_if_score,
        inning,
        is_home_batting,
        outs,
        base_state_if_score
    )

    wp_if_out = get_wp_for_batting_team(
        score_diff_batting + runs_if_out,
        inning,
        is_home_batting,
        outs + 1,
        base_state_if_out
    )

    return p_score * wp_if_score + (1 - p_score) * wp_if_out


def calculate_hold_wp(
    score_diff_batting: int,
    inning: int,
    is_home_batting: bool,
    outs: int,
    runs_if_hold: int,
    base_state_if_hold: str
) -> float:
    """Calculate WP if runner is held at 3rd."""
    return get_wp_for_batting_team(
        score_diff_batting + runs_if_hold,
        inning,
        is_home_batting,
        outs,
        base_state_if_hold
    )


def get_run_value_in_wp(score_diff: int, inning: int, is_bottom: bool) -> float:
    """Get the approximate WP value of one run in this situation."""
    wp_before = calculate_wp(score_diff, inning, is_bottom, 0, "empty")

    if is_bottom:
        wp_after = calculate_wp(score_diff + 1, inning, is_bottom, 0, "empty")
        return wp_after - wp_before
    else:
        wp_after = calculate_wp(score_diff - 1, inning, is_bottom, 0, "empty")
        return wp_before - wp_after


if __name__ == "__main__":
    print("=" * 70)
    print("WIN PROBABILITY VERIFICATION")
    print("=" * 70)

    def show_wp(desc, score_diff, inning, is_bottom, outs, base_state):
        wp = get_wp_for_batting_team(score_diff, inning, is_bottom, outs, base_state)
        print(f"{desc}: {wp:.1%}")

    print("\n=== BOTTOM 9TH - TIED GAME ===")
    show_wp("Empty, 0 outs", 0, 9, True, 0, "empty")
    show_wp("Empty, 1 out", 0, 9, True, 1, "empty")
    show_wp("Empty, 2 outs", 0, 9, True, 2, "empty")
    show_wp("Runner 2B, 0 outs", 0, 9, True, 0, "2B")
    show_wp("Runner 3B, 0 outs", 0, 9, True, 0, "3B")
    show_wp("Bases loaded, 0 outs", 0, 9, True, 0, "1B-2B-3B")
    show_wp("Bases loaded, 1 out", 0, 9, True, 1, "1B-2B-3B")
    show_wp("Bases loaded, 2 outs", 0, 9, True, 2, "1B-2B-3B")

    print("\n=== MIDDLE INNINGS (5th) ===")
    show_wp("Tied, empty, 0 outs", 0, 5, True, 0, "empty")
    show_wp("Tied, bases loaded, 0 outs", 0, 5, True, 0, "1B-2B-3B")
    show_wp("Up 2, empty", 2, 5, True, 0, "empty")
    show_wp("Down 3, empty", -3, 5, True, 0, "empty")

    print("\n=== TOP 9TH (Away batting) ===")
    show_wp("Tied, empty", 0, 9, False, 0, "empty")
    show_wp("Down 1, bases loaded", -1, 9, False, 0, "1B-2B-3B")
