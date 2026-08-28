"""
simulation_engine.py — Motor de simulación What-If y Path to Frontier
"""
import numpy as np
import pandas as pd
from dea_logic import perform_full_dea_analysis
from Pyfrontier.frontier_model import EnvelopDEA

DEA_MIN_POSITIVE_VALUE = 1e-5
MAX_ADJUSTMENT_PCT = 0.15  # 15% per quarter


def simulate_what_if(df_analysis, dmu_name, input_cols, output_cols,
                     targets, orientation='input', undesirable_outputs=None,
                     normalize=False):
    """
    Simulate changes for a single DMU and compute projected score + path to frontier.

    Args:
        df_analysis: Full DataFrame with all DMUs (indexed by DMU name).
        dmu_name: Name of the DMU to simulate.
        input_cols: List of input column names.
        output_cols: List of output column names.
        targets: Dict of {var_name: target_value}.
        orientation: 'input' or 'output'.
        undesirable_outputs: List of undesirable output column names.
        normalize: Whether normalization was applied.

    Returns:
        dict with projected_score, gain_pct, path_steps, new_peer_group,
        feasible, violations, current_score.
    """
    # --- Guard Clauses ---
    if dmu_name not in df_analysis.index:
        return {'feasible': False, 'violations': [f"DMU '{dmu_name}' not found."]}
    if not input_cols or not output_cols:
        return {'feasible': False, 'violations': ['No input or output columns specified.']}
    if not targets:
        return {'feasible': False, 'violations': ['No targets provided.']}

    # Get current DMU row
    dmu_row = df_analysis.loc[dmu_name]
    current_values = {}
    for col in input_cols + output_cols:
        if col in dmu_row.index:
            current_values[col] = float(dmu_row[col])

    # Validate targets
    violations = []
    all_vars = input_cols + output_cols
    for var, val in targets.items():
        if var not in all_vars:
            violations.append(f"Variable '{var}' is not a valid input/output.")
        elif val < 0:
            violations.append(f"Target for '{var}' cannot be negative ({val}).")

    if violations:
        return {'feasible': False, 'violations': violations, 'current_score': 0}

    # Run baseline DEA to get current score
    current_score = _compute_single_dmu_score(
        df_analysis, dmu_name, input_cols, output_cols, orientation,
        undesirable_outputs, normalize
    )

    # Build the simulated DMU row with target values
    simulated_row = dmu_row.copy()
    for var, val in targets.items():
        simulated_row[var] = val

    # Create a modified DataFrame with the simulated DMU
    df_simulated = df_analysis.copy()
    df_simulated.loc[dmu_name] = simulated_row

    # Run DEA on the simulated data to get projected score
    projected_score = _compute_single_dmu_score(
        df_simulated, dmu_name, input_cols, output_cols, orientation,
        undesirable_outputs, normalize
    )

    # Calculate gain
    if current_score > 1e-9:
        gain_pct = ((projected_score - current_score) / current_score) * 100
    else:
        gain_pct = 0.0

    # Calculate path to frontier (step-by-step trajectory)
    path_steps = calculate_path_to_frontier(
        current_values, targets, max_step_pct=MAX_ADJUSTMENT_PCT
    )

    # Estimate scores at each step via linear interpolation
    if current_score > 1e-9 and projected_score > 1e-9:
        for i, step in enumerate(path_steps):
            t = (i + 1) / len(path_steps) if path_steps else 1.0
            step['estimated_score'] = current_score + t * (projected_score - current_score)
    else:
        for step in path_steps:
            step['estimated_score'] = None

    # Determine new peer group from simulated lambdas
    new_peer_group = _get_peer_group(df_simulated, dmu_name, input_cols, output_cols,
                                     orientation, undesirable_outputs, normalize)

    return {
        'feasible': True,
        'violations': [],
        'current_score': round(current_score, 4),
        'projected_score': round(projected_score, 4),
        'gain_pct': round(gain_pct, 2),
        'path_steps': path_steps,
        'new_peer_group': new_peer_group,
    }


def calculate_path_to_frontier(current_values, target_values, max_step_pct=0.15):
    """
    Calculate step-by-step path from current to target values.

    Args:
        current_values: Dict of current variable values.
        target_values: Dict of target variable values.
        max_step_pct: Maximum adjustment per step (default 15%).

    Returns:
        List of step dicts with step_number, adjustments, and cumulative pct.
    """
    steps = []
    remaining = {}

    # Calculate required changes
    for var, target in target_values.items():
        if var not in current_values:
            continue
        current = current_values[var]
        diff = target - current
        remaining[var] = diff

    # Check if any adjustment needed
    if all(abs(v) < 1e-9 for v in remaining.values()):
        return [{'step_number': 1, 'adjustments': {}, 'cumulative_pct': 0,
                 'description': 'Already at target'}]

    step_num = 0
    max_remaining_pct = 1.0

    while max_remaining_pct > 1e-6 and step_num < 50:  # Safety limit
        step_num += 1
        adjustments = {}
        step_has_change = False

        for var, diff in remaining.items():
            if abs(diff) < 1e-9:
                continue

            current = current_values[var]
            if abs(current) < 1e-9:
                # Can't calculate pct for zero values; apply absolute step
                step_size = diff * max_step_pct
                adjustments[var] = round(step_size, 6)
                remaining[var] -= step_size
                step_has_change = True
            else:
                pct_of_total = abs(diff) / abs(current) if abs(current) > 0 else 0
                step_pct = min(pct_of_total, max_step_pct)
                step_size = current * step_pct * np.sign(diff)
                adjustments[var] = round(step_size, 6)
                remaining[var] -= step_size
                step_has_change = True

        if not step_has_change:
            break

        # Update current values for next step calculation
        for var in adjustments:
            current_values[var] = current_values.get(var, 0) + adjustments[var]

        # Calculate max remaining percentage
        max_remaining_pct = max(
            abs(remaining.get(v, 0)) / abs(target_values.get(v, 1))
            for v in target_values if v in remaining
        ) if target_values else 0

        steps.append({
            'step_number': step_num,
            'adjustments': adjustments,
            'cumulative_pct': round((1 - max_remaining_pct) * 100, 1),
            'description': _describe_step(adjustments),
        })

    return steps


def _compute_single_dmu_score(df, dmu_name, input_cols, output_cols,
                              orientation, undesirable_outputs=None, normalize=False):
    """Run full DEA and extract the score for a single DMU."""
    try:
        results, _ = perform_full_dea_analysis(
            df, df.index.name or 'DMU',
            input_cols, output_cols, orientation,
            undesirable_outputs or [], normalize
        )
        scores_df = results['scores']
        if dmu_name in scores_df.index:
            score = scores_df.loc[dmu_name, 'Score']
            return float(score) if not pd.isna(score) else 0.0
        return 0.0
    except Exception:
        return 0.0


def _get_peer_group(df, dmu_name, input_cols, output_cols,
                    orientation, undesirable_outputs=None, normalize=False):
    """Extract the peer group (lambdas > 0) for a DMU."""
    try:
        results, _ = perform_full_dea_analysis(
            df, df.index.name or 'DMU',
            input_cols, output_cols, orientation,
            undesirable_outputs or [], normalize
        )
        lambdas_df = results['lambdas']
        if dmu_name in lambdas_df.index:
            row = lambdas_df.loc[dmu_name]
            peers = row[row > 1e-6].index.tolist()
            peers = [p for p in peers if p != dmu_name]
            return peers[:10]  # Top 10 peers max
        return []
    except Exception:
        return []


def _describe_step(adjustments):
    """Generate a human-readable description of a step's adjustments."""
    if not adjustments:
        return "No changes"
    parts = []
    for var, val in adjustments.items():
        direction = "increase" if val > 0 else "decrease"
        parts.append(f"{direction} {var} by {abs(val):.4f}")
    return "; ".join(parts)
