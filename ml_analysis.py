"""
ml_analysis.py — Análisis de Importancia de Variables con SHAP y Reducción de Dimensionalidad
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import warnings

warnings.filterwarnings('ignore')


def audit_variable_importance(df, input_cols, output_cols, efficiency_scores):
    """
    M2: Análisis SHAP de importancia de variables.
    Returns dict with Plotly HTML divs and metadata.
    """
    from sklearn.ensemble import RandomForestRegressor
    import shap

    all_cols = input_cols + output_cols
    X = df[all_cols].fillna(0)
    y = np.array(efficiency_scores, dtype=float)

    if len(y) < 5:
        return {
            'shap_summary_html': '<p class="text-[var(--color-text-muted)]">Datos insuficientes para análisis SHAP (mínimo 5 DMUs).</p>',
            'shap_bar_html': '',
            'correlation_html': '',
            'importance_df': pd.DataFrame(),
            'n_important_variables': 0,
            'feature_names': all_cols,
        }

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    # --- 1. SHAP Beeswarm ---
    fig_beeswarm = go.Figure()
    for i, col in enumerate(all_cols):
        vals = shap_values[:, i]
        point_colors = ['#4f46e5' if v >= 0 else '#dc2626' for v in vals]
        # Box for the distribution shape (no points)
        fig_beeswarm.add_trace(go.Box(
            x=vals,
            name=col,
            orientation='h',
            boxpoints=False,
            line=dict(width=1, color='#94a3b8'),
            marker=dict(color='rgba(79,70,229,0.3)'),
            hoverinfo='skip',
            showlegend=False,
        ))
        # Scatter overlay with per-point colors
        fig_beeswarm.add_trace(go.Scatter(
            x=vals,
            y=[col] * len(vals),
            mode='markers',
            marker=dict(
                color=point_colors,
                opacity=0.6,
                size=4,
            ),
            hovertemplate='%{x:.4f}<extra>' + col + '</extra>',
            showlegend=False,
        ))
    fig_beeswarm.update_layout(
        title='SHAP — Impacto de cada variable en la eficiencia',
        xaxis_title='Valor SHAP (impacto en la predicción)',
        yaxis_title='',
        template='dea',
        height=max(300, len(all_cols) * 50 + 100),
        showlegend=False,
        margin=dict(l=120, r=30, t=50, b=40),
    )

    # --- 2. SHAP Bar (mean |SHAP|) ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Variable': all_cols,
        'Mean |SHAP|': mean_abs_shap,
    }).sort_values('Mean |SHAP|', ascending=True)

    n_important = int((importance_df['Mean |SHAP|'] > 0.001).sum())

    fig_bar = go.Figure()
    colors = ['#dc2626' if v == 0 else '#4f46e5' for v in importance_df['Mean |SHAP|']]
    fig_bar.add_trace(go.Bar(
        x=importance_df['Mean |SHAP|'],
        y=importance_df['Variable'],
        orientation='h',
        marker_color=colors,
        text=[f'{v:.4f}' for v in importance_df['Mean |SHAP|']],
        textposition='outside',
        hoverinfo='y+x',
    ))
    fig_bar.update_layout(
        title='Importancia Media de Variables (|SHAP|)',
        xaxis_title='Valor SHAP medio absoluto',
        yaxis_title='',
        template='dea',
        height=max(300, len(all_cols) * 40 + 100),
        showlegend=False,
        margin=dict(l=120, r=60, t=50, b=40),
    )

    # --- 3. Correlation matrix ---
    corr = df[all_cols].corr()
    mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_values = corr.values.copy()
    corr_values[mask_upper] = np.nan

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_values,
        x=corr.columns,
        y=corr.index,
        colorscale=[
            [0.0, '#dc2626'],
            [0.5, '#f5f5f5'],
            [1.0, '#4f46e5'],
        ],
        zmin=-1,
        zmax=1,
        text=np.round(corr_values, 2),
        texttemplate='%{text}',
        textfont=dict(size=11),
        hovertemplate='%{y} vs %{x}: %{z:.3f}<extra></extra>',
        colorbar=dict(title='r', thickness=12),
    ))
    fig_corr.update_layout(
        title='Matriz de Correlación — Variables de Entrada y Salida',
        template='dea',
        height=max(350, len(all_cols) * 35 + 120),
        xaxis=dict(tickangle=-45, side='bottom'),
        yaxis=dict(autorange='reversed'),
        margin=dict(l=100, r=30, t=50, b=100),
    )

    return {
        'shap_summary_html': fig_beeswarm.to_html(
            full_html=False, include_plotlyjs=False,
            config={'displayModeBar': False},
        ),
        'shap_bar_html': fig_bar.to_html(
            full_html=False, include_plotlyjs=False,
            config={'displayModeBar': False},
        ),
        'correlation_html': fig_corr.to_html(
            full_html=False, include_plotlyjs=False,
            config={'displayModeBar': False},
        ),
        'importance_df': importance_df,
        'n_important_variables': n_important,
        'feature_names': all_cols,
    }


def audit_dimensionality(df, input_cols, output_cols):
    """
    M1: Mitigación de dimensionalidad — reglas Cooper/Golany&Roll, correlaciones, PCA.
    Returns dict with analysis results and metadata.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n = len(df)
    m = len(input_cols)
    s = len(output_cols)

    # Cooper Rule
    cooper_min = max(m * s, 3 * (m + s))
    cooper_pass = n >= cooper_min

    # Golany & Roll Rule
    golany_min = 2 * (m + s)
    golany_pass = n >= golany_min

    # Correlation summary
    all_cols = input_cols + output_cols
    corr_matrix = df[all_cols].corr()
    high_corr_pairs = []
    for i in range(len(all_cols)):
        for j in range(i + 1, len(all_cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.7:
                high_corr_pairs.append({
                    'var1': all_cols[i],
                    'var2': all_cols[j],
                    'correlation': round(r, 3),
                })
    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

    # PCA variance explained
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[all_cols].fillna(0))
    pca = PCA()
    pca.fit(X_scaled)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_90 = int(np.searchsorted(cumulative_var, 0.90) + 1) if np.any(cumulative_var >= 0.90) else len(all_cols)
    n_components_95 = int(np.searchsorted(cumulative_var, 0.95) + 1) if np.any(cumulative_var >= 0.95) else len(all_cols)

    # Recommendation
    recommendations = []
    if not cooper_pass:
        recommendations.append(
            f'La regla de Cooper sugiere un mínimo de {cooper_min} DMUs (m*s={m*s}, 3*(m+s)={3*(m+s)}). '
            f'Actualmente tiene {n} DMUs. Considere reducir variables o agregar más datos.'
        )
    if not golany_pass:
        recommendations.append(
            f'La regla de Golany & Roll sugiere un mínimo de {golany_min} DMUs (2*(m+s)). '
            f'Actualmente tiene {n} DMUs.'
        )
    if high_corr_pairs:
        top = high_corr_pairs[0]
        recommendations.append(
            f'Alta correlación detectada entre "{top["var1"]}" y "{top["var2"]}" (r={top["correlation"]}). '
            'Considere eliminar una de las variables para reducir dimensionalidad.'
        )
    if n_components_90 < len(all_cols):
        recommendations.append(
            f'PCA indica que {n_components_90} componentes explican el 90% de la varianza '
            f'(de {len(all_cols)} variables originales).'
        )
    if not recommendations:
        recommendations.append('La dimensionalidad está dentro de rangos aceptables.')

    # Build PCA variance bar chart
    fig_pca = go.Figure()
    fig_pca.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(len(all_cols))],
        y=pca.explained_variance_ratio_ * 100,
        name='Varianza individual',
        marker_color='#4f46e5',
        text=[f'{v:.1f}%' for v in pca.explained_variance_ratio_ * 100],
        textposition='outside',
    ))
    fig_pca.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(len(all_cols))],
        y=cumulative_var * 100,
        name='Varianza acumulada',
        mode='lines+markers',
        line=dict(color='#dc2626', width=2, dash='dot'),
        marker=dict(size=6),
    ))
    fig_pca.add_hline(y=90, line_dash='dash', line_color='#059669',
                       annotation_text='90%', annotation_position='top right')
    fig_pca.update_layout(
        title='PCA — Varianza Explicada por Componente',
        xaxis_title='Componente Principal',
        yaxis_title='Varianza Explicada (%)',
        template='dea',
        height=350,
        margin=dict(l=50, r=30, t=50, b=40),
    )

    return {
        'n_dmus': n,
        'n_inputs': m,
        'n_outputs': s,
        'cooper_min': cooper_min,
        'cooper_pass': cooper_pass,
        'golany_min': golany_min,
        'golany_pass': golany_pass,
        'high_corr_pairs': high_corr_pairs,
        'pca_variance_explained': pca.explained_variance_ratio_.tolist(),
        'pca_cumulative': cumulative_var.tolist(),
        'n_components_90': n_components_90,
        'n_components_95': n_components_95,
        'recommendations': recommendations,
        'pca_chart_html': fig_pca.to_html(
            full_html=False, include_plotlyjs=False,
            config={'displayModeBar': False},
        ),
    }
