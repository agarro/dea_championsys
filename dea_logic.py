import pandas as pd
import numpy as np
from Pyfrontier.frontier_model import EnvelopDEA 
from sklearn.preprocessing import MinMaxScaler
import logging
import plotly.io as pio
import plotly.graph_objects as go

# Plotly template "dea" — glass light aesthetic
dea_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, system-ui, sans-serif', color='#18181b'),
        title=dict(
            font=dict(family='Space Grotesk, Inter, sans-serif', size=20, color='#4f46e5'),
            x=0.02,
            xanchor='left'
        ),
        xaxis=dict(
            gridcolor='#e4e4e7',
            gridwidth=1,
            griddash='dot',
            zerolinecolor='#d4d4d8',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            gridcolor='#e4e4e7',
            gridwidth=1,
            griddash='dot',
            zerolinecolor='#d4d4d8',
            tickfont=dict(size=12)
        ),
        colorway=['#4f46e5', '#06b6d4', '#059669', '#f59e0b', '#dc2626', '#8b5cf6', '#ec4899'],
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(99,102,241,0.15)',
            borderwidth=1,
            font=dict(size=12)
        ),
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='rgba(99,102,241,0.2)',
            font=dict(size=13)
        )
    )
)
pio.templates["dea"] = dea_template

logger = logging.getLogger(__name__)

DEA_MIN_POSITIVE_VALUE = 1e-5 # Mantener este valor, se puede ajustar si es necesario

# Guardar una referencia a los datos finales que se pasan al solver para logging
# Esta variable se poblará en perform_full_dea_analysis
# Es una solución simple para pasar datos al logger en _extract_from_envelop_result_list
# Una solución más elegante podría usar clases o contextos.
_final_inputs_for_log = None
_final_outputs_for_log = None
_original_dmu_names_for_log = None


def _extract_from_envelop_result_list(result_list, attribute_name, num_expected_rows, pyfrontier_orient, num_expected_cols=None, is_lambda=False):
    global _final_inputs_for_log, _final_outputs_for_log, _original_dmu_names_for_log
    
    error_shape = (num_expected_rows, num_expected_rows) if is_lambda else \
                  ((num_expected_rows, num_expected_cols) if num_expected_cols is not None else \
                   (num_expected_rows,))

    if not isinstance(result_list, list) or not result_list:
        logger.warning(f"Se esperaba una lista de EnvelopResult para '{attribute_name}', pero se recibió: {type(result_list)}. Devolviendo NaNs.")
        return np.full(error_shape, np.nan)

    data_list = [] 
    for i, res_item in enumerate(result_list): # 'i' aquí es el índice en la lista de resultados de Pyfrontier
        dmu_original_name = _original_dmu_names_for_log[i] if _original_dmu_names_for_log and i < len(_original_dmu_names_for_log) else f"DMU_Index_{i}"

        if res_item is None:
            logger.warning(f"DMU '{dmu_original_name}' (Índice {i}): EnvelopResult es None. Usando NaN para '{attribute_name}'.")
            if is_lambda: data_list.append(np.full(num_expected_rows, np.nan))
            elif num_expected_cols is not None: data_list.append(np.full(num_expected_cols, np.nan))
            else: data_list.append(np.nan)
            continue

        if hasattr(res_item, attribute_name):
            attr_value = getattr(res_item, attribute_name)
            
            if attribute_name == 'score' and attr_value is not None:
                current_score_val = float(attr_value) 
                is_problematic_score = False

                if pyfrontier_orient == 'out': 
                    if abs(current_score_val) < 1e-9: # Score crudo de Pyfrontier es (casi) cero
                        logger.warning(f"DMU '{dmu_original_name}' (Índice {i}): Score crudo Pyfrontier (output-orientado) anómalo o CERO ({current_score_val:.6f}). Estableciendo eficiencia DEA a NaN.")
                        data_list.append(np.nan)
                        is_problematic_score = True
                    else: # Score crudo Pyfrontier > 0
                        interpreted_score = 1.0 / current_score_val 
                        data_list.append(interpreted_score)
                        logger.debug(f"DMU '{dmu_original_name}' (Índice {i}): Score crudo Pyfrontier (output-orientado) {current_score_val:.4f} -> Eficiencia DEA interpretada {interpreted_score:.4f}")
                
                else: # Input-orientado
                    data_list.append(current_score_val)
                    if abs(current_score_val) < 1e-9:
                        logger.warning(f"DMU '{dmu_original_name}' (Índice {i}): Score crudo Pyfrontier (input-orientado) anómalo o CERO ({current_score_val:.6f}).")
                        is_problematic_score = True # Aún así se guarda el score 0, pero se loguea
                    else:
                        logger.debug(f"DMU '{dmu_original_name}' (Índice {i}): Score Pyfrontier (input-orientado) {current_score_val:.4f}")
                
                if is_problematic_score:
                    logger.debug(f"  DMU '{dmu_original_name}' (Índice {i}) con score problemático: x_slack={getattr(res_item, 'x_slack', 'N/A')}, y_slack={getattr(res_item, 'y_slack', 'N/A')}")
                    if _final_inputs_for_log is not None and i < _final_inputs_for_log.shape[0]:
                        logger.debug(f"    Valores de INPUT pasados al solver para DMU '{dmu_original_name}': {_final_inputs_for_log[i]}")
                    if _final_outputs_for_log is not None and i < _final_outputs_for_log.shape[0]:
                        logger.debug(f"    Valores de OUTPUT pasados al solver para DMU '{dmu_original_name}': {_final_outputs_for_log[i]}")

            elif attr_value is not None:
                data_list.append(attr_value)
            else: 
                logger.warning(f"DMU '{dmu_original_name}' (Índice {i}): Atributo '{attribute_name}' es None.")
                if is_lambda: data_list.append(np.full(num_expected_rows, np.nan))
                elif num_expected_cols is not None: data_list.append(np.full(num_expected_cols, np.nan))
                else: data_list.append(np.nan) 
        else: 
            logger.warning(f"DMU '{dmu_original_name}' (Índice {i}): Objeto EnvelopResult no tiene atributo '{attribute_name}'.")
            if is_lambda: data_list.append(np.full(num_expected_rows, np.nan))
            elif num_expected_cols is not None: data_list.append(np.full(num_expected_cols, np.nan))
            else: data_list.append(np.nan)
    
    if not data_list or len(data_list) != num_expected_rows:
        logger.error(f"No se pudieron extraer datos para '{attribute_name}' para todas las DMUs ({len(data_list)}/{num_expected_rows}). Devolviendo NaNs.")
        return np.full(error_shape, np.nan)

    try:
        # ... (resto de la lógica de conversión a NumPy se mantiene igual)
        np_array = None
        if attribute_name == 'score':
            np_array = np.array(data_list, dtype=float)
        elif attribute_name == 'weights': 
            processed_lambda_rows = []
            for item_weight_list in data_list:
                if isinstance(item_weight_list, (list, np.ndarray)) and len(item_weight_list) == num_expected_rows:
                    processed_lambda_rows.append(np.asarray(item_weight_list, dtype=float))
                else: 
                    processed_lambda_rows.append(np.full(num_expected_rows, np.nan, dtype=float))
            np_array = np.array(processed_lambda_rows, dtype=float)
        elif attribute_name in ['x_slack', 'y_slack']: 
            processed_slack_rows = []
            for item_slack_list in data_list:
                if isinstance(item_slack_list, (list, np.ndarray)) and len(item_slack_list) == num_expected_cols:
                    processed_slack_rows.append(np.asarray(item_slack_list, dtype=float))
                else: 
                    processed_slack_rows.append(np.full(num_expected_cols, np.nan, dtype=float))
            np_array = np.array(processed_slack_rows, dtype=float)
        else: 
            np_array = np.array(data_list, dtype=float)

        if np_array is None or np_array.shape != error_shape: 
            logger.warning(f"Forma de array para '{attribute_name}' ({np_array.shape if np_array is not None else 'None'}) no coincide con la esperada ({error_shape}). Rellenando con NaNs.")
            return np.full(error_shape, np.nan)
        return np_array
    except Exception as e:
        logger.error(f"Error al convertir lista de '{attribute_name}' a array NumPy: {e}. Devolviendo NaNs.", exc_info=True)
        return np.full(error_shape, np.nan)


def handle_undesirable_outputs_df(df, undesirable_cols):
    df_processed = df.copy()
    processed_undesirable_cols_map = {} 
    for col in undesirable_cols:
        if col in df_processed.columns:
            new_col_name = f"{col}_inv_desirable"
            mask_problematic = (df_processed[col] <= 1e-9) 
            if mask_problematic.any():
                logger.warning(f"Columna '{col}' (output no deseable) tiene {mask_problematic.sum()} valores <= 1e-9. Se reemplazarán con {DEA_MIN_POSITIVE_VALUE} antes de la inversión.")
                df_processed.loc[mask_problematic, col] = DEA_MIN_POSITIVE_VALUE
            
            if df_processed[col].isnull().any():
                logger.error(f"NaNs en columna '{col}' (output no deseable) ANTES de inversión. Producirá NaNs.")
            
            df_processed[new_col_name] = 1 / df_processed[col]
            
            if np.isinf(df_processed[new_col_name].values).any():
                logger.error(f"Valores INFINITOS en columna procesada '{new_col_name}' tras invertir '{col}'. Reemplazando con NaN.")
                df_processed[new_col_name].replace([np.inf, -np.inf], np.nan, inplace=True)
            
            processed_undesirable_cols_map[col] = new_col_name
            logger.info(f"Output no deseable '{col}' convertido a '{new_col_name}'.")
        else:
            logger.warning(f"Columna no deseable '{col}' no encontrada en DataFrame.")
    return df_processed, processed_undesirable_cols_map

def normalize_data_df(df, columns_to_normalize):
    df_normalized = df.copy()
    scaler = MinMaxScaler() 
    for col in columns_to_normalize:
        if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
            if df_normalized[col].isnull().any():
                mean_val = df_normalized[col].mean()
                logger.warning(f"NaNs en columna '{col}'. Rellenando con media ({mean_val:.4f}) antes de normalizar.")
                df_normalized[col] = df_normalized[col].fillna(mean_val)
            
            if df_normalized[col].isnull().all():
                 logger.error(f"Columna '{col}' es todo NaN incluso después de rellenar. No se puede normalizar.")
                 continue 
            
            # Si la columna es constante (después de tratar NaNs)
            if df_normalized[col].nunique(dropna=True) == 1:
                unique_val = df_normalized[col].dropna().unique()[0]
                logger.warning(f"Columna '{col}' es constante (valor={unique_val}) antes de normalizar. Se asignará 0.5 si no es NaN, o se mantendrá NaN.")
                if not pd.isna(unique_val):
                    df_normalized[col] = 0.5 
                # Si el valor único es NaN, se queda como NaN.
                continue # Saltar fit_transform

            try:
                col_data_reshaped = df_normalized[[col]].values.astype(float)
                if np.isinf(col_data_reshaped).any(): 
                    logger.error(f"Infinitos en columna '{col}' ANTES de scaler. Reemplazando con NaN.")
                    df_normalized.loc[np.isinf(df_normalized[col].values), col] = np.nan 
                    if df_normalized[col].isnull().all():
                        logger.error(f"Columna '{col}' es todo NaN tras reemplazar Inf. No se normalizará.")
                        continue
                    if df_normalized[col].isnull().any(): # Rellenar NaNs de nuevo
                         mean_val_after_inf = df_normalized[col].mean()
                         df_normalized[col] = df_normalized[col].fillna(mean_val_after_inf)
                    col_data_reshaped = df_normalized[[col]].values.astype(float)
                
                if np.isnan(col_data_reshaped).all(): 
                    logger.error(f"Columna '{col}' es todo NaN y no se puede escalar.")
                    continue
                
                # Nueva verificación de constante después de tratar Inf y NaN
                if pd.Series(col_data_reshaped.flatten()).nunique(dropna=False) == 1: # dropna=False para incluir NaN si es el único valor
                    const_val = col_data_reshaped.flatten()[0]
                    logger.warning(f"Columna '{col}' es constante (valor={const_val}) después de preproc. Asignando 0.5 si no es NaN.")
                    if not pd.isna(const_val):
                        df_normalized[col] = 0.5
                    # Si es NaN, se queda NaN
                else:
                    df_normalized[col] = scaler.fit_transform(col_data_reshaped)
                
                logger.info(f"Columna '{col}' normalizada. Min: {df_normalized[col].min():.4f}, Max: {df_normalized[col].max():.4f}.")
            except ValueError as e: 
                logger.error(f"Error al normalizar columna '{col}': {e}. Únicos (primeros 5): {df_normalized[col].dropna().unique()[:5]}")
        elif col not in df_normalized.columns:
             logger.warning(f"Columna '{col}' no encontrada para normalización.")
    return df_normalized

def determine_model_type_automatically_pyfrontier(inputs_df, outputs_df, pyfrontier_orient_for_test='in'):
    # ... (esta función se mantiene igual que en dea_logic_robust_preprocessing)
    # Asegúrate de que la lógica de manejo de ceros y NaNs aquí sea consistente
    # con el preprocesamiento principal.
    try:
        if inputs_df.empty or outputs_df.empty or not inputs_df.index.equals(outputs_df.index):
            logger.warning("AutoModel: Inputs/Outputs vacíos o índices no coinciden. Default BCC.")
            return 'BCC' 
        
        # Usar copias para no modificar los dataframes originales pasados a esta función
        inputs_test_df = inputs_df.copy()
        outputs_test_df = outputs_df.copy()

        # Preprocesamiento robusto para la prueba de determinación de modelo
        inputs_test_df = inputs_test_df.applymap(lambda x: x if pd.isna(x) or x > 1e-9 else DEA_MIN_POSITIVE_VALUE)
        outputs_test_df = outputs_test_df.applymap(lambda x: x if pd.isna(x) or x > 1e-9 else DEA_MIN_POSITIVE_VALUE)
        
        # Rellenar NaNs con la media para la prueba
        for col in inputs_test_df.columns:
            if inputs_test_df[col].isnull().any():
                inputs_test_df[col].fillna(inputs_test_df[col].mean(), inplace=True)
        for col in outputs_test_df.columns:
            if outputs_test_df[col].isnull().any():
                outputs_test_df[col].fillna(outputs_test_df[col].mean(), inplace=True)

        # Si aún hay NaNs (columnas enteras eran NaN), usar BCC
        if inputs_test_df.isnull().values.any() or outputs_test_df.isnull().values.any():
            logger.warning("AutoModel: NaNs persistentes tras rellenar. Default BCC.")
            return 'BCC'
        if np.isinf(inputs_test_df.values).any() or np.isinf(outputs_test_df.values).any():
            logger.warning("AutoModel: Infinitos en datos de prueba. Default BCC.") # Raro si ya se manejó
            return 'BCC'


        dea_vrs = EnvelopDEA(frontier='VRS', orient=pyfrontier_orient_for_test)
        dea_vrs.fit(inputs_test_df.values, outputs_test_df.values)
        scores_vrs_np = _extract_from_envelop_result_list(dea_vrs.result, "score", len(inputs_test_df), pyfrontier_orient_for_test)
        if np.all(np.isnan(scores_vrs_np)):
            logger.warning("AutoModel: Scores VRS son NaN. Default BCC.")
            return 'BCC' 
        eff_vrs = pd.Series(scores_vrs_np, index=inputs_test_df.index)

        dea_crs = EnvelopDEA(frontier='CRS', orient=pyfrontier_orient_for_test)
        dea_crs.fit(inputs_test_df.values, outputs_test_df.values)
        scores_crs_np = _extract_from_envelop_result_list(dea_crs.result, "score", len(inputs_test_df), pyfrontier_orient_for_test)
        if np.all(np.isnan(scores_crs_np)):
            logger.warning("AutoModel: Scores CRS son NaN. Default BCC.")
            return 'BCC'
        eff_crs = pd.Series(scores_crs_np, index=inputs_test_df.index)
        
        se = pd.Series(np.zeros(len(eff_vrs)), index=inputs_test_df.index)
        valid_vrs_mask = eff_vrs > 1e-9 
        se[valid_vrs_mask] = eff_crs[valid_vrs_mask] / eff_vrs[valid_vrs_mask]
        se[~valid_vrs_mask & (eff_crs < 1e-9)] = 1.0 
        se[~valid_vrs_mask & (eff_crs >= 1e-9)] = 0.0 
        se = se.clip(0, 1) 
        
        mean_se = se.mean()
        std_se = se.std()
        logger.info(f"AutoModel: Mean SE = {mean_se:.4f}, Std Dev SE = {std_se:.4f}")
        
        if pd.isna(mean_se) or pd.isna(std_se):
            logger.warning("AutoModel: Mean SE o Std Dev SE es NaN. Default BCC.")
            return 'BCC'
        if mean_se > 0.95 and std_se < 0.10: 
            return 'CCR'
        else: 
            return 'BCC'
    except Exception as e:
        logger.error(f"Error en AutoModel: {e}", exc_info=True)
        return 'BCC' 

def perform_full_dea_analysis(df_orig, dmu_col, input_cols, output_cols, orientation_str, undesirable_output_cols, normalize):
    global _final_inputs_for_log, _final_outputs_for_log, _original_dmu_names_for_log
    df = df_orig.copy()

    if dmu_col not in df.columns:
        raise ValueError(f"Columna DMU '{dmu_col}' no encontrada.")
    df.set_index(dmu_col, inplace=True)
    if not df.index.is_unique:
        raise ValueError(f"Índice DMU '{dmu_col}' no es único.")

    df_for_graph = df.copy() 
    _original_dmu_names_for_log = df.index.tolist() # Guardar para logging en _extract
    num_dmus = len(_original_dmu_names_for_log)
    num_inputs = len(input_cols)
    
    actual_output_cols = list(output_cols) 
    if undesirable_output_cols:
        df, _ = handle_undesirable_outputs_df(df, undesirable_output_cols)
        # Actualizar actual_output_cols si handle_undesirable_outputs_df los modificó
        # (Esto ya se hace en la lógica de app.py antes de llamar aquí, pero por si acaso)
        # La lógica actual de handle_undesirable_outputs_df crea nuevas columnas,
        # y app.py ya debería estar pasando los nombres correctos de las columnas procesadas.
        # Por simplicidad, asumimos que actual_output_cols ya contiene los nombres correctos.

    num_actual_outputs = len(actual_output_cols)

    # Validar columnas
    missing_inputs = [col for col in input_cols if col not in df.columns]
    if missing_inputs: raise ValueError(f"Inputs faltantes: {missing_inputs}")
    missing_outputs = [col for col in actual_output_cols if col not in df.columns]
    if missing_outputs: raise ValueError(f"Outputs faltantes: {missing_outputs}")

    inputs_df = df[input_cols].astype(float)
    outputs_df = df[actual_output_cols].astype(float)

    # Paso 1: Reemplazo inicial de valores <= 1e-9 (antes de normalización)
    inputs_df = inputs_df.applymap(lambda x: x if pd.isna(x) or x > 1e-9 else DEA_MIN_POSITIVE_VALUE)
    outputs_df = outputs_df.applymap(lambda x: x if pd.isna(x) or x > 1e-9 else DEA_MIN_POSITIVE_VALUE)
    
    if normalize:
        logger.info("Normalizando datos...")
        inputs_df_norm = normalize_data_df(inputs_df.copy(), input_cols)
        outputs_df_norm = normalize_data_df(outputs_df.copy(), actual_output_cols)
    else:
        inputs_df_norm = inputs_df.copy()
        outputs_df_norm = outputs_df.copy()

    # Paso 2: Asegurar que los datos que van al solver sean estrictamente positivos
    # Esto es crucial y se hace DESPUÉS de la normalización si está activa.
    # Si la normalización produce ceros, se ajustarán aquí.
    logger.info(f"Asegurando valores > 0 para el solver, usando {DEA_MIN_POSITIVE_VALUE} como mínimo.")
    final_inputs_np = inputs_df_norm.values.copy() # Trabajar con copia para no modificar df_norm
    final_outputs_np = outputs_df_norm.values.copy()

    final_inputs_np[~np.isnan(final_inputs_np) & (final_inputs_np <= 1e-9)] = DEA_MIN_POSITIVE_VALUE
    final_outputs_np[~np.isnan(final_outputs_np) & (final_outputs_np <= 1e-9)] = DEA_MIN_POSITIVE_VALUE
    
    # Guardar para logging en _extract_from_envelop_result_list
    _final_inputs_for_log = final_inputs_np
    _final_outputs_for_log = final_outputs_np

    if np.isnan(final_inputs_np).any() or np.isnan(final_outputs_np).any() or \
       np.isinf(final_inputs_np).any() or np.isinf(final_outputs_np).any():
        logger.error(f"NaNs/Inf en datos FINALES para DEA. Inputs NaNs: {np.isnan(final_inputs_np).sum()}, Outputs NaNs: {np.isnan(final_outputs_np).sum()}")
        # ... (logging de filas con problemas)
        raise ValueError("NaNs o Infinitos en datos preparados para DEA. Revise preprocesamiento y logs.")

    pyfrontier_orient = 'in' if orientation_str == 'input' else 'out' 
    # Para determinar el modelo, usar los datos que realmente irán al solver (o una copia muy similar)
    # Es importante que la determinación del modelo vea datos con características similares a los del fitting final.
    df_for_model_det_inputs = pd.DataFrame(final_inputs_np, index=inputs_df_norm.index, columns=inputs_df_norm.columns)
    df_for_model_det_outputs = pd.DataFrame(final_outputs_np, index=outputs_df_norm.index, columns=outputs_df_norm.columns)
    chosen_model_str = determine_model_type_automatically_pyfrontier(df_for_model_det_inputs, df_for_model_det_outputs, pyfrontier_orient) 
    
    pyfrontier_frontier = 'CRS' if chosen_model_str == 'CCR' else 'VRS' 
    dea_model = EnvelopDEA(frontier=pyfrontier_frontier, orient=pyfrontier_orient)

    logger.info(f"Datos para Pyfrontier.fit(): Orient={pyfrontier_orient}, Frontier={pyfrontier_frontier}, Inputs shape {final_inputs_np.shape}, Outputs shape {final_outputs_np.shape}")
    if num_dmus > 0:
        logger.debug(f"  Final Inputs (primeras {min(5, num_dmus)} filas) para fit:\n{final_inputs_np[:min(5, num_dmus)]}")
        logger.debug(f"  Final Outputs (primeras {min(5, num_dmus)} filas) para fit:\n{final_outputs_np[:min(5, num_dmus)]}")
        logger.debug(f"  Stats Inputs (final): min={np.nanmin(final_inputs_np):.4e}, max={np.nanmax(final_inputs_np):.4e}, mean={np.nanmean(final_inputs_np):.4e}")
        logger.debug(f"  Stats Outputs (final): min={np.nanmin(final_outputs_np):.4e}, max={np.nanmax(final_outputs_np):.4e}, mean={np.nanmean(final_outputs_np):.4e}")

    try:
        dea_model.fit(final_inputs_np, final_outputs_np)
    except Exception as e:
        logger.error(f"Error durante dea_model.fit(): {e}", exc_info=True)
        if "optimal" in str(e).lower() or "infeasible" in str(e).lower() or "unbounded" in str(e).lower():
            logger.error("El error de .fit() parece relacionado con el solver LP. Verifique datos y dimensionalidad.")
        raise RuntimeError(f"Pyfrontier .fit() falló: {e}") from e
    
    raw_result_list_from_pyfrontier = dea_model.result
    # ... (resto de la función: extracción de scores, slacks, lambdas, cálculo de targets)
    # Esta parte se mantiene mayormente igual, pero _extract_from_envelop_result_list ahora tiene mejor logging
    logger.info(f"Tipo de raw_result_list_from_pyfrontier: {type(raw_result_list_from_pyfrontier)}")
    if isinstance(raw_result_list_from_pyfrontier, list) and raw_result_list_from_pyfrontier:
        logger.info(f"Pyfrontier devolvió una lista con {len(raw_result_list_from_pyfrontier)} elementos.")
        if len(raw_result_list_from_pyfrontier) > 0 and raw_result_list_from_pyfrontier[0] is not None:
            first_res_attrs = {attr: getattr(raw_result_list_from_pyfrontier[0], attr, 'No encontrado') 
                               for attr in ['id', 'score', 'is_efficient', 'has_slack', 'weights', 'x_slack', 'y_slack']}
            logger.info(f"Atributos del primer elemento de la lista de resultados: {first_res_attrs}")
    else:
        logger.warning("Pyfrontier no devolvió una lista válida para .result o la lista está vacía.")

    scores_np = _extract_from_envelop_result_list(raw_result_list_from_pyfrontier, "score", num_dmus, pyfrontier_orient)
    input_slacks_np = _extract_from_envelop_result_list(raw_result_list_from_pyfrontier, "x_slack", num_dmus, pyfrontier_orient, num_inputs)
    output_slacks_np = _extract_from_envelop_result_list(raw_result_list_from_pyfrontier, "y_slack", num_dmus, pyfrontier_orient, num_actual_outputs)
    lambdas_np = _extract_from_envelop_result_list(raw_result_list_from_pyfrontier, "weights", num_dmus, pyfrontier_orient, is_lambda=True)
    
    analysis_successful = not np.all(np.isnan(scores_np)) if scores_np is not None and scores_np.size > 0 else False

    if analysis_successful:
        logger.info("Scores obtenidos y procesados.")
        if scores_np is not None and np.all(np.isclose(scores_np, 1.0, equal_nan=True)) and num_dmus > 1 : 
            logger.warning("Todos los puntajes de eficiencia interpretados son 1 (o NaN).")
    else:
        logger.error("Análisis DEA con Pyfrontier no produjo scores válidos (todos NaN o error al extraer).")

    scores_df = pd.DataFrame({'Score': scores_np}, index=_original_dmu_names_for_log)
    
    if input_slacks_np.shape != (num_dmus, num_inputs):
        input_slacks_np = np.full((num_dmus, num_inputs), np.nan)
    if output_slacks_np.shape != (num_dmus, num_actual_outputs):
        output_slacks_np = np.full((num_dmus, num_actual_outputs), np.nan)
    slacks_df = pd.DataFrame(np.hstack((input_slacks_np, output_slacks_np)), 
                             index=_original_dmu_names_for_log, 
                             columns=[f'Slack_{col}' for col in input_cols] + \
                                     [f'Slack_{col}' for col in actual_output_cols])
    
    if lambdas_np.shape != (num_dmus, num_dmus):
        lambdas_np = np.full((num_dmus, num_dmus), np.nan)
    lambdas_df = pd.DataFrame(lambdas_np, index=_original_dmu_names_for_log, columns=_original_dmu_names_for_log)
    
    # Usar los dataframes que contienen los datos que fueron al solver (final_inputs_np, final_outputs_np)
    # pero con el índice y columnas originales de inputs_df_norm / outputs_df_norm
    base_inputs_for_targets = pd.DataFrame(final_inputs_np, index=inputs_df_norm.index, columns=inputs_df_norm.columns)
    base_outputs_for_targets = pd.DataFrame(final_outputs_np, index=outputs_df_norm.index, columns=outputs_df_norm.columns)

    target_inputs_df = base_inputs_for_targets.copy()
    target_outputs_df = base_outputs_for_targets.copy()

    if analysis_successful and scores_np is not None:
        for i in range(num_dmus):
            score_dmu_i = scores_np[i]
            if pd.isna(score_dmu_i):
                target_inputs_df.iloc[i, :] = np.nan
                target_outputs_df.iloc[i, :] = np.nan
                continue

            if score_dmu_i < 0.99999: # Ajustar para ineficientes (considerar tolerancia)
                if pyfrontier_orient == 'in':
                    target_inputs_df.iloc[i, :] = score_dmu_i * base_inputs_for_targets.iloc[i, :] - input_slacks_np[i, :]
                    target_outputs_df.iloc[i, :] = base_outputs_for_targets.iloc[i, :] + output_slacks_np[i, :]
                elif pyfrontier_orient == 'out':
                    target_inputs_df.iloc[i, :] = base_inputs_for_targets.iloc[i, :] - input_slacks_np[i, :]
                    if score_dmu_i > 1e-9:
                        target_outputs_df.iloc[i, :] = (1.0 / score_dmu_i) * base_outputs_for_targets.iloc[i, :] + output_slacks_np[i, :]
                    else:
                        target_outputs_df.iloc[i, :] = np.nan
        
        target_inputs_df[target_inputs_df < 0] = 0 # Los targets no deben ser negativos
        target_outputs_df[target_outputs_df < 0] = 0
    else:
        logger.warning("Análisis no exitoso o scores NaN, Targets serán NaN.")
        target_inputs_df.iloc[:,:] = np.nan
        target_outputs_df.iloc[:,:] = np.nan
            
    targets_combined_df = pd.concat([target_inputs_df.rename(columns=lambda c: f'Target_{c}'),
                                     target_outputs_df.rename(columns=lambda c: f'Target_{c}')], axis=1)
    targets_combined_df.index = _original_dmu_names_for_log

    # Para graficar, usar los datos originales de las DMUs que efectivamente se analizaron
    # (después de dropna y con el índice correcto)
    analyzed_dmu_indices = inputs_df_norm.index 
    original_inputs_for_graph = df_for_graph.loc[analyzed_dmu_indices, input_cols].copy()
    original_outputs_for_graph = df_for_graph.loc[analyzed_dmu_indices, output_cols].copy()

    results = {
        'scores': scores_df,
        'slacks': slacks_df, 
        'targets': targets_combined_df, 
        'lambdas': lambdas_df,
        'original_inputs_for_graph': original_inputs_for_graph,
        'original_outputs_for_graph': original_outputs_for_graph 
    }
    logger.info("Análisis DEA y formateo de resultados completados.")
    # Limpiar las variables globales de logging
    _final_inputs_for_log = None
    _final_outputs_for_log = None
    _original_dmu_names_for_log = None
    return results, chosen_model_str
