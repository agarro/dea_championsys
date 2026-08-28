import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, jsonify
from werkzeug.utils import secure_filename
import io
import logging
import datetime 

from dea_logic import perform_full_dea_analysis 
from pdf_utils import generate_pdf_report 

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dea-championsys-stable-key-2024')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GRAPH_IMAGE_FOLDER'] = os.path.join('static', 'graph_images') 
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['GRAPH_IMAGE_FOLDER']):
    os.makedirs(app.config['GRAPH_IMAGE_FOLDER'])

@app.context_processor
def inject_current_year():
    return {'current_year': datetime.datetime.now().year}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No se encontró el archivo en la solicitud.', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No se seleccionó ningún archivo.', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                session['filepath'] = filepath
                session['filename'] = filename

                if filename.endswith('.csv'):
                    df_peek = pd.read_csv(filepath, nrows=5) 
                else:
                    df_peek = pd.read_excel(filepath, nrows=5) 
                
                potential_dmu_cols = [col for col in df_peek.columns if df_peek[col].dtype == 'object' or (df_peek[col].nunique() == len(df_peek) and df_peek[col].dtype != 'datetime64[ns]')]
                if not potential_dmu_cols and len(df_peek.columns) > 0:
                    potential_dmu_cols = [df_peek.columns[0]]

                session['columns'] = df_peek.columns.tolist()
                session['potential_dmu_cols'] = potential_dmu_cols
                session['numeric_columns'] = df_peek.select_dtypes(include=np.number).columns.tolist()
                
                session.pop('dea_results_raw', None)
                session.pop('analysis_config', None)
                session.pop('error_message', None)
                session.pop('current_graph_html', None)
                session.pop('current_graph_image_path', None)

                logger.info(f"Archivo '{filename}' cargado. Columnas: {session['columns']}")
                return redirect(url_for('configure_analysis'))
            except Exception as e:
                logger.error(f"Error al procesar el archivo '{filename}': {e}", exc_info=True)
                flash(f"Error al procesar el archivo: {str(e)}", 'error')
                return redirect(request.url)
        else:
            flash('Tipo de archivo no permitido.', 'error')
            return redirect(request.url)

    if 'config_step' not in request.args: 
        for key in ['filepath', 'filename', 'columns', 'potential_dmu_cols', 'numeric_columns', 'dea_results_raw', 'analysis_config', 'current_graph_html', 'current_graph_image_path']:
            session.pop(key, None)

    return render_template('index.html', 
                           filename=session.get('filename'),
                           upload_page=True)

@app.route('/configure', methods=['GET'])
def configure_analysis():
    if 'columns' not in session:
        flash("Por favor, suba un archivo primero.", 'info')
        return redirect(url_for('index'))
    
    error_message = session.pop('error_message', None) 
    if error_message:
        flash(error_message, 'error')

    return render_template('index.html', 
                           filename=session.get('filename'),
                           columns=session.get('columns'),
                           potential_dmu_cols=session.get('potential_dmu_cols'),
                           numeric_columns=session.get('numeric_columns'),
                           analysis_config=session.get('analysis_config', {}), 
                           config_step=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'filepath' not in session or 'columns' not in session:
        session['error_message'] = "Por favor, suba un archivo primero." 
        return redirect(url_for('index'))

    try:
        filepath = session['filepath']
        dmu_column = request.form.get('dmu_column')
        input_columns = request.form.getlist('input_columns')
        output_columns = request.form.getlist('output_columns')
        undesirable_outputs = request.form.getlist('undesirable_outputs')
        orientation = request.form.get('orientation', 'input') 
        normalize = 'normalize' in request.form

        if not dmu_column:
            session['error_message'] = "Debe seleccionar una columna de DMU."
            return redirect(url_for('configure_analysis'))
        if not input_columns:
            session['error_message'] = "Debe seleccionar al menos una columna de input."
            return redirect(url_for('configure_analysis'))
        if not output_columns:
            session['error_message'] = "Debe seleccionar al menos una columna de output."
            return redirect(url_for('configure_analysis'))
        
        for u_output in undesirable_outputs:
            if u_output not in output_columns:
                session['error_message'] = f"La columna no deseable '{u_output}' también debe ser seleccionada como output."
                return redirect(url_for('configure_analysis'))
        
        current_config = {
            'dmu_column': dmu_column,
            'input_columns': input_columns,
            'output_columns': output_columns, 
            'undesirable_outputs': undesirable_outputs,
            'orientation': orientation,
            'normalize': normalize,
            'filename': session.get('filename')
        }
        session['analysis_config'] = current_config
        
        logger.info(f"Iniciando análisis para el archivo: {session['filename']} con config: {current_config}")
        if session['filename'].endswith('.csv'):
            df_full = pd.read_csv(filepath)
        else:
            df_full = pd.read_excel(filepath)
        
        if not df_full[dmu_column].is_unique:
            error_msg_dmu = (f"Error: La columna de DMU ('{dmu_column}') contiene valores duplicados. "
                             "Cada DMU debe tener un identificador único.")
            logger.error(error_msg_dmu)
            session['error_message'] = error_msg_dmu
            return redirect(url_for('configure_analysis'))

        df_analysis = df_full.copy() 

        num_dmus_actual = len(df_analysis)
        num_inputs_selected = len(input_columns)
        num_outputs_selected = len(output_columns)
        min_dmus_rule1 = num_inputs_selected * num_outputs_selected
        min_dmus_rule2 = 3 * (num_inputs_selected + num_outputs_selected)
        recommended_min_dmus = max(min_dmus_rule1, min_dmus_rule2) if num_inputs_selected > 0 and num_outputs_selected > 0 else (3 * (num_inputs_selected + num_outputs_selected) if (num_inputs_selected + num_outputs_selected) > 0 else 1)
        
        if num_dmus_actual < recommended_min_dmus and num_dmus_actual > 0 : 
            warning_message = (
                f"Advertencia de Dimensionalidad: {num_dmus_actual} DMUs, {num_inputs_selected} inputs, {num_outputs_selected} outputs. "
                f"Se recomiendan al menos {recommended_min_dmus} DMUs. "
                "Resultados podrían no ser significativos."
            )
            flash(warning_message, 'warning')
            logger.warning(warning_message)

        cols_to_check_numeric = [col for col in input_columns + output_columns if col != dmu_column]
        for col in cols_to_check_numeric:
            if col not in df_analysis.columns:
                session['error_message'] = f"La columna '{col}' seleccionada no existe en el archivo."
                return redirect(url_for('configure_analysis'))
            if not pd.api.types.is_numeric_dtype(df_analysis[col]):
                try:
                    df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
                    if df_analysis[col].isnull().any():
                        nan_count = df_analysis[col].isnull().sum()
                        session['error_message'] = f"La columna '{col}' contiene {nan_count} valor(es) no numérico(s) o NaNs. Revise los datos."
                        return redirect(url_for('configure_analysis'))
                except ValueError: 
                    session['error_message'] = f"La columna '{col}' contiene valores no numéricos. Revise los datos."
                    return redirect(url_for('configure_analysis'))
        
        relevant_cols_for_nan_check = input_columns + output_columns
        initial_rows = len(df_analysis)
        df_analysis.dropna(subset=relevant_cols_for_nan_check, inplace=True)
        rows_dropped = initial_rows - len(df_analysis)

        if df_analysis.empty:
            session['error_message'] = "No quedan datos para analizar después de eliminar filas con valores faltantes."
            return redirect(url_for('configure_analysis'))
        
        logger.info(f"Columnas: DMU='{dmu_column}', Inputs={input_columns}, Outputs={output_columns}, Undesirable={undesirable_outputs}")

        results_data, chosen_model = perform_full_dea_analysis(
            df_analysis,
            dmu_column,
            input_columns,
            output_columns, 
            orientation,
            undesirable_outputs,
            normalize
        )
        
        current_config['chosen_model'] = chosen_model 
        session['analysis_config'] = current_config 

        # Almacenar SOLO las cadenas JSON necesarias en la sesión
        dea_results_for_session_raw = {
            'raw_scores_df_json': results_data['scores'].to_json(orient='split'),
            'raw_slacks_df_json': results_data['slacks'].to_json(orient='split'),
            'raw_targets_df_json': results_data['targets'].to_json(orient='split'),
            'raw_lambdas_df_json': results_data['lambdas'].to_json(orient='split'),
            'raw_original_inputs_df_json': results_data['original_inputs_for_graph'].to_json(orient='split'),
            'raw_original_outputs_df_json': results_data['original_outputs_for_graph'].to_json(orient='split'),
            'summary': f"Análisis DEA ({chosen_model}, {orientation}-orientado) para {len(results_data['scores'])} DMUs. " + 
                       (f"Se eliminaron {rows_dropped} filas por valores faltantes." if rows_dropped > 0 else "No se eliminaron filas.")
        }
        
        session['dea_results_raw'] = dea_results_for_session_raw
        logger.info(f"Análisis DEA completado. Modelo: {chosen_model}. Resultados (raw JSON) almacenados en sesión.")

        # --- ML Analysis (F4: M1 Dimensionalidad + M2 SHAP) ---
        try:
            from ml_analysis import audit_variable_importance, audit_dimensionality
            efficiency_scores = results_data['scores']['Score'].tolist()
            importance_result = audit_variable_importance(
                df_analysis, input_columns, output_columns, efficiency_scores
            )
            # Convert DataFrame to JSON-serializable format for session storage
            importance_result['importance_df'] = importance_result['importance_df'].to_dict(orient='records')
            ml_results = {
                'dimensionality': audit_dimensionality(df_analysis, input_columns, output_columns),
                'importance': importance_result,
            }
            session['ml_results'] = ml_results
            logger.info("Análisis ML (dimensionalidad + SHAP) completado.")
        except Exception as e:
            logger.error(f"Error en análisis ML: {e}", exc_info=True)
            session['ml_results'] = {'error': str(e)}
        
        session.pop('error_message', None) 
        flash(f"Análisis DEA completado con éxito usando el modelo {chosen_model}.", 'success')
        return redirect(url_for('show_results'))

    except ValueError as ve: 
        logger.error(f"Error de valor durante el análisis DEA: {ve}", exc_info=True)
        session['error_message'] = f"Error en los datos o configuración: {str(ve)}"
        return redirect(url_for('configure_analysis'))
    except RuntimeError as re:
        logger.error(f"Error de ejecución durante el análisis DEA (solver): {re}", exc_info=True)
        session['error_message'] = f"Error durante la ejecución del análisis (solver): {str(re)}"
        return redirect(url_for('configure_analysis'))
    except Exception as e:
        logger.error(f"Error general durante el análisis DEA: {e}", exc_info=True)
        session['error_message'] = f"Ocurrió un error inesperado durante el análisis: {str(e)}"
        return redirect(url_for('configure_analysis'))


@app.route('/results')
def show_results():
    raw_results_from_session = session.get('dea_results_raw')
    analysis_cfg = session.get('analysis_config')

    if not raw_results_from_session or not analysis_cfg:
        error_msg = session.pop('error_message', "No hay resultados para mostrar o la sesión ha expirado. Intente de nuevo.")
        flash(error_msg, 'warning')
        return redirect(url_for('index'))

    session.pop('current_graph_html', None)
    session.pop('current_graph_image_path', None)
    session.pop('current_graph_config', None)
    
    error_message = session.pop('error_message', None) 
    if error_message:
        flash(error_message, 'error')

    try:
        table_classes = 'table table-striped table-hover text-sm w-full'
        results_for_html = {
            'scores_html': pd.read_json(raw_results_from_session['raw_scores_df_json'], orient='split').to_html(classes=table_classes, index=True, border=0, float_format='{:,.4f}'.format),
            'slacks_html': pd.read_json(raw_results_from_session['raw_slacks_df_json'], orient='split').to_html(classes=table_classes, index=True, border=0, float_format='{:,.4f}'.format),
            'targets_html': pd.read_json(raw_results_from_session['raw_targets_df_json'], orient='split').to_html(classes=table_classes, index=True, border=0, float_format='{:,.2f}'.format),
            'lambdas_html': pd.read_json(raw_results_from_session['raw_lambdas_df_json'], orient='split').to_html(classes=table_classes, index=True, border=0, float_format='{:,.4f}'.format),
            'original_inputs_html': pd.read_json(raw_results_from_session['raw_original_inputs_df_json'], orient='split').to_html(classes=table_classes, index=True, border=0, float_format='{:,.4f}'.format),
            'original_outputs_html': pd.read_json(raw_results_from_session['raw_original_outputs_df_json'], orient='split').to_html(classes=table_classes, index=True, border=0, float_format='{:,.4f}'.format),
            'summary': raw_results_from_session['summary']
        }
    except Exception as e:
        logger.error(f"Error al convertir datos JSON de la sesión a HTML: {e}", exc_info=True)
        flash("Error al procesar los resultados para visualización. Intente el análisis de nuevo.", "error")
        return redirect(url_for('index'))

    return render_template('results.html', 
                           results=results_for_html, 
                           analysis_config=analysis_cfg,
                           ml_results=session.get('ml_results', {}))

@app.route('/graphs', methods=['GET', 'POST'])
def show_graphs():
    analysis_cfg = session.get('analysis_config')
    dea_results_session_raw = session.get('dea_results_raw') 

    if not analysis_cfg or not dea_results_session_raw:
        flash("Complete el análisis primero para generar gráficas.", 'warning')
        return redirect(url_for('index'))
    
    try:
        scores_df = pd.read_json(dea_results_session_raw['raw_scores_df_json'], orient='split')
        inputs_df_graph = pd.read_json(dea_results_session_raw['raw_original_inputs_df_json'], orient='split')
        outputs_df_graph = pd.read_json(dea_results_session_raw['raw_original_outputs_df_json'], orient='split')
    except KeyError as ke:
        logger.error(f"Error de clave cargando datos para graficar (KeyError): {ke}. Verifique 'dea_results_raw'.", exc_info=True)
        flash(f"Error al cargar datos para la gráfica (faltan datos en sesión): {str(ke)}. Intente el análisis de nuevo.", "error")
        return redirect(url_for('show_results'))
    except Exception as e:
        logger.error(f"Error general cargando datos para graficar: {e}", exc_info=True)
        flash("Error al cargar datos para la gráfica. Intente el análisis de nuevo.", "error")
        return redirect(url_for('show_results'))

    graph_html_output = session.get('current_graph_html')
    current_graph_cfg = session.get('current_graph_config', {})

    if request.method == 'POST':
        selected_input = request.form.get('selected_input')
        selected_output_col_original_name = request.form.get('selected_output') 

        if not selected_input or not selected_output_col_original_name:
            flash("Debe seleccionar un input y un output para generar la gráfica.", 'error')
        else:
            try:
                plot_df = pd.DataFrame(index=scores_df.index) 
                plot_df['DMU'] = scores_df.index
                
                if selected_input not in inputs_df_graph.columns:
                    raise KeyError(f"Input seleccionado '{selected_input}' no encontrado.")
                plot_df[selected_input] = inputs_df_graph[selected_input]
                
                if selected_output_col_original_name not in outputs_df_graph.columns:
                    raise KeyError(f"Output seleccionado '{selected_output_col_original_name}' no encontrado.")
                plot_df[selected_output_col_original_name] = outputs_df_graph[selected_output_col_original_name]
                
                plot_df['Efficiency'] = scores_df['Score'] 
                plot_df['Status'] = np.where(plot_df['Efficiency'] >= 0.99999, 'Eficiente', 'Ineficiente') 

                fig = px.scatter(plot_df, x=selected_input, y=selected_output_col_original_name, 
                                 text='DMU', color='Status',
                                 color_discrete_map={'Eficiente': 'green', 'Ineficiente': 'red'},
                                 title=f"Frontera: {selected_output_col_original_name} vs. {selected_input}",
                                 hover_data={'DMU': True, selected_input: True, selected_output_col_original_name: True, 'Efficiency':':.4f', 'Status':False},
                                 labels={selected_input: f"Input: {selected_input}", selected_output_col_original_name: f"Output: {selected_output_col_original_name}"})
                fig.update_traces(textposition='top center', marker=dict(size=8))
                fig.update_layout(title_x=0.5, legend_title_text='Estado DMU')
                
                efficient_dmus = plot_df[plot_df['Status'] == 'Eficiente'].sort_values(by=selected_input)
                if not efficient_dmus.empty and len(efficient_dmus) > 1: 
                     fig.add_trace(go.Scatter(x=efficient_dmus[selected_input], y=efficient_dmus[selected_output_col_original_name],
                                             mode='lines', name='Frontera Observada',
                                             line=dict(color='rgba(0,100,80,0.7)', dash='dash', width=2),
                                             hoverinfo='skip')) 
                
                graph_html_output = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
                session['current_graph_html'] = graph_html_output
                
                graph_filename = f"graph_{session.get('filename', 'data').split('.')[0]}_{selected_input}_{selected_output_col_original_name}.png"
                graph_image_path = os.path.join(app.config['GRAPH_IMAGE_FOLDER'], secure_filename(graph_filename))
                try:
                    pio.write_image(fig, graph_image_path, width=800, height=600, scale=1.5)
                    session['current_graph_image_path'] = graph_image_path
                    logger.info(f"Gráfica generada y guardada en: {graph_image_path}")
                except Exception as img_err:
                    logger.warning(f"Kaleido no disponible, imagen estática no generada: {img_err}")
                    session['current_graph_image_path'] = None

                current_graph_cfg = {'input': selected_input, 'output': selected_output_col_original_name}
                session['current_graph_config'] = current_graph_cfg 
                flash("Gráfica generada con éxito.", "success")

            except KeyError as ke:
                logger.error(f"Error de clave generando gráfica: {ke}", exc_info=True)
                flash(f"Error al generar la gráfica: Columna no encontrada. {str(ke)}", 'error')
                graph_html_output = None # Limpiar para no mostrar gráfica vieja
            except Exception as e:
                logger.error(f"Error generando gráfica: {e}", exc_info=True)
                flash(f"Error al generar la gráfica: {str(e)}", 'error')
                graph_html_output = None # Limpiar
            # No es necesario limpiar current_graph_image_path o current_graph_config aquí,
            # se limpian al entrar a show_results o al cargar nueva gráfica.

    error_message = session.pop('error_message', None) 
    if error_message:
        flash(error_message, 'error')

    return render_template('graphs.html', 
                           analysis_config=analysis_cfg, 
                           graph_html=graph_html_output,
                           selected_input=current_graph_cfg.get('input'),
                           selected_output=current_graph_cfg.get('output'))


# ===== F5: Simulador What-If =====

@app.route('/simulator')
def simulator():
    """Simulador What-If page."""
    if 'dea_results_raw' not in session:
        flash('Primero realiza un análisis DEA.', 'warning')
        return redirect(url_for('index'))

    raw_results = session['dea_results_raw']
    config = session.get('analysis_config', {})

    dmu_name = request.args.get('dmu', '')
    if not dmu_name:
        dmu_name = session.get('selected_dmu', '')

    # Load DataFrames
    try:
        scores_df = pd.read_json(raw_results['raw_scores_df_json'], orient='split')
        inputs_df = pd.read_json(raw_results['raw_original_inputs_df_json'], orient='split')
        outputs_df = pd.read_json(raw_results['raw_original_outputs_df_json'], orient='split')
    except Exception as e:
        logger.error(f"Error loading data for simulator: {e}", exc_info=True)
        flash("Error al cargar datos para el simulador.", "error")
        return redirect(url_for('show_results'))

    # Default to first DMU if none specified
    if not dmu_name or dmu_name not in scores_df.index:
        dmu_name = scores_df.index[0] if len(scores_df) > 0 else ''

    input_cols = config.get('input_columns', [])
    output_cols = config.get('output_columns', [])
    orientation = config.get('orientation', 'input')

    # Build variable info with min/max/current for sliders
    input_variables = []
    for col in input_cols:
        if col in inputs_df.columns:
            vals = inputs_df[col].dropna()
            if len(vals) > 0:
                min_val = float(vals.min())
                max_val = float(vals.max())
                current_val = float(inputs_df.loc[dmu_name, col]) if dmu_name in inputs_df.index else float(vals.mean())
                step_val = max((max_val - min_val) / 200, 0.0001)
                input_variables.append({
                    'name': col,
                    'min': round(min_val * 0.5, 4),
                    'max': round(max_val * 1.5, 4),
                    'current': round(current_val, 4),
                    'step': round(step_val, 6),
                })

    output_variables = []
    for col in output_cols:
        if col in outputs_df.columns:
            vals = outputs_df[col].dropna()
            if len(vals) > 0:
                min_val = float(vals.min())
                max_val = float(vals.max())
                current_val = float(outputs_df.loc[dmu_name, col]) if dmu_name in outputs_df.index else float(vals.mean())
                step_val = max((max_val - min_val) / 200, 0.0001)
                output_variables.append({
                    'name': col,
                    'min': round(min_val * 0.5, 4),
                    'max': round(max_val * 1.5, 4),
                    'current': round(current_val, 4),
                    'step': round(step_val, 6),
                })

    current_score = 0.0
    if dmu_name in scores_df.index:
        current_score = float(scores_df.loc[dmu_name, 'Score'])

    # Get peer group from lambdas
    peer_group = []
    try:
        lambdas_df = pd.read_json(raw_results['raw_lambdas_df_json'], orient='split')
        if dmu_name in lambdas_df.index:
            row = lambdas_df.loc[dmu_name]
            peer_group = row[row > 1e-6].index.tolist()
            peer_group = [p for p in peer_group if p != dmu_name][:10]
    except Exception:
        pass

    return render_template('simulator.html',
                           dmu_name=dmu_name,
                           current_score=current_score,
                           input_variables=input_variables,
                           output_variables=output_variables,
                           orientation=orientation,
                           peer_group=peer_group)


@app.route('/api/simulate-what-if', methods=['POST'])
def api_simulate():
    """API endpoint for What-If simulation."""
    raw_results = session.get('dea_results_raw')
    config = session.get('analysis_config')

    if not raw_results or not config:
        return jsonify({'feasible': False, 'violations': ['No hay datos de análisis en sesión.']}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'feasible': False, 'violations': ['Request body inválido.']}), 400

    targets = data.get('targets', {})
    if not targets:
        return jsonify({'feasible': False, 'violations': ['No se proporcionaron metas de ajuste.']}), 400

    dmu_name = data.get('dmu', session.get('selected_dmu', ''))
    input_cols = config.get('input_columns', [])
    output_cols = config.get('output_columns', [])
    orientation = config.get('orientation', 'input')
    undesirable_outputs = config.get('undesirable_outputs', [])
    normalize = config.get('normalize', False)

    try:
        # Load full dataframe
        filepath = session.get('filepath')
        if not filepath:
            return jsonify({'feasible': False, 'violations': ['No se encontró el archivo de datos.']}), 400

        if filepath.endswith('.csv'):
            df_full = pd.read_csv(filepath)
        else:
            df_full = pd.read_excel(filepath)

        dmu_col = config.get('dmu_column', '')
        if dmu_col and dmu_col in df_full.columns:
            df_full.set_index(dmu_col, inplace=True)

        # Default DMU to first if not specified
        if not dmu_name or dmu_name not in df_full.index:
            dmu_name = df_full.index[0] if len(df_full) > 0 else ''

        from simulation_engine import simulate_what_if
        result = simulate_what_if(
            df_full, dmu_name, input_cols, output_cols,
            targets, orientation, undesirable_outputs, normalize
        )
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in simulation: {e}", exc_info=True)
        return jsonify({'feasible': False, 'violations': [f'Error interno: {str(e)}']}), 500


@app.route('/api/diagnosis', methods=['POST'])
def api_diagnosis():
    """API endpoint para generar diagnóstico LLM de una DMU."""
    from ai_insights import generate_dmu_diagnosis

    data = request.get_json(force=True)
    dmu_index = data.get("dmu_index")

    if "dea_results_raw" not in session:
        return jsonify({"error": "No hay resultados de análisis disponibles"}), 400

    try:
        results = session["dea_results_raw"]
        config = session.get("analysis_config", {})
        input_cols = config.get("input_columns", [])
        output_cols = config.get("output_columns", [])
        orientation = config.get("orientation", "output")

        scores_df = pd.read_json(results['raw_scores_df_json'], orient='split')
        slacks_df = pd.read_json(results['raw_slacks_df_json'], orient='split')

        if dmu_index is None or dmu_index < 0 or dmu_index >= len(scores_df):
            return jsonify({"error": "Índice de DMU inválido"}), 400

        score_row = scores_df.iloc[dmu_index]
        dmu_name = str(score_row.iloc[0]) if len(score_row) > 0 else f"DMU {dmu_index}"
        score = float(score_row.iloc[-1]) if len(score_row) > 0 else 0.0

        slacks_dict = {}
        if dmu_index < len(slacks_df):
            slack_row = slacks_df.iloc[dmu_index]
            for col in slack_row.index:
                if col != slacks_df.columns[0]:
                    try:
                        slacks_dict[str(col)] = float(slack_row[col])
                    except (ValueError, TypeError):
                        slacks_dict[str(col)] = 0.0

        dmu_data = {"name": dmu_name, "_dmu_name": dmu_name}

        diagnosis = generate_dmu_diagnosis(
            dmu_row=dmu_data,
            peers=[],
            slacks=slacks_dict,
            input_cols=input_cols,
            output_cols=output_cols,
            score=score,
            orientation=orientation,
            provider=data.get("provider", "auto"),
        )

        return jsonify(diagnosis)

    except Exception as e:
        logger.error(f"Error generating diagnosis: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/download_report')
def download_report():
    dea_results_raw_from_session = session.get('dea_results_raw')
    analysis_config_from_session = session.get('analysis_config')

    if not dea_results_raw_from_session or not analysis_config_from_session:
        flash("No hay resultados para descargar. Realice un análisis primero.", 'warning')
        return redirect(url_for('show_results'))

    try:
        # Construir el diccionario que generate_pdf_report espera
        # generate_pdf_report espera claves como 'raw_scores_df', no 'raw_scores_df_json'
        data_for_pdf = {
            'raw_scores_df': dea_results_raw_from_session.get('raw_scores_df_json'),
            'raw_slacks_df': dea_results_raw_from_session.get('raw_slacks_df_json'),
            'raw_targets_df': dea_results_raw_from_session.get('raw_targets_df_json'),
            'raw_lambdas_df': dea_results_raw_from_session.get('raw_lambdas_df_json'),
            # 'original_inputs_df' y 'original_outputs_df' no son usados directamente por
            # generate_pdf_report para tablas, pero se usan para la gráfica si está presente.
            # La gráfica se pasa por graph_image_path.
            'summary': dea_results_raw_from_session.get('summary')
        }
        
        current_analysis_config = analysis_config_from_session.copy()
        if session.get('current_graph_config'):
            current_analysis_config['current_graph_config'] = session.get('current_graph_config')

        graph_image_path = session.get('current_graph_image_path') 

        logger.info(f"Iniciando generación de PDF. Graph path: {graph_image_path}")
        pdf_content = generate_pdf_report(data_for_pdf, current_analysis_config, graph_image_path)
        
        if pdf_content:
            buffer = io.BytesIO(pdf_content)
            buffer.seek(0)
            report_filename = f"dea_report_{current_analysis_config.get('filename', 'data').split('.')[0]}.pdf"
            logger.info(f"PDF generado: {report_filename}")
            return send_file(buffer,
                             as_attachment=True,
                             download_name=report_filename,
                             mimetype='application/pdf')
        else:
            flash("Error al generar el reporte PDF. Verifique los logs.", 'error')
            return redirect(url_for('show_results'))

    except Exception as e:
        logger.error(f"Error al descargar el reporte PDF: {e}", exc_info=True)
        flash(f"Error al generar el PDF: {str(e)}", 'error')
        return redirect(url_for('show_results'))

@app.route('/clear_session', methods=['POST', 'GET']) 
def clear_session_route():
    keys_to_clear = [
        'filepath', 'filename', 'columns', 'potential_dmu_cols', 'numeric_columns',
        'analysis_config', 'dea_results_raw', 'error_message', 
        'current_graph_html', 'current_graph_config', 'current_graph_image_path',
        'ml_results', 'selected_dmu',
    ]
    for key in keys_to_clear:
        session.pop(key, None)
    
    logger.info("Sesión limpiada.")
    flash("Puede comenzar un nuevo análisis.", 'info')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
