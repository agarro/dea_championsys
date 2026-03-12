import io
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
import html
import os
import logging
import re

logger = logging.getLogger(__name__)

# Clase wrapper para depurar el dibujado de Paragraphs
class DebuggableParagraph(Flowable):
    def __init__(self, paragraph_obj, context_info="N/A"):
        Flowable.__init__(self)
        self.paragraph_obj = paragraph_obj
        self.context_info = context_info
        # Asegurar que el estilo del párrafo interno tenga un nombre único si es posible
        if hasattr(self.paragraph_obj, 'style') and not self.paragraph_obj.style.name.startswith(f"DebugStyle_{self.context_info.replace(' ','_')[:20]}"):
             # Clonar el estilo para evitar modificar el original si es compartido
            cloned_style = ParagraphStyle(name=f"DebugStyle_{self.context_info.replace(' ','_')[:30]}_{id(self)}", parent=self.paragraph_obj.style)
            self.paragraph_obj.style = cloned_style


    def wrap(self, availWidth, availHeight):
        # Delegar wrap al Paragraph interno
        try:
            return self.paragraph_obj.wrapOn(None, availWidth, availHeight) # Pasamos None como canvas para wrapOn
        except Exception as e:
            logger.error(f"Error en DebuggableParagraph.wrap para contexto '{self.context_info}': {e}", exc_info=True)
            # Fallback si wrap falla: intentar crear un párrafo de error simple
            error_style = ParagraphStyle(name=f'ErrorWrapStyle_{self.context_info.replace(" ","_")[:20]}', parent=getSampleStyleSheet()['Normal'])
            error_style.textColor = colors.red
            error_p = Paragraph(f"[WRAP ERR: {str(self.paragraph_obj.text)[:20]}]", error_style)
            return error_p.wrapOn(None, availWidth, availHeight)


    def draw(self):
        # Registrar antes de intentar dibujar
        logger.info(f"Intentando dibujar DebuggableParagraph para contexto: '{self.context_info}', Texto (primeros 50): '{str(self.paragraph_obj.text)[:50]}'")
        try:
            self.paragraph_obj.canv = self.canv # Asegurar que el párrafo interno use el canvas correcto
            self.paragraph_obj.draw()
        except UnboundLocalError as ule_draw: # Captura específica del error dpl si ocurre aquí
            logger.error(f"UnboundLocalError (dpl?) DENTRO DE DebuggableParagraph.draw para contexto '{self.context_info}'. Texto: '{str(self.paragraph_obj.text)[:50]}'. Error: {ule_draw}", exc_info=True)
            # Intentar dibujar un mensaje de error en el canvas si es posible
            try:
                error_style = ParagraphStyle(name=f'ErrorDrawStyle_{self.context_info.replace(" ","_")[:20]}', parent=getSampleStyleSheet()['Normal'])
                error_style.textColor = colors.red
                error_p = Paragraph(f"[DRAW ULE ERR: {str(self.paragraph_obj.text)[:20]}]", error_style)
                # Necesitamos hacer wrap antes de drawOn
                w, h = error_p.wrapOn(self.canv, self.width if hasattr(self, 'width') else 100, self.height if hasattr(self, 'height') else 20) # Usar self.width/height si están disponibles
                error_p.drawOn(self.canv, 0, 0) # Dibujar en la posición actual del flowable
            except Exception as e_draw_err:
                logger.error(f"Error al intentar dibujar mensaje de error en canvas para DebuggableParagraph: {e_draw_err}")
            raise # Relanzar la excepción original para que el proceso de build falle como antes
        except Exception as e_draw:
            logger.error(f"Excepción general DENTRO DE DebuggableParagraph.draw para contexto '{self.context_info}'. Texto: '{str(self.paragraph_obj.text)[:50]}'. Error: {e_draw}", exc_info=True)
            raise


def insert_zero_width_spaces(text, max_len_no_space=50):
    """
    Inserta espacios de ancho cero en palabras largas para ayudar con el ajuste de línea.
    """
    if not text or text == "\u00A0":
        return text
    
    new_text_parts = []
    for word in text.split(' '): # Dividir por espacios existentes
        if len(word) > max_len_no_space:
            # Insertar ZWS cada max_len_no_space caracteres dentro de la palabra larga
            new_word = ''
            for i in range(0, len(word), max_len_no_space):
                new_word += word[i:i+max_len_no_space] + '\u200B' # Añadir ZWS
            new_text_parts.append(new_word.rstrip('\u200B')) # Quitar el último ZWS si se añadió
        else:
            new_text_parts.append(word)
    return ' '.join(new_text_parts)


def create_safe_paragraph(text_content, style, context_info="N/A", enable_zws=True):
    """
    Crea un objeto DebuggableParagraph (que envuelve un Paragraph de ReportLab),
    asegurándose de que el texto sea seguro.
    """
    original_text_for_log = str(text_content)[:200] 

    if text_content is None or (isinstance(text_content, float) and np.isnan(text_content)):
        processed_text = "\u00A0"
    else:
        processed_text = str(text_content).strip()

    if not processed_text or processed_text.lower() == 'nan' or processed_text.lower() == 'none':
        processed_text = "\u00A0"
    
    if processed_text != "\u00A0": 
        processed_text = html.escape(processed_text)

    processed_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', processed_text)

    if enable_zws and processed_text != "\u00A0":
        processed_text = insert_zero_width_spaces(processed_text) # Aplicar ZWS

    if not processed_text.strip(): 
        processed_text = "\u00A0"
    
    try:
        # Crear el Paragraph real
        para_obj = Paragraph(processed_text, style)
        # Envolverlo en DebuggableParagraph
        return DebuggableParagraph(para_obj, context_info)
    except Exception as e: # Captura cualquier error durante la creación del Paragraph
        logger.error(f"Excepción AL CREAR Paragraph (antes de DebuggableWrapper) para contexto '{context_info}'. Texto (procesado, max 200): '{processed_text[:200]}'. Original (max 200): '{original_text_for_log}'. Error: {e}", exc_info=True)
        
        # Fallback: crear un Paragraph de error y envolverlo también
        error_style = ParagraphStyle(name=f'ErrorCreationStyle_{context_info.replace(" ","_")[:20]}', parent=getSampleStyleSheet()['Normal'])
        error_style.textColor = colors.red
        error_style.fontSize = style.fontSize if hasattr(style, 'fontSize') else 7
        error_para_obj = Paragraph(f"[ERROR CREACIÓN TEXTO: {html.escape(processed_text[:30])}...]", error_style)
        return DebuggableParagraph(error_para_obj, f"ERROR_FALLBACK_{context_info}")


def df_to_reportlab_table(df_original, available_width, fixed_col_widths=None, default_font_size=7, header_font_size=8, table_name="GenericTable"):
    if df_original is None or df_original.empty:
        logger.warning(f"DataFrame de entrada para df_to_reportlab_table ('{table_name}') es None o vacío.")
        return None

    df = df_original.copy()
    header_list_str = []
    include_index_col = False
    index_col_name_for_log = "Index"

    if df.index.name is not None and str(df.index.name).strip():
        index_col_name_for_log = str(df.index.name)
        header_list_str.append(index_col_name_for_log)
        include_index_col = True
    elif isinstance(df.index, pd.MultiIndex):
        index_names = [str(name) if name is not None else f"Idx{i}" for i, name in enumerate(df.index.names)]
        index_col_name_for_log = " / ".join(index_names)
        header_list_str.append(index_col_name_for_log)
        include_index_col = True
    elif not isinstance(df.index, pd.RangeIndex): 
        header_list_str.append(index_col_name_for_log)
        include_index_col = True
    
    header_list_str.extend([str(col) for col in df.columns])
    
    data_list_str = []
    for idx_row, (index_val, row_series) in enumerate(df.iterrows()):
        current_row_str = []
        if include_index_col:
            if isinstance(index_val, tuple): 
                current_row_str.append(str(" / ".join(map(str, index_val))))
            else:
                current_row_str.append(str(index_val))
        
        for col_name in df.columns:
            val = row_series[col_name]
            if isinstance(val, (int, float, np.number)):
                if pd.isna(val):
                    current_row_str.append("") 
                elif isinstance(val, float):
                    current_row_str.append(f"{val:.4f}")
                else: 
                    current_row_str.append(str(val))
            else:
                current_row_str.append(str(val)) 
        data_list_str.append(current_row_str)

    if not header_list_str: 
        logger.warning(f"No hay encabezados para la tabla '{table_name}' (después de procesar índice).")
        return None

    base_style_sheet = getSampleStyleSheet()
    # Crear estilos con nombres únicos para evitar conflictos
    cell_style_name = f'Cell_{table_name}_{id(df)}'
    cell_style = ParagraphStyle(name=cell_style_name, parent=base_style_sheet['Normal'])
    cell_style.fontSize = default_font_size
    cell_style.leading = default_font_size + 2.5 # Ajustar leading
    cell_style.alignment = 0 # TA_LEFT
    cell_style.spaceBefore = 1
    cell_style.spaceAfter = 1
    
    header_cell_style_name = f'HeaderCell_{table_name}_{id(df)}'
    header_cell_style = ParagraphStyle(name=header_cell_style_name, parent=base_style_sheet['Normal'])
    header_cell_style.fontSize = header_font_size
    header_cell_style.leading = header_font_size + 2.5 # Ajustar leading
    header_cell_style.alignment = 0 # TA_LEFT
    header_cell_style.fontName = 'Helvetica-Bold'
    header_cell_style.spaceBefore = 1
    header_cell_style.spaceAfter = 1

    formatted_data = []
    # Para los encabezados, generalmente no queremos ZWS
    formatted_data.append([create_safe_paragraph(cell_value, header_cell_style, context_info=f"Table '{table_name}', Header: {str(cell_value)[:50]}", enable_zws=False) for cell_value in header_list_str])
    
    for idx_r, row_str_values in enumerate(data_list_str):
        formatted_row = []
        actual_row_index_for_log = df.index[idx_r] if idx_r < len(df.index) else f"OOB_{idx_r}"
        col_names_for_log = ([index_col_name_for_log] if include_index_col else []) + df.columns.tolist()

        for idx_c, cell_value in enumerate(row_str_values):
            col_log_name = col_names_for_log[idx_c] if idx_c < len(col_names_for_log) else f"OOB_Col_{idx_c}"
            context = f"Table '{table_name}', Row_idx '{str(actual_row_index_for_log)[:50]}', Col_name '{str(col_log_name)[:50]}'"
            # Habilitar ZWS para el contenido de las celdas
            formatted_row.append(create_safe_paragraph(cell_value, cell_style, context_info=context, enable_zws=True))
        formatted_data.append(formatted_row)
        
    num_cols = len(header_list_str)
    col_widths = None

    if fixed_col_widths and len(fixed_col_widths) == num_cols:
        col_widths = fixed_col_widths
    else:
        if num_cols > 0:
            effective_available_width = available_width * 0.98 
            base_col_width = effective_available_width / num_cols
            min_col_width = 0.20 * inch 
            col_width_val = max(min_col_width, base_col_width)
            if col_width_val * num_cols > effective_available_width and num_cols > 0:
                 col_width_val = effective_available_width / num_cols
                 col_width_val = max(min_col_width, col_width_val) 
            col_widths = [col_width_val] * num_cols
        else: 
            logger.warning(f"df_to_reportlab_table ('{table_name}') llamada sin columnas en el header_list_str.")
            return None
    
    if col_widths is None: 
        logger.error(f"col_widths es None para tabla '{table_name}', no se puede crear la tabla.")
        return create_safe_paragraph(f"Error: No se pudieron determinar los anchos de columna para la tabla {table_name}.", base_style_sheet['Normal'], enable_zws=False)

    try:
        table = Table(formatted_data, colWidths=col_widths, repeatRows=1) 
    except Exception as e:
        logger.error(f"Error al crear la tabla '{table_name}' con ReportLab: {e}", exc_info=True)
        return create_safe_paragraph(f"Error: No se pudo generar la tabla {table_name}.", base_style_sheet['Normal'], enable_zws=False)

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E0E0E0")), 
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'), 
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5), 
        ('TOPPADDING', (0, 0), (-1, 0), 3), 
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('TOPPADDING', (0, 1), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey), 
        ('LEFTPADDING', (0,0), (-1,-1), 3), 
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ])
    table.setStyle(style)
    return table

def generate_pdf_report(dea_results_json, analysis_config, graph_image_path=None):
    try:
        buffer = io.BytesIO()
        page_width, page_height = A4
        margin_horizontal = 1.0 * cm 
        margin_vertical = 1.5 * cm 
        available_width = page_width - 2 * margin_horizontal
        
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=margin_horizontal, leftMargin=margin_horizontal,
                                topMargin=margin_vertical, bottomMargin=margin_vertical)
        
        styles = getSampleStyleSheet()
        story = []

        style_h1 = ParagraphStyle(name='CustomHeading1_Main', parent=styles['h1'], fontSize=16, spaceAfter=12, alignment=1, textColor=colors.HexColor("#1E40AF"))
        style_h2 = ParagraphStyle(name='CustomHeading2_Main', parent=styles['h2'], fontSize=12, spaceAfter=8, spaceBefore=10, textColor=colors.HexColor("#1D4ED8"), fontName='Helvetica-Bold')
        style_body = ParagraphStyle(name='CustomBody_Main', parent=styles['Normal'])
        style_body.fontSize = 9
        style_body.leading = 11
        
        style_config_label = ParagraphStyle(name='CustomConfigLabel_Main', parent=style_body, fontName='Helvetica-Bold')
        
        style_small_body = ParagraphStyle(name='CustomSmallBody_Main', parent=styles['Normal'])
        style_small_body.fontSize = 7
        style_small_body.leading = 9

        # Usar enable_zws=False para títulos y elementos donde no se espera texto largo problemático
        story.append(create_safe_paragraph("Reporte de Análisis DEA", style_h1, context_info="Main Title", enable_zws=False))
        story.append(Spacer(1, 0.2*inch))

        story.append(create_safe_paragraph("Configuración del Análisis", style_h2, context_info="Config Subtitle", enable_zws=False))
        
        config_items = [
            ("Archivo de Datos:", str(analysis_config.get('filename', 'N/A'))),
            ("Columna DMU:", str(analysis_config.get('dmu_column', 'N/A'))),
            ("Orientación:", str(analysis_config.get('orientation', 'N/A')).capitalize() + "-orientado"),
            ("Modelo Elegido:", str(analysis_config.get('chosen_model', 'N/A')) + " (automático)"),
            ("Normalización Aplicada:", "Sí" if analysis_config.get('normalize') else "No"),
            ("Inputs Seleccionados:", ", ".join(analysis_config.get('input_columns', []))),
            ("Outputs Seleccionados:", ", ".join(analysis_config.get('output_columns', []))),
        ]
        if analysis_config.get('undesirable_outputs'):
            config_items.append(("Outputs No Deseables (original):", ", ".join(analysis_config.get('undesirable_outputs', []))))

        config_data_p = []
        for idx, (label_text, value_text) in enumerate(config_items):
            # Habilitar ZWS para value_text si puede ser largo, deshabilitar para label_text
            label_p = create_safe_paragraph(label_text, style_config_label, context_info=f"Config Item Label {idx}: {str(label_text)[:50]}", enable_zws=False)
            value_p = create_safe_paragraph(value_text, style_body, context_info=f"Config Item Value {idx}: {str(value_text)[:50]}", enable_zws=True)
            config_data_p.append([label_p, value_p])
        
        config_table_p = Table(config_data_p, colWidths=[2.5*inch, available_width - 2.5*inch - 0.1*inch])
        config_table_p.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 4)]))
        story.append(config_table_p)
        story.append(Spacer(1, 0.1*inch))
        
        summary_text_val = dea_results_json.get('summary', '')
        story.append(create_safe_paragraph(summary_text_val, style_body, context_info=f"Summary Text: {str(summary_text_val)[:100]}", enable_zws=True))
        story.append(Spacer(1, 0.2*inch))

        def footer_canvas(canvas, doc):
            canvas.saveState()
            footer_style = ParagraphStyle(name='CustomFooterMain_Page', parent=styles['Normal'], fontSize=8, alignment=1, textColor=colors.grey)
            p = create_safe_paragraph(f"Reporte generado por la Herramienta de Análisis DEA - Página {doc.page}", 
                                      footer_style, 
                                      context_info=f"Footer Page {doc.page}", enable_zws=False)
            w, h = p.wrapOn(canvas, doc.width, doc.bottomMargin) 
            p.drawOn(canvas, doc.leftMargin, doc.bottomMargin - h - 0.2*cm) 
            canvas.restoreState()

        try:
            scores_df = pd.read_json(dea_results_json['raw_scores_df'], orient='split') if dea_results_json.get('raw_scores_df') else pd.DataFrame()
            slacks_df = pd.read_json(dea_results_json['raw_slacks_df'], orient='split') if dea_results_json.get('raw_slacks_df') else pd.DataFrame()
            targets_df = pd.read_json(dea_results_json['raw_targets_df'], orient='split') if dea_results_json.get('raw_targets_df') else pd.DataFrame()
            lambdas_df = pd.read_json(dea_results_json['raw_lambdas_df'], orient='split') if dea_results_json.get('raw_lambdas_df') else pd.DataFrame()
        except KeyError as e:
            logger.error(f"Falta una clave JSON para los DataFrames de resultados: {e}")
            story.append(create_safe_paragraph(f"Error: Faltan datos para generar las tablas ({e}).", styles['Normal'], "Error Msg Missing Data", enable_zws=False))
            doc.build(story, onLaterPages=footer_canvas, onFirstPage=footer_canvas) 
            return buffer.getvalue() 

        table_sections = [
            ("Puntajes de Eficiencia", "Un score de 1 indica una DMU eficiente. Menor a 1 indica ineficiencia.", scores_df, 8, 9, "ScoresTable"),
            ("Holguras (Slacks)", "Excesos de inputs o déficits de outputs para DMUs ineficientes.", slacks_df, 6, 7, "SlacksTable"),
            ("Objetivos (Targets)", "Niveles de inputs/outputs para que una DMU ineficiente alcance la frontera.", targets_df, 6, 7, "TargetsTable"),
            ("Lambdas (Pesos de Peers)", "Pesos de las DMUs eficientes (peers) en la proyección de las DMUs ineficientes.", lambdas_df, 5, 6, "LambdasTable")
        ]

        for title, desc, df_data, font_s, header_font_s, table_id_name in table_sections:
            if df_data is not None and not df_data.empty: 
                story.append(PageBreak())
                story.append(create_safe_paragraph(title, style_h2, context_info=f"Table Title: {title}", enable_zws=False))
                story.append(create_safe_paragraph(desc, style_small_body, context_info=f"Table Desc: {title}", enable_zws=True))
                story.append(Spacer(1, 0.1*inch))
                
                current_font_size = font_s
                current_header_font_size = header_font_s
                if title == "Lambdas (Pesos de Peers)":
                    num_lambda_cols = len(df_data.columns)
                    if df_data.index.name or not isinstance(df_data.index, pd.RangeIndex):
                        num_lambda_cols +=1
                        
                    if num_lambda_cols > 15: 
                        current_font_size = max(3, font_s - 2) 
                        current_header_font_size = max(4, header_font_s - 2)
                        logger.info(f"Tabla de Lambdas con {num_lambda_cols} columnas. Usando fuente {current_font_size}pt.")
                
                table_obj = df_to_reportlab_table(df_data, available_width, 
                                                  default_font_size=current_font_size, 
                                                  header_font_size=current_header_font_size,
                                                  table_name=table_id_name)
                if table_obj: 
                    story.append(table_obj)
                else:
                    story.append(create_safe_paragraph(f"No se pudo generar la tabla para {title}.", style_small_body, f"Error Msg Table Gen Fail: {title}", enable_zws=False))
                story.append(Spacer(1, 0.2*inch))
            else:
                logger.warning(f"DataFrame de {title} vacío o None, no se añadirá la tabla al PDF.")

        if graph_image_path and os.path.exists(graph_image_path):
            story.append(PageBreak())
            story.append(create_safe_paragraph("Gráfica de Frontera de Eficiencia", style_h2, "Graph Title", enable_zws=False))
            graph_desc = "Visualización 2D de un par input/output seleccionado."
            if analysis_config.get('current_graph_config'):
                cg_conf = analysis_config['current_graph_config']
                graph_desc += f" Input: {cg_conf.get('input', 'N/A')}, Output: {cg_conf.get('output', 'N/A')}."
            story.append(create_safe_paragraph(graph_desc, style_small_body, "Graph Description", enable_zws=True))
            story.append(Spacer(1, 0.2*inch))
            try:
                img = Image(graph_image_path)
                img_width_orig = img.imageWidth 
                img_height_orig = img.imageHeight
                max_img_width = available_width * 0.98 
                max_img_height = page_height * 0.65 
                scale_factor_w = max_img_width / img_width_orig if img_width_orig > 0 else 1
                scale_factor_h = max_img_height / img_height_orig if img_height_orig > 0 else 1
                scale_factor = min(scale_factor_w, scale_factor_h, 1) 
                img.drawWidth = img_width_orig * scale_factor
                img.drawHeight = img_height_orig * scale_factor
                img_table = Table([[img]], colWidths=[available_width])
                img_table.setStyle(TableStyle([('ALIGN', (0,0), (0,0), 'CENTER')]))
                story.append(img_table)
            except Exception as e:
                logger.error(f"No se pudo cargar o procesar la imagen de la gráfica para el PDF: {e}", exc_info=True)
                story.append(create_safe_paragraph(f"Error al cargar la imagen de la gráfica: {e}", style_body, "Error Msg Graph Image", enable_zws=False))
        else:
            logger.info(f"No se incluyó la gráfica en el PDF: ruta no válida o no proporcionada ('{graph_image_path}')")

        doc.build(story, onLaterPages=footer_canvas, onFirstPage=footer_canvas)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        logger.info("Reporte PDF generado exitosamente con ReportLab.")
        return pdf_bytes

    except Exception as e:
        logger.error(f"Error generando el reporte PDF con ReportLab (excepción principal): {e}", exc_info=True)
        try:
            buffer_err = io.BytesIO()
            styles_err_fallback = getSampleStyleSheet() 
            doc_err = SimpleDocTemplate(buffer_err, pagesize=A4)
            
            story_err = [Paragraph("Error Crítico en Generación de PDF", styles_err_fallback['h1'])]
            story_err.append(Paragraph(f"Ocurrió un error que impidió la generación completa del reporte: {html.escape(str(e))}", styles_err_fallback['Normal']))
            story_err.append(Paragraph("Por favor, revise los logs del servidor para más detalles.", styles_err_fallback['Normal']))
            
            def footer_err_canvas(canvas, doc):
                canvas.saveState()
                footer_err_style = ParagraphStyle(name='CustomFooterError_Fallback', parent=styles_err_fallback['Normal'], fontSize=8, alignment=1, textColor=colors.grey)
                p_err = Paragraph(f"Error Report - Page {doc.page}", footer_err_style) 
                w, h = p_err.wrapOn(canvas, doc.width, doc.bottomMargin)
                p_err.drawOn(canvas, doc.leftMargin, doc.bottomMargin - h - 0.2*cm)
                canvas.restoreState()

            doc_err.build(story_err, onFirstPage=footer_err_canvas, onLaterPages=footer_err_canvas)
            pdf_bytes_err = buffer_err.getvalue()
            buffer_err.close()
            logger.info("PDF de error generado.")
            return pdf_bytes_err
        except Exception as e_final:
            logger.error(f"Error generando el PDF de error (excepción de fallback): {e_final}", exc_info=True)
            return None
