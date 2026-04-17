# DEA ChampionSys

Sistema de análisis de Eficiencia de Datos (Data Envelopment Analysis - DEA).

## Requisitos Previos

- Python 3.9+
- Pip (última versión instalada automáticamente en el entorno virtual)

## Instalación

Para configurar el entorno de desarrollo y ejecutar la aplicación desde la consola, siga estos pasos:

1. **Crear el entorno virtual:**
   ```bash
   python3 -m venv venv
   ```

2. **Activar el entorno virtual:**
   - En macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - En Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Instalar dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Ejecución

Una vez activado el entorno virtual y con las dependencias instaladas, ejecute:

```bash
python app.py
```

La aplicación estará disponible en: [http://localhost:5001](http://localhost:5001)

## Estructura del Proyecto

- `app.py`: Punto de entrada de la aplicación Flask.
- `dea_logic.py`: Lógica para los cálculos de DEA y PyFrontier.
- `pdf_utils.py`: Utilidades para la generación de reportes en PDF.
- `templates/`: Archivos HTML para la interfaz de usuario.
- `static/`: Recursos estáticos (CSS, JS e imágenes generadas).
- `uploads/`: Directorio temporal para archivos subidos por el usuario.
- `requirements.txt`: Lista de dependencias del proyecto.
