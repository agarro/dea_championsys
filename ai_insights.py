import json
import os

# LLM providers: httpx for Ollama, openai/google-genai are optional
try:
    import httpx
except ImportError:
    httpx = None


def _heuristic_diagnosis(dmu_name, score, slacks, input_cols, output_cols):
    """Genera diagnóstico basado en reglas cuando no hay LLM disponible."""
    critical_slacks = []
    for col, val in slacks.items():
        if val > 0.01:
            critical_slacks.append((col, val))

    critical_slacks.sort(key=lambda x: x[1], reverse=True)

    diagnosis = {
        "dmu_name": dmu_name,
        "efficiency_score": round(score, 4),
        "status": "INEFICIENTE" if score < 1.0 else "EFICIENTE",
        "summary": "",
        "short_term": [],
        "medium_term": [],
        "long_term": [],
        "meta_inputs": {},
        "meta_outputs": {},
        "raw_slacks": slacks,
    }

    if score >= 1.0:
        diagnosis["summary"] = (
            f"La DMU '{dmu_name}' es eficiente (score={score}). "
            "Se encuentra en la frontera de eficiencia."
        )
        return diagnosis

    n_slacks = len(critical_slacks)
    diagnosis["summary"] = (
        f"La DMU '{dmu_name}' tiene un score de eficiencia de {score:.2%}. "
        f"Se detectaron {n_slacks} variables con holguras que requieren ajuste."
    )

    theta = score
    for col in input_cols:
        if col in slacks and slacks[col] > 0.01:
            diagnosis["meta_inputs"][col] = f"Reducir en {slacks[col]:.2f} unidades"
            diagnosis["short_term"].append(
                f"Reducir '{col}' en {slacks[col]:.2f} unidades (holgura)"
            )

    for col in output_cols:
        if col in slacks and slacks[col] > 0.01:
            diagnosis["meta_outputs"][col] = f"Aumentar en {slacks[col]:.2f} unidades"
            diagnosis["medium_term"].append(
                f"Incrementar '{col}' en {slacks[col]:.2f} unidades (holgura)"
            )

    if not diagnosis["short_term"] and not diagnosis["medium_term"]:
        diagnosis["long_term"].append("Revisar estrategia operativa general")

    return diagnosis


def _call_ollama(prompt, model="llama3.2"):
    """Llama a Ollama local."""
    if httpx is None:
        return None
    try:
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        print(f"[ai_insights] Ollama error: {e}")
        return None


def _call_openai(prompt, model="gpt-4o-mini"):
    """Llama a OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[ai_insights] OpenAI error: {e}")
        return None


def _call_gemini(prompt, model="gemini-2.0-flash"):
    """Llama a Google Gemini."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text
    except Exception as e:
        print(f"[ai_insights] Gemini error: {e}")
        return None


DIAGNOSIS_PROMPT = """Eres un analista de eficiencia DEA experto. Analiza la siguiente DMU ineficiente y genera un diagnóstico prescriptivo.

DMU: {dmu_name}
Score de Eficiencia: {score}
Orientación: {orientation}
Holguras detectadas:
{slacks_summary}

Para cada variable con holgura > 0:
- Input con holgura s_i: Meta = θ·x₀ᵢ − s_i⁻ (reducir input en s_i⁻ unidades)
- Output con holgura s_r: Meta = y₀ᵣ + s_r⁺ (aumentar output en s_r⁺ unidades)

Genera un JSON con esta estructura exacta:
{{
  "summary": "Resumen ejecutivo de 1-2 oraciones",
  "short_term": ["Acción inmediata 1", "Acción inmediata 2"],
  "medium_term": ["Acción mediano plazo 1"],
  "long_term": ["Estrategia largo plazo 1"]
}}

Sé conciso y accionable. Máximo 5 acciones por categoría."""


def generate_dmu_diagnosis(
    dmu_row,
    peers,
    slacks,
    input_cols,
    output_cols,
    score,
    orientation="output",
    provider="auto",
):
    """
    Genera diagnóstico y plan prescriptivo para una DMU ineficiente.

    Args:
        dmu_row: dict con datos de la DMU
        peers: lista de DMUs eficientes de referencia
        slacks: dict {variable: slack_value}
        input_cols: columnas de inputs
        output_cols: columnas de outputs
        score: efficiency score
        orientation: 'input' o 'output'
        provider: 'auto', 'ollama', 'openai', 'gemini', 'heuristic'

    Returns:
        dict con diagnóstico estructurado
    """
    dmu_name = dmu_row.get("name", dmu_row.get("_dmu_name", "DMU"))

    if score >= 1.0:
        return {
            "dmu_name": dmu_name,
            "efficiency_score": round(score, 4),
            "status": "EFICIENTE",
            "summary": f"La DMU '{dmu_name}' es eficiente (score={score:.4f}).",
            "short_term": [],
            "medium_term": [],
            "long_term": [],
            "meta_inputs": {},
            "meta_outputs": {},
            "provider_used": "none",
        }

    llm_result = None
    provider_used = "heuristic"

    if provider != "heuristic":
        slacks_summary = "\n".join(
            [f"- {k}: {v:.4f}" for k, v in slacks.items() if v > 0.01]
        )
        prompt = DIAGNOSIS_PROMPT.format(
            dmu_name=dmu_name,
            score=f"{score:.4f}",
            orientation=orientation,
            slacks_summary=slacks_summary,
        )

        if provider in ("auto", "ollama"):
            llm_result = _call_ollama(prompt)
            if llm_result:
                provider_used = "ollama"

        if not llm_result and provider in ("auto", "openai"):
            llm_result = _call_openai(prompt)
            if llm_result:
                provider_used = "openai"

        if not llm_result and provider in ("auto", "gemini"):
            llm_result = _call_gemini(prompt)
            if llm_result:
                provider_used = "gemini"

    if llm_result:
        try:
            clean = llm_result.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(clean)
            parsed["dmu_name"] = dmu_name
            parsed["efficiency_score"] = round(score, 4)
            parsed["status"] = "INEFICIENTE"
            parsed["provider_used"] = provider_used

            theta = score
            parsed["meta_inputs"] = {}
            parsed["meta_outputs"] = {}
            for col in input_cols:
                if col in slacks and slacks[col] > 0.01:
                    parsed["meta_inputs"][col] = f"Reducir en {slacks[col]:.2f}"
            for col in output_cols:
                if col in slacks and slacks[col] > 0.01:
                    parsed["meta_outputs"][col] = f"Aumentar en {slacks[col]:.2f}"

            return parsed
        except (json.JSONDecodeError, KeyError):
            pass

    result = _heuristic_diagnosis(dmu_name, score, slacks, input_cols, output_cols)
    result["provider_used"] = provider_used
    return result
