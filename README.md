# Stock Analyzer — Herramienta profesional de análisis de empresas cotizadas

Herramienta de uso personal para análisis riguroso de acciones combinando:
- Análisis fundamental (calidad del negocio, márgenes, deuda, crecimiento, FCF)
- Análisis de valoración (múltiplos + DCF con rangos conservador/base/optimista)
- Análisis técnico (tendencia, soportes, RSI, MACD, volatilidad)
- Gestión del riesgo (stop técnico, invalidación de tesis, tamaño de posición)
- Síntesis con plan orientativo de entrada/salida

> **Aviso importante**: Esta herramienta es de apoyo a la decisión, no asesoramiento financiero.
> Los niveles de entrada/salida son probabilísticos, no predicciones.

---

## Arquitectura

```
stock-analyzer/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI — endpoints REST
│   │   ├── config.py             # Configuración desde .env
│   │   ├── models/
│   │   │   ├── inputs.py         # Pydantic: parámetros de entrada
│   │   │   └── outputs.py        # Pydantic: estructura de respuesta completa
│   │   ├── providers/
│   │   │   ├── base.py           # Interfaz abstracta DataProvider
│   │   │   └── yfinance_provider.py  # Implementación con Yahoo Finance
│   │   └── analysis/
│   │       ├── fundamental.py    # Análisis fundamental + scoring
│   │       ├── technical.py      # Indicadores técnicos + interpretación
│   │       ├── valuation.py      # Múltiplos + DCF
│   │       ├── entry_exit.py     # Zonas de entrada/salida
│   │       ├── risk.py           # Análisis de riesgo
│   │       └── synthesis.py      # Motor de síntesis final
│   ├── tests/
│   │   ├── test_fundamental.py
│   │   ├── test_technical.py
│   │   └── test_valuation.py
│   └── pytest.ini
├── frontend/
│   ├── app.py                    # Streamlit — interfaz web
│   └── components/
│       ├── charts.py             # Gráficos Plotly
│       ├── tables.py             # Tablas de datos
│       └── narrative.py          # Paneles de texto/conclusiones
├── .env.example
├── requirements.txt
└── README.md
```

### Flujo de datos

```
Usuario (Streamlit)
    → POST /analyze (FastAPI)
    → DataProvider.get_*() (yfinance)
    → analyze_fundamental()
    → analyze_technical()
    → analyze_valuation()
    → analyze_entry_exit()
    → analyze_risk()
    → analyze_synthesis()
    → AnalysisResponse (JSON)
    → Streamlit renderiza resultados
```

---

## Requisitos previos

- Python 3.11+
- Conexión a internet (para descargar datos de Yahoo Finance)
- pip o uv

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <repo-url>
cd stock-analyzer
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
# Edita .env si necesitas cambiar puertos o añadir API keys opcionales
```

---

## Ejecución

### Opción A: Interfaz completa (backend + frontend)

**Terminal 1 — Backend FastAPI:**
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend Streamlit:**
```bash
cd frontend
streamlit run app.py --server.port 8501
```

Abre el navegador en `http://localhost:8501`

### Opción B: Solo API (para integración o pruebas)

```bash
cd backend
uvicorn app.main:app --reload
```

Documentación interactiva en `http://localhost:8000/docs`

### Opción C: Análisis desde Python directamente

```python
from backend.app.providers import get_provider
from backend.app.analysis import analyze_fundamental, analyze_technical, analyze_valuation

provider = get_provider("yfinance")
fundamentals = provider.get_fundamentals("AAPL")
statements = provider.get_financial_statements("AAPL")
prices = provider.get_price_history("AAPL")

result = analyze_fundamental(fundamentals, statements)
print(f"Score: {result.overall_fundamental_score}/10")
print(result.narrative)
```

---

## Tests

```bash
cd backend
pytest                          # todos los tests
pytest tests/test_fundamental.py -v   # módulo específico
pytest --cov=app tests/         # con cobertura
```

Los tests usan datos sintéticos y no requieren conexión a internet.

---

## Parámetros de análisis

| Parámetro | Opciones | Descripción |
|---|---|---|
| `ticker` | cualquier símbolo | AAPL, IBE.MC, LLOY.L... |
| `time_horizon` | short / medium / long | Ajusta targets de salida |
| `risk_profile` | conservative / balanced / aggressive | Ajusta entradas y stops |
| `analysis_style` | fundamental / technical / mixed | Pesos del score final |
| `valuation_approach` | base / conservative | +30% margen de seguridad en DCF |
| `dcf_growth_rate_override` | float (0.10 = 10%) | Sobreescribe la tasa de crecimiento del DCF |
| `dcf_discount_rate_override` | float | Sobreescribe el WACC del DCF |

---

## Cómo se calcula cada cosa

### Puntuaciones (0–10)

Cada módulo produce una puntuación con factores explícitos:

- **Calidad del negocio**: margen bruto (2pts), margen operativo (2pts), FCF consistency (1.5pts), ROE (1.5pts)
- **Fortaleza financiera**: ND/EBITDA (3pts), cobertura intereses (2pts), current ratio (1pt)
- **Crecimiento**: crecimiento ingresos 1A+3A CAGR (4.5pts), EPS growth (2pts), FCF growth (1.5pts)
- **Eficiencia**: ROE (3pts), ROA (2pts), margen neto (2pts)

El score global pondera los módulos según el estilo elegido:
- `mixed`: Fundamental 40% + Valoración 30% + Técnico 30%
- `fundamental`: Fundamental 55% + Valoración 35% + Técnico 10%
- `technical`: Fundamental 15% + Valoración 20% + Técnico 65%

### Zonas de entrada/salida

La lógica es multicapa:

1. **Ancla de valoración**: valor razonable conservador/base/optimista
2. **Ancla técnica**: soportes/resistencias por swing highs-lows
3. **Ajuste por extensión**: si RSI>72 o precio >15% sobre SMA50, las entradas se alejan del precio actual
4. **Ajuste por tendencia**: bajista → entradas más conservadoras
5. **Ajuste por perfil**: conservador exige 20% de descuento vs. valor base; agresivo acepta 3%
6. **Stop técnico**: soporte más cercano menos N×ATR (N depende del perfil)
7. **Targets**: valor base (parcial) y optimista (revisión/salida)

### DCF simplificado

- Flujo base: FCF por acción (o EPS ajustado si no hay FCF)
- Fase 1 (años 1-5): crecimiento estimado del histórico
- Fase 2 (años 6-10): 60% del crecimiento de fase 1
- Valor terminal: Gordon Growth Model con 2.5% de crecimiento perpetuo
- WACC estimado: CAPM (Rf=4.5%, ERP=5.5%) + peso de deuda
- Sensibilidad: ±2pp en tasa de crecimiento

---

## Fuentes de datos

### Primaria: Yahoo Finance (via yfinance)
- Cobertura: mercados globales
- Datos: histórico OHLCV, fundamentales, estados financieros
- Limitaciones: datos institucionales limitados, sin estimaciones de analistas detalladas
- Coste: gratuito, sin API key

### Cómo añadir un proveedor nuevo

1. Crea `backend/app/providers/nuevo_provider.py`
2. Hereda de `DataProvider` e implementa los tres métodos abstractos:
   - `get_price_history(ticker, period, interval) → PriceHistory`
   - `get_fundamentals(ticker) → FundamentalData`
   - `get_financial_statements(ticker) → FinancialStatements`
3. Registra en `backend/app/providers/__init__.py`:
   ```python
   from .nuevo_provider import NuevoProvider
   registry["nuevo"] = NuevoProvider
   ```
4. Cambia `DATA_PROVIDER=nuevo` en `.env`

---

## Extensiones futuras

### Fáciles de añadir
- **Estimaciones de analistas**: Alpha Vantage o Finnhub tienen precio consenso y estimaciones EPS forward
- **Comparativa sectorial real**: Obtener mediana de ratios de empresas del mismo sector
- **Alertas**: Webhook o email cuando el precio entra en zona de entrada
- **Historial de análisis**: Guardar análisis en SQLite para comparar evolución
- **Exportar PDF**: Generar informe en PDF con weasyprint o reportlab

### Requieren más trabajo
- **Modelo de scoring ML**: Entrenar con histórico de empresas y rentabilidades reales
- **Screener**: Buscar empresas que cumplan criterios (sector + valoración + técnico)
- **Backtesting**: Evaluar qué tal habrían funcionado las zonas de entrada históricamente

---

## Limitaciones y advertencias

1. Los datos de Yahoo Finance pueden tener retrasos o imprecisiones
2. Los múltiplos sectoriales son benchmarks genéricos, no comparativas precisas
3. El DCF es una simplificación — la valoración real requiere análisis profundo del negocio
4. El análisis técnico describe el pasado, no predice el futuro
5. No se tienen en cuenta factores macroeconómicos ni eventos corporativos próximos
6. Esta herramienta no sustituye a un asesor financiero cualificado

---

## Licencia

Uso personal. No redistribuir sin autorización.
