import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import mplfinance as mpf
import pandas as pd
import statsmodels.api as sm
import os
from dotenv import load_dotenv
import numpy as np
import anthropic
import matplotlib.pyplot as plt
import requests  # para SerpAPI

# ================== CONFIGURACIÓN PÁGINA ==================
st.set_page_config(
    page_title="Modelo de Regresión y Cálculo de Alphas",
    page_icon="📈",
    layout="wide",
)

# ================== ESTILO PERSONALIZADO (MODO OSCURO) ==================
custom_css = """
<style>
/* Fondo general */
.stApp {
    background: radial-gradient(circle at top left, #111827 0, #020617 45%, #000000 100%);
    color: #e5e7eb;
}

/* Contenedor principal tipo tarjeta de cristal */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
    padding-left: 2rem;
    padding-right: 2rem;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(18px);
    border-radius: 24px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.7);
}

/* Título principal */
h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #38bdf8, #a855f7, #f97316);
    -webkit-background-clip: text;
    color: transparent;
    letter-spacing: 0.05em;
}

/* Subtítulos */
h2, h3 {
    color: #e5e7eb !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #e5e7eb;
}

/* Tablas */
.dataframe tbody tr:nth-child(even) {
    background-color: rgba(15, 23, 42, 0.9);
}
.dataframe tbody tr:nth-child(odd) {
    background-color: rgba(15, 23, 42, 0.7);
}
.dataframe th {
    background-color: #020617;
    color: #e5e7eb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617 !important;
    border-right: 1px solid rgba(148, 163, 184, 0.4);
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ================== CONFIGURACIÓN ANTHROPIC / SERPAPI ==================
load_dotenv()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

if ANTHROPIC_KEY:
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
else:
    client = None

# ================== TÍTULO DEL APLICATIVO ==================
st.title("MODELO DE REGRESIÓN Y CÁLCULO DE ALPHAS")

# ================== GUÍA DE USO Y REFLEXIÓN ==================
with st.expander("¿Cómo usar este aplicativo?"):
    st.markdown("""
    1. Escribe el **ticker** de una empresa que cotiza en bolsa (por ejemplo: AAPL, MSFT, TSLA).
    2. La app descargará datos históricos y mostrará:
       - Un **gráfico de velas**.
       - Un modelo de **regresión CAPM** contra un índice (SPY, QQQ, DIA).
       - Los **múltiplos financieros** clave.
       - Una **interpretación automática** con Anthropic.
       - Un **modelo autorregresivo (AR)** del promedio móvil de 20 días.
    """)

with st.expander("Reflexión sobre el proyecto"):
    st.markdown("""
    Este proyecto integra series de tiempo y finanzas cuantitativas en una sola herramienta.
    Conecta el análisis de riesgo sistemático (CAPM), la valuación relativa por múltiplos
    y la dinámica del precio mediante modelos AR. Anthropic permite traducir resultados
    técnicos a lenguaje natural para usuarios no especializados.
    """)

# ================== CONTROLES SUPERIORES (GLOBALES) ==================
col_top1, col_top2 = st.columns([2, 1])

with col_top1:
    ticker = st.text_input("Ticker de la acción", value="AAPL", help="Ejemplos: AAPL, MSFT, TSLA, NVDA")

with col_top2:
    horizonte_opcion = st.selectbox(
        "Horizonte histórico",
        ["1 año", "3 años", "5 años"],
        index=2
    )

horizonte_map = {
    "1 año": 1,
    "3 años": 3,
    "5 años": 5,
}

# ================== SIDEBAR (SOLO BRANDING) ==================
st.sidebar.markdown("### 📊 Proyecto Series de Tiempo")
st.sidebar.markdown("Desarrollado por **Emiliano**")
st.sidebar.markdown("Usa Anthropic + SerpAPI + yfinance")

# ================== FECHAS ==================
end_date = datetime.today().date()
start_date = end_date - timedelta(days=365 * horizonte_map[horizonte_opcion])

# ================== LÓGICA PRINCIPAL ==================
if not ticker:
    st.warning("Hey compa; no sea migajero; ingrese un ticker")
else:
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.warning("No hay datos, revisa el ticker.")
    else:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)

        # ========== TABS PRINCIPALES ==========
        tab_ceo, tab_precio, tab_multiples, tab_ar, tab_news = st.tabs(
            ["👔 CEO & Management", "📊 Precio & CAPM", "📈 Múltiplos", "🔮 Modelo AR", "📰 Noticias"]
        )

        # ============ TAB 1: CEO & MANAGEMENT ============
        with tab_ceo:
            st.subheader(f"CEO y contexto ejecutivo de {ticker}")

            tk = yf.Ticker(ticker)
            try:
                info_ceo = tk.info
            except Exception:
                info_ceo = {}
                st.warning("No se pudo cargar la información ejecutiva desde yfinance.")

            company_name = info_ceo.get("longName", ticker)

            ceo_name = None
            officers = info_ceo.get("companyOfficers", [])

            if isinstance(officers, list):
                for o in officers:
                    title = str(o.get("title", "")).lower()
                    name = o.get("name")
                    if "chief executive officer" in title or "ceo" in title:
                        ceo_name = name
                        break
                if ceo_name is None and len(officers) > 0:
                    ceo_name = officers[0].get("name")

            if ceo_name is None:
                st.warning("No pude identificar el nombre del CEO.")
            else:
                st.markdown(f"### 👔 {ceo_name}")
                st.caption(f"Máximo responsable ejecutivo de **{company_name}**")

                # FOTO DEL CEO (SerpAPI)
                if not SERPAPI_KEY:
                    st.warning("Falta SERPAPI_KEY en el .env para buscar la foto del CEO.")
                else:
                    try:
                        params_img = {
                            "engine": "google_images",
                            "q": f"{ceo_name} CEO {company_name}",
                            "api_key": SERPAPI_KEY,
                            "ijn": "0",
                        }
                        resp_img = requests.get(
                            "https://serpapi.com/search.json",
                            params=params_img,
                            timeout=10
                        )
                        data_img = resp_img.json()
                        images = data_img.get("images_results", [])
                    except Exception as e:
                        images = []
                        st.error(f"Error al consultar imágenes: {e}")

                    if images:
                        img_url = (
                            images[0].get("original")
                            or images[0].get("thumbnail")
                            or images[0].get("link")
                        )
                        if img_url:
                            st.image(
                                img_url,
                                caption=f"{ceo_name} — {company_name}",
                                width=300,
                            )
                    else:
                        st.info("No se encontró imagen del CEO.")

                # DESCRIPCIÓN DE LA EMPRESA (Anthropic)
                if client is not None:
                    business_summary = info_ceo.get("longBusinessSummary", "")
                    business_summary = business_summary[:1500]

                    prompt_ceo = (
                        f"Nombre del CEO: {ceo_name}\n"
                        f"Empresa: {company_name}\n\n"
                        f"Resumen oficial del negocio:\n{business_summary}\n\n"
                        "Escribe una explicación clara y profesional (máximo 2 párrafos) que describa:\n"
                        "- Qué hace esta empresa y en qué industrias opera.\n"
                        "- Cómo encaja el rol del CEO dentro de la estrategia general del negocio.\n"
                        "- Evita inventar datos; usa solo la información del resumen.\n"
                        "- No hables de la vida personal del CEO, solo del negocio."
                    )

                    try:
                        ceo_analysis = client.messages.create(
                            model="claude-sonnet-4-5-20250929",
                            max_tokens=600,
                            temperature=0.2,
                            messages=[{"role": "user", "content": prompt_ceo}],
                        )
                        st.subheader("🏢 ¿Qué hace la empresa? (Anthropic)")
                        st.write(ceo_analysis.content[0].text)
                    except Exception as e:
                        st.error(f"Error al generar análisis del CEO: {e}")
                else:
                    st.warning("No se pudo generar análisis porque falta la ANTHROPIC_KEY.")

        # ============ TAB 2: PRECIO & CAPM ============
        with tab_precio:
            st.subheader(f"📊 Precio y modelo CAPM para {ticker}")

            # Control local: días para velas
            dias_velas = st.slider(
                "Días para gráfico de velas",
                min_value=60,
                max_value=365,
                value=183,
                step=5,
                help="Rango de días recientes que se mostrará en el gráfico de velas."
            )

            # Gráfico de velas
            df_plot = df[-dias_velas:].copy()
            df_plot.dropna(inplace=True)
            cols = ["Open", "High", "Low", "Close", "Volume"]
            df_plot[cols] = df_plot[cols].astype(float)

            mpf.plot(
                df_plot,
                type="candle",
                style="charles",
                title=f"Gráfico de velas: {ticker}",
                ylabel="Precio",
                volume=True,
                mav=(5, 10),
                figsize=(10, 6),
                tight_layout=True,
                show_nontrading=False,
                savefig="candlestick.png",
            )
            st.image("candlestick.png")

            # Control local: índice de referencia
            benchmark = st.selectbox(
                "Índice de referencia para CAPM",
                ["SPY", "QQQ", "DIA"],
                index=0,
                help="Elige contra qué índice quieres estimar Beta."
            )

            tickers_capm = [ticker, benchmark]
            df_capm = yf.download(tickers_capm, start=start_date, end=end_date)["Close"]
            df_ret = df_capm.pct_change()
            df_ret.columns = [f"Pc_{c}" for c in df_ret.columns]

            y_var = f"Pc_{ticker}"
            x_var = f"Pc_{benchmark}"
            df_reg = df_ret[[y_var, x_var]].dropna()

            X = sm.add_constant(df_reg[x_var])
            Y = df_reg[y_var]

            model = sm.OLS(Y, X).fit()

            alpha = model.params["const"]
            beta = model.params[x_var]
            r_squared = model.rsquared

            st.subheader("Resumen del modelo CAPM")
            c1, c2, c3 = st.columns(3)
            c1.metric("Alpha diario", f"{alpha:.6f}")
            c2.metric(f"Beta vs {benchmark}", f"{beta:.3f}")
            c3.metric("R²", f"{r_squared:.3f}")

            with st.expander("Ver detalle completo de la regresión"):
                st.text(model.summary())

            # Interpretación con Anthropic
            if client is not None:
                message = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=1000,
                    temperature=0.1,
                    system=(
                        "Rol:\nEres un analista cuantitativo especializado en riesgo financiero y análisis "
                        "de rendimiento relativo al mercado. Tu tarea es interpretar resultados de una regresión "
                        "lineal simple entre los retornos diarios de una acción (Y) y un índice de referencia (X).\n"
                        "Explica como a jóvenes interesados en inversiones, sin jerga pesada, usando términos "
                        "como riesgo sistemático, rendimiento alfa y significancia estadística.\n"
                        "Entrega una tabla compacta con Alpha, Alpha anualizado (252 días), Beta y R², y un texto "
                        "máximo de 800 caracteres explicando la relación con el mercado y el papel del alpha.\n"
                        "No des consejos de inversión ni uses títulos o subtítulos."
                    ),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Resultados de la regresión entre los rendimientos de {ticker} y {benchmark}:\n"
                                        f"Alpha (intercepto): {alpha}\n"
                                        f"Beta: {beta}\n"
                                        f"R²: {r_squared}\n"
                                        "Calcula el alpha anualizado (252 días) y genera el análisis solicitado."
                                    ),
                                }
                            ],
                        }
                    ],
                )
                st.subheader("Interpretación del CAPM (Anthropic)")
                st.write(message.content[0].text)
            else:
                st.warning("No se encontró ANTHROPIC_KEY en el .env, no se puede llamar a Claude.")

        # ============ TAB 3: MÚLTIPLOS ============
        with tab_multiples:
            st.subheader("📈 Múltiplos financieros clave")

            tk = yf.Ticker(ticker)
            try:
                info = tk.info
            except Exception:
                info = {}
                st.warning("No se pudieron cargar los fundamentales.")

            multiples_dict = {
                "Precio actual": info.get("currentPrice", None),
                "P/E (trailing)": info.get("trailingPE", None),
                "P/E (forward)": info.get("forwardPE", None),
                "P/B": info.get("priceToBook", None),
                "EV/EBITDA": info.get("enterpriseToEbitda", None),
                "EV/Ventas": info.get("enterpriseToRevenue", None),
                "Margen neto": info.get("profitMargins", None),
                "ROE": info.get("returnOnEquity", None),
                "ROA": info.get("returnOnAssets", None),
            }

            multiples_limpios = {k: v for k, v in multiples_dict.items() if v is not None}

            if len(multiples_limpios) == 0:
                st.warning("No hay múltiplos disponibles para este ticker.")
            else:
                df_multiples = pd.DataFrame(multiples_limpios, index=[ticker]).T
                df_multiples.columns = ["Valor"]
                st.dataframe(df_multiples)

                texto_multiples = "\n".join(
                    [f"{nombre}: {valor}" for nombre, valor in multiples_limpios.items()]
                )

                if client is not None:
                    mensaje_multiples = client.messages.create(
                        model="claude-sonnet-4-5-20250929",
                        max_tokens=900,
                        temperature=0.2,
                        system=(
                            "Rol:\nEres un analista financiero especializado en valoración relativa mediante múltiplos.\n"
                            "Analiza los múltiplos de una empresa pública y explica de forma clara qué implican "
                            "sobre valuación, rentabilidad y perfil de riesgo. No recomiendes comprar o vender."
                        ),
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"Múltiplos financieros de {ticker}:\n\n"
                                            f"{texto_multiples}\n\n"
                                            "1) Resume los datos en una tabla sencilla.\n"
                                            "2) Explica en menos de 900 caracteres qué implican sobre valuación, "
                                            "rentabilidad, estructura de capital y si el perfil es de crecimiento, "
                                            "defensivo o mixto.\n"
                                            "No des recomendaciones de inversión."
                                        ),
                                    }
                                ],
                            }
                        ],
                    )
                    st.subheader("Interpretación de Múltiplos (Anthropic)")
                    st.write(mensaje_multiples.content[0].text)
                else:
                    st.warning("No se puede interpretar múltiplos con Claude porque falta ANTHROPIC_KEY.")

                st.subheader("Ejercicios del curso aplicados")
                st.markdown("""
                - **Ejercicio 1:** Regresión de retornos contra índice de referencia (CAPM) para estimar Alpha y Beta.  
                - **Ejercicio 2:** Modelo autorregresivo AR sobre el promedio móvil de 20 días.
                """)

        # ============ TAB 4: MODELO AR ============
        with tab_ar:
            st.subheader("🔮 Modelo Autorregresivo sobre Promedio Móvil 20 días")

            # Controles locales de AR
            horizonte_regresion = st.slider(
                "Número de rezagos del modelo AR",
                min_value=5,
                max_value=60,
                value=20,
                step=5,
                help="Número de rezagos usados en el modelo AR."
            )

            horizonte_prediccion = st.slider(
                "Horizonte de predicción (días hábiles)",
                min_value=5,
                max_value=60,
                value=20,
                step=5,
                help="Cuántos días hábiles hacia adelante quieres proyectar."
            )

            df["Moving_average_20"] = df["Close"].rolling(window=horizonte_regresion).mean()
            df["m_cyclical_20"] = df["Close"] - df["Moving_average_20"]

            st.dataframe(
                df[["Close", "Moving_average_20", "m_cyclical_20"]]
                .dropna()
                .tail(10)
            )

            serie_ma = df["Moving_average_20"].dropna()
            serie_ma_ultimos_datos = serie_ma.tail(horizonte_regresion)

            if len(serie_ma_ultimos_datos) < horizonte_regresion:
                st.warning("No seas mensito, no hay datos suficientes para el modelo AR.")
            else:
                model_ar = sm.tsa.AutoReg(
                    serie_ma, lags=horizonte_regresion, old_names=False
                ).fit()
                with st.expander("Ver detalle del modelo AR"):
                    st.text(model_ar.summary())

                pred = model_ar.get_prediction(
                    start=len(serie_ma),
                    end=len(serie_ma) + horizonte_prediccion - 1,
                )

                pred_mean = pred.predicted_mean
                conf_int = pred.conf_int(alpha=0.05)

                last_date = df.index[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=horizonte_prediccion,
                    freq="B",
                )

                df_pred = pd.DataFrame(
                    {
                        "Fecha": future_dates,
                        "Predicción": pred_mean.values,
                        "Límite Inferior (95%)": conf_int.iloc[:, 0].values,
                        "Límite Superior (95%)": conf_int.iloc[:, 1].values,
                    }
                )

                st.subheader(
                    f"Predicción del Promedio Móvil a {horizonte_prediccion} días "
                    "(Intervalo de Confianza 95%)"
                )
                st.dataframe(df_pred, use_container_width=True)

                fig, ax = plt.subplots(figsize=(10, 5))

                ax.plot(
                    df.index,
                    df["Moving_average_20"],
                    label="Promedio Móvil 20 días (Histórico)",
                    color="blue",
                    alpha=0.7,
                )

                ax.plot(
                    df_pred["Fecha"],
                    df_pred["Predicción"],
                    label=f"Predicción AR({horizonte_regresion})",
                    color="orange",
                    linewidth=2,
                )

                ax.fill_between(
                    df_pred["Fecha"],
                    df_pred["Límite Inferior (95%)"],
                    df_pred["Límite Superior (95%)"],
                    color="orange",
                    alpha=0.2,
                    label="Intervalo de Confianza 95%",
                )

                ax.errorbar(
                    df_pred["Fecha"],
                    df_pred["Predicción"],
                    yerr=[
                        pred_mean - conf_int.iloc[:, 0],
                        conf_int.iloc[:, 1] - pred_mean,
                    ],
                    fmt="o",
                    color="black",
                    alpha=0.6,
                    capsize=3,
                    label="Error ±95%",
                )

                ax.set_title(
                    f"Predicción del Promedio Móvil (AR({horizonte_regresion})) - {ticker}"
                )
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Nivel del Promedio Móvil (USD)")
                ax.legend()
                ax.grid(True)

                st.pyplot(fig)

        # ============ TAB 5: NOTICIAS (SERPAPI) ============
        with tab_news:
            st.subheader(f"📰 Noticias recientes sobre {ticker}")

            if not SERPAPI_KEY:
                st.warning("No se encontró SERPAPI_KEY en el .env, no se pueden consultar noticias.")
            else:
                try:
                    params = {
                        "engine": "google_news",
                        "q": f"{ticker} stock",
                        "api_key": SERPAPI_KEY,
                        "hl": "en",
                        "num": 5
                    }
                    resp = requests.get("https://serpapi.com/search.json", params=params, timeout=10)
                    data = resp.json()
                    news_results = data.get("news_results", [])
                except Exception as e:
                    news_results = []
                    st.error(f"Ocurrió un error al consultar SerpAPI: {e}")

                rows = []
                for item in news_results[:5]:
                    title = item.get("title")
                    link = item.get("link")
                    source = item.get("source") or {}
                    publisher = source.get("name") if isinstance(source, dict) else source
                    date_str = item.get("date") or item.get("published_at")

                    rows.append(
                        {
                            "Fecha": date_str,
                            "Título": title,
                            "Fuente": publisher,
                            "Link": link,
                        }
                    )

                if not rows:
                    st.warning("No se encontraron noticias recientes para este ticker en SerpAPI.")
                else:
                    df_news = pd.DataFrame(rows)
                    st.dataframe(df_news, use_container_width=True)

                    titulares_texto = ""
                    for r in rows:
                        if r["Título"]:
                            titulares_texto += f"- {r['Título']} (Fuente: {r['Fuente']}, Fecha: {r['Fecha']})\n"

                    if client is not None and titulares_texto.strip():
                        mensaje_noticias = client.messages.create(
                            model="claude-sonnet-4-5-20250929",
                            max_tokens=900,
                            temperature=0.2,
                            system=(
                                "Rol:\nEres un analista de riesgo financiero que resume noticias relevantes "
                                "para un inversionista institucional. Debes identificar tono general, riesgos "
                                "y posibles catalizadores para el precio de la acción. No recomiendes comprar "
                                "o vender, solo interpreta la narrativa."
                            ),
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                f"Titulares recientes sobre la empresa {ticker}:\n\n"
                                                f"{titulares_texto}\n"
                                                "Resume el sentimiento general, identifica riesgos clave y posibles "
                                                "catalizadores (positivos o negativos) para el precio. "
                                                "Máximo 900 caracteres."
                                            ),
                                        }
                                    ],
                                }
                            ],
                        )
                        st.subheader("Análisis cualitativo de noticias (Anthropic)")
                        st.write(mensaje_noticias.content[0].text)
                    else:
                        st.info("No hay titulares suficientes para analizar con Anthropic.")
