import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Body, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import os
import pandas as pd
import base64
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langsmith import Client
from mcp.server import MCPServer

# ============================================================
# 1. GEMINI 3.5 FLASH MODEL
# ============================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=os.environ["GEMINI_API_KEY"]
)

# ============================================================
# 2. LANGSMITH TRACING
# ============================================================
ls_client = Client()

def trace_run(input_text: str, output_text: dict):
    try:
        ls_client.create_run(
            name="merchant-multi-agent-run",
            inputs={"input": input_text},
            outputs=output_text,
        )
    except Exception:
        # tracing should never break the app
        pass

# ============================================================
# 3. MCP TOOLS
# ============================================================
mcp = MCPServer("MerchantTools")

@mcp.tool()
def search_orders(query: str) -> list:
    return [f"Search result for: {query}"]

# ============================================================
# 4. ERROR‑PROOF CSV LOADING (PANDAS ONLY)
# ============================================================
def load_clean_data() -> pd.DataFrame:
    csv_path = CSV_FILENAME

    df = pd.read_csv(csv_path)

    # --- Fix date parsing ---
    # Try multiple formats instead of dropping everything
    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce",
        format=None  # allow flexible parsing
    )

    # If still NaT, try common formats manually
    if df["order_date"].isna().all():
        df["order_date"] = pd.to_datetime(
            df["order_date"],
            errors="coerce",
            infer_datetime_format=True
        )

    # --- Fix numeric parsing ---
    # Remove currency symbols and commas
    df["total"] = (
        df["total"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    df["total"] = pd.to_numeric(df["total"], errors="coerce")

    # --- Do NOT drop rows aggressively ---
    # Only drop rows where BOTH date and total are missing
    df = df.dropna(subset=["order_date", "total"], how="all")

    # Fill missing totals with 0 instead of dropping
    df["total"] = df["total"].fillna(0)

    # Fill missing dates with today's date (or leave NaT)
    df["order_date"] = df["order_date"].fillna(pd.Timestamp.today())

    return df
    
# ============================================================
# 5. FORECAST / ANOMALY / RISK
# ============================================================
def run_forecasting(df: pd.DataFrame) -> dict:
    daily_sales = df.groupby("order_date")["total"].sum().reset_index()
    return {"daily_sales": daily_sales}

def run_anomaly_detection(df: pd.DataFrame) -> dict:
    mean_total = df["total"].mean()
    df["is_anomaly"] = df["total"] > 3 * mean_total
    anomalies = df[df["is_anomaly"]]
    return {"anomalies": anomalies}

def run_risk_scoring(df: pd.DataFrame) -> dict:
    mean_total = df["total"].mean()
    df["risk_score"] = df["total"] / mean_total
    top_risk = df.sort_values("risk_score", ascending=False).head(10)
    return {"top_risk": top_risk}

# ============================================================
# 6. SAFE MATPLOTLIB DASHBOARD GENERATOR
# ============================================================
def generate_dashboard(df: pd.DataFrame):
    required_cols = ["order_date", "total", "quantity", "channel", "customer_name"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Missing columns: {missing}",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        return buf

    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "CSV loaded but contains no rows",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        return buf

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Merchant Behavior & Store Health Dashboard', fontsize=20)

    daily_sales = df.groupby("order_date")["total"].sum().reset_index()
    axes[0, 0].plot(daily_sales["order_date"], daily_sales["total"], marker="o")
    axes[0, 0].set_title("Daily Total Sales")
    axes[0, 0].tick_params(axis='x', rotation=45)

    mean_total = df["total"].mean()
    df["risk_score"] = df["total"] / mean_total
    top_risk = df.sort_values("risk_score", ascending=False).head(10)
    axes[0, 1].barh(top_risk["customer_name"], top_risk["risk_score"], color="red")
    axes[0, 1].set_title("Top 10 High-Risk Customers")

    axes[1, 0].scatter(df["total"], df["quantity"], alpha=0.6)
    axes[1, 0].set_title("Total vs Quantity")
    axes[1, 0].set_xlabel("Total")
    axes[1, 0].set_ylabel("Quantity")

    channel_sales = df.groupby("channel")["total"].sum().reset_index()
    axes[1, 1].barh(channel_sales["channel"], channel_sales["total"], color="purple")
    axes[1, 1].set_title("Total Sales by Channel")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

# ============================================================
# 7. MULTI‑AGENT WORKFLOW (SAFE)
# ============================================================
class AgentState(TypedDict):
    input: str
    forecast: dict
    anomalies: dict
    risk: dict
    dashboard_image_b64: str
    analysis: str
    decision: str
    action_report: str

def forecast_step(state: AgentState):
    df = load_clean_data()
    return {"forecast": run_forecasting(df)}

def anomaly_step(state: AgentState):
    df = load_clean_data()
    return {"anomalies": run_anomaly_detection(df)}

def risk_step(state: AgentState):
    df = load_clean_data()
    return {"risk": run_risk_scoring(df)}

def dashboard_step(state: AgentState):
    df = load_clean_data()
    buf = generate_dashboard(df)
    image_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"dashboard_image_b64": image_b64}

def vision_analysis_step(state: AgentState):
    try:
        forecast_sample = (
            state.get("forecast", {})
            .get("daily_sales", pd.DataFrame())
            .head()
            .to_string()
        )
    except Exception:
        forecast_sample = "No forecast data"

    try:
        risk_sample = (
            state.get("risk", {})
            .get("top_risk", pd.DataFrame())
            .head()
            .to_string()
        )
    except Exception:
        risk_sample = "No risk data"

    prompt = [
        {
            "role": "user",
            "content": (
                "You are an AI analyzing a merchant health dashboard.\n"
                f"Forecast sample:\n{forecast_sample}\n\n"
                f"Top risk sample:\n{risk_sample}\n"
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{state.get('dashboard_image_b64', '')}"
                    }
                }
            ]
        }
    ]

    try:
        result = llm.invoke(prompt)
        analysis = result.content if isinstance(result.content, str) else str(result.content)
    except Exception as e:
        analysis = f"Vision analysis failed: {e}"

    return {"analysis": analysis}

def decision_step(state: AgentState):
    text = state.get("analysis", "").lower()
    if any(word in text for word in ["anomaly", "risk", "outlier"]):
        decision = "Escalate to sales and risk team; review high-risk and anomalous orders."
    else:
        decision = "Monitor normally; no immediate intervention required."
    return {"decision": decision}

def action_step(state: AgentState):
    report = (
        f"Decision: {state.get('decision', 'No decision')}\n"
        "Suggested actions:\n"
        "- Notify account managers for top-risk customers.\n"
        "- Review anomalous orders for potential fraud or data issues.\n"
        "- Adjust marketing or inventory strategy based on forecast trends."
    )
    return {"action_report": report}

builder = StateGraph(AgentState)
builder.add_node("forecast", forecast_step)
builder.add_node("anomaly", anomaly_step)
builder.add_node("risk", risk_step)
builder.add_node("dashboard", dashboard_step)
builder.add_node("vision", vision_analysis_step)
builder.add_node("decide", decision_step)
builder.add_node("act", action_step)

builder.set_entry_point("forecast")
builder.add_edge("forecast", "anomaly")
builder.add_edge("anomaly", "risk")
builder.add_edge("risk", "dashboard")
builder.add_edge("dashboard", "vision")
builder.add_edge("vision", "decide")
builder.add_edge("decide", "act")
builder.add_edge("act", END)

agent_app = builder.compile()

# ============================================================
# DATASET CONFIGURATION
# ============================================================
CSV_FILENAME = "order_id_order_date_36.csv"

# ============================================================
# 8. FASTAPI APP
# ============================================================
app = FastAPI()

# ============================================================
# 9. HEALTH ENDPOINT
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "message": "FastAPI is running"}

# ============================================================
# 10. DEBUG ENDPOINTS
# ============================================================
@app.get("/debug")
def debug():
    csv_path = "order_id_order_date_36.csv"
    info = {
        "csv_exists": os.path.exists(csv_path),
        "csv_path": csv_path,
    }

    if not os.path.exists(csv_path):
        info["error"] = "CSV file not found in repo root."
        return info

    try:
        df = pd.read_csv(csv_path)
        info["columns"] = list(df.columns)
        info["head"] = df.head().to_dict(orient="records")
        info["row_count"] = len(df)
    except Exception as e:
        info["error"] = f"Failed to read CSV: {e}"

    return info

@app.get("/debug-dashboard")
def debug_dashboard():
    try:
        df = load_clean_data()
        return {
            "row_count": len(df),
            "columns": list(df.columns),
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# 11. HOMEPAGE
# ============================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h1>Autonomous Merchant Growth Multi-Agent System</h1>
            <ul>
                <li><a href="/cleaned-table">Cleaned CSV Table</a></li>
                <li><a href="/dashboard">Matplotlib Dashboard</a></li>
                <li><a href="/agent-ui">Gemini gemini 3.5 flash</a></li>
                <li><a href="/debug">Debug CSV</a></li>
                <li><a href="/debug-dashboard">Debug Dashboard</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
        </body>
    </html>
    """

# ============================================================
# 12. CLEANED TABLE ENDPOINT (PANDAS ONLY, SAFE)
# ============================================================
@app.get("/cleaned-table")
def cleaned_table():
    try:
        df = pd.read_csv("order_id_order_date_36.csv")
    except Exception as e:
        return HTMLResponse(f"<h2>Error reading CSV</h2><pre>{e}</pre>", status_code=500)

    text_cols = ["customer_name", "email", "product_title",
                 "category", "channel", "status"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    df["order_date"] = pd.to_datetime(df.get("order_date"), errors="coerce")
    df = df.dropna(subset=["order_date"])

    html_table = df.head(50).to_html(index=False)
    return HTMLResponse(f"<h2>Cleaned CSV Data</h2>{html_table}")

# ============================================================
# 13. MATPLOTLIB DASHBOARD ENDPOINT (SAFE)
# ============================================================
@app.get("/dashboard")
def dashboard_plot():
    try:
        df = load_clean_data()
        buf = generate_dashboard(df)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Dashboard error: {e}",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        return StreamingResponse(buf, media_type="image/png")

# ============================================================
# 14. MULTI-AGENT API ENDPOINT (SAFE)
# ============================================================
@app.post("/agent")
def agent_api(prompt: str = Body(..., embed=True)):
    try:
        result = agent_app.invoke({
            "input": prompt,
            "forecast": {},
            "anomalies": {},
            "risk": {},
            "dashboard_image_b64": "",
            "analysis": "",
            "decision": "",
            "action_report": ""
        })
        trace_run(prompt, result)
        return {
            "analysis": result.get("analysis", ""),
            "decision": result.get("decision", ""),
            "action_report": result.get("action_report", ""),
        }
    except Exception as e:
        return {
            "analysis": f"Agent pipeline failed: {e}",
            "decision": "Error",
            "action_report": "Agent encountered an error; please check logs and CSV.",
        }

# ============================================================
# 15. AGENT CHAT UI
# ============================================================
@app.get("/agent-ui", response_class=HTMLResponse)
def agent_ui():
    return """
    <html>
        <body>
            <h2>Gemini 3.5 Flash Multi-Agent</h2>
            <form method="post" action="/agent-ui">
                <textarea name="prompt" rows="4" style="width:100%;"></textarea>
                <button type="submit">Run Multi-Agent Workflow</button>
            </form>
        </body>
    </html>
    """

@app.post("/agent-ui", response_class=HTMLResponse)
async def agent_ui_post(request: Request):
    form = await request.form()
    prompt = form.get("prompt", "")
    try:
        result = agent_app.invoke({
            "input": prompt,
            "forecast": {},
            "anomalies": {},
            "risk": {},
            "dashboard_image_b64": "",
            "analysis": "",
            "decision": "",
            "action_report": ""
        })
        trace_run(prompt, result)
    except Exception as e:
        result = {
            "analysis": f"Agent pipeline failed: {e}",
            "decision": "Error",
            "action_report": "Agent encountered an error; please check logs and CSV.",
        }

    return f"""
    <html>
        <body>
            <h2>Gemini 3.5 Flash Multi-Agent</h2>
            <form method="post" action="/agent-ui">
                <textarea name="prompt" rows="4" style="width:100%;">{prompt}</textarea>
                <button type="submit">Run Multi-Agent Workflow</button>
            </form>
            <h3>Analysis:</h3>
            <pre>{result.get('analysis', '')}</pre>
            <h3>Decision:</h3>
            <pre>{result.get('decision', '')}</pre>
            <h3>Action Report:</h3>
            <pre>{result.get('action_report', '')}</pre>
        </body>
    </html>
    """
