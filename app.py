from fastapi import FastAPI, Body, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import os
import polars as pl
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langsmith import Client
from mcp.server import MCPServer

# ============================================================
# 1. GEMINI 3.4 FLASH PREVIEW MODEL
# ============================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-3.4-flash-preview",
    api_key=os.environ["GEMINI_API_KEY"]
)

# ============================================================
# 2. LANGSMITH TRACING
# ============================================================
ls_client = Client()

def trace_run(input_text: str, output_text: dict):
    ls_client.create_run(
        name="merchant-multi-agent-run",
        inputs={"input": input_text},
        outputs=output_text,
    )

# ============================================================
# 3. MCP TOOLS
# ============================================================
mcp = MCPServer("MerchantTools")

@mcp.tool()
def search_orders(query: str) -> list:
    return [f"Search result for: {query}"]

# ============================================================
# 4. DATA UTILITIES: CLEANING, FORECASTING, ANOMALY, RISK
# ============================================================
def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv("order_id_order_date_36.csv")
    df = df.drop_duplicates()

    text_cols = ["customer_name", "email", "product_title",
                 "category", "channel", "status"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])

    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df = df.dropna(subset=["total"])

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1)
    else:
        df["quantity"] = 1

    return df

def run_forecasting(df: pd.DataFrame) -> dict:
    daily_sales = df.groupby("order_date")["total"].sum().reset_index()
    return {"daily_sales": daily_sales}

def run_anomaly_detection(df: pd.DataFrame) -> dict:
    # Simple anomaly proxy: mark orders above 3x mean as anomalies
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
# 5. MATPLOTLIB DASHBOARD GENERATOR
# ============================================================
def generate_dashboard(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Merchant Behavior & Store Health Dashboard', fontsize=20)

    # 1. Daily Sales Trend
    daily_sales = df.groupby("order_date")["total"].sum().reset_index()
    axes[0, 0].plot(daily_sales["order_date"], daily_sales["total"], marker="o")
    axes[0, 0].set_title("Daily Total Sales")
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Top Risk Customers
    mean_total = df["total"].mean()
    df["risk_score"] = df["total"] / mean_total
    top_risk = df.sort_values("risk_score", ascending=False).head(10)
    axes[0, 1].barh(top_risk["customer_name"], top_risk["risk_score"], color="red")
    axes[0, 1].set_title("Top 10 High-Risk Customers")

    # 3. Total vs Quantity (Anomaly View)
    axes[1, 0].scatter(df["total"], df["quantity"], alpha=0.6)
    axes[1, 0].set_title("Total vs Quantity")
    axes[1, 0].set_xlabel("Total")
    axes[1, 0].set_ylabel("Quantity")

    # 4. Sales by Channel
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
# 6. MULTI-AGENT STATE + NODES (FORECAST, ANOMALY, RISK, DASHBOARD, VISION, DECISION, ACTION)
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
    forecast = run_forecasting(df)
    return {"forecast": forecast}

def anomaly_step(state: AgentState):
    df = load_clean_data()
    anomalies = run_anomaly_detection(df)
    return {"anomalies": anomalies}

def risk_step(state: AgentState):
    df = load_clean_data()
    risk = run_risk_scoring(df)
    return {"risk": risk}

def dashboard_step(state: AgentState):
    df = load_clean_data()
    buf = generate_dashboard(df)
    image_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"dashboard_image_b64": image_b64}

def vision_analysis_step(state: AgentState):
    prompt = [
        {
            "type": "text",
            "text": (
                "You are an AI analyzing a merchant health dashboard.\n"
                "Use the image plus the following context:\n"
                f"Forecast sample: {state['forecast'].get('daily_sales').head().to_string() if state.get('forecast') else 'N/A'}\n"
                f"Top risk sample: {state['risk'].get('top_risk').head().to_string() if state.get('risk') else 'N/A'}\n"
            )
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['dashboard_image_b64']}"}
        }
    ]
    result = llm.invoke(prompt)
    analysis = result.content if isinstance(result.content, str) else str(result.content)
    return {"analysis": analysis}

def decision_step(state: AgentState):
    text = state["analysis"].lower()
    if "anomaly" in text or "risk" in text or "outlier" in text:
        decision = "Escalate to sales and risk team; review high-risk and anomalous orders."
    else:
        decision = "Monitor normally; no immediate intervention required."
    return {"decision": decision}

def action_step(state: AgentState):
    report = (
        f"Decision: {state['decision']}\n"
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
# 7. FASTAPI APP
# ============================================================
app = FastAPI()

# ============================================================
# 8. HOMEPAGE
# ============================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h1>Autonomous Merchant Growth Multi-Agent System</h1>
            <ul>
                <li><a href="/cleaned-table">Cleaned Polars Table</a></li>
                <li><a href="/dashboard">Matplotlib Dashboard</a></li>
                <li><a href="/agent-ui">Gemini 3.4 Flash Multi-Agent</a></li>
            </ul>
        </body>
    </html>
    """

# ============================================================
# 9. CLEANED POLARS TABLE ENDPOINT
# ============================================================
@app.get("/cleaned-table")
def cleaned_table():
    df = pl.read_csv("order_id_order_date_36.csv")

    cleaned_df = df.with_columns([
        pl.col("customer_name").str.strip().str.to_lowercase(),
        pl.col("email").str.strip().str.to_lowercase(),
        pl.col("product_title").str.strip().str.to_lowercase(),
        pl.col("category").str.strip().str.to_lowercase(),
        pl.col("channel").str.strip().str.to_lowercase(),
        pl.col("status").str.strip().str.to_lowercase()
    ]).unique()

    cleaned_df = cleaned_df.with_columns([
        pl.col("order_date").str.to_datetime(strict=False).alias("order_date")
    ]).drop_nulls(["order_date"])

    html_table = cleaned_df.head(50).to_pandas().to_html(index=False)

    return HTMLResponse(f"<h2>Cleaned Polars Data</h2>{html_table}")

# ============================================================
# 10. MATPLOTLIB DASHBOARD ENDPOINT
# ============================================================
@app.get("/dashboard")
def dashboard_plot():
    df = load_clean_data()
    buf = generate_dashboard(df)
    return StreamingResponse(buf, media_type="image/png")

# ============================================================
# 11. MULTI-AGENT API ENDPOINT
# ============================================================
@app.post("/agent")
def agent_api(prompt: str = Body(..., embed=True)):
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
        "analysis": result["analysis"],
        "decision": result["decision"],
        "action_report": result["action_report"],
    }

# ============================================================
# 12. AGENT CHAT UI
# ============================================================
@app.get("/agent-ui", response_class=HTMLResponse)
def agent_ui():
    return """
    <html>
        <body>
            <h2>Gemini 3.4 Flash Multi-Agent</h2>
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

    return f"""
    <html>
        <body>
            <h2>Gemini 3.4 Flash Multi-Agent</h2>
            <form method="post" action="/agent-ui">
                <textarea name="prompt" rows="4" style="width:100%;">{prompt}</textarea>
                <button type="submit">Run Multi-Agent Workflow</button>
            </form>
            <h3>Analysis:</h3>
            <pre>{result['analysis']}</pre>
            <h3>Decision:</h3>
            <pre>{result['decision']}</pre>
            <h3>Action Report:</h3>
            <pre>{result['action_report']}</pre>
        </body>
    </html>
    """
