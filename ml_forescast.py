import pandas as pd
import numpy as np

# import the dataset from the inventory of the order of the clients.

df = pd.read_csv('/content/order_id_order_date_36.csv')

# Basic cleaning: remove exact duplicates and standardize text columns
df = df.drop_duplicates()
text_cols = ['customer_name', 'email', 'product_title', 'category', 'channel', 'status']
for col in text_cols:
    df[col] = df[col].str.strip().str.lower()

# Convert date to datetime objects
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

# Show cleaned info
print(f'Cleaned dataset shape: {df.shape}')
display(df.head())

# Further deduplication based on order and product identity
df = df.drop_duplicates(subset=['order_id', 'product_id'], keep='first')

# Handle missing dates by dropping or filling (dropping for forecasting accuracy)
df = df.dropna(subset=['order_date'])

# Convert numeric columns to ensure they are ready for modeling
df['total'] = pd.to_numeric(df['total'], errors='coerce')
df = df.dropna(subset=['total'])

print("Data summary for modeling:")
print(df.info())
display(df.describe())

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Feature Engineering
model_df = df.copy()
model_df['day_of_week'] = model_df['order_date'].dt.dayofweek
model_df['day_of_month'] = model_df['order_date'].dt.day

# Encode categorical features
le = LabelEncoder()
for col in ['category', 'channel', 'status']:
    model_df[col] = le.fit_transform(model_df[col])

# Define features and target
features = ['category', 'channel', 'status', 'day_of_week', 'day_of_month', 'price']
X = model_df[features]
y = model_df['total']

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f'Forecasting Model Mean Absolute Error: {mae:.2f}')
display(pd.DataFrame({'Actual': y_test, 'Predicted': preds}).head())

from sklearn.ensemble import IsolationForest

# 1. Anomaly Detection
# Using 'total' and 'quantity' (need to ensure quantity is numeric first)
model_df['quantity'] = pd.to_numeric(model_df['quantity'], errors='coerce').fillna(1)
iso_forest = IsolationForest(contamination=0.1, random_state=42)
model_df['anomaly_score'] = iso_forest.fit_predict(model_df[['total', 'quantity']])
# -1 indicates an anomaly, 1 indicates normal

# 2. Risk Scoring
# Define a simple risk score based on high total value relative to the mean and anomaly status
mean_total = model_df['total'].mean()
model_df['risk_score'] = (model_df['total'] / mean_total) + (model_df['anomaly_score'].apply(lambda x: 5 if x == -1 else 0))

print("Top 5 High-Risk / Anomalous Orders:")
display(model_df[['order_id', 'customer_name', 'total', 'anomaly_score', 'risk_score']].sort_values(by='risk_score', ascending=False).head())

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Merchant Behavior & Store Health Dashboard', fontsize=20)

# 1. Sales Trend over Time
daily_sales = model_df.groupby('order_date')['total'].sum().reset_index()
sns.lineplot(ax=axes[0, 0], x='order_date', y='total', data=daily_sales, marker='o', color='b')
axes[0, 0].set_title('Daily Total Sales (Forecasting View)')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Risk Score by Customer
top_risk = model_df.sort_values(by='risk_score', ascending=False).head(10)
sns.barplot(ax=axes[0, 1], x='risk_score', y='customer_name', data=top_risk, hue='customer_name', palette='Reds_r', legend=False)
axes[0, 1].set_title('Top 10 High-Risk Customers')

# 3. Anomaly Distribution (Total vs Quantity)
sns.scatterplot(ax=axes[1, 0], x='total', y='quantity', hue='anomaly_score', palette={1: 'blue', -1: 'red'}, data=model_df, s=100)
axes[1, 0].set_title('Anomaly Detection: Total vs Quantity')

# 4. Total Sales by Channel
channel_sales_agg = model_df.groupby('channel')['total'].sum().sort_values(ascending=False).reset_index()
sns.barplot(ax=axes[1, 1], x='total', y='channel', data=channel_sales_agg, hue='channel', palette='viridis', legend=False)
axes[1, 1].set_title('Total Sales by Channel')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

!pip install --upgrade langchain-google-genai google-generativeai langgraph==1.1.1 langsmith

import os
os.environ["LANGSMITH_API_KEY"] = "LangSmith"

!pip install -U "mcp[cli]"
from mcp.server import MCPServer

mcp = MCPServer("GeminiTools")

@mcp.tool()
def search(query: str) -> list:
    # Your search logic here
    return ["Result 1", "Result 2"]

!mkdir -p mcp-server-demo
import os
os.chdir('/content/mcp-server-demo')

!cd mcp-server-demo && uv add langchain-google-genai langgraph langsmith

import base64
import io
import matplotlib.pyplot as plt
from IPython.display import Markdown
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os
from google.colab import userdata

# 1. Initialize Gemini model.
api_key = userdata.get("Ben856")
os.environ["GOOGLE_API_KEY"] = api_key
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=api_key)

# 2. Capture the Dashboard Figure
buf = io.BytesIO()
try:
    current_fig = fig if 'fig' in globals() else plt.gcf()
    current_fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
except Exception as e:
    print(f"Error capturing figure: {e}")

# 3. Analyze with Gemini 3 Flash
prompt = """Analyze this 'Merchant Behavior & Store Health Dashboard'.
Provide insights on sales growth, customer churn risk based on the RFM scores,
and an interpretation of the anomalies in the Total vs Quantity scatter plot."""

message = HumanMessage(
    content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
    ]
)

print("--- Analyzing Dashboard with Gemini 3 Flash Preview ---")
response = llm.invoke([message])
display(Markdown("### Automated Store Health Analysis"))
display(Markdown(response.content if isinstance(response.content, str) else response.content[0]['text']))

# 4. Agent Workflow Definition
class AgentState(TypedDict):
    input: str
    logs: str
    symptom: str
    diagnostic_report: str
    network_ok: bool

workflow = StateGraph(AgentState)

def observe_step(state: AgentState):
    print("--- STEP: OBSERVE ---")
    return {"logs": "Metrics observed.", "network_ok": True}

def diagnose_step(state: AgentState):
    print("--- STEP: DIAGNOSE ---")
    return {"symptom": "anomaly_detected", "diagnostic_report": "High volume outlier identified."}

workflow.add_node("observe", observe_step)
workflow.add_node("diagnose", diagnose_step)
workflow.set_entry_point("observe")
workflow.add_edge("observe", "diagnose")
workflow.add_edge("diagnose", END)

app = workflow.compile()
print("Agent Logic Initialized.")

# 4. Refined Agent Workflow Definition (Complete Loop)
class AgentState(TypedDict):
    input: str
    logs: str
    symptom: str
    diagnostic_report: str
    network_ok: bool

def observe_step(state: AgentState):
    print("--- STEP: OBSERVE ---")
    return {"logs": "Metrics observed.", "network_ok": True}

def diagnose_step(state: AgentState):
    print("--- STEP: DIAGNOSE ---")
    # In a real scenario, this would ingest the Gemini analysis results
    return {"symptom": "anomaly_detected", "diagnostic_report": "High volume outlier identified via dashboard analysis."}

def decide_step(state: AgentState):
    print("--- STEP: DECIDE ---")
    if state['symptom'] == "anomaly_detected":
        decision = "Trigger B2B Whale Customer Onboarding Workflow"
    else:
        decision = "Monitor and Maintain Standard Operations"
    return {"logs": f"Decision made: {decision}"}

def act_step(state: AgentState):
    print("--- STEP: ACT ---")
    action_report = f"EXECUTION PLAN: {state['logs']}. Notify Sales Team for manual verification."
    return {"logs": "Action successfully executed.", "diagnostic_report": action_report}

# Initialize and Build Graph cleanly
builder = StateGraph(AgentState)
builder.add_node("observe", observe_step)
builder.add_node("diagnose", diagnose_step)
builder.add_node("decide", decide_step)
builder.add_node("act", act_step)

builder.set_entry_point("observe")
builder.add_edge("observe", "diagnose")
builder.add_edge("diagnose", "decide")
builder.add_edge("decide", "act")
builder.add_edge("act", END)

# Compile once
app = builder.compile()
print("Autonomous Reasoning Loop (ODA) successfully compiled without warnings.")

# Execute the full loop
print("--- Running Full Autonomous Agent Workflow ---")
final_output = app.invoke({"input": "Perform store growth audit", "logs": "", "symptom": "", "diagnostic_report": "", "network_ok": False})

print("\n--- FINAL AGENT REPORT ---")
print(f"Summary: {final_output['diagnostic_report']}")
print(f"Execution Status: {final_output['logs']}")
