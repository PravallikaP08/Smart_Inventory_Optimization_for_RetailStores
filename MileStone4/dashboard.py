import os
import io
import subprocess
import time
import tempfile
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Optional PDF library; show friendly message if not installed
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

st.set_page_config(page_title="Smart Inventory Dashboard (Final)", layout="wide")
st.title("📦 Milestone 4 — Smart Inventory Dashboard (Final)")

# ---------- CONFIG / PATHS ----------
FORECAST_PATH = r"C:\Users\PRAVALLIKA\Smart_Stock_Project\MileStone2\data\forecast_results.csv"
CLEANED_SALES_PATH = r"C:\Users\PRAVALLIKA\Smart_Stock_Project\MileStone1\cleaned_retail_dataset_single_store_2020_2025.csv"
# Path to the forecasting script you want to run from the dashboard:
FORECAST_SCRIPT_PATH = r"C:\Users\PRAVALLIKA\Smart_Stock_Project\MileStone2\forecast.py"

# ---------- Load forecast_results.csv ----------
if not os.path.exists(FORECAST_PATH):
    st.warning(f"Forecast file not found at {FORECAST_PATH}. Upload or run forecasting from the sidebar.")
    forecast_df = None
else:
    forecast_df = pd.read_csv(FORECAST_PATH)
    forecast_df.columns = forecast_df.columns.str.strip()

if forecast_df is not None:
    if "product_id" in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"product_id": "product_id"})
    if "Product_ID" in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"Product_ID": "product_id"})
    if "Product ID" in forecast_df.columns and "product_id" not in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"Product ID": "product_id"})


sales_df = None
if os.path.exists(CLEANED_SALES_PATH):
    try:
        sales_df = pd.read_csv(CLEANED_SALES_PATH)
        sales_df.columns = sales_df.columns.str.strip()
        # normalize names
        if "date" in sales_df.columns:
            sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
        if "product_id" not in sales_df.columns and "product id" in sales_df.columns:
            sales_df = sales_df.rename(columns={"product id": "product_id"})
    except Exception:
        sales_df = None


st.sidebar.header("Controls")
lead = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.slider("Ordering Cost ($)", 10, 200, 50)
holding_cost = st.sidebar.slider("Holding Cost ($/unit/year)", 1, 20, 5)
unit_cost = st.sidebar.slider("Unit Purchase Cost ($)", 10, 200, 50)
stockout_cost = st.sidebar.slider("Stockout Cost ($/unit)", 10, 200, 25)
service_level = st.sidebar.selectbox("Service Level", ["90%", "95%", "99%"], index=1)
z_map = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
z = z_map[service_level]

st.sidebar.markdown("---")
st.sidebar.subheader("Files / Uploads")


uploaded_onhand = st.sidebar.file_uploader("Upload current_stock CSV (columns: product_id,current_stock)", type=["csv"], key="onhand")


uploaded_forecast = st.sidebar.file_uploader("Upload forecast_results.csv (optional)", type=["csv"], key="forecast_up")

if uploaded_forecast is not None:
    try:
        forecast_df = pd.read_csv(uploaded_forecast)
        forecast_df.columns = forecast_df.columns.str.strip()
        if "product_id" not in forecast_df.columns and "Product ID" in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={"Product ID": "product_id"})
        st.sidebar.success("Uploaded forecast_results.csv loaded into session")
    except Exception as e:
        st.sidebar.error("Uploaded forecast file couldn't be read.")

onhand_df = None
if uploaded_onhand is not None:
    try:
        onhand_df = pd.read_csv(uploaded_onhand)
        onhand_df.columns = onhand_df.columns.str.strip()
        
        col_candidates = {c.lower(): c for c in onhand_df.columns}
     
        pid_col = None
        for c in ["product_id", "product id", "product", "sku"]:
            if c in col_candidates:
                pid_col = col_candidates[c]
                break
        stock_col = None
        for c in ["current_stock", "current stock", "onhand", "on_hand", "quantity", "qty"]:
            if c in col_candidates:
                stock_col = col_candidates[c]
                break
        if pid_col is None or stock_col is None:
            st.sidebar.warning("Uploaded on-hand CSV loaded but could not find 'product_id' or 'current_stock' columns. Please ensure columns are present.")
            onhand_df = None
        else:
            onhand_df = onhand_df[[pid_col, stock_col]].rename(columns={pid_col: "product_id", stock_col: "current_stock"})
            st.sidebar.success("On-hand inventory loaded")
    except Exception as e:
        st.sidebar.error("Failed to parse uploaded on-hand CSV.")
        onhand_df = None


def build_30d_forecast_from_data(pid):

    if forecast_df is not None and "forecast_best" in forecast_df.columns and "date" in forecast_df.columns:
        prod = forecast_df[forecast_df["product_id"] == pid].sort_values("date")
        try:
            arr = prod["forecast_best"].dropna().astype(float).values
            if len(arr) >= 30:
                return arr[-30:]
        except Exception:
            pass

    if sales_df is not None and "product_id" in sales_df.columns and "units_sold" in sales_df.columns:
        s = sales_df[sales_df["product_id"] == pid].sort_values("date")
        if len(s) >= 1:
            daily = s.groupby("date")["units_sold"].sum().reset_index()
            avg_daily = daily["units_sold"].mean()
            if np.isnan(avg_daily) or avg_daily <= 0:
                avg_daily = 5.0
            noise = np.random.normal(0, 0.05 * avg_daily, 30)
            return np.maximum(0, avg_daily + noise)

    return np.maximum(0, np.random.normal(20, 5, 30))


product_list = []
if forecast_df is not None and "product_id" in forecast_df.columns:
    product_list = list(pd.Series(forecast_df["product_id"].unique()))
else:
  
    if sales_df is not None and "product_id" in sales_df.columns:
        product_list = list(pd.Series(sales_df["product_id"].unique()))
    else:
        product_list = []

if len(product_list) == 0:
    st.warning("No products found in forecast_results.csv or cleaned sales file. Upload files in sidebar or run forecasting.")
else:

    product_forecasts = {pid: build_30d_forecast_from_data(pid) for pid in product_list}


st.sidebar.markdown("---")
st.sidebar.subheader("Run Forecasting Script")
st.sidebar.write("If you updated sales data and want fresh forecasts, click below to run your Milestone2 forecasting script.")
run_forecast_btn = st.sidebar.button("▶️ Run Milestone-2 Forecasting Script")

if run_forecast_btn:
    if not os.path.exists(FORECAST_SCRIPT_PATH):
        st.sidebar.error(f"Forecast script not found at: {FORECAST_SCRIPT_PATH}")
    else:
        st.sidebar.info("Running forecasting script — this may take a while. Output CSV (forecast_results.csv) will be reloaded if successful.")
        try:
      
            proc = subprocess.run(
                ["python", FORECAST_SCRIPT_PATH],
                capture_output=True, text=True, timeout=60*30  # 30 minutes max
            )
            st.sidebar.text("=== Forecasting script stdout ===")
            st.sidebar.code(proc.stdout[:1000])
            if proc.stderr:
                st.sidebar.text("=== Forecasting script stderr ===")
                st.sidebar.code(proc.stderr[:1000])
      
            time.sleep(2)
            if os.path.exists(FORECAST_PATH):
                forecast_df = pd.read_csv(FORECAST_PATH)
                forecast_df.columns = forecast_df.columns.str.strip()
                if "product_id" in forecast_df.columns:
                    product_list = list(pd.Series(forecast_df["product_id"].unique()))
                    product_forecasts = {pid: build_30d_forecast_from_data(pid) for pid in product_list}
                st.sidebar.success("Forecasting script finished and forecast_results.csv reloaded (if produced).")
            else:
                st.sidebar.warning("Forecasting script finished but forecast_results.csv was not created at expected path.")
        except subprocess.TimeoutExpired:
            st.sidebar.error("Forecasting script took too long and was terminated.")
        except Exception as e:
            st.sidebar.error(f"Failed to run forecasting script: {e}")


tab1, tab2, tab3, tab4 = st.tabs(["Forecasts", "Inventory", "Stock Alerts", "Reports & Downloads"])

# ---------- TAB1: Forecasts ----------
with tab1:
    st.header("📈 Forecasts & Historical Trends")

    if len(product_list) == 0:
        st.info("No products to display. Upload forecast_results.csv or cleaned sales data.")
    else:
        # ---- KPI Summary ----
        total_products = len(product_list)
        avg_demand = np.mean([np.mean(v) for v in product_forecasts.values()])
        reorder_count = sum(
            (onhand_df["current_stock"] < inv_df["ReorderPoint"]).sum()
        ) if onhand_df is not None and 'inv_df' in locals() else 0
        avg_eoq = inv_df["EOQ"].mean() if 'inv_df' in locals() else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Products", total_products)
        col2.metric("Avg Daily Demand", f"{avg_demand:.2f}")
        col3.metric("Reorder Needed", reorder_count)
        col4.metric("Avg EOQ", f"{avg_eoq:.2f}")

        st.markdown("---")

        sel = st.selectbox("Select Product to View Forecast", product_list)
        fc = product_forecasts.get(sel, np.zeros(30))
        future_dates = pd.date_range(start=pd.Timestamp.today().normalize(), periods=len(fc), freq="D")

        fig = px.line(
            x=future_dates,
            y=fc,
            title=f"📊 30-Day Forecast for Product {sel}",
            labels={"x": "Date", "y": "Forecasted Units"},
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------- TAB2: Inventory ----------
with tab2:
    st.header("Inventory Plan (from forecasts)")
    if len(product_list) == 0:
        st.info("No products to compute inventory for.")
    else:
        rows = []
        for pid in product_list:
            fc = product_forecasts.get(pid, np.zeros(30))
            total_30 = float(np.sum(fc))
            avg_daily = float(np.mean(fc))
            std = float(np.std(fc))
            yearly_demand = avg_daily * 365
            eoq = np.sqrt((2 * yearly_demand * ordering_cost) / max(holding_cost, 1e-6))
            ss = z * std * np.sqrt(lead)
            rop = (avg_daily * lead) + ss
            
            # Calculate annual value for ABC analysis
            annual_value = yearly_demand * unit_cost
            
            rows.append({
                "product_id": pid,
                "AvgDailySales": round(avg_daily, 3),
                "Total30dDemand": round(total_30, 2),
                "EOQ": round(eoq, 2),
                "SafetyStock": round(ss, 2),
                "ReorderPoint": round(rop, 2),
                "StdDev": round(std, 2),
                "Value": round(annual_value, 2)
            })
        
        inv_df = pd.DataFrame(rows).sort_values("AvgDailySales", ascending=False).reset_index(drop=True)
        
        # Add ABC Analysis
        inv_df = inv_df.sort_values("Value", ascending=False)
        inv_df["Cumulative%"] = (inv_df["Value"].cumsum() / inv_df["Value"].sum() * 100).round(4)
        
        # Assign ABC categories
        def assign_abc_category(cumulative_pct):
            if cumulative_pct <= 80:
                return "A"
            elif cumulative_pct <= 95:
                return "B"
            else:
                return "C"
        
        inv_df["ABC_Category"] = inv_df["Cumulative%"].apply(assign_abc_category)

        # merge on-hand stock if provided
        if onhand_df is not None:
            inv_df = inv_df.merge(onhand_df, on="product_id", how="left")
        else:
            inv_df["current_stock"] = np.nan

        # Reorder columns to match your desired format
        column_order = ["product_id", "AvgDailySales", "Total30dDemand", "EOQ", "SafetyStock", 
                       "ReorderPoint", "StdDev", "Value", "Cumulative%", "ABC_Category", "current_stock"]
        
        # Only include columns that exist in the dataframe
        final_columns = [col for col in column_order if col in inv_df.columns]
        inv_df = inv_df[final_columns]

        st.dataframe(inv_df, use_container_width=True)
        
        # Display ABC Analysis Summary
        st.subheader("📊 ABC Analysis Summary")
        abc_summary = inv_df["ABC_Category"].value_counts().sort_index()
        col1, col2, col3 = st.columns(3)
        col1.metric("A Items (Top 80%)", abc_summary.get("A", 0))
        col2.metric("B Items (Next 15%)", abc_summary.get("B", 0))
        col3.metric("C Items (Bottom 5%)", abc_summary.get("C", 0))


with tab3:
    st.header("⚠️ Stock Health Overview")

    if 'inv_df' not in locals():
        st.info("Generate inventory plan first in the Inventory tab.")
    else:
        if inv_df["current_stock"].isna().all():
            st.info("No on-hand stock provided. Upload a CSV in the sidebar or simulate data.")
            if st.button("Simulate Current Stock"):
                inv_df["current_stock"] = np.random.randint(0, 200, size=len(inv_df))

        # Compute stock health %
        inv_df["StockHealth(%)"] = (inv_df["current_stock"] / inv_df["ReorderPoint"]) * 100
        inv_df["StockHealth(%)"] = inv_df["StockHealth(%)"].clip(0, 200)

        # Professional Action labels
        conditions = [
            (inv_df["StockHealth(%)"] < 80),
            (inv_df["StockHealth(%)"].between(80, 110)),
            (inv_df["StockHealth(%)"] > 110)
        ]
        choices = ["🔴 Critical - Reorder Now", "🟡 Watch - Near Threshold", "🟢 Healthy"]
        inv_df["Status"] = np.select(conditions, choices, default="Unknown")

        st.dataframe(inv_df[["product_id", "current_stock", "ReorderPoint", "StockHealth(%)", "Status", "ABC_Category"]], use_container_width=True)

        st.markdown("#### Stock vs. Reorder Point Comparison")
        fig = px.bar(
            inv_df,
            x="product_id",
            y=["current_stock", "ReorderPoint"],
            barmode="group",
            title="Stock Levels vs Reorder Points",
            labels={"value": "Units", "variable": "Metric"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Demand Variability (Forecast Std. Dev)")
        fig2 = px.bar(
            inv_df,
            x="product_id",
            y="SafetyStock",
            title="Demand Variability Across Products",
            labels={"SafetyStock": "Safety Stock (Demand Variability)"},
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("#### ABC Category Analysis")
        fig3 = px.pie(
            inv_df,
            names="ABC_Category",
            title="Product Distribution by ABC Category",
        )
        st.plotly_chart(fig3, use_container_width=True)


with tab4:
    st.header("Reports & Downloads")
    if 'inv_df' not in locals():
        st.info("Inventory plan not yet created.")
    else:
  
        csv_bytes = inv_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download inventory_plan.csv", data=csv_bytes, file_name="inventory_plan.csv", mime="text/csv")

       
        try:
            import xlsxwriter
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                inv_df.to_excel(writer, index=False, sheet_name="InventoryPlan")
                writer.save()
            excel_data = output.getvalue()
            st.download_button("📥 Download inventory_plan.xlsx", data=excel_data, file_name="inventory_plan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            st.info("Excel export requires xlsxwriter. Install via: pip install xlsxwriter")

        st.markdown("---")
        # Generate small charts to embed in PDF
        st.write("Generate PDF report (KPIs + charts):")
        pdf_name = f"inventory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        generate_pdf_btn = st.button("🖨️ Generate PDF Report")

        if generate_pdf_btn:
            if not REPORTLAB_AVAILABLE:
                st.error("PDF generation requires reportlab. Install with: pip install reportlab")
            else:
                # Create images for top charts in temp files
                try:
                    tmp_dir = tempfile.mkdtemp()
                    # Chart 1: Inventory bar chart (top 10 by Annualized demand)
                    top = inv_df.sort_values("AvgDailySales", ascending=False).head(10)
                    fig1 = px.bar(top, x="product_id", y="AvgDailySales", title="Top 10 Avg Daily Sales")
                    img1_path = os.path.join(tmp_dir, "chart1.png")
                    fig1.write_image(img1_path, scale=2)

                    # Chart 2: EOQ vs ROP scatter
                    fig2 = px.scatter(inv_df, x="EOQ", y="ReorderPoint", hover_data=["product_id"], title="EOQ vs ROP")
                    img2_path = os.path.join(tmp_dir, "chart2.png")
                    fig2.write_image(img2_path, scale=2)
                    
                    # Chart 3: ABC Analysis pie chart
                    fig3 = px.pie(inv_df, names="ABC_Category", title="ABC Category Distribution")
                    img3_path = os.path.join(tmp_dir, "chart3.png")
                    fig3.write_image(img3_path, scale=2)

                    # KPI text
                    total_cost_est = (inv_df["EOQ"].sum() * 0)  # placeholder, user can adapt
                    # Create PDF with reportlab
                    pdf_path = os.path.join(tmp_dir, pdf_name)
                    c = canvas.Canvas(pdf_path, pagesize=A4)
                    w, h = A4
                    # Header
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(40, h - 40, "Inventory Optimization Report")
                    c.setFont("Helvetica", 10)
                    c.drawString(40, h - 60, f"Generated: {datetime.now().isoformat(sep=' ', timespec='seconds')}")
                    # KPIs block
                    c.drawString(40, h - 90, f"Products analyzed: {len(inv_df)}")
                    c.drawString(40, h - 110, f"Service level: {service_level}   Lead time (days): {lead}")
                    c.drawString(40, h - 130, f"Unit Cost: ${unit_cost}   Ordering Cost: ${ordering_cost}")
                    
                    # ABC Summary
                    abc_counts = inv_df["ABC_Category"].value_counts()
                    c.drawString(40, h - 160, f"A Items: {abc_counts.get('A', 0)}   B Items: {abc_counts.get('B', 0)}   C Items: {abc_counts.get('C', 0)}")
                    
                    # Chart images
                    img1 = ImageReader(img1_path)
                    c.drawImage(img1, 40, h - 400, width=500, preserveAspectRatio=True, mask='auto')
                    img2 = ImageReader(img2_path)
                    c.drawImage(img2, 40, h - 650, width=500, preserveAspectRatio=True, mask='auto')
                    img3 = ImageReader(img3_path)
                    c.drawImage(img3, 40, h - 900, width=500, preserveAspectRatio=True, mask='auto')
                    c.showPage()
                    
                    # Add a table page (top 20 rows)
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(40, h - 40, "Top Products (summary)")
                    c.setFont("Helvetica", 9)
                    y = h - 70
                    cols = ["product_id", "AvgDailySales", "EOQ", "SafetyStock", "ReorderPoint", "current_stock", "ABC_Category"]
                    top_table = inv_df[cols].head(25).fillna("")
                    for _, row in top_table.iterrows():
                        line = ", ".join([str(row[c]) for c in cols])
                        c.drawString(40, y, line[:120])
                        y -= 12
                        if y < 40:
                            c.showPage()
                            y = h - 40
                    c.save()

                    # Serve PDF to user
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name=pdf_name, mime="application/pdf")
                    st.success("PDF generated. Use the download button above.")
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Dashboard loaded.")