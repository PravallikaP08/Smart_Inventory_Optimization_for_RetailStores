import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO


st.set_page_config(page_title="Smart Inventory Optimization Dashboard", layout="wide")
st.title("📦 Milestone 3 — Smart Inventory Optimization Dashboard (Enhanced Version)")


df = pd.read_csv(r"C:\Users\PRAVALLIKA\Smart_Stock_Project\MileStone2\data\forecast_results.csv")


if "product_name" not in df.columns:
    df["product_name"] = ["Product_" + str(i + 1) for i in range(len(df))]
if "product_id" not in df.columns:
    st.error("❌ Missing 'product_id' column in dataset.")
    st.stop()


st.sidebar.header("⚙️ Inventory Parameters")

lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.slider("Ordering Cost ($)", 10, 200, 50)
holding_cost = st.sidebar.slider("Holding Cost ($/unit/year)", 1, 20, 5)
unit_cost = st.sidebar.slider("Unit Purchase Cost ($)", 10, 200, 50)
stockout_cost = st.sidebar.slider("Stockout Cost ($/unit)", 10, 100, 25)

service_levels = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
service_choice = st.sidebar.selectbox("Service Level", list(service_levels.keys()), 1)
z = service_levels[service_choice]

products = df["product_id"].unique()
selected_product = st.sidebar.selectbox("Select Product", products)


inventory_plan = []

for _, row in df.iterrows():
    product = row["product_id"]
    name = row["product_name"]
    avg_demand = np.random.randint(10, 60)
    std_demand = np.random.uniform(2, 10)

  
    if row["Prophet_MAPE"] < row["LSTM_MAPE"]:
        best_model = "Prophet"
        reliability = 100 - row["Prophet_MAPE"]
    else:
        best_model = "LSTM"
        reliability = 100 - row["LSTM_MAPE"]

    demand = avg_demand * 30
    eoq = np.sqrt((2 * demand * ordering_cost) / (holding_cost))
    ss = z * std_demand * np.sqrt(lead_time)
    rop = (avg_demand * lead_time) + ss
    annual_cost = ((demand / eoq) * ordering_cost) + ((eoq / 2) * holding_cost) + (demand * unit_cost)

    inventory_plan.append({
        "Product_ID": product,
        "Product_Name": name,
        "Best_Model": best_model,
        "Forecast_Reliability(%)": round(reliability, 2),
        "AvgDailyDemand": round(avg_demand, 2),
        "Demand_StdDev": round(std_demand, 2),
        "EOQ": round(eoq, 2),
        "SafetyStock": round(ss, 2),
        "ReorderPoint": round(rop, 2),
        "Annual_Cost($)": round(annual_cost, 2)
    })

inv_df = pd.DataFrame(inventory_plan)


inv_df["Annual_Value"] = inv_df["AvgDailyDemand"] * 365 * unit_cost
inv_df = inv_df.sort_values(by="Annual_Value", ascending=False)
inv_df["Cumulative%"] = inv_df["Annual_Value"].cumsum() / inv_df["Annual_Value"].sum() * 100

def abc_category(x):
    if x <= 20:
        return "A"
    elif x <= 50:
        return "B"
    else:
        return "C"

inv_df["ABC"] = inv_df["Cumulative%"].apply(abc_category)

def xyz_category(std):
    if std < 4:
        return "X"
    elif std < 7:
        return "Y"
    else:
        return "Z"

inv_df["XYZ"] = inv_df["Demand_StdDev"].apply(xyz_category)
inv_df["Class"] = inv_df["ABC"] + inv_df["XYZ"]


st.sidebar.header("🔍 Filters")
abc_filter = st.sidebar.multiselect("Filter by ABC Category", ["A", "B", "C"], default=["A", "B", "C"])
reliability_threshold = st.sidebar.slider("Minimum Forecast Reliability (%)", 50, 100, 70)
filtered_df = inv_df[(inv_df["ABC"].isin(abc_filter)) & (inv_df["Forecast_Reliability(%)"] >= reliability_threshold)]


tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📈 Analytics", "🔍 Comparisons", "🧠 Recommendations", "📥 Downloads"])


with tab1:
    st.subheader("📦 Inventory Overview")

    selected_row = inv_df[inv_df["Product_ID"] == selected_product].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🛒 EOQ", f"{selected_row['EOQ']:.2f}")
    c2.metric("⚠️ Safety Stock", f"{selected_row['SafetyStock']:.2f}")
    c3.metric("📦 Reorder Point", f"{selected_row['ReorderPoint']:.2f}")
    c4.metric("💰 Annual Cost", f"${selected_row['Annual_Cost($)']:.2f}")

    # Inventory Simulation Chart
    weeks = np.arange(1, 13)
    inv_level = np.maximum(0, 120 - np.cumsum(np.random.randint(5, 20, size=12)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks, y=inv_level, mode='lines+markers', name='Inventory Level'))
    fig.add_hline(y=selected_row["ReorderPoint"], line_dash="dash", line_color="orange", annotation_text="ROP")
    fig.add_hline(y=selected_row["SafetyStock"], line_dash="dash", line_color="red", annotation_text="Safety Stock")
    fig.update_layout(title=f"📊 Inventory Simulation for {selected_row['Product_Name']}",
                      xaxis_title="Weeks", yaxis_title="Units", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.subheader("📈 Inventory Performance Insights")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(filtered_df, x="Product_Name", y="Annual_Cost($)", color="ABC", title="Annual Cost by Product (ABC Category)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.pie(filtered_df, values="Forecast_Reliability(%)", names="Best_Model", title="Forecast Model Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(filtered_df.style.highlight_max(axis=0, color='lightgreen'))


with tab3:
    st.subheader("🔍 Prophet vs LSTM Comparison")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=df["product_name"], y=df["Prophet_MAPE"], name="Prophet MAPE"))
    fig3.add_trace(go.Bar(x=df["product_name"], y=df["LSTM_MAPE"], name="LSTM MAPE"))
    fig3.update_layout(barmode="group", xaxis_title="Product", yaxis_title="MAPE (%)", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)


with tab4:
    st.subheader("🧠 AI Recommendations")
    recs = []
    for _, r in inv_df.iterrows():
        if r["Forecast_Reliability(%)"] < 75:
            rec = f"🔸 {r['Product_Name']}: Improve forecast accuracy."
        elif r["SafetyStock"] < (0.15 * r["ReorderPoint"]):
            rec = f"⚠️ {r['Product_Name']}: Increase safety stock."
        elif r["EOQ"] > 600:
            rec = f"💼 {r['Product_Name']}: Reduce EOQ to minimize holding cost."
        else:
            rec = f"✅ {r['Product_Name']}: Parameters optimized."
        recs.append(rec)
    st.markdown("\n".join(recs))


with tab5:
    st.subheader("📥 Download Inventory Reports")

    csv_data = inv_df.to_csv(index=False).encode("utf-8")

    # Excel download
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        inv_df.to_excel(writer, index=False, sheet_name="Inventory Plan")
        writer.close()
    excel_data = excel_buffer.getvalue()

    st.download_button("⬇️ Download CSV", data=csv_data, file_name="smart_inventory_plan.csv", mime="text/csv")
    st.download_button("📊 Download Excel", data=excel_data, file_name="smart_inventory_plan.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.success("✅ You can now successfully download your CSV and Excel reports!")


st.sidebar.markdown("---")
st.sidebar.subheader("📊 Summary KPIs")
st.sidebar.metric("Total Annual Cost", f"${inv_df['Annual_Cost($)'].sum():,.2f}")
st.sidebar.metric("Average Reliability", f"{inv_df['Forecast_Reliability(%)'].mean():.2f}%")
st.sidebar.metric("Products Optimized", f"{len(inv_df)}")
