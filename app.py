
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="InduTech",
    page_icon="🏭",
    layout="wide",
)
st.markdown("""
<style>
    .stApp {
        background-color: #0F2A43;
    }

    [data-testid="stSidebar"] {
        background-color: #211A3C;
    }

    h1, h2, h3 {
        color: #F0FDF4;
    }

    p, label {
        color: #DCFCE7;
    }
</style>
""", unsafe_allow_html=True)

DATA = "smart_manufacturing_production_dataset.csv"
MODEL = "production_model.pkl"

FEATURES = [
    "machine_id", "product_type", "shift", "workers", "machine_hours",
    "downtime_hours", "maintenance_hours", "material_availability_pct",
    "quality_rate_pct", "temperature_c", "energy_consumption_kwh",
    "production_target"
]

#Automatic model training
@st.cache_resource
def load_model():
    if not Path(MODEL).exists():
        from sklearn.model_selection import train_test_split
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestRegressor

        df = pd.read_csv(DATA)
        X = df[FEATURES]
        y = df["actual_production"]

        categorical = ["machine_id", "product_type", "shift"]
        preprocessor = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
            remainder="passthrough"
        )

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=250,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ))
        ])

        pipe.fit(X, y)
        Path("model").mkdir(exist_ok=True)
        joblib.dump(pipe, MODEL)

    return joblib.load(MODEL)

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

model = load_model()
df = load_data()

# Sidebar 
st.sidebar.title("🏭 Production Planner")
st.sidebar.write("Enter the current production conditions.")

machine = st.sidebar.selectbox(
    "Machine ID", sorted(df["machine_id"].unique())
)
product = st.sidebar.selectbox(
    "Product Type", sorted(df["product_type"].unique())
)
shift = st.sidebar.selectbox(
    "Shift", ["Morning", "Evening", "Night"]
)

workers = st.sidebar.slider("Workers", 5, 50, 20)
machine_hours = st.sidebar.slider("Machine Hours", 4.0, 16.0, 8.0, 0.1)
downtime = st.sidebar.slider("Downtime Hours", 0.0, 5.0, 1.0, 0.1)
maintenance = st.sidebar.slider("Maintenance Hours", 0.0, 4.0, 0.5, 0.1)
material = st.sidebar.slider("Material Availability (%)", 50.0, 100.0, 95.0, 0.5)
quality = st.sidebar.slider("Quality Rate (%)", 70.0, 100.0, 95.0, 0.5)
temperature = st.sidebar.slider("Temperature (°C)", 15.0, 45.0, 28.0, 0.5)
energy = st.sidebar.slider("Energy Consumption (kWh)", 100.0, 800.0, 400.0, 5.0)
target = st.sidebar.number_input(
    "Production Target (units)", min_value=50, max_value=2000, value=500, step=10
)

#Main
st.title("🏭InduTech")
st.caption(
    "Predict. Plan. Produce. Smarter."
)

input_data = pd.DataFrame([{
    "machine_id": machine,
    "product_type": product,
    "shift": shift,
    "workers": workers,
    "machine_hours": machine_hours,
    "downtime_hours": downtime,
    "maintenance_hours": maintenance,
    "material_availability_pct": material,
    "quality_rate_pct": quality,
    "temperature_c": temperature,
    "energy_consumption_kwh": energy,
    "production_target": target
}])

if st.button("🔮 Predict Production & OT", type="primary"):
    predicted = float(model.predict(input_data[FEATURES])[0])
    predicted = max(0, predicted)

    shortfall = max(0, target - predicted)

    # Practical baseline production rate by product.
    # This is used only for OT planning after the ML prediction.
    base_rate = {
        "Product_A": 52,
        "Product_B": 46,
        "Product_C": 40
    }[product]

    shift_factor = {
        "Morning": 1.03,
        "Evening": 1.00,
        "Night": 0.94
    }[shift]

    # Estimate hourly capacity for the OT scenario.
    # Keep a minimum rate to avoid division by zero.
    production_rate = max(
        10,
        base_rate * (quality / 100) * (material / 100) * shift_factor
        * min(1.10, max(0.70, workers / 20))
    )

    ot_hours = shortfall / production_rate if shortfall > 0 else 0

    # OT rate can vary by shift.
    ot_rate = {
        "Morning": 180,
        "Evening": 200,
        "Night": 230
    }[shift]

    # Use a manageable OT team rather than automatically assigning everyone.
    ot_workers = int(np.ceil(workers * 0.60)) if shortfall > 0 else 0
    ot_cost = ot_hours * ot_workers * ot_rate

    status = "Target Met" if shortfall == 0 else "Shortfall Detected"

    st.divider()
    st.subheader("📊 Prediction Result")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Production Target", f"{target:,.0f}")
    c2.metric("Predicted Production", f"{predicted:,.0f}")
    c3.metric("Shortfall", f"{shortfall:,.0f}")
    c4.metric("Status", status)

    st.divider()
    st.subheader("⏱️ Overtime Planning")

    if shortfall > 0:
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Production Rate", f"{production_rate:.1f} units/hr")
        o2.metric("Required OT", f"{ot_hours:.2f} hrs")
        o3.metric("OT Workers", f"{ot_workers}")
        o4.metric("Estimated OT Cost", f"₹{ot_cost:,.2f}")

        st.warning(
            f"Production shortfall of {shortfall:.0f} units detected. "
            f"Approximately {ot_hours:.2f} hours of overtime is required."
        )

        st.info(
            f"Recommended plan: assign {ot_workers} workers for approximately "
            f"{ot_hours:.2f} OT hours. Estimated additional OT cost: ₹{ot_cost:,.2f}."
        )
    else:
        st.success(
            "The predicted production meets or exceeds the target. "
            "No additional overtime is required."
        )

    #  Comparison chart
    st.divider()
    st.subheader("📈 Target vs Predicted Production")

    chart_df = pd.DataFrame({
        "Production": [target, predicted]
    }, index=["Target", "Predicted"])

    st.bar_chart(chart_df)

# ---------- Historical analysis ----------
st.divider()
st.subheader("📋 Historical Production Overview")

a, b, c = st.columns(3)
a.metric("Total Records", f"{len(df):,}")
b.metric("Average Target", f"{df['production_target'].mean():.0f}")
c.metric("Average Actual", f"{df['actual_production'].mean():.0f}")

with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)

st.caption(
    "Note: OT is calculated only when production falls below the target. "
)
