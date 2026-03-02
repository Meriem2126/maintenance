import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Henkel Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Custom CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    .stApp {background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%); font-family: 'Roboto', sans-serif;}
    .stApp::before {content: "HENKEL"; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%) rotate(-45deg); font-size: 15rem; font-weight: 700; color: rgba(227, 6, 19, 0.03); z-index: 0; pointer-events: none; letter-spacing: 2rem;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #E30613 0%, #B8050F 100%); box-shadow: 4px 0 15px rgba(0,0,0,0.1);}
    [data-testid="stSidebar"] * {color: white !important;}
    h1 {color: #E30613; font-weight: 700; border-bottom: 3px solid #E30613; padding-bottom: 10px; margin-bottom: 30px;}
    h2, h3 {color: #E30613; font-weight: 600;}
    .metric-card {background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(227, 6, 19, 0.1); border-left: 5px solid #E30613; margin: 15px 0; transition: transform 0.3s ease, box-shadow 0.3s ease;}
    .metric-card:hover {transform: translateY(-5px); box-shadow: 0 8px 25px rgba(227, 6, 19, 0.2);}
    .stButton > button {background: linear-gradient(135deg, #E30613 0%, #B8050F 100%); color: white; border: none; border-radius: 8px; padding: 12px 30px; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 4px 10px rgba(227, 6, 19, 0.3);}
    .stButton > button:hover {transform: translateY(-2px); box-shadow: 0 6px 15px rgba(227, 6, 19, 0.4);}
    .info-box {background: linear-gradient(135deg, #fff5f5 0%, #ffe6e6 100%); border: 2px solid #E30613; border-radius: 12px; padding: 20px; margin: 20px 0;}
    .chart-wrapper {background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 20px 0;}
</style>
""", unsafe_allow_html=True)

# ----------------- Load Data -----------------
data = pd.read_csv('machinery_data.csv')
data.fillna(method='ffill', inplace=True)

# ----------------- Features & Model -----------------
features = ['sensor_1', 'sensor_2', 'sensor_3', 'operational_hours', 'carbon_sensor']
target_rul = 'RUL'
target_maintenance = 'maintenance'

# If carbon_sensor column does not exist, create dummy
if 'carbon_sensor' not in data.columns:
    data['carbon_sensor'] = np.random.uniform(0, 100, size=len(data))

scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(data[features], data[target_rul], test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(data[features], data[target_maintenance], test_size=0.2, random_state=42)

# Models
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)

clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_clf, y_train_clf)

kmeans = KMeans(n_clusters=2, random_state=42)
data['cluster'] = kmeans.fit_predict(data[features])

# Prediction function
def predict_maintenance(features):
    rul_pred = reg_model.predict([features])[0]
    maint_pred = clf_model.predict([features])[0]
    cluster_pred = kmeans.predict([features])[0]
    return {
        'RUL Prediction': rul_pred,
        'Maintenance Prediction': 'Needs Maintenance' if maint_pred == 1 else 'Normal',
        'Anomaly Detection': 'Anomaly' if cluster_pred == 1 else 'Normal'
    }

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: white; font-size: 2.5rem; margin: 0;'>HENKEL</h1>
        </div><hr style='border: 1px solid rgba(255,255,255,0.2); margin: 20px 0;'>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Home", "Historical Data", "Input Data", "Results", "Visualizations",
                  "Machine Status", "Reports", "Settings"],
        icons=["house-fill", "table", "sliders", "check-circle-fill", "bar-chart-fill",
               "graph-up", "gear-fill", "file-earmark-text-fill", "sliders"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "black", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px", "color": "black", "background-color": "rgba(255,255,255,0.1)", "border-radius": "8px"},
            "nav-link-selected": {"background-color": "white", "color": "#E30613", "font-weight": "600"},
        }
    )

# ----------------- Home Page -----------------
if selected == "Home":
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
            <div style='padding: 40px 0;'>
                <h1 style='font-size: 3rem; margin-bottom: 20px;'> Predictive Maintenance Dashboard</h1>
                <p style='font-size: 1.2rem; color: #666; line-height: 1.8;'>
                    Advanced AI-powered predictive maintenance system for industrial machinery. 
                    Leveraging machine learning to predict equipment failures before they happen, 
                    reducing downtime and optimizing maintenance schedules.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class='metric-card'>
                <h3 style='text-align: center; margin-bottom: 15px;'>System Status</h3>
                <p style='text-align: center; font-size: 3rem; margin: 10px 0;'>✓</p>
                <p style='text-align: center; color: #28a745; font-weight: 600;'>All Systems Operational</p>
            </div>
        """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("<h2>Key Features</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col, title, desc in zip([col1, col2, col3],
                                ["🎯 RUL Prediction", "🔍 Anomaly Detection", "⚙️ Maintenance Alerts"],
                                ["Predict Remaining Useful Life of components",
                                 "Real-time detection of unusual sensor patterns",
                                 "Smart notifications when maintenance is required"]):
        col.markdown(f"""
            <div class='metric-card'>
                <h3>{title}</h3>
                <p style='color: #666;'>{desc}</p>
            </div>
        """, unsafe_allow_html=True)

    # Quick stats
    st.markdown("<h2>Dashboard Overview</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📊 Total Records", f"{len(data):,}")
    maintenance_count = data['maintenance'].sum()
    col2.metric("⚠️ Maintenance Required", maintenance_count, f"{(maintenance_count/len(data)*100):.1f}%")
    col3.metric("⏱️ Avg RUL", f"{data['RUL'].mean():.1f} hrs")
    col4.metric("🤖 Models Trained", "3")
    col5.metric("🌱 Avg Carbon Sensor", f"{data['carbon_sensor'].mean():.1f} kg CO₂")

elif selected == "Historical Data":
    st.title("📂 Historical Data Overview")
    
    st.markdown("""
        <div class='info-box'>
            <h3>📋 Dataset Information</h3>
            <p>This dataset contains historical sensor readings and maintenance records from industrial machinery.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Data preview
    st.markdown("<h3>Recent Records</h3>", unsafe_allow_html=True)
    st.dataframe(data.head(10), use_container_width=True)
    
    # Summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Summary Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(data.describe(), use_container_width=True)
    
    with col2:
        st.markdown("<h3>Data Information</h3>", unsafe_allow_html=True)
        st.write(f"**Total Rows:** {len(data):,}")
        st.write(f"**Total Columns:** {len(data.columns)}")
        st.write(f"**Memory Usage:** {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ----------------- Input Data -----------------
elif selected == "Input Data":
    st.title("🔧 Input Sensor Data")
    st.markdown("<div class='info-box'><p>Generate random values or manually adjust the sliders to input sensor readings and operational hours.</p></div>", unsafe_allow_html=True)
    
    if 'generated_values' not in st.session_state:
        st.session_state['generated_values'] = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button('🎲 Generate Random Values', use_container_width=True):
            values = [
                np.random.uniform(data['sensor_1'].min(), data['sensor_1'].max()),
                np.random.uniform(data['sensor_2'].min(), data['sensor_2'].max()),
                np.random.uniform(data['sensor_3'].min(), data['sensor_3'].max()),
                np.random.uniform(data['operational_hours'].min(), data['operational_hours'].max()),
                np.random.uniform(data['carbon_sensor'].min(), data['carbon_sensor'].max())
            ]
            st.session_state['generated_values'] = values
            st.success("✓ Random values generated successfully!")

    # Display generated values
    if st.session_state['generated_values']:
        st.markdown("<h3>Generated Values</h3>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        labels = ["Sensor 1","Sensor 2","Sensor 3","Op. Hours","Carbon Sensor"]
        for col, val, label in zip([col1,col2,col3,col4,col5], st.session_state['generated_values'], labels):
            col.markdown(f"<div class='metric-card'><h4>{label}</h4><p style='font-size:1.5rem;font-weight:700;color:#E30613'>{val:.2f}</p></div>", unsafe_allow_html=True)
        if st.button('✓ Use Generated Values', use_container_width=True):
            st.session_state['input_features'] = st.session_state['generated_values']
            st.success("✓ Values applied! Navigate to Results to see predictions.")

    # Manual input
    st.markdown("<hr><h3>Manual Input</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        sensor_1 = st.slider('🔴 Sensor 1', float(data['sensor_1'].min()), float(data['sensor_1'].max()), float(data['sensor_1'].mean()))
        sensor_2 = st.slider('🔴 Sensor 2', float(data['sensor_2'].min()), float(data['sensor_2'].max()), float(data['sensor_2'].mean()))
        sensor_3 = st.slider('🔴 Sensor 3', float(data['sensor_3'].min()), float(data['sensor_3'].max()), float(data['sensor_3'].mean()))
    with col2:
        operational_hours = st.slider('⏱️ Operational Hours', int(data['operational_hours'].min()), int(data['operational_hours'].max()), int(data['operational_hours'].mean()))
        carbon_sensor = st.slider('🌱 Carbon Sensor', float(data['carbon_sensor'].min()), float(data['carbon_sensor'].max()), float(data['carbon_sensor'].mean()))
    if st.button('📤 Submit Manual Input', use_container_width=True):
        st.session_state['input_features'] = [sensor_1,sensor_2,sensor_3,operational_hours,carbon_sensor]
        st.success("✓ Input data submitted! Navigate to Results to see predictions.")

# ----------------- Results -----------------
elif selected == "Results":
    st.title("📊 Prediction Results")
    
    if 'input_features' not in st.session_state:
        st.markdown("""
            <div class='info-box'>
                <h3>⚠️ No Input Data</h3>
                <p>Please navigate to the <strong>Input Data</strong> section and submit sensor readings first.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        input_features = st.session_state['input_features']
        prediction = predict_maintenance(input_features)
        
        # Input summary
        st.markdown("<h3>Input Summary</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sensor 1", f"{input_features[0]:.2f}")
        with col2:
            st.metric("Sensor 2", f"{input_features[1]:.2f}")
        with col3:
            st.metric("Sensor 3", f"{input_features[2]:.2f}")
        with col4:
            st.metric("Op. Hours", f"{input_features[3]:.2f}")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3>Prediction Results</h3>", unsafe_allow_html=True)
        
        # Results cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4>⏱️ Remaining Useful Life</h4>
                    <p style='font-size: 2.5rem; font-weight: 700; color: #E30613; margin: 20px 0;'>{prediction['RUL Prediction']:.2f}</p>
                    <p style='color: #666;'>hours</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status_color = "#E30613" if prediction['Maintenance Prediction'] == 'Needs Maintenance' else "#28a745"
            status_icon = "⚠️" if prediction['Maintenance Prediction'] == 'Needs Maintenance' else "✓"
            st.markdown(f"""
                <div class='metric-card'>
                    <h4>⚙️ Maintenance Status</h4>
                    <p style='font-size: 2.5rem; margin: 20px 0;'>{status_icon}</p>
                    <p style='color: {status_color}; font-weight: 700;'>{prediction['Maintenance Prediction']}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            anomaly_color = "#ffc107" if prediction['Anomaly Detection'] == 'Anomaly' else "#28a745"
            anomaly_icon = "⚠️" if prediction['Anomaly Detection'] == 'Anomaly' else "✓"
            st.markdown(f"""
                <div class='metric-card'>
                    <h4>🔍 Anomaly Detection</h4>
                    <p style='font-size: 2.5rem; margin: 20px 0;'>{anomaly_icon}</p>
                    <p style='color: {anomaly_color}; font-weight: 700;'>{prediction['Anomaly Detection']}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Alerts
        if prediction['Maintenance Prediction'] == 'Needs Maintenance':
            st.error('🚨 ALERT: Maintenance is required! Schedule immediate inspection.')
        
        if prediction['Anomaly Detection'] == 'Anomaly':
            st.warning('⚠️ WARNING: Anomaly detected in sensor readings. Further investigation recommended.')
        
        # ------------------ Carbon Emission & Financial Impact ------------------
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3>🌱 Carbon Emission & Financial Impact</h3>", unsafe_allow_html=True)
        
        # Simple carbon calculation (example)
        base_emission = input_features[3] * (input_features[0] + input_features[1] + input_features[2]) * 0.1
        emission_status = "Low"
        emission_color = "#28a745"
        
        if base_emission > 50:
            emission_status = "High"
            emission_color = "#E30613"
        elif base_emission > 20:
            emission_status = "Medium"
            emission_color = "#ffc107"
        
        # Financial impact calculation (example: 0.2 € per kg CO2)
        financial_impact = base_emission * 0.2
        
        # Display in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4>💨 Estimated CO₂ Emission</h4>
                    <p style='font-size: 2.5rem; font-weight: 700; color: {emission_color}; margin: 20px 0;'>{base_emission:.2f} kg CO₂</p>
                    <p style='color: {emission_color}; font-weight: 600;'>{emission_status}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4>💰 Estimated Financial Impact</h4>
                    <p style='font-size: 2.5rem; font-weight: 700; color: {emission_color}; margin: 20px 0;'>{financial_impact:.2f} €</p>
                    <p style='color: {emission_color}; font-weight: 600;'>{emission_status}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Alerts
        if emission_status == "High":
            st.warning("⚠️ High carbon emission detected! Consider maintenance or energy-saving actions.")
        elif emission_status == "Medium":
            st.info("ℹ️ Moderate emission level. Monitor for improvements.")
        else:
            st.success("✓ Low emission. System running efficiently.")

       
elif selected == "Visualizations":
    st.title("📊 Data Visualizations")
    
    # Set Henkel color scheme for plots
    henkel_colors = ['#E30613', '#B8050F', '#8B0410', '#666666', '#999999']
    sns.set_palette(henkel_colors)
    
    # Histogram for sensor readings
    st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
    st.subheader("Histogram of Sensor Readings")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    
    sns.histplot(data['sensor_1'], bins=30, ax=axs[0], kde=True, color='#E30613')
    axs[0].set_title('Sensor 1', fontweight='bold', color='#E30613')
    axs[0].set_facecolor('#f9f9f9')
    
    sns.histplot(data['sensor_2'], bins=30, ax=axs[1], kde=True, color='#E30613')
    axs[1].set_title('Sensor 2', fontweight='bold', color='#E30613')
    axs[1].set_facecolor('#f9f9f9')
    
    sns.histplot(data['sensor_3'], bins=30, ax=axs[2], kde=True, color='#E30613')
    axs[2].set_title('Sensor 3', fontweight='bold', color='#E30613')
    axs[2].set_facecolor('#f9f9f9')
    
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Scatter plot for sensor readings vs operational hours
    st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
    st.subheader("Scatter Plot: Sensor Readings vs Operational Hours")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    
    axs[0].scatter(data['operational_hours'], data['sensor_1'], alpha=0.6, c='#E30613', edgecolors='#B8050F')
    axs[0].set_title('Operational Hours vs Sensor 1', fontweight='bold', color='#E30613')
    axs[0].set_xlabel('Operational Hours')
    axs[0].set_ylabel('Sensor 1')
    axs[0].set_facecolor('#f9f9f9')
    axs[0].grid(True, alpha=0.3)
    
    axs[1].scatter(data['operational_hours'], data['sensor_2'], alpha=0.6, c='#E30613', edgecolors='#B8050F')
    axs[1].set_title('Operational Hours vs Sensor 2', fontweight='bold', color='#E30613')
    axs[1].set_xlabel('Operational Hours')
    axs[1].set_ylabel('Sensor 2')
    axs[1].set_facecolor('#f9f9f9')
    axs[1].grid(True, alpha=0.3)
    
    axs[2].scatter(data['operational_hours'], data['sensor_3'], alpha=0.6, c='#E30613', edgecolors='#B8050F')
    axs[2].set_title('Operational Hours vs Sensor 3', fontweight='bold', color='#E30613')
    axs[2].set_xlabel('Operational Hours')
    axs[2].set_ylabel('Sensor 3')
    axs[2].set_facecolor('#f9f9f9')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Line chart for RUL over time
    st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
    st.subheader("Line Chart: RUL Over Operational Hours")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    
    ax.plot(data['operational_hours'], data['RUL'], marker='o', linestyle='-', 
            color='#E30613', linewidth=2, markersize=4, markeredgecolor='#B8050F')
    ax.set_title('RUL Over Operational Hours', fontweight='bold', fontsize=14, color='#E30613')
    ax.set_xlabel('Operational Hours', fontweight='bold')
    ax.set_ylabel('RUL (hours)', fontweight='bold')
    ax.set_facecolor('#f9f9f9')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    if 'input_features' in st.session_state:
        input_features = st.session_state['input_features']

        if input_features is not None:
            st.markdown("<h2>Visualizations with Input Data Overlay</h2>", unsafe_allow_html=True)
            
            # Histogram with generated input
            st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
            st.subheader("Histogram with Input Value Markers")
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.patch.set_facecolor('white')
            
            sns.histplot(data['sensor_1'], bins=30, ax=axs[0], kde=True, color='#E30613')
            axs[0].axvline(input_features[0], color='black', linestyle='--', linewidth=2, label='Your Input')
            axs[0].set_title('Sensor 1', fontweight='bold', color='#E30613')
            axs[0].legend()
            axs[0].set_facecolor('#f9f9f9')
            
            sns.histplot(data['sensor_2'], bins=30, ax=axs[1], kde=True, color='#E30613')
            axs[1].axvline(input_features[1], color='black', linestyle='--', linewidth=2, label='Your Input')
            axs[1].set_title('Sensor 2', fontweight='bold', color='#E30613')
            axs[1].legend()
            axs[1].set_facecolor('#f9f9f9')
            
            sns.histplot(data['sensor_3'], bins=30, ax=axs[2], kde=True, color='#E30613')
            axs[2].axvline(input_features[2], color='black', linestyle='--', linewidth=2, label='Your Input')
            axs[2].set_title('Sensor 3', fontweight='bold', color='#E30613')
            axs[2].legend()
            axs[2].set_facecolor('#f9f9f9')
            
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

            # Scatter plot with generated input
            st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
            st.subheader("Scatter Plot with Input Data Highlighted")
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.patch.set_facecolor('white')
            
            axs[0].scatter(data['operational_hours'], data['sensor_1'], alpha=0.6, c='#E30613', edgecolors='#B8050F')
            axs[0].axvline(input_features[3], color='black', linestyle='--', linewidth=2, label='Your Input')
            axs[0].set_title('Operational Hours vs Sensor 1', fontweight='bold', color='#E30613')
            axs[0].set_xlabel('Operational Hours')
            axs[0].set_ylabel('Sensor 1')
            axs[0].legend()
            axs[0].set_facecolor('#f9f9f9')
            axs[0].grid(True, alpha=0.3)
            
            axs[1].scatter(data['operational_hours'], data['sensor_2'], alpha=0.6, c='#E30613', edgecolors='#B8050F')
            axs[1].axvline(input_features[3], color='black', linestyle='--', linewidth=2, label='Your Input')
            axs[1].set_title('Operational Hours vs Sensor 2', fontweight='bold', color='#E30613')
            axs[1].set_xlabel('Operational Hours')
            axs[1].set_ylabel('Sensor 2')
            axs[1].legend()
            axs[1].set_facecolor('#f9f9f9')
            axs[1].grid(True, alpha=0.3)
            
            axs[2].scatter(data['operational_hours'], data['sensor_3'], alpha=0.6, c='#E30613', edgecolors='#B8050F')
            axs[2].axvline(input_features[3], color='black', linestyle='--', linewidth=2, label='Your Input')
            axs[2].set_title('Operational Hours vs Sensor 3', fontweight='bold', color='#E30613')
            axs[2].set_xlabel('Operational Hours')
            axs[2].set_ylabel('Sensor 3')
            axs[2].legend()
            axs[2].set_facecolor('#f9f9f9')
            axs[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

            # Line chart with generated input
            st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
            st.subheader("RUL Over Time with Input Data Marker")
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('white')
            
            ax.plot(data['operational_hours'], data['RUL'], marker='o', linestyle='-', 
                    color='#E30613', linewidth=2, markersize=4, markeredgecolor='#B8050F')
            ax.axvline(input_features[3], color='black', linestyle='--', linewidth=2, label='Your Input')
            ax.set_title('RUL Over Operational Hours', fontweight='bold', fontsize=14, color='#E30613')
            ax.set_xlabel('Operational Hours', fontweight='bold')
            ax.set_ylabel('RUL (hours)', fontweight='bold')
            ax.legend()
            ax.set_facecolor('#f9f9f9')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)


# ================= Machine Status Page ==================
elif selected == "Machine Status":
    st.title("⚙️ Machine Status Overview")
    st.markdown("<p>Monitor real-time machine health, anomalies, and operational metrics.</p>", unsafe_allow_html=True)

    st.markdown("<h3>Cluster-Based Health Status</h3>", unsafe_allow_html=True)
    cluster_counts = data['cluster'].value_counts()
    st.bar_chart(cluster_counts)

    st.markdown("<h3>Machines Needing Maintenance</h3>", unsafe_allow_html=True)
    maintenance_needed = data[data['maintenance'] == 1]
    st.dataframe(maintenance_needed[features + ['RUL']])


# ================= Reports Page ==================
elif selected == "Reports":
    st.title("📄 Reports")
    st.markdown("<p>Generate and download reports of sensor data, predictions, and maintenance logs.</p>", unsafe_allow_html=True)

    report_type = st.selectbox("Select Report Type", ["All Data", "Maintenance Required", "Cluster Overview"])
    if st.button("📥 Generate Report"):
        if report_type == "All Data":
            report_df = data.copy()
        elif report_type == "Maintenance Required":
            report_df = data[data['maintenance'] == 1]
        else:
            report_df = data[['sensor_1','sensor_2','sensor_3','operational_hours','cluster']]

        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Report", data=csv, file_name=f"{report_type.replace(' ','_')}_report.csv", mime='text/csv')


# ================= Settings Page ==================
elif selected == "Settings":
    st.title("⚙️ Settings")
    st.markdown("<p>Adjust model parameters, visualization options, and dashboard preferences.</p>", unsafe_allow_html=True)

    st.subheader("Model Settings")
    n_estimators = st.slider("Random Forest Estimators", 50, 500, 100, step=50)
    test_size = st.slider("Test Data Size (%)", 10, 50, 20, step=5)
    if st.button("Apply Settings"):
        st.success(f"Settings updated! Estimators: {n_estimators}, Test Size: {test_size}%")

# ----------------- Footer -----------------
st.markdown("""
    <hr style='margin-top: 50px; border: 1px solid #E30613;'>
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p><strong>HENKEL</strong> Predictive Maintenance System</p>
        <p style='font-size: 0.9rem;'>Powered by Advanced Machine Learning | © 2025</p>
    </div>
""", unsafe_allow_html=True)
