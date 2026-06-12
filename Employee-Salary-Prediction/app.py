import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="SmartSalary Predictor & Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Geographic market adjustments configuration to align synthetic data with real-world scales
# Format: { "Location": (Realistic USD Adjustment Factor, Exchange Rate from USD, Symbol, Currency Code, Experience Boost Factor) }
LOCATION_CONFIG = {
    "USA": (1.00, 1.00, "$", "USD", 0.25),
    "Canada": (0.75, 1.36, "C$", "CAD", 0.35),
    "UK": (0.70, 0.78, "£", "GBP", 0.35),
    "Germany": (0.65, 0.92, "€", "EUR", 0.30),
    "Netherlands": (0.62, 0.92, "€", "EUR", 0.30),
    "Sweden": (0.45, 10.50, "kr", "SEK", 0.30),
    "Australia": (0.72, 1.50, "A$", "AUD", 0.35),
    "Singapore": (0.80, 1.35, "S$", "SGD", 0.35),
    "India": (0.16, 83.50, "₹", "INR", 0.90), # Steeper career growth curve (at least 1.5x, up to 2.7x over 20 yrs)
    "Remote": (0.85, 1.00, "$", "USD", 0.30)
}

@st.cache_resource
def load_model_and_encoders():
    model_path = os.path.join(BASE_DIR, "salary_model.joblib")
    encoders_path = os.path.join(BASE_DIR, "label_encoders.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(encoders_path):
        st.error("Model or Label Encoders not found! Please run train_and_save.py first.")
        st.stop()
        
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    return model, label_encoders

@st.cache_data
def load_dataset_and_stats():
    file_path = os.path.join(BASE_DIR, "dataset", "salary_prediction.csv")
    if not os.path.exists(file_path):
        st.error("Dataset not found!")
        st.stop()
        
    df = pd.read_csv(file_path)
    
    # Apply market adjustments to the dataset to align with geographic realities
    df["salary_usd_adj"] = df.apply(
        lambda r: r["salary"] * LOCATION_CONFIG.get(r["location"], (1.0, 1.0, "$", "USD", 0.25))[0] * (
            1.0 + (r["experience_years"] / 20.0) * LOCATION_CONFIG.get(r["location"], (1.0, 1.0, "$", "USD", 0.25))[4]
        ),
        axis=1
    )
    df["salary_local"] = df.apply(
        lambda r: r["salary_usd_adj"] * LOCATION_CONFIG.get(r["location"], (1.0, 1.0, "$", "USD", 0.25))[1],
        axis=1
    )
    
    # Calculate dataset-wide statistics in adjusted USD
    avg_role_salaries_adj = df.groupby("job_title")["salary_usd_adj"].mean()
    
    stats = {
        "total_records": len(df),
        "avg_salary_adj": df["salary_usd_adj"].mean(),
        "max_salary_adj": df["salary_usd_adj"].max(),
        "min_salary_adj": df["salary_usd_adj"].min(),
        "avg_by_role_adj": avg_role_salaries_adj.to_dict(),
        "min_by_role_adj": df.groupby("job_title")["salary_usd_adj"].min().to_dict(),
        "max_by_role_adj": df.groupby("job_title")["salary_usd_adj"].max().to_dict(),
        "roles": sorted(df["job_title"].unique().tolist()),
        "education": sorted(df["education_level"].unique().tolist()),
        "industries": sorted(df["industry"].unique().tolist()),
        "sizes": sorted(df["company_size"].unique().tolist()),
        "locations": sorted(df["location"].unique().tolist()),
        "remotes": sorted(df["remote_work"].unique().tolist()),
        "top_role": avg_role_salaries_adj.idxmax(),
        "top_role_salary_adj": avg_role_salaries_adj.max()
    }
    return df, stats

# Initialize app dependencies
model, label_encoders = load_model_and_encoders()
df, stats = load_dataset_and_stats()

# Inject Premium CSS Styling
st.markdown("""
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* Global typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
    }

    /* Gradient header layout */
    .hero-container {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2.5rem;
        background: linear-gradient(135deg, rgba(77, 150, 255, 0.06) 0%, rgba(107, 203, 119, 0.06) 100%);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #4D96FF, #6BCB77, #FFD93D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #8F9CAE;
        font-size: 1.15rem;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Cards */
    .luxury-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease-in-out;
    }
    .luxury-card:hover {
        transform: translateY(-2px);
        border-color: rgba(77, 150, 255, 0.25);
        box-shadow: 0 12px 40px 0 rgba(77, 150, 255, 0.08);
    }

    /* Dynamic Output Cards */
    .prediction-card {
        background: linear-gradient(135deg, rgba(77, 150, 255, 0.08) 0%, rgba(107, 203, 119, 0.08) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(77, 150, 255, 0.2);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 40px 0 rgba(77, 150, 255, 0.12);
        margin-top: 10px;
        margin-bottom: 25px;
    }
    .salary-value {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4D96FF, #6BCB77);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 15px 0;
        font-family: 'Outfit', sans-serif;
    }
    .salary-label {
        font-size: 1.1rem;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }

    /* KPI Stats */
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    .kpi-box {
        flex: 1;
        background: rgba(255, 255, 255, 0.015);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }
    .kpi-box:hover {
        background: rgba(255, 255, 255, 0.03);
        border-color: rgba(107, 203, 119, 0.2);
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 5px;
        font-family: 'Outfit', sans-serif;
    }
    .kpi-lbl {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Custom Delta Badges */
    .delta-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 8px;
    }
    .delta-positive {
        background-color: rgba(107, 203, 119, 0.15);
        color: #6BCB77;
        border: 1px solid rgba(107, 203, 119, 0.2);
    }
    .delta-negative {
        background-color: rgba(255, 107, 107, 0.15);
        color: #FF6B6B;
        border: 1px solid rgba(255, 107, 107, 0.2);
    }

    /* Customizing Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.015);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        margin-bottom: 25px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 10px;
        color: #A0AEC0 !important;
        font-weight: 600;
        font-size: 1rem;
        border: none !important;
        padding: 0 20px;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.03);
        color: #FFFFFF !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(77, 150, 255, 0.1) !important;
        color: #4D96FF !important;
        border: 1px solid rgba(77, 150, 255, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to predict salary and apply currency / geographic scale conversions
def make_prediction(job_title, experience, education, skills, industry, company_size, location, remote, certifications):
    features = [
        "job_title",
        "experience_years",
        "education_level",
        "skills_count",
        "industry",
        "company_size",
        "location",
        "remote_work",
        "certifications"
    ]
    encoded_input = pd.DataFrame([[
        label_encoders["job_title"].transform([job_title])[0],
        experience,
        label_encoders["education_level"].transform([education])[0],
        skills,
        label_encoders["industry"].transform([industry])[0],
        label_encoders["company_size"].transform([company_size])[0],
        label_encoders["location"].transform([location])[0],
        label_encoders["remote_work"].transform([remote])[0],
        certifications
    ]], columns=features)
    
    # Model returns the base prediction in dataset scale (inflated USD)
    raw_pred = model.predict(encoded_input)[0]
    
    # Apply adjustment factor and exchange rate to localize the salary
    adj_factor, ex_rate, symbol, code, exp_boost = LOCATION_CONFIG.get(location, (1.00, 1.00, "$", "USD", 0.25))
    
    # Calculate non-linear experience scaling boost
    exp_factor = 1.0 + (experience / 20.0) * exp_boost
    
    adjusted_usd = raw_pred * adj_factor * exp_factor
    local_currency_sal = adjusted_usd * ex_rate
    
    return raw_pred, adjusted_usd, local_currency_sal, symbol, code

# Career boost recommendations simulator in local currency
def simulate_career_boosts_local(curr_profile):
    boosts = []
    _, _, base_local, symbol, code = make_prediction(
        curr_profile["job_title"],
        curr_profile["experience"],
        curr_profile["education"],
        curr_profile["skills"],
        curr_profile["industry"],
        curr_profile["company_size"],
        curr_profile["location"],
        curr_profile["remote"],
        curr_profile["certifications"]
    )
    
    def check_boost(modified_profile, action_name, category):
        _, _, new_local, _, _ = make_prediction(
            modified_profile["job_title"],
            modified_profile["experience"],
            modified_profile["education"],
            modified_profile["skills"],
            modified_profile["industry"],
            modified_profile["company_size"],
            modified_profile["location"],
            modified_profile["remote"],
            modified_profile["certifications"]
        )
        if new_local > base_local:
            boosts.append({
                "Category": category,
                "Action": action_name,
                "Salary Increase": new_local - base_local,
                "Pct": ((new_local - base_local) / base_local) * 100
            })
            
    # 1. Higher Education Scenario
    curr_edu = curr_profile["education"]
    target_edu = None
    if curr_edu in ["High School", "Diploma", "Bachelor"]:
        target_edu = "Master"
    elif curr_edu == "Master":
        target_edu = "PhD"
        
    if target_edu:
        mod = curr_profile.copy()
        mod["education"] = target_edu
        check_boost(mod, f"Upgrade degree to {target_edu}", "Education Upgradation")
        
    # 2. Certifications Scenario (limit to 5)
    if curr_profile["certifications"] < 4:
        mod = curr_profile.copy()
        mod["certifications"] = curr_profile["certifications"] + 1
        check_boost(mod, "Earn 1 extra Professional Certification", "Certifications")

    # 3. Skills Expansion Scenario (limit to 19)
    if curr_profile["skills"] < 17:
        mod = curr_profile.copy()
        mod["skills"] = curr_profile["skills"] + 2
        check_boost(mod, "Acquire 2 new specialized skills", "Skills Expansion")
        
    # 4. Company Scale Scenario
    curr_size = curr_profile["company_size"]
    if curr_size in ["Startup", "Small", "Medium"]:
        target_size = "Enterprise"
        mod = curr_profile.copy()
        mod["company_size"] = target_size
        check_boost(mod, f"Transition to an {target_size} corporation", "Company Shift")

    # 5. Remote Work Status Shift
    if curr_profile["remote"] == "No":
        mod = curr_profile.copy()
        mod["remote"] = "Yes"
        check_boost(mod, "Transition to a fully Remote role", "Remote Premium")
        
    # 6. Promotion / Experience Gain (limit to 20)
    if curr_profile["experience"] < 19:
        mod = curr_profile.copy()
        mod["experience"] = curr_profile["experience"] + 2
        check_boost(mod, "Gain 2 more Years of Experience", "Tenure / Experience")
        
    return pd.DataFrame(boosts)

# Header Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">SmartSalary Predictor</div>
    <div class="hero-subtitle">
        Leverage machine learning algorithms to evaluate professional profiles, predict competitive localized salaries, and discover strategic career progression tracks.
    </div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR: Profile Inputs for Primary Analysis
st.sidebar.markdown("### 👤 Candidate Profile")
st.sidebar.markdown("Adjust attributes below to simulate predictions dynamically.")

# Categorical selects
selected_role = st.sidebar.selectbox("Job Title", stats["roles"], index=stats["roles"].index("Software Engineer") if "Software Engineer" in stats["roles"] else 0)
education = st.sidebar.selectbox("Education Level", stats["education"], index=stats["education"].index("Bachelor") if "Bachelor" in stats["education"] else 0)
industry = st.sidebar.selectbox("Industry", stats["industries"], index=stats["industries"].index("Technology") if "Technology" in stats["industries"] else 0)
company_size = st.sidebar.selectbox("Company Size", stats["sizes"], index=stats["sizes"].index("Medium") if "Medium" in stats["sizes"] else 0)
location = st.sidebar.selectbox("Location", stats["locations"], index=stats["locations"].index("India") if "India" in stats["locations"] else 0)
remote = st.sidebar.selectbox("Remote Work Status", stats["remotes"], index=stats["remotes"].index("Hybrid") if "Hybrid" in stats["remotes"] else 0)

# Sliders - restricted to real training data limits (0-20 experience, 1-19 skills, 0-5 certs)
experience = st.sidebar.slider("Years of Experience", 0, 20, 5)
skills = st.sidebar.slider("Skills Count", 1, 19, 6)
certifications = st.sidebar.slider("Certifications Count", 0, 5, 2)

# Pack sidebar inputs into a profile dictionary
current_profile = {
    "job_title": selected_role,
    "experience": experience,
    "education": education,
    "skills": skills,
    "industry": industry,
    "company_size": company_size,
    "location": location,
    "remote": remote,
    "certifications": certifications
}

# Run primary prediction instantly
_, predicted_salary_usd, predicted_salary_local, symbol, code = make_prediction(**current_profile)

# Define Tabs
tab_predict, tab_pathfinder, tab_analytics, tab_compare = st.tabs([
    "🔮 Predict Salary",
    "📈 Career Pathfinder",
    "📊 Market Insights",
    "👥 Profile Comparator"
])

# ==========================================
# TAB 1: PREDICT SALARY
# ==========================================
with tab_predict:
    # Get config variables for local scale conversion
    _, ex_rate, symbol, code, _ = LOCATION_CONFIG.get(location, (1.00, 1.00, "$", "USD", 0.25))
    
    role_avg_local = stats["avg_by_role_adj"].get(selected_role, stats["avg_salary_adj"]) * ex_rate
    role_min_local = stats["min_by_role_adj"].get(selected_role, stats["min_salary_adj"]) * ex_rate
    role_max_local = stats["max_by_role_adj"].get(selected_role, stats["max_salary_adj"]) * ex_rate
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-box">
            <div class="kpi-lbl">Current Job Title</div>
            <div class="kpi-val" style="color: #4D96FF;">{selected_role}</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-lbl">Market Average ({code})</div>
            <div class="kpi-val">{symbol}{role_avg_local:,.0f}</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-lbl">Top Salary ({code})</div>
            <div class="kpi-val" style="color: #6BCB77;">{symbol}{role_max_local:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_out, col_gauge = st.columns([1, 1.2])
    
    with col_out:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="salary-label">Estimated Annual Salary</div>
            <div class="salary-value">{symbol}{predicted_salary_local:,.2f} {code}</div>
            <p style="color: #8F9CAE; font-size: 0.95rem; margin-top: 5px;">
                Equivalent to <strong>${predicted_salary_usd:,.2f} USD</strong> per year.
            </p>
            <p style="color: #718096; font-size: 0.85rem; margin-top: 10px; line-height: 1.4;">
                Prediction uses a Random Forest Regressor trained on 250,000 profile records and is mathematically scaled to match the local market compensation rates.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison delta badge in local currency
        diff = predicted_salary_local - role_avg_local
        diff_pct = (diff / role_avg_local) * 100
        if diff >= 0:
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="delta-badge delta-positive">
                    ▲ {symbol}{diff:,.0f} (+{diff_pct:.1f}%) above local average
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="delta-badge delta-negative">
                    ▼ {symbol}{abs(diff):,.0f} ({diff_pct:.1f}%) below local average
                </span>
            </div>
            """, unsafe_allow_html=True)
            
    with col_gauge:
        # Build interactive Plotly Gauge Indicator in local currency
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_salary_local,
            number={'prefix': symbol, 'valueformat': ",.0f", 'font': {'size': 26, 'color': "#FFFFFF"}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [role_min_local, role_max_local], 'tickformat': f"{symbol},.0f", 'tickcolor': "#A0AEC0"},
                'bar': {'color': '#4D96FF'},
                'bgcolor': "rgba(255,255,255,0.03)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [role_min_local, role_avg_local], 'color': 'rgba(255, 107, 107, 0.08)'},
                    {'range': [role_avg_local, role_max_local], 'color': 'rgba(107, 203, 119, 0.08)'}
                ],
                'threshold': {
                    'line': {'color': '#FFD93D', 'width': 3},
                    'thickness': 0.75,
                    'value': role_avg_local
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=10, l=40, r=40),
            height=260,
            title={'text': f"Standing Indicator ({code})", 'font': {'size': 14, 'color': '#718096'}, 'x': 0.5, 'xanchor': 'center'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="luxury-card">
        <h4 style="margin-top:0; color:#4D96FF;">ℹ️ Profile Details Summarized</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 15px;">
            <div><strong>Education:</strong> {edu}</div>
            <div><strong>Experience:</strong> {exp} Years</div>
            <div><strong>Industry:</strong> {ind}</div>
            <div><strong>Company Size:</strong> {size}</div>
            <div><strong>Skills Count:</strong> {skills}</div>
            <div><strong>Location:</strong> {loc}</div>
            <div><strong>Remote Status:</strong> {rem}</div>
            <div><strong>Certifications:</strong> {cert}</div>
        </div>
    </div>
    """.format(
        edu=education, exp=experience, ind=industry, size=company_size,
        skills=skills, loc=location, rem=remote, cert=certifications
    ), unsafe_allow_html=True)

# ==========================================
# TAB 2: CAREER PATHFINDER
# ==========================================
with tab_pathfinder:
    st.markdown("### 📈 Career Salary Optimizer")
    st.markdown("Discover which strategic adjustments to your profile would drive the highest financial returns in your local market.")
    
    # Run simulation
    df_boosts = simulate_career_boosts_local(current_profile)
    
    if len(df_boosts) > 0:
        df_boosts = df_boosts.sort_values(by="Salary Increase", ascending=True)
        
        col_list, col_plot = st.columns([1.1, 1.2])
        
        with col_list:
            st.write("#### Recommended Boosters")
            for idx, row in df_boosts.sort_values(by="Salary Increase", ascending=False).iterrows():
                st.markdown(f"""
                <div class="luxury-card" style="padding: 16px; margin-bottom: 12px; border-left: 4px solid #6BCB77;">
                    <div style="font-weight:700; color:#FFFFFF; font-size:1.05rem;">{row['Action']}</div>
                    <div style="color:#A0AEC0; font-size:0.9rem; margin: 4px 0;">Category: {row['Category']}</div>
                    <div style="font-size:1.1rem; color:#6BCB77; font-weight:700;">+{symbol}{row['Salary Increase']:,.0f} (+{row['Pct']:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
                
        with col_plot:
            fig_boost = px.bar(
                df_boosts,
                x="Salary Increase",
                y="Action",
                orientation="h",
                title=f"Simulated Salary Increments ({code})",
                labels={"Salary Increase": f"Potential Salary Increase ({symbol})", "Action": "Strategic Profile Change"},
                color="Salary Increase",
                color_continuous_scale="GnBu",
                template="plotly_dark"
            )
            fig_boost.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                margin=dict(t=40, b=40, l=10, r=20),
                height=400
            )
            st.plotly_chart(fig_boost, use_container_width=True)
    else:
        st.info("You've already maximized all features in the simulator! Add more variables or lower some baseline parameters to run predictions.")

# ==========================================
# TAB 3: MARKET INSIGHTS
# ==========================================
with tab_analytics:
    st.markdown("### 📊 Market Payroll Analysis")
    st.markdown("Comparative metrics scaled to **Realistic USD** (adjusted for local purchasing power index) to ensure fair comparisons.")
    
    # Render global dataset statistics in adjusted USD
    st.markdown(f"""
    <div style="display: flex; gap: 15px; margin-bottom: 25px;">
        <div style="flex:1; background:rgba(255,255,255,0.01); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:10px; text-align:center;">
            <div style="color:#A0AEC0; font-size:0.85rem; text-transform:uppercase;">Adjusted Global Salary Avg</div>
            <div style="font-size:1.6rem; font-weight:700; color:#FFFFFF; margin-top:5px;">${stats['avg_salary_adj']:,.0f} USD</div>
        </div>
        <div style="flex:1; background:rgba(255,255,255,0.01); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:10px; text-align:center;">
            <div style="color:#A0AEC0; font-size:0.85rem; text-transform:uppercase;">Highest Industry Avg</div>
            <div style="font-size:1.6rem; font-weight:700; color:#6BCB77; margin-top:5px;">${df.groupby('industry')['salary_usd_adj'].mean().max():,.0f} USD</div>
        </div>
        <div style="flex:1; background:rgba(255,255,255,0.01); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:10px; text-align:center;">
            <div style="color:#A0AEC0; font-size:0.85rem; text-transform:uppercase;">Top Paying Job Title</div>
            <div style="font-size:1.6rem; font-weight:700; color:#4D96FF; margin-top:5px;">{stats['top_role']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        # 1. Experience curves (in adjusted USD)
        exp_role_sal = df.groupby(["experience_years", "job_title"])["salary_usd_adj"].mean().reset_index()
        # Filterable multi-select roles
        selected_roles_plot = st.multiselect(
            "Select Roles to Compare Experience Curves",
            stats["roles"],
            default=stats["roles"][:3]
        )
        
        if len(selected_roles_plot) > 0:
            filtered_exp = exp_role_sal[exp_role_sal["job_title"].isin(selected_roles_plot)]
            fig_exp = px.line(
                filtered_exp,
                x="experience_years",
                y="salary_usd_adj",
                color="job_title",
                title="Salary Progression Curves by Experience (Adjusted USD)",
                labels={"experience_years": "Years of Experience", "salary_usd_adj": "Average Salary ($ USD)", "job_title": "Role"},
                template="plotly_dark",
                line_shape="spline"
            )
            fig_exp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, b=40, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0),
                height=350
            )
            st.plotly_chart(fig_exp, use_container_width=True)
        else:
            st.info("Please select at least one role to plot the experience curve.")
            
    with row1_col2:
        # 2. Industry Averages (in adjusted USD)
        ind_sal = df.groupby("industry")["salary_usd_adj"].mean().reset_index().sort_values("salary_usd_adj", ascending=True)
        fig_ind = px.bar(
            ind_sal,
            x="salary_usd_adj",
            y="industry",
            orientation="h",
            title="Industry Sectors Ranked by Average Salary (Adjusted USD)",
            labels={"salary_usd_adj": "Average Salary ($ USD)", "industry": "Industry"},
            color="salary_usd_adj",
            color_continuous_scale="Blues",
            template="plotly_dark"
        )
        fig_ind.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            margin=dict(t=50, b=40, l=10, r=20),
            height=350
        )
        st.plotly_chart(fig_ind, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # 3. Education levels (in adjusted USD)
        edu_sal = df.groupby("education_level")["salary_usd_adj"].mean().reset_index().sort_values("salary_usd_adj", ascending=True)
        fig_edu = px.bar(
            edu_sal,
            x="education_level",
            y="salary_usd_adj",
            title="Education Level Average Valuation (Adjusted USD)",
            labels={"education_level": "Degree", "salary_usd_adj": "Average Salary ($ USD)"},
            color="salary_usd_adj",
            color_continuous_scale="Purples",
            template="plotly_dark"
        )
        fig_edu.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            margin=dict(t=50, b=40, l=10, r=20),
            height=350
        )
        st.plotly_chart(fig_edu, use_container_width=True)
        
    with row2_col2:
        # 4. Remote work premium (in adjusted USD)
        rem_loc_sal = df.groupby(["location", "remote_work"])["salary_usd_adj"].mean().reset_index()
        fig_rem = px.bar(
            rem_loc_sal,
            x="location",
            y="salary_usd_adj",
            color="remote_work",
            barmode="group",
            title="Remote Work Premium by Geographic Location (Adjusted USD)",
            labels={"location": "Location", "salary_usd_adj": "Avg Salary ($ USD)", "remote_work": "Remote?"},
            template="plotly_dark",
            color_discrete_sequence=["#4D96FF", "#FF6B6B", "#6BCB77"]
        )
        fig_rem.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=40, l=10, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0),
            height=350
        )
        st.plotly_chart(fig_rem, use_container_width=True)

# ==========================================
# TAB 4: PROFILE COMPARATOR
# ==========================================
with tab_compare:
    st.markdown("### 👥 Profile Side-by-Side Comparison")
    st.markdown("Compare predictions and parameters for two distinct candidates or job offers side by side.")
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown("""
        <div style="background:rgba(77, 150, 255, 0.03); border:1px solid rgba(77, 150, 255, 0.1); border-radius:12px; padding:15px; margin-bottom:15px;">
            <h4 style="margin:0; color:#4D96FF;">👤 Profile A</h4>
        </div>
        """, unsafe_allow_html=True)
        role_a = st.selectbox("Job Title (A)", stats["roles"], index=stats["roles"].index("Software Engineer") if "Software Engineer" in stats["roles"] else 0, key="role_a")
        edu_a = st.selectbox("Education Level (A)", stats["education"], index=stats["education"].index("Bachelor") if "Bachelor" in stats["education"] else 0, key="edu_a")
        ind_a = st.selectbox("Industry (A)", stats["industries"], index=stats["industries"].index("Technology") if "Technology" in stats["industries"] else 0, key="ind_a")
        size_a = st.selectbox("Company Size (A)", stats["sizes"], index=stats["sizes"].index("Startup") if "Startup" in stats["sizes"] else 0, key="size_a")
        loc_a = st.selectbox("Location (A)", stats["locations"], index=stats["locations"].index("India") if "India" in stats["locations"] else 0, key="loc_a")
        remote_a = st.selectbox("Remote Work (A)", stats["remotes"], index=stats["remotes"].index("No") if "No" in stats["remotes"] else 0, key="remote_a")
        
        # Sliders aligned with boundaries
        exp_a = st.slider("Experience (A)", 0, 20, 4, key="exp_a")
        skills_a = st.slider("Skills Count (A)", 1, 19, 5, key="skills_a")
        cert_a = st.slider("Certifications (A)", 0, 5, 1, key="cert_a")
        
    with comp_col2:
        st.markdown("""
        <div style="background:rgba(107, 203, 119, 0.03); border:1px solid rgba(107, 203, 119, 0.1); border-radius:12px; padding:15px; margin-bottom:15px;">
            <h4 style="margin:0; color:#6BCB77;">👤 Profile B</h4>
        </div>
        """, unsafe_allow_html=True)
        role_b = st.selectbox("Job Title (B)", stats["roles"], index=stats["roles"].index("AI Engineer") if "AI Engineer" in stats["roles"] else 0, key="role_b")
        edu_b = st.selectbox("Education Level (B)", stats["education"], index=stats["education"].index("Master") if "Master" in stats["education"] else 0, key="edu_b")
        ind_b = st.selectbox("Industry (B)", stats["industries"], index=stats["industries"].index("Technology") if "Technology" in stats["industries"] else 0, key="ind_b")
        size_b = st.selectbox("Company Size (B)", stats["sizes"], index=stats["sizes"].index("Large") if "Large" in stats["sizes"] else 0, key="size_b")
        loc_b = st.selectbox("Location (B)", stats["locations"], index=stats["locations"].index("USA") if "USA" in stats["locations"] else 0, key="loc_b")
        remote_b = st.selectbox("Remote Work (B)", stats["remotes"], index=stats["remotes"].index("Yes") if "Yes" in stats["remotes"] else 0, key="remote_b")
        
        # Sliders aligned with boundaries
        exp_b = st.slider("Experience (B)", 0, 20, 6, key="exp_b")
        skills_b = st.slider("Skills Count (B)", 1, 19, 8, key="skills_b")
        cert_b = st.slider("Certifications (B)", 0, 5, 3, key="cert_b")
        
    # Predictions for both profiles
    _, pred_usd_a, pred_local_a, sym_a, code_a = make_prediction(role_a, exp_a, edu_a, skills_a, ind_a, size_a, loc_a, remote_a, cert_a)
    _, pred_usd_b, pred_local_b, sym_b, code_b = make_prediction(role_b, exp_b, edu_b, skills_b, ind_b, size_b, loc_b, remote_b, cert_b)
    
    st.markdown("<br><hr style='border-color:rgba(255,255,255,0.05);'><br>", unsafe_allow_html=True)
    
    # Results display in local currencies and USD equivalent
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown(f"""
        <div class="luxury-card" style="border-top: 4px solid #4D96FF; text-align:center;">
            <div style="color:#A0AEC0; font-size:1rem; text-transform:uppercase;">Profile A Predicted Salary</div>
            <div style="font-size:2.8rem; font-weight:800; color:#4D96FF; margin:10px 0;">{sym_a}{pred_local_a:,.2f} {code_a}</div>
            <div style="color:#718096; font-size:0.9rem;">(${pred_usd_a:,.2f} USD equivalent)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        st.markdown(f"""
        <div class="luxury-card" style="border-top: 4px solid #6BCB77; text-align:center;">
            <div style="color:#A0AEC0; font-size:1rem; text-transform:uppercase;">Profile B Predicted Salary</div>
            <div style="font-size:2.8rem; font-weight:800; color:#6BCB77; margin:10px 0;">{sym_b}{pred_local_b:,.2f} {code_b}</div>
            <div style="color:#718096; font-size:0.9rem;">(${pred_usd_b:,.2f} USD equivalent)</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Standardized Comparison chart (in USD equivalent)
    comp_df = pd.DataFrame({
        "Profile": ["Profile A", "Profile B"],
        "Adjusted USD Salary": [pred_usd_a, pred_usd_b]
    })
    fig_comp = px.bar(
        comp_df,
        x="Profile",
        y="Adjusted USD Salary",
        color="Profile",
        color_discrete_map={"Profile A": "#4D96FF", "Profile B": "#6BCB77"},
        title="Direct Purchasing Power Parity Comparison (USD Equivalent)",
        labels={"Adjusted USD Salary": "Adjusted Salary ($ USD)"},
        template="plotly_dark"
    )
    fig_comp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=40, l=40, r=40),
        height=320,
        showlegend=False
    )
    st.plotly_chart(fig_comp, use_container_width=True)