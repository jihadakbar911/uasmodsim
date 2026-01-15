import random
import simpy
import pandas as pd
import statistics as stats
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# Helper stats
# =========================
def safe_mean(xs):
    return stats.mean(xs) if xs else float("nan")

def percentile(xs, p):
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] + (k - f) * (xs_sorted[c] - xs_sorted[f])

# =========================
# Sampling functions
# =========================
def sample_interarrival(mean):
    return random.expovariate(1.0 / mean)

def sample_triangular(min_val, mode_val, max_val):
    return random.triangular(min_val, max_val, mode_val)

def get_service_time(service_time_dict, name):
    a, m, b = service_time_dict[name]
    return sample_triangular(a, m, b)

def sample_priority(p_priority):
    return 0 if random.random() < p_priority else 1  # 0=prioritas, 1=normal

def sample_test_type(p_fast):
    return "fast" if random.random() < p_fast else "slow"

def log_event(env, msg, verbose):
    if verbose:
        print(f"Waktu: {env.now:6.2f} | {msg}")

# =========================
# Logging structure
# =========================
def make_log():
    return {
        "patients": [],
        "wait_reg": [],
        "wait_sample": [],
        "wait_machine": [],
        "wait_verify": [],
        "system_time": [],
        "counts": {"priority": 0, "normal": 0, "fast": 0, "slow": 0}
    }

# =========================
# Patient process
# =========================
def patient(env, pid, priority, test_type, resources, log, params):
    arrival_time = env.now

    # REG
    q_start = env.now
    if pid < params["max_debug_patients"]:
        log_event(env, f"Pasien {pid} antre REGISTRASI", params["verbose"])
    with resources["reg"].request(priority=priority) as req:
        yield req
        wait_reg = env.now - q_start
        if pid < params["max_debug_patients"]:
            log_event(env, f"Mulai REGISTRASI pasien {pid} (tunggu {wait_reg:.2f})", params["verbose"])
        service_reg = get_service_time(params["service_time"], "registration")
        yield env.timeout(service_reg)
        if pid < params["max_debug_patients"]:
            log_event(env, f"Selesai REGISTRASI pasien {pid} (durasi {service_reg:.2f})", params["verbose"])

    # SAMPLING
    q_start = env.now
    if pid < params["max_debug_patients"]:
        log_event(env, f"Pasien {pid} antre AMBIL SAMPEL", params["verbose"])
    with resources["phleb"].request(priority=priority) as req:
        yield req
        wait_sample = env.now - q_start
        if pid < params["max_debug_patients"]:
            log_event(env, f"Mulai AMBIL SAMPEL pasien {pid} (tunggu {wait_sample:.2f})", params["verbose"])
        service_sample = get_service_time(params["service_time"], "sampling")
        yield env.timeout(service_sample)
        if pid < params["max_debug_patients"]:
            log_event(env, f"Selesai AMBIL SAMPEL pasien {pid} (durasi {service_sample:.2f})", params["verbose"])

    # TEST
    q_start = env.now
    if test_type == "fast":
        machine = resources["machine_fast"]
        service_test = get_service_time(params["service_time"], "test_fast")
        stage = "TES CEPAT"
    else:
        machine = resources["machine_slow"]
        service_test = get_service_time(params["service_time"], "test_slow")
        stage = "TES LAMA"

    if pid < params["max_debug_patients"]:
        log_event(env, f"Pasien {pid} antre {stage}", params["verbose"])
    with machine.request(priority=priority) as req:
        yield req
        wait_machine = env.now - q_start
        if pid < params["max_debug_patients"]:
            log_event(env, f"Mulai {stage} pasien {pid} (tunggu {wait_machine:.2f})", params["verbose"])
        yield env.timeout(service_test)
        if pid < params["max_debug_patients"]:
            log_event(env, f"Selesai {stage} pasien {pid} (durasi {service_test:.2f})", params["verbose"])

    # VERIFY (optional)
    wait_verify = 0.0
    if resources.get("verify") is not None:
        q_start = env.now
        if pid < params["max_debug_patients"]:
            log_event(env, f"Pasien {pid} antre VERIFIKASI", params["verbose"])
        with resources["verify"].request(priority=priority) as req:
            yield req
            wait_verify = env.now - q_start
            if pid < params["max_debug_patients"]:
                log_event(env, f"Mulai VERIFIKASI pasien {pid} (tunggu {wait_verify:.2f})", params["verbose"])
            service_verify = get_service_time(params["service_time"], "verification")
            yield env.timeout(service_verify)
            if pid < params["max_debug_patients"]:
                log_event(env, f"Selesai VERIFIKASI pasien {pid} (durasi {service_verify:.2f})", params["verbose"])

    finish_time = env.now
    system_time = finish_time - arrival_time

    if pid < params["max_debug_patients"]:
        log_event(env, f"Pasien {pid} SELESAI total waktu sistem {system_time:.2f}\n", params["verbose"])

    record = {
        "pid": pid,
        "arrival": arrival_time,
        "finish": finish_time,
        "priority": priority,
        "test_type": test_type,
        "wait_reg": wait_reg,
        "wait_sample": wait_sample,
        "wait_machine": wait_machine,
        "wait_verify": wait_verify,
        "system_time": system_time,
    }
    log["patients"].append(record)

    # warm-up filter
    if arrival_time >= params["warm_up"]:
        log["wait_reg"].append(wait_reg)
        log["wait_sample"].append(wait_sample)
        log["wait_machine"].append(wait_machine)
        if resources.get("verify") is not None:
            log["wait_verify"].append(wait_verify)
        log["system_time"].append(system_time)

# =========================
# Arrivals generator
# =========================
def arrivals(env, resources, log, params):
    pid = 0
    while True:
        priority = sample_priority(params["p_priority"])
        test_type = sample_test_type(params["p_fast"])

        if priority == 0:
            log["counts"]["priority"] += 1
        else:
            log["counts"]["normal"] += 1

        if test_type == "fast":
            log["counts"]["fast"] += 1
        else:
            log["counts"]["slow"] += 1

        if pid < params["max_debug_patients"]:
            pr_txt = "PRIORITAS" if priority == 0 else "NORMAL"
            log_event(env, f"Kedatangan pasien {pid} [{pr_txt}] | Tes: {test_type}", params["verbose"])

        env.process(patient(env, pid, priority, test_type, resources, log, params))
        pid += 1

        iat = sample_interarrival(params["mean_interarrival"])
        yield env.timeout(iat)

# =========================
# Run simulation
# =========================
def run_simulation(params, seed=123):
    random.seed(seed)
    env = simpy.Environment()

    resources = {
        "reg": simpy.PriorityResource(env, capacity=params["c_reg"]),
        "phleb": simpy.PriorityResource(env, capacity=params["c_phleb"]),
        "machine_fast": simpy.PriorityResource(env, capacity=params["c_fast"]),
        "machine_slow": simpy.PriorityResource(env, capacity=params["c_slow"]),
    }
    if params["use_verify"] and params["c_verify"] > 0:
        resources["verify"] = simpy.PriorityResource(env, capacity=params["c_verify"])
    else:
        resources["verify"] = None

    log = make_log()
    env.process(arrivals(env, resources, log, params))
    env.run(until=params["sim_time"])
    return log

def summarize(log):
    return {
        "Total pasien (termasuk warm-up)": len(log["patients"]),
        "Pasien dihitung (setelah warm-up)": len(log["system_time"]),
        "Mean wait registrasi": safe_mean(log["wait_reg"]),
        "Mean wait sampel": safe_mean(log["wait_sample"]),
        "Mean wait mesin": safe_mean(log["wait_machine"]),
        "Mean total time in system": safe_mean(log["system_time"]),
        "P95 total time in system": percentile(log["system_time"], 95),
        "Komposisi": log["counts"],
    }

# =========================
# Analysis Functions
# =========================
def analyze_bottleneck(summary, log):
    """Analyze which stage is the bottleneck"""
    stages = {
        "Registrasi": summary["Mean wait registrasi"],
        "Pengambilan Sampel": summary["Mean wait sampel"],
        "Mesin Tes": summary["Mean wait mesin"],
        "Verifikasi": safe_mean(log["wait_verify"]) if log["wait_verify"] else 0
    }
    
    # Find bottleneck (highest wait time)
    bottleneck = max(stages, key=stages.get)
    bottleneck_time = stages[bottleneck]
    
    # Calculate severity (percentage of total wait)
    total_wait = sum(stages.values())
    severity = (bottleneck_time / total_wait * 100) if total_wait > 0 else 0
    
    return {
        "bottleneck": bottleneck,
        "wait_time": bottleneck_time,
        "severity": severity,
        "all_stages": stages
    }

def calculate_utilization(log, params):
    """Calculate resource utilization"""
    df = pd.DataFrame(log["patients"])
    if df.empty:
        return {}
    
    sim_time = params["sim_time"]
    service_time = params["service_time"]
    
    # Average service times (mode of triangular)
    avg_reg = service_time["registration"][1]
    avg_sample = service_time["sampling"][1]
    avg_fast = service_time["test_fast"][1]
    avg_slow = service_time["test_slow"][1]
    avg_verify = service_time["verification"][1] if params["use_verify"] else 0
    
    total_patients = len(df)
    fast_patients = log["counts"]["fast"]
    slow_patients = log["counts"]["slow"]
    
    # Utilization = (total service time) / (capacity * sim_time)
    util_reg = (total_patients * avg_reg) / (params["c_reg"] * sim_time) * 100
    util_phleb = (total_patients * avg_sample) / (params["c_phleb"] * sim_time) * 100
    util_fast = (fast_patients * avg_fast) / (params["c_fast"] * sim_time) * 100
    util_slow = (slow_patients * avg_slow) / (params["c_slow"] * sim_time) * 100
    util_verify = (total_patients * avg_verify) / (params["c_verify"] * sim_time) * 100 if params["use_verify"] else 0
    
    return {
        "Registrasi": min(util_reg, 100),
        "Pengambilan Sampel": min(util_phleb, 100),
        "Mesin Tes Cepat": min(util_fast, 100),
        "Mesin Tes Lama": min(util_slow, 100),
        "Verifikasi": min(util_verify, 100)
    }

def get_recommendations(bottleneck_analysis, utilization, params):
    """Generate optimization recommendations"""
    recommendations = []
    
    bottleneck = bottleneck_analysis["bottleneck"]
    severity = bottleneck_analysis["severity"]
    
    # Recommendation based on bottleneck
    if severity > 40:
        if bottleneck == "Registrasi":
            new_capacity = params["c_reg"] + 1
            recommendations.append({
                "type": "critical",
                "icon": "🚨",
                "title": "Tambah Petugas Registrasi",
                "desc": f"Registrasi adalah bottleneck utama ({severity:.1f}% dari total waktu tunggu). Pertimbangkan menambah dari {params['c_reg']} menjadi {new_capacity} petugas.",
                "impact": "Estimasi pengurangan waktu tunggu: 30-50%"
            })
        elif bottleneck == "Pengambilan Sampel":
            new_capacity = params["c_phleb"] + 1
            recommendations.append({
                "type": "critical",
                "icon": "🚨",
                "title": "Tambah Petugas Pengambilan Sampel",
                "desc": f"Pengambilan sampel adalah bottleneck utama ({severity:.1f}% dari total waktu tunggu). Pertimbangkan menambah dari {params['c_phleb']} menjadi {new_capacity} petugas.",
                "impact": "Estimasi pengurangan waktu tunggu: 25-45%"
            })
        elif bottleneck == "Mesin Tes":
            recommendations.append({
                "type": "critical",
                "icon": "🚨",
                "title": "Tambah Kapasitas Mesin Tes",
                "desc": f"Mesin tes adalah bottleneck utama ({severity:.1f}% dari total waktu tunggu). Pertimbangkan menambah mesin tes cepat atau lama.",
                "impact": "Estimasi pengurangan waktu tunggu: 35-55%"
            })
    
    # Check high utilization
    for resource, util in utilization.items():
        if util > 85:
            recommendations.append({
                "type": "warning",
                "icon": "⚠️",
                "title": f"Utilisasi Tinggi: {resource}",
                "desc": f"Utilisasi {resource} mencapai {util:.1f}%. Resource hampir overloaded.",
                "impact": "Risiko antrian panjang saat peak hours"
            })
    
    # Check low utilization
    for resource, util in utilization.items():
        if util < 30 and util > 0:
            recommendations.append({
                "type": "info",
                "icon": "💡",
                "title": f"Utilisasi Rendah: {resource}",
                "desc": f"Utilisasi {resource} hanya {util:.1f}%. Mungkin bisa dikurangi kapasitasnya.",
                "impact": "Potensi efisiensi biaya operasional"
            })
    
    if not recommendations:
        recommendations.append({
            "type": "success",
            "icon": "✅",
            "title": "Sistem Optimal",
            "desc": "Tidak ada bottleneck signifikan terdeteksi. Sistem berjalan dengan baik.",
            "impact": "Pertahankan konfigurasi saat ini"
        })
    
    return recommendations

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="Simulasi Antrean Lab Klinik",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for scenario comparison
if "scenarios" not in st.session_state:
    st.session_state.scenarios = []

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Dark mode toggle function
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Custom CSS with dark mode support
dark_mode = st.session_state.dark_mode

if dark_mode:
    css_vars = """
        --bg-primary: #1a1a2e;
        --bg-secondary: #16213e;
        --bg-card: #0f3460;
        --text-primary: #eaeaea;
        --text-secondary: #a0a0a0;
        --accent: #00d9ff;
        --accent-dark: #0099b8;
        --border: #2d4a6f;
        --sidebar-bg: linear-gradient(180deg, #16213e 0%, #1a1a2e 100%);
        --header-bg: linear-gradient(135deg, #0f3460 0%, #1a1a2e 50%, #16213e 100%);
    """
else:
    css_vars = """
        --bg-primary: #ffffff;
        --bg-secondary: #f8fbfc;
        --bg-card: #ffffff;
        --text-primary: #1e3a5f;
        --text-secondary: #3d5a73;
        --accent: #2d5a7b;
        --accent-dark: #1e3a5f;
        --border: #b8d4e8;
        --sidebar-bg: linear-gradient(180deg, #f0f7fa 0%, #e3f2f6 100%);
        --header-bg: linear-gradient(135deg, #1e3a5f 0%, #2d5a7b 50%, #3d7a9e 100%);
    """

st.markdown(f"""
<style>
    :root {{
        {css_vars}
    }}
    
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Main container styling */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    
    /* Header styling */
    .main-header {{
        background: var(--header-bg);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}
    
    .main-header h1 {{
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        margin: 0;
        font-size: 2rem;
    }}
    
    .main-header p {{
        color: #b8d4e8;
        font-family: 'Poppins', sans-serif;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: var(--sidebar-bg);
    }}
    
    /* Section headers */
    .section-header {{
        background: linear-gradient(90deg, var(--accent) 0%, transparent 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        border-radius: 25px;
        box-shadow: 0 4px 12px rgba(45, 90, 123, 0.3);
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(45, 90, 123, 0.4);
    }}
    
    /* Info box styling */
    .info-box {{
        background: linear-gradient(135deg, #e8f4f8 0%, #d4eaf1 100%);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1rem;
    }}
    
    .info-box h4 {{
        color: var(--text-primary);
        font-family: 'Poppins', sans-serif;
        margin: 0 0 0.8rem 0;
    }}
    
    .info-box ul {{
        color: var(--text-secondary);
        font-family: 'Poppins', sans-serif;
        margin: 0;
        padding-left: 1.2rem;
    }}
    
    /* Recommendation cards */
    .rec-card {{
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        font-family: 'Poppins', sans-serif;
    }}
    
    .rec-critical {{
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
    }}
    
    .rec-warning {{
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
    }}
    
    .rec-info {{
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
    }}
    
    .rec-success {{
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
    }}
    
    .rec-title {{
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.3rem;
        color: #1e293b;
    }}
    
    .rec-desc {{
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 0.3rem;
    }}
    
    .rec-impact {{
        font-size: 0.8rem;
        color: #64748b;
        font-style: italic;
    }}
    
    /* Bottleneck indicator */
    .bottleneck-card {{
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a7b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }}
    
    .bottleneck-title {{
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 0.3rem;
    }}
    
    .bottleneck-value {{
        font-size: 1.8rem;
        font-weight: 700;
    }}
    
    .bottleneck-severity {{
        font-size: 0.85rem;
        margin-top: 0.5rem;
        padding: 0.3rem 0.8rem;
        background: rgba(255,255,255,0.2);
        border-radius: 20px;
        display: inline-block;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        color: var(--text-secondary);
        font-family: 'Poppins', sans-serif;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid var(--border);
    }}
    
    /* Utilization gauge */
    .util-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1rem 0;
    }}
    
    .util-item {{
        flex: 1;
        min-width: 150px;
        background: var(--bg-card);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    /* Animation for loading */
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    
    .loading-pulse {{
        animation: pulse 1.5s ease-in-out infinite;
    }}
</style>
""", unsafe_allow_html=True)

# Header dengan tema medis
st.markdown("""
<div class="main-header">
    <h1>🏥 Simulasi Antrean Laboratorium Klinik Rumah Sakit</h1>
    <p>📊 Sistem Pemodelan & Simulasi Berbasis SimPy untuk Optimasi Layanan Laboratorium</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🩺 Parameter Kedatangan")
    mean_interarrival = st.number_input("⏱️ Mean interarrival (menit/pasien)", min_value=0.5, max_value=60.0, value=5.0, step=0.5)
    p_priority = st.slider("🚨 Proporsi pasien prioritas", 0.0, 1.0, 0.20, 0.05)
    p_fast = st.slider("⚡ Proporsi tes cepat (fast)", 0.0, 1.0, 0.70, 0.05)

    st.markdown("---")
    st.markdown("### 👥 Kapasitas Resource")
    c_reg = st.slider("📝 Petugas registrasi", 1, 5, 1)
    c_phleb = st.slider("💉 Petugas pengambilan sampel", 1, 6, 2)
    c_fast = st.slider("🔬 Mesin tes cepat", 1, 5, 1)
    c_slow = st.slider("🧪 Mesin tes lama", 1, 5, 1)

    st.markdown("---")
    st.markdown("### ⏰ Simulasi")
    sim_time = st.slider("📅 Durasi simulasi (menit)", 60, 720, 480, 30)
    warm_up = st.slider("🔥 Warm-up (menit)", 0, 120, 30, 10)

    st.markdown("---")
    st.markdown("### ✅ Verifikasi (opsional)")
    use_verify = st.checkbox("Aktifkan tahap verifikasi", value=True)
    c_verify = st.slider("👨‍⚕️ Petugas verifikasi", 1, 3, 1)

    st.markdown("---")
    st.markdown("### 📊 Output Detail")
    verbose = st.checkbox("Tampilkan log detail (di terminal)", value=False)
    max_debug_patients = st.slider("Batas pasien untuk log detail", 1, 100, 30, 1)

    st.markdown("---")
    seed = st.number_input("🎲 Random seed", min_value=1, max_value=999999, value=123, step=1)
    
    st.markdown("---")
    st.markdown("### 🌙 Tampilan")
    if st.button("🌓 Toggle Dark Mode"):
        toggle_dark_mode()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📁 Skenario")
    scenario_name = st.text_input("Nama skenario", value=f"Skenario {len(st.session_state.scenarios) + 1}")

service_time = {
    "registration": (2, 3, 5),
    "sampling": (3, 5, 8),
    "test_fast": (8, 12, 18),
    "test_slow": (20, 30, 45),
    "verification": (1, 2, 4),
}

params = {
    "mean_interarrival": float(mean_interarrival),
    "p_priority": float(p_priority),
    "p_fast": float(p_fast),
    "c_reg": int(c_reg),
    "c_phleb": int(c_phleb),
    "c_fast": int(c_fast),
    "c_slow": int(c_slow),
    "use_verify": bool(use_verify),
    "c_verify": int(c_verify),
    "sim_time": int(sim_time),
    "warm_up": int(warm_up),
    "verbose": bool(verbose),
    "max_debug_patients": int(max_debug_patients),
    "service_time": service_time,
}

colA, colB = st.columns([2, 1])

with colA:
    if st.button("🚀 Jalankan Simulasi", use_container_width=True):
        with st.spinner("⏳ Menjalankan simulasi..."):
            log = run_simulation(params, seed=seed)
            summary = summarize(log)

        # KPI Cards
        st.markdown('<div class="section-header">📊 Ringkasan KPI (Key Performance Indicators)</div>', unsafe_allow_html=True)
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.metric(
                label="👥 Total Pasien",
                value=f"{summary['Total pasien (termasuk warm-up)']}",
                delta=f"{summary['Pasien dihitung (setelah warm-up)']} dihitung"
            )
        
        with kpi_col2:
            st.metric(
                label="⏱️ Rata-rata Waktu Sistem",
                value=f"{summary['Mean total time in system']:.2f} menit"
            )
        
        with kpi_col3:
            st.metric(
                label="📈 P95 Waktu Sistem",
                value=f"{summary['P95 total time in system']:.2f} menit"
            )
        
        with kpi_col4:
            st.metric(
                label="🔬 Rasio Tes",
                value=f"Fast: {summary['Komposisi']['fast']}",
                delta=f"Slow: {summary['Komposisi']['slow']}"
            )
        
        # Additional KPI row
        kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(4)
        
        with kpi_col5:
            st.metric(
                label="📝 Tunggu Registrasi",
                value=f"{summary['Mean wait registrasi']:.2f} menit"
            )
        
        with kpi_col6:
            st.metric(
                label="💉 Tunggu Sampel",
                value=f"{summary['Mean wait sampel']:.2f} menit"
            )
        
        with kpi_col7:
            st.metric(
                label="🔬 Tunggu Mesin",
                value=f"{summary['Mean wait mesin']:.2f} menit"
            )
        
        with kpi_col8:
            st.metric(
                label="🚨 Pasien Prioritas",
                value=f"{summary['Komposisi']['priority']}",
                delta=f"Normal: {summary['Komposisi']['normal']}"
            )

        # Data Pasien Table
        df = pd.DataFrame(log["patients"])
        
        # Sort by PID untuk tampilan terurut
        df = df.sort_values(by="pid", ascending=True).reset_index(drop=True)
        
        # Rename columns untuk tampilan lebih deskriptif
        df_display = df.head(30).copy()
        df_display = df_display.rename(columns={
            "pid": "ID Pasien",
            "arrival": "Waktu Tiba",
            "finish": "Waktu Selesai",
            "priority": "Prioritas",
            "test_type": "Jenis Tes",
            "wait_reg": "Tunggu Registrasi",
            "wait_sample": "Tunggu Sampel",
            "wait_machine": "Tunggu Mesin",
            "wait_verify": "Tunggu Verifikasi",
            "system_time": "Total Waktu Sistem"
        })
        
        # Format prioritas untuk lebih readable
        df_display["Prioritas"] = df_display["Prioritas"].map({0: "🔴 Prioritas", 1: "🟢 Normal"})
        
        st.markdown('<div class="section-header">📋 Data Pasien (30 Baris Pertama - Terurut berdasarkan ID)</div>', unsafe_allow_html=True)
        
        # Styling tabel dengan gradient warna
        styled_df = df_display.style.format({
            "Waktu Tiba": "{:.2f}",
            "Waktu Selesai": "{:.2f}",
            "Tunggu Registrasi": "{:.2f}",
            "Tunggu Sampel": "{:.2f}",
            "Tunggu Mesin": "{:.2f}",
            "Tunggu Verifikasi": "{:.2f}",
            "Total Waktu Sistem": "{:.2f}"
        }).background_gradient(
            subset=["Tunggu Registrasi", "Tunggu Sampel", "Tunggu Mesin", "Tunggu Verifikasi"],
            cmap="YlOrRd",
            vmin=0
        ).background_gradient(
            subset=["Total Waktu Sistem"],
            cmap="Blues",
            vmin=0
        )
        
        st.dataframe(styled_df, use_container_width=True, height=500)

        # =====================
        # GRAFIK INTERAKTIF
        # =====================
        st.markdown('<div class="section-header">📈 Grafik Interaktif</div>', unsafe_allow_html=True)
        
        df_post = df[df["arrival"] >= params["warm_up"]]
        
        # Tab untuk grafik berbeda
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribusi Waktu", "🥧 Komposisi Pasien", "📉 Perbandingan Tunggu", "📈 Timeline Pasien"])
        
        with tab1:
            # Histogram interaktif dengan Plotly
            fig_hist = px.histogram(
                df_post, 
                x="system_time", 
                nbins=30,
                title="Distribusi Total Waktu dalam Sistem",
                labels={"system_time": "Total Waktu (menit)", "count": "Frekuensi"},
                color_discrete_sequence=["#2d5a7b"]
            )
            fig_hist.update_layout(
                xaxis_title="Total Waktu dalam Sistem (menit)",
                yaxis_title="Jumlah Pasien",
                template="plotly_white",
                hoverlabel=dict(bgcolor="white", font_size=12),
                title_font_size=16,
                title_font_color="#1e3a5f"
            )
            fig_hist.update_traces(
                hovertemplate="<b>Waktu:</b> %{x:.2f} menit<br><b>Jumlah:</b> %{y} pasien<extra></extra>"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab2:
            # Pie chart untuk komposisi pasien
            col_pie1, col_pie2 = st.columns(2)
            
            with col_pie1:
                # Pie chart prioritas
                priority_data = pd.DataFrame({
                    "Kategori": ["Prioritas", "Normal"],
                    "Jumlah": [summary["Komposisi"]["priority"], summary["Komposisi"]["normal"]]
                })
                fig_pie1 = px.pie(
                    priority_data, 
                    values="Jumlah", 
                    names="Kategori",
                    title="Komposisi Prioritas Pasien",
                    color_discrete_sequence=["#e74c3c", "#27ae60"],
                    hole=0.4
                )
                fig_pie1.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>"
                )
                fig_pie1.update_layout(title_font_size=14, title_font_color="#1e3a5f")
                st.plotly_chart(fig_pie1, use_container_width=True)
            
            with col_pie2:
                # Pie chart jenis tes
                test_data = pd.DataFrame({
                    "Jenis Tes": ["Fast", "Slow"],
                    "Jumlah": [summary["Komposisi"]["fast"], summary["Komposisi"]["slow"]]
                })
                fig_pie2 = px.pie(
                    test_data, 
                    values="Jumlah", 
                    names="Jenis Tes",
                    title="Komposisi Jenis Tes",
                    color_discrete_sequence=["#3498db", "#9b59b6"],
                    hole=0.4
                )
                fig_pie2.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>"
                )
                fig_pie2.update_layout(title_font_size=14, title_font_color="#1e3a5f")
                st.plotly_chart(fig_pie2, use_container_width=True)
        
        with tab3:
            # Bar chart perbandingan waktu tunggu
            wait_data = pd.DataFrame({
                "Tahap": ["Registrasi", "Pengambilan Sampel", "Mesin Tes", "Verifikasi"],
                "Rata-rata Tunggu": [
                    summary["Mean wait registrasi"],
                    summary["Mean wait sampel"],
                    summary["Mean wait mesin"],
                    safe_mean(log["wait_verify"]) if log["wait_verify"] else 0
                ]
            })
            
            fig_bar = px.bar(
                wait_data,
                x="Tahap",
                y="Rata-rata Tunggu",
                title="Perbandingan Waktu Tunggu per Tahap",
                color="Tahap",
                color_discrete_sequence=["#1abc9c", "#3498db", "#9b59b6", "#e67e22"]
            )
            fig_bar.update_layout(
                xaxis_title="Tahap Layanan",
                yaxis_title="Rata-rata Waktu Tunggu (menit)",
                template="plotly_white",
                showlegend=False,
                title_font_size=16,
                title_font_color="#1e3a5f"
            )
            fig_bar.update_traces(
                hovertemplate="<b>%{x}</b><br>Waktu Tunggu: %{y:.2f} menit<extra></extra>"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Box plot untuk distribusi waktu tunggu
            wait_melted = df_post.melt(
                id_vars=["pid"],
                value_vars=["wait_reg", "wait_sample", "wait_machine", "wait_verify"],
                var_name="Tahap",
                value_name="Waktu Tunggu"
            )
            wait_melted["Tahap"] = wait_melted["Tahap"].map({
                "wait_reg": "Registrasi",
                "wait_sample": "Pengambilan Sampel",
                "wait_machine": "Mesin Tes",
                "wait_verify": "Verifikasi"
            })
            
            fig_box = px.box(
                wait_melted,
                x="Tahap",
                y="Waktu Tunggu",
                title="Distribusi Waktu Tunggu per Tahap",
                color="Tahap",
                color_discrete_sequence=["#1abc9c", "#3498db", "#9b59b6", "#e67e22"]
            )
            fig_box.update_layout(
                xaxis_title="Tahap Layanan",
                yaxis_title="Waktu Tunggu (menit)",
                template="plotly_white",
                showlegend=False,
                title_font_size=16,
                title_font_color="#1e3a5f"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with tab4:
            # Line chart timeline pasien (sample pertama 50 pasien)
            df_timeline = df_post.head(50).copy()
            
            fig_timeline = go.Figure()
            
            # Add traces untuk setiap metric
            fig_timeline.add_trace(go.Scatter(
                x=df_timeline["pid"],
                y=df_timeline["system_time"],
                mode="lines+markers",
                name="Total Waktu Sistem",
                line=dict(color="#2d5a7b", width=2),
                marker=dict(size=6),
                hovertemplate="<b>Pasien %{x}</b><br>Total: %{y:.2f} menit<extra></extra>"
            ))
            
            fig_timeline.add_trace(go.Scatter(
                x=df_timeline["pid"],
                y=df_timeline["wait_reg"],
                mode="lines+markers",
                name="Tunggu Registrasi",
                line=dict(color="#1abc9c", width=2),
                marker=dict(size=6),
                hovertemplate="<b>Pasien %{x}</b><br>Registrasi: %{y:.2f} menit<extra></extra>"
            ))
            
            fig_timeline.add_trace(go.Scatter(
                x=df_timeline["pid"],
                y=df_timeline["wait_sample"],
                mode="lines+markers",
                name="Tunggu Sampel",
                line=dict(color="#3498db", width=2),
                marker=dict(size=6),
                hovertemplate="<b>Pasien %{x}</b><br>Sampel: %{y:.2f} menit<extra></extra>"
            ))
            
            fig_timeline.add_trace(go.Scatter(
                x=df_timeline["pid"],
                y=df_timeline["wait_machine"],
                mode="lines+markers",
                name="Tunggu Mesin",
                line=dict(color="#9b59b6", width=2),
                marker=dict(size=6),
                hovertemplate="<b>Pasien %{x}</b><br>Mesin: %{y:.2f} menit<extra></extra>"
            ))
            
            fig_timeline.update_layout(
                title="Timeline Waktu Layanan per Pasien (50 Pasien Pertama)",
                xaxis_title="ID Pasien",
                yaxis_title="Waktu (menit)",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                title_font_size=16,
                title_font_color="#1e3a5f"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Scatter plot arrival vs system time
            fig_scatter = px.scatter(
                df_post,
                x="arrival",
                y="system_time",
                color="test_type",
                title="Waktu Tiba vs Total Waktu Sistem",
                labels={"arrival": "Waktu Tiba (menit)", "system_time": "Total Waktu Sistem (menit)", "test_type": "Jenis Tes"},
                color_discrete_map={"fast": "#3498db", "slow": "#e74c3c"}
            )
            fig_scatter.update_layout(
                template="plotly_white",
                title_font_size=16,
                title_font_color="#1e3a5f"
            )
            fig_scatter.update_traces(
                marker=dict(size=8, opacity=0.7),
                hovertemplate="<b>Waktu Tiba:</b> %{x:.2f} menit<br><b>Total Waktu:</b> %{y:.2f} menit<extra></extra>"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # =====================
        # TAB ANALISIS LANJUTAN
        # =====================
        st.markdown('<div class="section-header">🔍 Analisis Lanjutan</div>', unsafe_allow_html=True)
        
        tab_a1, tab_a2, tab_a3, tab_a4 = st.tabs(["🎯 Bottleneck Analysis", "📊 Utilisasi Resource", "📅 Gantt Chart", "🔥 Heatmap"])
        
        # Run analysis
        bottleneck_analysis = analyze_bottleneck(summary, log)
        utilization = calculate_utilization(log, params)
        recommendations = get_recommendations(bottleneck_analysis, utilization, params)
        
        with tab_a1:
            col_bn1, col_bn2 = st.columns([1, 2])
            
            with col_bn1:
                # Bottleneck indicator card
                st.markdown(f"""
                <div class="bottleneck-card">
                    <div class="bottleneck-title">🎯 Bottleneck Utama</div>
                    <div class="bottleneck-value">{bottleneck_analysis['bottleneck']}</div>
                    <div class="bottleneck-severity">Severity: {bottleneck_analysis['severity']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric(
                    label="Waktu Tunggu Rata-rata",
                    value=f"{bottleneck_analysis['wait_time']:.2f} menit"
                )
            
            with col_bn2:
                # Bar chart of all stages wait time
                stages_df = pd.DataFrame({
                    "Tahap": list(bottleneck_analysis['all_stages'].keys()),
                    "Waktu Tunggu": list(bottleneck_analysis['all_stages'].values())
                })
                
                colors = ['#ef4444' if s == bottleneck_analysis['bottleneck'] else '#3b82f6' for s in stages_df['Tahap']]
                
                fig_bn = go.Figure(data=[
                    go.Bar(
                        x=stages_df['Tahap'],
                        y=stages_df['Waktu Tunggu'],
                        marker_color=colors,
                        hovertemplate="<b>%{x}</b><br>Waktu: %{y:.2f} menit<extra></extra>"
                    )
                ])
                fig_bn.update_layout(
                    title="Perbandingan Waktu Tunggu (Merah = Bottleneck)",
                    xaxis_title="Tahap",
                    yaxis_title="Waktu Tunggu (menit)",
                    template="plotly_white",
                    height=350
                )
                st.plotly_chart(fig_bn, use_container_width=True)
            
            # Recommendations
            st.markdown("#### 💡 Rekomendasi Optimasi")
            for rec in recommendations:
                rec_class = f"rec-{rec['type']}"
                st.markdown(f"""
                <div class="rec-card {rec_class}">
                    <div class="rec-title">{rec['icon']} {rec['title']}</div>
                    <div class="rec-desc">{rec['desc']}</div>
                    <div class="rec-impact">📈 {rec['impact']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab_a2:
            # Utilization gauges
            st.markdown("#### 📊 Tingkat Utilisasi Resource")
            
            util_cols = st.columns(5)
            util_items = list(utilization.items())
            
            for i, (resource, util_val) in enumerate(util_items):
                with util_cols[i]:
                    # Color based on utilization level
                    if util_val > 85:
                        color = "#ef4444"
                        status = "⚠️ Tinggi"
                    elif util_val > 60:
                        color = "#f59e0b"
                        status = "📊 Normal"
                    else:
                        color = "#10b981"
                        status = "✅ Rendah"
                    
                    # Gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=util_val,
                        title={'text': resource, 'font': {'size': 12}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': color},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 60], 'color': '#d1fae5'},
                                {'range': [60, 85], 'color': '#fef3c7'},
                                {'range': [85, 100], 'color': '#fee2e2'}
                            ],
                        }
                    ))
                    fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    st.caption(status)
        
        with tab_a3:
            # Gantt Chart for patient flow
            st.markdown("#### 📅 Gantt Chart - Alur 20 Pasien Pertama")
            
            df_gantt = df_post.head(20).copy()
            
            gantt_data = []
            for _, row in df_gantt.iterrows():
                pid = row['pid']
                arrival = row['arrival']
                
                # Calculate stage times
                reg_start = arrival
                reg_end = reg_start + row['wait_reg'] + 3  # avg service time
                
                sample_start = reg_end
                sample_end = sample_start + row['wait_sample'] + 5
                
                test_start = sample_end
                test_end = test_start + row['wait_machine'] + (12 if row['test_type'] == 'fast' else 30)
                
                verify_start = test_end
                verify_end = row['finish']
                
                gantt_data.append(dict(Task=f"Pasien {pid}", Start=reg_start, Finish=reg_end, Stage="Registrasi"))
                gantt_data.append(dict(Task=f"Pasien {pid}", Start=sample_start, Finish=sample_end, Stage="Pengambilan Sampel"))
                gantt_data.append(dict(Task=f"Pasien {pid}", Start=test_start, Finish=test_end, Stage="Tes Laboratorium"))
                if params["use_verify"]:
                    gantt_data.append(dict(Task=f"Pasien {pid}", Start=verify_start, Finish=verify_end, Stage="Verifikasi"))
            
            df_gantt_chart = pd.DataFrame(gantt_data)
            
            fig_gantt = px.timeline(
                df_gantt_chart,
                x_start="Start",
                x_end="Finish",
                y="Task",
                color="Stage",
                color_discrete_map={
                    "Registrasi": "#1abc9c",
                    "Pengambilan Sampel": "#3498db",
                    "Tes Laboratorium": "#9b59b6",
                    "Verifikasi": "#e67e22"
                }
            )
            fig_gantt.update_layout(
                xaxis_title="Waktu (menit)",
                yaxis_title="Pasien",
                template="plotly_white",
                height=500,
                xaxis_type='linear'
            )
            fig_gantt.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_gantt, use_container_width=True)
        
        with tab_a4:
            # Heatmap of queue density
            st.markdown("#### 🔥 Heatmap Kepadatan Antrian per Periode")
            
            # Create time bins (every 30 minutes)
            df_heatmap = df_post.copy()
            df_heatmap['time_bin'] = (df_heatmap['arrival'] // 30).astype(int) * 30
            
            # Aggregate by time bin
            heatmap_data = []
            for time_bin in sorted(df_heatmap['time_bin'].unique()):
                bin_data = df_heatmap[df_heatmap['time_bin'] == time_bin]
                heatmap_data.append({
                    'Periode': f"{int(time_bin)}-{int(time_bin+30)} menit",
                    'Registrasi': bin_data['wait_reg'].mean(),
                    'Sampel': bin_data['wait_sample'].mean(),
                    'Mesin': bin_data['wait_machine'].mean(),
                    'Verifikasi': bin_data['wait_verify'].mean()
                })
            
            df_heat = pd.DataFrame(heatmap_data)
            
            # Create heatmap
            fig_heat = go.Figure(data=go.Heatmap(
                z=[df_heat['Registrasi'], df_heat['Sampel'], df_heat['Mesin'], df_heat['Verifikasi']],
                x=df_heat['Periode'],
                y=['Registrasi', 'Sampel', 'Mesin', 'Verifikasi'],
                colorscale='YlOrRd',
                hovertemplate='Periode: %{x}<br>Tahap: %{y}<br>Waktu Tunggu: %{z:.2f} menit<extra></extra>'
            ))
            fig_heat.update_layout(
                title="Rata-rata Waktu Tunggu per Periode (30 menit)",
                xaxis_title="Periode Waktu",
                yaxis_title="Tahap Layanan",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        
        # =====================
        # SAVE SCENARIO
        # =====================
        st.markdown('<div class="section-header">💾 Simpan & Bandingkan Skenario</div>', unsafe_allow_html=True)
        
        col_save1, col_save2 = st.columns([1, 3])
        
        with col_save1:
            if st.button("💾 Simpan Skenario Ini", use_container_width=True):
                scenario_data = {
                    "name": scenario_name,
                    "params": params.copy(),
                    "summary": summary.copy(),
                    "seed": seed
                }
                st.session_state.scenarios.append(scenario_data)
                st.success(f"✅ Skenario '{scenario_name}' tersimpan!")
        
        with col_save2:
            if st.button("🗑️ Hapus Semua Skenario", use_container_width=True):
                st.session_state.scenarios = []
                st.info("Semua skenario telah dihapus.")
        
        # Display saved scenarios comparison
        if len(st.session_state.scenarios) > 0:
            st.markdown("#### 📊 Perbandingan Skenario Tersimpan")
            
            comparison_data = []
            for sc in st.session_state.scenarios:
                comparison_data.append({
                    "Skenario": sc["name"],
                    "Pasien": sc["summary"]["Total pasien (termasuk warm-up)"],
                    "Mean Waktu Sistem": f"{sc['summary']['Mean total time in system']:.2f}",
                    "P95 Waktu Sistem": f"{sc['summary']['P95 total time in system']:.2f}",
                    "Reg": sc["params"]["c_reg"],
                    "Phleb": sc["params"]["c_phleb"],
                    "Fast": sc["params"]["c_fast"],
                    "Slow": sc["params"]["c_slow"]
                })
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            # Comparison chart
            if len(st.session_state.scenarios) > 1:
                fig_compare = go.Figure()
                
                for sc in st.session_state.scenarios:
                    fig_compare.add_trace(go.Bar(
                        name=sc["name"],
                        x=["Mean Waktu Sistem", "P95 Waktu Sistem"],
                        y=[sc["summary"]["Mean total time in system"], sc["summary"]["P95 total time in system"]],
                        hovertemplate=f"<b>{sc['name']}</b><br>%{{x}}: %{{y:.2f}} menit<extra></extra>"
                    ))
                
                fig_compare.update_layout(
                    title="Perbandingan KPI antar Skenario",
                    barmode='group',
                    template="plotly_white",
                    yaxis_title="Waktu (menit)"
                )
                st.plotly_chart(fig_compare, use_container_width=True)

with colB:
    st.markdown("""
    <div class="info-box">
        <h4>📌 Panduan Penggunaan</h4>
        <ul>
            <li><strong>Parameter Kedatangan:</strong> Atur rata-rata waktu antar kedatangan pasien dan proporsi jenis pasien</li>
            <li><strong>Kapasitas Resource:</strong> Tentukan jumlah petugas dan mesin yang tersedia</li>
            <li><strong>Simulasi:</strong> Atur durasi dan periode warm-up simulasi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>💡 Tips Analisis</h4>
        <ul>
            <li>Jika <strong>waktu tunggu tinggi</strong>, coba tambah kapasitas resource terkait</li>
            <li>Gunakan <strong>P95</strong> untuk melihat waktu tunggu kasus terburuk</li>
            <li>Bandingkan beberapa skenario dengan mengubah parameter</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🔬 Tentang Simulasi</h4>
        <ul>
            <li>Model berbasis <strong>SimPy</strong> (Discrete Event Simulation)</li>
            <li>Waktu layanan menggunakan <strong>distribusi triangular</strong></li>
            <li>Kedatangan pasien mengikuti <strong>proses Poisson</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if verbose:
        st.info("📝 **Log detail aktif** - Cek terminal untuk melihat output verbose")

# Footer
st.markdown("""
<div class="footer">
    🏥 Simulasi Antrean Laboratorium Klinik | Praktikum Pemodelan & Simulasi | 2026
</div>
""", unsafe_allow_html=True)
