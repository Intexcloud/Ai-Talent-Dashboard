import streamlit as st
import pandas as pd
import psycopg2
import json
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(layout="wide", page_title="AI Talent Match Intelligence")
st.title("🚀 AI Talent Match Intelligence Dashboard")
st.markdown("Menemukan kandidat internal terbaik berdasarkan profil benchmark.")

# --- Fungsi Ai Generated Profile ---
@st.cache_data(show_spinner=False) 
def generate_job_profile_ai(role, level, purpose):
    """
    Menghasilkan draf profil pekerjaan menggunakan AI.
    """
    try:
        # 1. Ambil API Key dari Streamlit Secrets
        api_key = st.secrets.get("openrouter_api_key")
        if not api_key:
            st.error("OpenRouter API Key (openrouter_api_key) tidak ditemukan di Streamlit Secrets.")
            st.error("Silakan tambahkan `openrouter_api_key = '...'` ke file `.streamlit/secrets.toml` Anda.")
            return None

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=st.secrets["openrouter_api_key"],
        )

        # 2. Buat Prompt
        system_role_instruction = "Anda adalah asisten HR yang ahli dalam membuat draf profil pekerjaan (job profile) internal."
        
        user_prompt_content = f"""
        Tugas Anda adalah membuat draf profil pekerjaan yang profesional dan menarik berdasarkan informasi berikut:

        - Nama Peran: {role}
        - Level Pekerjaan / Grade: {level}
        - Tujuan Peran (Role Purpose): {purpose}

        Harap buat draf yang mencakup bagian-bagian berikut:
        1.  **Ringkasan Peran** (Perluas 'Tujuan Peran' yang diberikan menjadi paragraf singkat)
        2.  **Tanggung Jawab Utama** (Buat daftar 5-7 poin penting dalam format bullet points)
        3.  **Kualifikasi Minimum** (Perkirakan berdasarkan level dan peran, misal: pendidikan, pengalaman)
        4.  **Keterampilan yang Diutamakan** (Sebutkan 3-5 keterampilan teknis atau soft skills yang relevan)

        Gunakan format Markdown yang jelas dan profesional.
        """ # Gunakan ini sebagai content dari role 'user'

        # 3. Panggil Model
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free", 
            messages=[
                {"role": "system", "content": system_role_instruction},
            ],
            max_tokens=1000,
            temperature=0.7,
        )

        # 4. Kembalikan Teks
        return completion.choices[0].message.content

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memanggil AI: {e}")
        st.error("Pastikan API Key Anda valid dan memiliki kuota.")
        return None
        
# --- Fungsi Bantu ---

# Cache koneksi database
@st.cache_resource
def init_connection():
    """Menginisialisasi koneksi ke database PostgreSQL."""
    try:
        # GANTI DENGAN DETAIL KONEKSI ANDA 
        conn = psycopg2.connect(
            host=st.secrets["postgres"]["host"],
            database=st.secrets["postgres"]["dbname"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
            port=st.secrets["postgres"]["port"]
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        st.error("Pastikan Anda telah membuat file .streamlit/secrets.toml dengan benar.")
        return None

# Cache data karyawan
@st.cache_data
def get_employee_list(_conn):
    """Mengambil daftar employee_id dan fullname dari database."""
    if _conn is None:
        return pd.DataFrame({'employee_id': [], 'label': []}) 
    try:
        query = "SELECT employee_id, fullname FROM employees ORDER BY fullname;"
        df = pd.read_sql(query, _conn)
        df['label'] = df['fullname'] + " (" + df['employee_id'] + ")"
        return df[['employee_id', 'label']]
    except Exception as e:
        st.error(f"Error fetching employee list: {e}")
        return pd.DataFrame({'employee_id': [], 'label': []})

def run_talent_match_query(conn, bench_ids, weights):
    """Menjalankan fungsi SQL fn_talent_management.""" 
    if not conn or not bench_ids:
        st.warning("Koneksi database gagal atau tidak ada benchmark ID yang dipilih.")
        return pd.DataFrame() 

    try:
        weights_json = json.dumps(weights)
        query = "SELECT * FROM fn_talent_management(%s::TEXT[], %s::JSONB);" 
        df_results = pd.read_sql(query, conn, params=(bench_ids, weights_json))
        df_results.columns = [col.replace('o_', '') for col in df_results.columns]
        
        return df_results
    except Exception as e:
        st.error(f"Error running talent match function (fn_talent_management): {e}") 
        st.error(f"Query parameters: bench_ids={bench_ids}, weights={weights_json}")
        if hasattr(e, 'pgcode'): st.error(f"PostgreSQL Error Code: {e.pgcode}")
        if hasattr(e, 'pgerror'): st.error(f"PostgreSQL Error Message: {e.pgerror}")
        return pd.DataFrame()

# --- Sidebar Input Form ---
st.sidebar.header("🔍 Konfigurasi Pencocokan")

conn = init_connection()
df_employees = get_employee_list(conn)

# 1. Input Metadata Lowongan
job_vacancy_id = st.sidebar.text_input("Job Vacancy ID", "VAC-2025-DA-01")
role_name = st.sidebar.text_input("Role Name", "Data Analyst")
job_level = st.sidebar.selectbox("Job Level / Grade", ["I", "II", "III", "IV", "V", "VI"], index=0) 
role_purpose = st.sidebar.text_area("Role Purpose (1-2 sentences)", "Analyze complex data sets to identify trends, develop insights, and support data-driven decision making.")

# 2. Input Benchmark Talenta
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Pilih Benchmark Talenta (Rating 5)")
# Menggunakan label informatif untuk multiselect
selected_labels = st.sidebar.multiselect(
    "Pilih Karyawan Benchmark:",
    options=df_employees['label'].tolist(),
    # Default selection (contoh, ambil 3 ID pertama dari daftar)
    default=df_employees['label'].head(3).tolist() if not df_employees.empty else [] 
)
# Ekstrak employee_id dari label yang dipilih
selected_talent_ids = df_employees[df_employees['label'].isin(selected_labels)]['employee_id'].tolist()

# 3. Input Bobot TGV
st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Konfigurasi Bobot TGV")
default_weights = {"cognitive": 0.35, "competency": 0.30, "behavioral": 0.25, "contextual": 0.10}
weight_input_method = st.sidebar.radio("Metode Input Bobot:", ("Sliders", "JSON"), index=0)

weights_config = {}
if weight_input_method == "Sliders":
    st.sidebar.write("Sesuaikan bobot (total harus 1.0):")
    weights_config["cognitive"] = st.sidebar.slider("Cognitive (%)", 0.0, 1.0, default_weights["cognitive"], 0.05)
    weights_config["competency"] = st.sidebar.slider("Competency (%)", 0.0, 1.0, default_weights["competency"], 0.05)
    weights_config["behavioral"] = st.sidebar.slider("Behavioral (%)", 0.0, 1.0, default_weights["behavioral"], 0.05)
    weights_config["contextual"] = st.sidebar.slider("Contextual (%)", 0.0, 1.0, default_weights["contextual"], 0.05)
    current_total = sum(weights_config.values())
    st.sidebar.caption(f"Total Bobot Saat Ini: {current_total:.2f}")
    if abs(current_total - 1.0) > 0.01:
         st.sidebar.warning("Total bobot idealnya adalah 1.0")

else: # Input JSON
    weights_json_str = st.sidebar.text_area("Masukkan JSON Bobot:", json.dumps(default_weights, indent=2), height=150)
    try:
        weights_config = json.loads(weights_json_str)
        if not all(key in weights_config for key in default_weights.keys()):
            st.sidebar.error("JSON bobot tidak valid. Pastikan ada key: cognitive, competency, behavioral, contextual.")
            weights_config = {} 
    except json.JSONDecodeError:
        st.sidebar.error("Format JSON tidak valid.")
        weights_config = {} 

# Tombol untuk menjalankan pencocokan
run_button = st.sidebar.button("🚀 Jalankan Pencocokan Talenta")

# --- Area Tampilan Hasil ---
results_df = pd.DataFrame() # Inisialisasi dataframe kosong

# Jalankan query jika tombol ditekan dan koneksi/input valid
if run_button and conn:
    if not selected_talent_ids:
        st.sidebar.error("Harap pilih minimal satu karyawan benchmark.")
    elif not weights_config:
         st.sidebar.error("Konfigurasi bobot tidak valid.")
    else:
        with st.spinner("Mencari kandidat terbaik..."):
            # Memanggil fungsi run_talent_match_query (yang memanggil fn_talent_management)
            results_df = run_talent_match_query(conn, selected_talent_ids, weights_config) 

            # Join dengan nama karyawan (jika belum ada di fungsi SQL)
            if not results_df.empty and 'fullname' not in results_df.columns:
                 # Pastikan df_employees tidak kosong sebelum join
                 if not df_employees.empty:
                    employee_names = df_employees.set_index('employee_id')['label'].str.split(' \(').str[0] # Ambil nama saja
                    results_df = results_df.join(employee_names.rename('fullname'), on='employee_id')
                    if 'fullname' in results_df.columns:
                        cols = ['employee_id', 'fullname'] + [col for col in results_df.columns if col not in ['employee_id', 'fullname']]
                        if all(c in results_df.columns for c in cols):
                            results_df = results_df[cols]
                        else: 
                            st.warning("Beberapa kolom tidak ditemukan untuk reordering.")
                 else:
                    st.warning("Data karyawan tidak dapat dimuat untuk menampilkan nama lengkap.")


# Tampilkan hasil jika dataframe tidak kosong
if not results_df.empty:
    st.header(f"🏆 Top 10 Kandidat untuk {role_name}")

    # 1. Tabel Hasil Peringkat Ringkas
    st.subheader("Tabel Peringkat Kandidat (Ringkasan)")
    summary_cols = ['employee_id', 'fullname', 'directorate', 'role', 'grade', 'final_match_rate']
    tgv_cols_for_summary = ['tgv_cognitive_match_rate', 'tgv_competency_match_rate', 'tgv_behavioral_match_rate', 'tgv_contextual_match_rate']
    
    # Ambil skor TGV unik
    tgv_scores_unique = results_df.drop_duplicates(subset=['employee_id'])[['employee_id'] + tgv_cols_for_summary]
    tgv_scores_unique = tgv_scores_unique.set_index('employee_id')
    tgv_scores_unique.columns = ['TGV Cognitive', 'TGV Competency', 'TGV Behavioral', 'TGV Contextual'] # Rename TGV columns

    # Gabungkan ke tabel ringkasan
    summary_df = results_df.drop_duplicates(subset=['employee_id'])
    # Hanya pilih kolom yang ada di summary_df sebelum join
    cols_to_select = [col for col in summary_cols if col in summary_df.columns]
    summary_df = summary_df[cols_to_select]
    summary_df = summary_df.join(tgv_scores_unique, on='employee_id')
    
    # Format & Tampilkan Top 10 Unik
    st.dataframe(summary_df.head(10).style.format({
        'final_match_rate': '{:.2f}%',
        'TGV Cognitive': '{:.2f}', 
        'TGV Competency': '{:.2f}', 
        'TGV Behavioral': '{:.2f}', 
        'TGV Contextual': '{:.2f}'
        }))

    st.markdown("---")

    # 2. Visualisasi Radar Chart & Breakdown Detail TV
    st.subheader("📊 Perbandingan Kandidat vs Benchmark & Detail Skor TV")
    
    unique_candidates = results_df['employee_id'].unique()
    # Buat label nama + ID untuk selectbox
    candidate_labels = {}
    if not df_employees.empty:
         candidate_labels = df_employees[df_employees['employee_id'].isin(unique_candidates)]\
            .set_index('employee_id')['label'].to_dict()

    selected_candidate_label = st.selectbox(
        "Pilih Kandidat untuk Detail:",
        options=[candidate_labels.get(eid, eid) for eid in unique_candidates] 
    )
    selected_candidate_id = next((eid for eid, label in candidate_labels.items() if label == selected_candidate_label), unique_candidates[0] if len(unique_candidates)>0 else None) # Default ke ID pertama jika label tidak ditemukan

    if selected_candidate_id:
        candidate_data = results_df[results_df['employee_id'] == selected_candidate_id]
        
        col1, col2 = st.columns([2, 3]) # Lebarkan kolom detail
        
        with col1:
            st.markdown(f"#### Radar Chart: {selected_candidate_label}")
            tgv_data_radar = candidate_data.drop_duplicates(subset=['tgv_name'])
            categories = ['Cognitive', 'Competency', 'Behavioral', 'Contextual'] 
            
            candidate_scores_radar = []
            for cat in categories:
                 score = tgv_data_radar[tgv_data_radar['tgv_name'] == cat]['tgv_match_rate'].iloc[0] if not tgv_data_radar[tgv_data_radar['tgv_name'] == cat].empty else 0
                 candidate_scores_radar.append(score)

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=candidate_scores_radar, theta=categories, fill='toself', name=f'Kandidat', line=dict(color='blue')))
            fig_radar.add_trace(go.Scatterpolar(r=[100] * len(categories), theta=categories, fill=None, name='Benchmark (100%)', line=dict(color='grey', dash='dash')))
            
            # Cari nilai max untuk range, pastikan minimal 100
            max_val = max([100] + candidate_scores_radar) * 1.1 
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max_val])),
                showlegend=True, height=400, margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(yanchor="bottom", y= -0.2, xanchor="center", x=0.5) 
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("Membandingkan skor TGV kandidat (area biru) dengan skor benchmark ideal (garis putus-putus 100%).")

        with col2:
            st.markdown(f"#### Detail Skor TV: {selected_candidate_label}")
            detail_cols = ['tgv_name', 'tv_name', 'baseline_score', 'user_score', 'tv_match_rate']
            st.dataframe(candidate_data[detail_cols].style.format({
                 'baseline_score': '{:.2f}',
                 'user_score': '{:.2f}',
                 'tv_match_rate': '{:.2f}%'
            }), height=410) # Sesuaikan tinggi tabel
            
        st.markdown("---")

        # 4. Ringkasan Insight (Placeholder)
        st.subheader("💡 Ringkasan Insight")
        top_candidate_name_insight = candidate_labels.get(results_df.iloc[0]['employee_id'], results_df.iloc[0]['employee_id'])
        top_score_insight = results_df.iloc[0]['final_match_rate']
        st.markdown(f"Kandidat teratas, **{top_candidate_name_insight}**, mencapai skor **{top_score_insight:.2f}%**. "
                    f"Ini menunjukkan keselarasan yang kuat dengan profil benchmark. Analisis lebih lanjut pada Radar Chart dan Detail Skor TV dapat mengungkapkan area kekuatan dan pengembangan spesifik.")
     

    # 3. Distribusi Skor 
    st.subheader("Distribusi Skor Kecocokan (Top 10)")
    fig_hist = px.histogram(summary_df.head(10), x="final_match_rate", nbins=5, title="Distribusi Final Match Rate (Top 10)")
    st.plotly_chart(fig_hist)


# Kondisi jika tombol ditekan tapi tidak ada hasil
elif run_button:
     st.info("Tidak ada hasil yang ditemukan. Periksa kembali ID benchmark yang dipilih atau konfigurasi bobot. Pastikan juga benchmark memiliki data yang cukup.")

# --- Placeholder untuk AI Job Profile Generator ---
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Job Profile Generator")
generate_profile = st.sidebar.button("Buat Profil Pekerjaan (AI)")

if 'generated_profile' in st.session_state:
    st.markdown("---")
    st.header("🤖 Draf Profil Pekerjaan (Dibuat oleh AI)")
    st.markdown(st.session_state['generated_profile'])
    
    # Tambahkan tombol untuk menghapus/membersihkan hasil
    if st.button("Bersihkan Draf Profil"):
        del st.session_state['generated_profile']
        st.rerun() 

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("Talent Match App v1.0")
