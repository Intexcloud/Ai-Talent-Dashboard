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
        api_key_or = st.secrets.get("openrouter", {}).get("api_key")
        if not api_key_or:
            return "Error: OpenRouter API Key (st.secrets['openrouter']['api_key']) tidak ditemukan di Streamlit Secrets."

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key_or, 
        )

        # 2. Buat Prompt 
        system_prompt = "Anda adalah asisten HR yang ahli dalam membuat draf profil pekerjaan (job profile) internal. Tugas Anda adalah membuat draf profil pekerjaan yang profesional dan menarik berdasarkan informasi yang diberikan pengguna."
        user_prompt_template = f"""
        Buat draf profil pekerjaan yang profesional dan menarik berdasarkan informasi berikut:

        - Nama Peran: {role}
        - Level Pekerjaan / Grade: {level}
        - Tujuan Peran (Role Purpose): {purpose}

        Harap buat draf yang mencakup bagian-bagian berikut:
        1.  **Ringkasan Peran** (Perluas 'Tujuan Peran' yang diberikan menjadi paragraf singkat)
        2.  **Tanggung Jawab Utama** (Buat daftar 5-7 poin penting dalam format bullet points)
        3.  **Kualifikasi Minimum** (Perkirakan berdasarkan level dan peran, misal: pendidikan, pengalaman)
        4.  **Keterampilan yang Diutamakan** (Sebutkan 3-5 keterampilan teknis atau soft skills yang relevan)

        Gunakan format Markdown yang jelas dan profesional.
        """

        # 3. Panggil Model
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template},
            ],
            max_tokens=1000,
            temperature=0.7,
        )

        # 4. Kembalikan Teks
        return completion.choices[0].message.content

    except Exception as e:
        # Menangkap error koneksi, API key, atau lainnya
        st.error(f"Terjadi kesalahan saat memanggil AI: {e}")
        st.error("Pastikan API Key OpenRouter Anda valid dan memiliki kuota.")
        return None

# --- Fungsi Bantu ---

# Cache koneksi database
@st.cache_resource
def init_connection():
    """Menginisialisasi koneksi ke database PostgreSQL."""
    try:
        # Pastikan key 'postgres' ada sebelum mencoba mengakses sub-key
        if "postgres" not in st.secrets:
            st.error("Section [postgres] tidak ditemukan di Streamlit Secrets.")
            return None

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
        st.error("Pastikan Anda telah membuat file .streamlit/secrets.toml dengan benar dan detail koneksi PostgreSQL valid.")
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
        # Pastikan kolom fullname dan employee_id ada
        if 'fullname' in df.columns and 'employee_id' in df.columns:
            df['label'] = df['fullname'] + " (" + df['employee_id'] + ")"
            return df[['employee_id', 'label']]
        else:
            st.error("Tabel 'employees' tidak memiliki kolom 'fullname' atau 'employee_id'.")
            return pd.DataFrame({'employee_id': [], 'label': []})
    except Exception as e:
        st.error(f"Error fetching employee list: {e}")
        return pd.DataFrame({'employee_id': [], 'label': []})

def run_talent_match_query(conn, bench_ids, weights):
    """Menjalankan fungsi SQL fn_talent_management.""" 
    if not conn:
        st.warning("Koneksi database gagal.")
        return pd.DataFrame() 
    if not bench_ids:
        st.warning("Tidak ada benchmark ID yang dipilih.")
        return pd.DataFrame() 
    if not weights:
         st.warning("Konfigurasi bobot tidak valid.")
         return pd.DataFrame()

    weights_json = json.dumps(weights) 
    
    try:
        query = "SELECT * FROM fn_talent_management(%s::TEXT[], %s::JSONB);" 
        df_results = pd.read_sql(query, conn, params=(bench_ids, weights_json))
        df_results.columns = [col.replace('o_', '') for col in df_results.columns]
        
        return df_results
    except Exception as e:
        st.error(f"Error running talent match function (fn_talent_management): {e}") 
        pg_error_message = ""
        if hasattr(e, 'pgcode'): pg_error_message += f"PostgreSQL Error Code: {e.pgcode}. "
        if hasattr(e, 'pgerror'): pg_error_message += f"PostgreSQL Error Message: {e.pgerror}"
        st.error(f"Detail Database Error: {pg_error_message}")
        st.error(f"Query parameters: bench_ids={bench_ids}, weights={weights_json}")
        return pd.DataFrame()

# --- Sidebar Input Form ---
st.sidebar.header("🔍 Konfigurasi Pencocokan")

conn = init_connection()
df_employees = get_employee_list(conn)

# Inisialisasi session state untuk AI Profile
if 'generated_profile' not in st.session_state:
    st.session_state['generated_profile'] = None

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
is_weights_valid = False
if weight_input_method == "Sliders":
    st.sidebar.write("Sesuaikan bobot (total harus 1.0):")
    weights_config["cognitive"] = st.sidebar.slider("Cognitive (%)", 0.0, 1.0, default_weights["cognitive"], 0.05, key="slider_cog")
    weights_config["competency"] = st.sidebar.slider("Competency (%)", 0.0, 1.0, default_weights["competency"], 0.05, key="slider_comp")
    weights_config["behavioral"] = st.sidebar.slider("Behavioral (%)", 0.0, 1.0, default_weights["behavioral"], 0.05, key="slider_beh")
    weights_config["contextual"] = st.sidebar.slider("Contextual (%)", 0.0, 1.0, default_weights["contextual"], 0.05, key="slider_cont")
    current_total = sum(weights_config.values())
    st.sidebar.caption(f"Total Bobot Saat Ini: {current_total:.2f}")
    if abs(current_total - 1.0) > 0.01:
         st.sidebar.warning("Total bobot idealnya adalah 1.0")
    else:
        is_weights_valid = True

else: 
    weights_json_str = st.sidebar.text_area("Masukkan JSON Bobot:", json.dumps(default_weights, indent=2), height=150, key="json_weight_input")
    try:
        weights_config = json.loads(weights_json_str)
        if not all(key in weights_config for key in default_weights.keys()):
            st.sidebar.error("JSON bobot tidak valid. Pastikan ada key: cognitive, competency, behavioral, contextual.")
            weights_config = {} 
        else:
             # Cek total bobot
            current_total = sum(weights_config.values())
            st.sidebar.caption(f"Total Bobot Saat Ini: {current_total:.2f}")
            if abs(current_total - 1.0) > 0.01:
                st.sidebar.warning("Total bobot idealnya adalah 1.0")
            is_weights_valid = True

    except json.JSONDecodeError:
        st.sidebar.error("Format JSON tidak valid.")
        weights_config = {} 
        is_weights_valid = False

# Tombol untuk menjalankan pencocokan
run_button = st.sidebar.button("🚀 Jalankan Pencocokan Talenta")

# --- Area Tampilan Hasil ---
results_df = pd.DataFrame()

# Jalankan query jika tombol ditekan dan koneksi/input valid
if run_button:
    if not conn:
        st.error("Koneksi database gagal. Harap periksa `secrets.toml` Anda.")
    elif not selected_talent_ids:
        st.sidebar.error("Harap pilih minimal satu karyawan benchmark.")
    elif not weights_config or not is_weights_valid:
         st.sidebar.error("Konfigurasi bobot tidak valid. Harap periksa input Anda.")
    else:
        with st.spinner("Mencari kandidat terbaik..."):
            # Memanggil fungsi run_talent_match_query (yang memanggil fn_talent_management)
            results_df = run_talent_match_query(conn, selected_talent_ids, weights_config) 

            # Join dengan nama karyawan (jika belum ada di fungsi SQL)
            if not results_df.empty:
                 if not df_employees.empty and 'fullname' not in results_df.columns:
                    employee_names = df_employees.set_index('employee_id')['label'].str.split(' \(').str[0] 
                    results_df = results_df.join(employee_names.rename('fullname'), on='employee_id')
                    cols_to_reorder = ['employee_id', 'fullname', 'directorate', 'role', 'grade']
                    cols = [c for c in cols_to_reorder if c in results_df.columns] + [col for col in results_df.columns if col not in cols_to_reorder]
                    results_df = results_df[cols]


# Tampilkan hasil jika dataframe tidak kosong
if not results_df.empty:
    st.header(f"🏆 Top {len(results_df.drop_duplicates(subset=['employee_id']))} Kandidat untuk {role_name}")

    # 1. Tabel Hasil Peringkat Ringkas
    st.subheader("Tabel Peringkat Kandidat (Ringkasan)")
    # Kolom ringkasan + skor TGV unik per karyawan
    summary_cols = ['employee_id', 'fullname', 'directorate', 'role', 'grade', 'final_match_rate']
    tgv_cols_for_summary = ['tgv_cognitive_match_rate', 'tgv_competency_match_rate', 'tgv_behavioral_match_rate', 'tgv_contextual_match_rate']
    
    # Ambil skor TGV unik
    tgv_scores_unique = results_df.drop_duplicates(subset=['employee_id'])[['employee_id'] + [c for c in tgv_cols_for_summary if c in results_df.columns]]
    if not tgv_scores_unique.empty:
        tgv_scores_unique = tgv_scores_unique.set_index('employee_id')
        tgv_scores_unique.columns = ['TGV Cognitive', 'TGV Competency', 'TGV Behavioral', 'TGV Contextual'] # Rename TGV columns
    
        # Gabungkan ke tabel ringkasan
        summary_df = results_df.drop_duplicates(subset=['employee_id'])
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
            }), use_container_width=True)
    else:
        st.warning("Kolom skor TGV tidak ditemukan dalam hasil query.")


    st.markdown("---")

    # 2. Visualisasi Radar Chart & Breakdown Detail TV
    st.subheader("📊 Perbandingan Kandidat vs Benchmark & Detail Skor TV")
    
    unique_candidates = results_df['employee_id'].unique()
    candidate_labels = {}
    if not df_employees.empty:
         candidate_labels = df_employees[df_employees['employee_id'].isin(unique_candidates)]\
            .set_index('employee_id')['label'].to_dict()
    default_candidate_label = candidate_labels.get(results_df.iloc[0]['employee_id'], results_df.iloc[0]['employee_id'])

    selected_candidate_label = st.selectbox(
        "Pilih Kandidat untuk Detail:",
        options=[candidate_labels.get(eid, eid) for eid in unique_candidates],
        index=([candidate_labels.get(eid, eid) for eid in unique_candidates].index(default_candidate_label) if default_candidate_label in [candidate_labels.get(eid, eid) for eid in unique_candidates] else 0)
    )
    selected_candidate_id = next((eid for eid, label in candidate_labels.items() if label == selected_candidate_label), unique_candidates[0] if len(unique_candidates)>0 else None) # Default ke ID pertama jika label tidak ditemukan

    if selected_candidate_id:
        candidate_data = results_df[results_df['employee_id'] == selected_candidate_id]
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown(f"#### Radar Chart: {selected_candidate_label}")
            # Data untuk Radar Chart (Skor TGV)
            tgv_data_radar = candidate_data.drop_duplicates(subset=['tgv_name'])
            categories = ['Cognitive', 'Competency', 'Behavioral', 'Contextual'] 
            
            candidate_scores_radar = []
            for cat in categories:
                 score_row = tgv_data_radar[tgv_data_radar['tgv_name'].str.lower() == cat.lower()]
                 score = score_row['tgv_match_rate'].iloc[0] if not score_row.empty and 'tgv_match_rate' in score_row.columns else 0
                 candidate_scores_radar.append(score)


            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=candidate_scores_radar, theta=categories, fill='toself', name=f'Kandidat', line=dict(color='blue')))
            fig_radar.add_trace(go.Scatterpolar(r=[100] * len(categories), theta=categories, fill=None, name='Benchmark (100%)', line=dict(color='grey', dash='dash')))
            
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
            cols_to_display = [c for c in detail_cols if c in candidate_data.columns]
            st.dataframe(candidate_data[cols_to_display].style.format({
                 'baseline_score': '{:.2f}',
                 'user_score': '{:.2f}',
                 'tv_match_rate': '{:.2f}%'
            }), height=410, use_container_width=True)
            
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
    st.plotly_chart(fig_hist, use_container_width=True)


# Kondisi jika tombol ditekan tapi tidak ada hasil
elif run_button:
     st.info("Tidak ada hasil yang ditemukan. Periksa kembali ID benchmark yang dipilih atau konfigurasi bobot. Pastikan juga benchmark memiliki data yang cukup.")

# --- Bagian AI Job Profile Generator ---
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Job Profile Generator")
generate_profile = st.sidebar.button("Buat Profil Pekerjaan (AI)")

if generate_profile:
    # Pengecekan API Key sebelum memanggil fungsi AI
    api_key_check = st.secrets.get("openrouter", {}).get("api_key")
    if not api_key_check:
        st.sidebar.error("OpenRouter API Key (st.secrets['openrouter']['api_key']) tidak ditemukan di Streamlit Secrets.")
    elif not role_name or not role_purpose:
         st.sidebar.error("Nama peran dan Tujuan Peran tidak boleh kosong.")
    else:
        with st.spinner("Membuat draf profil pekerjaan dengan AI..."):
            profile_text = generate_job_profile_ai(role_name, job_level, role_purpose)
            if profile_text and not profile_text.startswith("Error:"):
                 st.session_state['generated_profile'] = profile_text
            elif profile_text and profile_text.startswith("Error:"):
                 st.session_state['generated_profile'] = None 
                 st.error(profile_text.replace("Error: ", ""))
            else:
                 st.session_state['generated_profile'] = None 

        st.rerun() 


if st.session_state['generated_profile']:
    st.markdown("---")
    st.header("🤖 Draf Profil Pekerjaan (Dibuat oleh AI)")
    st.markdown(st.session_state['generated_profile'])
    
    if st.button("Bersihkan Draf Profil"):
        st.session_state['generated_profile'] = None
        st.rerun() 

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("Talent Match App v1.0 (Diperbaiki)")
