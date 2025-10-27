import streamlit as st
import pandas as pd
import psycopg2
import json
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
import sys # Digunakan untuk menangani pengecekan error database lebih baik

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
    if not role or not level or not purpose:
        return "Error: Nama Peran, Level, atau Tujuan Peran tidak boleh kosong."

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

        db_config = st.secrets["postgres"]
        
        
        conn = psycopg2.connect(
            host=db_config["host"],
            database=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
            port=db_config["port"],
            options="-c client_encoding=utf8"
        )
        return conn
    except psycopg2.OperationalError as e:
        # Mengidentifikasi error jaringan/koneksi spesifik
        st.error(f"Error koneksi database: Periksa Firewall atau Koneksi Jaringan. Detail: {e}")
        st.error("Pastikan Anda menggunakan Port dan Host yang benar (misalnya, Port 6543 untuk Supabase Pooler).")
        return None
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
        
        if 'fullname' in df.columns and 'employee_id' in df.columns:
            df['label'] = df['fullname'].astype(str) + " (" + df['employee_id'].astype(str) + ")"
            return df[['employee_id', 'label']]
        else:
            st.error("Tabel 'employees' tidak memiliki kolom 'fullname' atau 'employee_id'.")
            return pd.DataFrame({'employee_id': [], 'label': []})
    except Exception as e:
        st.error(f"Error fetching employee list: {e}")
        return pd.DataFrame({'employee_id': [], 'label': []})

@st.cache_data(show_spinner="Menjalankan Pencocokan...")
def run_talent_match_query(conn, bench_ids, weights):
    """Menjalankan fungsi SQL fn_talent_management.""" 
    if not conn:
        st.error("Koneksi database gagal.")
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
        
        if hasattr(e, 'diag'):
             st.error(f"PostgreSQL Error Details: {e.diag.message_primary}")
             if e.diag.context:
                 st.caption(f"Context: {e.diag.context}")
        else:
             st.error(f"Detail Error Python: {e}")

        st.error(f"Query parameters: bench_ids={bench_ids}, weights={weights_json}")
        return pd.DataFrame()

# --- Sidebar Input Form ---
st.sidebar.header("🔍 Konfigurasi Pencocokan")

# --- Inisialisasi & Ambil Data Awal ---
conn = init_connection()
df_employees = get_employee_list(conn)

# Inisialisasi session state untuk AI Profile
if 'generated_profile' not in st.session_state:
    st.session_state['generated_profile'] = None

# 1. Input Metadata Lowongan
job_vacancy_id = st.sidebar.text_input("Job Vacancy ID")
role_name = st.sidebar.text_input("Role Name") 
job_level = st.sidebar.selectbox("Job Level / Grade", ["I", "II", "III", "IV", "V", "VI"], index=2) 
role_purpose = st.sidebar.text_area("Role Purpose (1-2 sentences)")

# 2. Input Benchmark Talenta
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Pilih Benchmark Talenta (Rating 5)")

# Cek agar tidak error jika df_employees kosong
default_selection = []
if not df_employees.empty:
    default_selection = df_employees['label'].head(3).tolist()

selected_labels = st.sidebar.multiselect(
    "Pilih Karyawan Benchmark:",
    options=df_employees['label'].tolist(),
    default=default_selection
)

# Ekstrak employee_id dari label yang dipilih (Logika ini sudah benar)
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
    
    # Toleransi floating point diperluas
    if abs(current_total - 1.0) <= 0.02: 
        is_weights_valid = True
    else:
        st.sidebar.warning("Total bobot idealnya adalah 1.0 (toleransi 0.02).")

else: 
    weights_json_str = st.sidebar.text_area("Masukkan JSON Bobot:", json.dumps(default_weights, indent=2), height=150, key="json_weight_input")
    try:
        weights_config = json.loads(weights_json_str)
        # Pengecekan semua key wajib ada dan nilainya numerik
        if all(key in weights_config and isinstance(weights_config[key], (int, float)) for key in default_weights.keys()):
            current_total = sum(weights_config.values())
            st.sidebar.caption(f"Total Bobot Saat Ini: {current_total:.2f}")
            
            if abs(current_total - 1.0) <= 0.02:
                is_weights_valid = True
            else:
                 st.sidebar.warning("Total bobot idealnya adalah 1.0 (toleransi 0.02).")
        else:
            st.sidebar.error("JSON bobot tidak valid. Pastikan semua key ada dan nilainya numerik.")
            weights_config = {} 
            is_weights_valid = False

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
         st.sidebar.error("Konfigurasi bobot tidak valid. Total harus mendekati 1.0.")
    else:
        with st.spinner("Mencari kandidat terbaik..."):
            # Memanggil fungsi run_talent_match_query (yang memanggil fn_talent_management)
            results_df = run_talent_match_query(conn, selected_talent_ids, weights_config) 

            # Join dengan nama karyawan (Logika Join sudah efisien dan dipertahankan)
            if not results_df.empty:
                if not df_employees.empty and 'fullname' not in results_df.columns:
                    # Perbaikan: menggunakan 'employee_id' untuk join
                    employee_names = df_employees.set_index('employee_id')['label'].str.split(' \(').str[0] 
                    results_df = results_df.join(employee_names.rename('fullname'), on='employee_id')
                    
                    # Reorder Columns
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
    
    # Perbaikan: Tambahkan pengecekan jika kolom TGV ada
    if not tgv_scores_unique.empty and all(c in tgv_scores_unique.columns for c in ['tgv_cognitive_match_rate', 'tgv_competency_match_rate', 'tgv_behavioral_match_rate', 'tgv_contextual_match_rate']):
        tgv_scores_unique = tgv_scores_unique.set_index('employee_id')
        tgv_scores_unique.columns = ['TGV Cognitive', 'TGV Competency', 'TGV Behavioral', 'TGV Contextual'] # Rename TGV columns
        
        # Gabungkan ke tabel ringkasan
        summary_df = results_df.drop_duplicates(subset=['employee_id'])
        cols_to_select = [col for col in summary_cols if col in summary_df.columns]
        summary_df = summary_df[cols_to_select]
        summary_df = summary_df.join(tgv_scores_unique, on='employee_id')
        
        # Format & Tampilkan Top 10 Unik
        st.dataframe(summary_df.head(10).sort_values(by='final_match_rate', ascending=False).style.format({ # Sorting ditambahkan
            'final_match_rate': '{:.2f}%',
            'TGV Cognitive': '{:.2f}', 
            'TGV Competency': '{:.2f}', 
            'TGV Behavioral': '{:.2f}', 
            'TGV Contextual': '{:.2f}'
            }), use_container_width=True)
    else:
        st.warning("Kolom skor TGV tidak ditemukan dalam hasil query atau nama kolom tidak sesuai.")


    st.markdown("---")

    # 2. Visualisasi Radar Chart & Breakdown Detail TV
    st.subheader("📊 Perbandingan Kandidat vs Benchmark & Detail Skor TV")
    
    unique_candidates = results_df['employee_id'].unique()
    candidate_labels = {}
    
    # Perbaikan: Membuat dictionary label yang lebih robust
    if not df_employees.empty:
        candidate_labels = df_employees.set_index('employee_id')['label'].to_dict()
    
    # Menentukan default kandidat (yang skornya paling tinggi)
    top_candidate_id = summary_df.iloc[0]['employee_id'] if not summary_df.empty else unique_candidates[0]
    default_candidate_label = candidate_labels.get(top_candidate_id, top_candidate_id)

    selected_candidate_label = st.selectbox(
        "Pilih Kandidat untuk Detail:",
        options=[candidate_labels.get(eid, eid) for eid in unique_candidates],
        index=0 
    )
    
    # Mendapatkan ID dari label yang dipilih
    selected_candidate_id = next((eid for eid, label in candidate_labels.items() if label == selected_candidate_label), 
                                 selected_candidate_label.split(' (')[-1].strip(')'))

    if selected_candidate_id:
        candidate_data = results_df[results_df['employee_id'].astype(str) == str(selected_candidate_id)] 
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown(f"#### Radar Chart: {selected_candidate_label}")
            
            # Data untuk Radar Chart (Skor TGV)
            tgv_data_radar = candidate_data.drop_duplicates(subset=['tgv_name'])
            categories = ['Cognitive', 'Competency', 'Behavioral', 'Contextual'] 
            
            candidate_scores_radar = []
            for cat in categories:
                # Menggunakan nama kolom yang sudah di-rename dari fn_talent_management
                score_col_name = f'tgv_{cat.lower()}_match_rate' 
                score_row = candidate_data[candidate_data['tgv_name'].str.lower() == cat.lower()]
                
                # Mengambil nilai dari kolom TGV match rate (jika ada)
                score = score_row[score_col_name].iloc[0] if not score_row.empty and score_col_name in score_row.columns else 0
                candidate_scores_radar.append(score)


            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=candidate_scores_radar, theta=categories, fill='toself', name=f'Kandidat', line=dict(color='blue')))
            # Skala 100 untuk benchmark ideal
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
            
            # Sorting berdasarkan TGV Name untuk keterbacaan
            st.dataframe(candidate_data[cols_to_display].sort_values(by='tgv_name').style.format({
                 'baseline_score': '{:.2f}',
                 'user_score': '{:.2f}',
                 'tv_match_rate': '{:.2f}%'
            }), height=410, use_container_width=True)
            
        st.markdown("---")

        # 4. Ringkasan Insight
        st.subheader("💡 Ringkasan Insight")
        top_candidate = summary_df.iloc[0]
        top_candidate_name_insight = top_candidate['fullname']
        top_score_insight = top_candidate['final_match_rate']
        
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
    # Pengecekan input sebelum memanggil AI
    if not role_name or not role_purpose or not job_level:
        st.sidebar.error("Nama peran, Level, dan Tujuan Peran tidak boleh kosong.")
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
