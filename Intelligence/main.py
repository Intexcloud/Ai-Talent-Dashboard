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

# --- Fungsi Utility Pengecekan Secrets ---
def check_secrets_key(key_path, description):
    """Memeriksa keberadaan key di st.secrets dan menampilkan error jika hilang."""
    keys = key_path.split('.')
    current = st.secrets
    for key in keys:
        current = current.get(key)
        if current is None:
            st.error(f"Error: Kunci '{key_path}' ({description}) tidak ditemukan di Streamlit Secrets.")
            return None
    return current

# --- Fungsi Ai Generated Profile ---
@st.cache_data(show_spinner=False) 
def generate_job_profile_ai(role, level, purpose):
    """
    Menghasilkan draf profil pekerjaan menggunakan AI.
    """
    # 1. Ambil API Key dari Streamlit Secrets (Pengecekan awal)
    api_key_or = check_secrets_key("openrouter.api_key", "OpenRouter API Key")
    if not api_key_or:
        return "Error: OpenRouter API Key tidak ditemukan."

    if not role or not purpose:
         return "Error: Nama Peran dan Tujuan Peran harus diisi untuk membuat profil AI."

    try:
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
        return f"Error: Terjadi kesalahan saat memanggil AI: {e}"

# --- Fungsi Bantu Database ---

# Cache koneksi database
@st.cache_resource
def init_connection():
    """Menginisialisasi koneksi ke database PostgreSQL."""
    try:
        # Pengecekan keberadaan kunci secara kolektif
        host = check_secrets_key("postgres.host", "Database Host")
        dbname = check_secrets_key("postgres.dbname", "Database Name")
        user = check_secrets_key("postgres.user", "Database User")
        password = check_secrets_key("postgres.password", "Database Password")
        port = check_secrets_key("postgres.port", "Database Port")
        
        if any(v is None for v in [host, dbname, user, password, port]):
            return None

        conn = psycopg2.connect(
            host=host,
            database=dbname,
            user=user,
            password=password,
            port=port
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
    empty_df = pd.DataFrame({'employee_id': [], 'label': []})
    if _conn is None:
        return empty_df
    try:
        query = "SELECT employee_id, fullname FROM employees ORDER BY fullname;"
        df = pd.read_sql(query, _conn)
        
        if df.empty:
            return empty_df

        # Pastikan kolom fullname dan employee_id ada
        if 'fullname' in df.columns and 'employee_id' in df.columns:
            df['label'] = df['fullname'] + " (" + df['employee_id'] + ")"
            return df[['employee_id', 'label']]
        else:
            st.error("Tabel 'employees' tidak memiliki kolom 'fullname' atau 'employee_id'.")
            return empty_df
    except Exception as e:
        st.error(f"Error fetching employee list: {e}")
        return empty_df

def run_talent_match_query(conn, bench_ids, weights):
    """Menjalankan fungsi SQL fn_talent_management.""" 
    if not conn:
        st.warning("Koneksi database gagal.")
        return pd.DataFrame() 
    if not bench_ids:
        st.warning("Tidak ada benchmark ID yang dipilih.")
        return pd.DataFrame() 
    
    # Pengecekan weights lebih detail di fungsi ini untuk keandalan
    required_weights = {"cognitive", "competency", "behavioral", "contextual"}
    if not all(key in weights for key in required_weights):
         st.warning(f"Konfigurasi bobot tidak valid. Diperlukan kunci: {', '.join(required_weights)}.")
         return pd.DataFrame()

    weights_json = json.dumps(weights) 
    
    try:
        # Menggunakan tuple untuk parameter tunggal di read_sql
        query = "SELECT * FROM fn_talent_management(%s::TEXT[], %s::JSONB);" 
        df_results = pd.read_sql(query, conn, params=(bench_ids, weights_json))
        
        # Penamaan ulang kolom
        df_results.columns = [col.replace('o_', '') for col in df_results.columns]
        
        return df_results
    except Exception as e:
        # Penanganan error PostgreSQL yang lebih umum
        st.error(f"Error running talent match function (fn_talent_management): {e}") 
        st.error(f"Query parameters: bench_ids={bench_ids}, weights={weights_json}")
        return pd.DataFrame()

# --- Sidebar Input Form ---
st.sidebar.header("🔍 Konfigurasi Pencocokan")

# Inisialisasi koneksi dan ambil data karyawan
conn = init_connection()
df_employees = get_employee_list(conn)

# Inisialisasi session state untuk AI Profile
if 'generated_profile' not in st.session_state:
    st.session_state['generated_profile'] = None

# 1. Input Metadata Lowongan
job_vacancy_id = st.sidebar.text_input("Job Vacancy ID")
role_name = st.sidebar.text_input("Role Name", value="") 
job_level = st.sidebar.selectbox("Job Level / Grade", ["I", "II", "III", "IV", "V", "VI"], index=0) 
role_purpose = st.sidebar.text_area("Role Purpose (1-2 sentences)", value="") 

# 2. Input Benchmark Talenta
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Pilih Benchmark Talenta (Rating 5)")
selected_labels = st.sidebar.multiselect(
    "Pilih Karyawan Benchmark:",
    options=df_employees['label'].tolist(),
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
    
    # Bobot dianggap valid jika semua kunci ada
    is_weights_valid = True
    if abs(current_total - 1.0):
         st.sidebar.warning("Total bobot idealnya adalah 1.0")

else:
    weights_json_str = st.sidebar.text_area("Masukkan JSON Bobot:", json.dumps(default_weights, indent=2), height=150, key="json_weight_input")
    try:
        weights_config = json.loads(weights_json_str)
        required_weights = set(default_weights.keys())
        if not required_weights.issubset(weights_config.keys()):
            st.sidebar.error(f"JSON bobot tidak valid. Pastikan ada key: {', '.join(required_weights)}.")
            weights_config = {} 
        else:
            current_total = sum(weights_config.values())
            st.sidebar.caption(f"Total Bobot Saat Ini: {current_total:.2f}")
            if abs(current_total - 1.0) > 0.01:
                st.sidebar.warning("Total bobot idealnya adalah 1.0")
            is_weights_valid = True

    except json.JSONDecodeError:
        st.sidebar.error("Format JSON tidak valid.")
        weights_config = {} 
        is_weights_valid = False

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
            results_df = run_talent_match_query(conn, selected_talent_ids, weights_config) 

            if not results_df.empty:
                 # Pastikan fullname ditambahkan jika belum ada (terutama jika query tidak mengembalikan fullname)
                 if not df_employees.empty and 'fullname' not in results_df.columns:

                    # Perbaikan: Menggunakan apply untuk memastikan penamaan karyawan akurat
                    employee_name_map = df_employees.set_index('employee_id')['label'].apply(lambda x: x.split(' (')[0]).rename('fullname')
                    results_df = results_df.join(employee_name_map, on='employee_id')
                    
                    # Susun ulang kolom
                    cols_to_reorder = ['employee_id', 'fullname', 'directorate', 'role', 'grade']
                    cols = [c for c in cols_to_reorder if c in results_df.columns] + [col for col in results_df.columns if col not in cols_to_reorder]
                    results_df = results_df[cols]
            else:
                # Set results_df menjadi kosong secara eksplisit jika query gagal/kosong
                results_df = pd.DataFrame()


# Tampilkan hasil jika dataframe tidak kosong
if not results_df.empty:
    st.header(f"🏆 Top {len(results_df.drop_duplicates(subset=['employee_id']))} Kandidat untuk {role_name}")

    # 1. Tabel Hasil Peringkat Ringkas
    st.subheader("Tabel Peringkat Kandidat (Ringkasan)")
    summary_cols = ['employee_id', 'fullname', 'directorate', 'role', 'grade', 'final_match_rate']
    tgv_cols_for_summary = ['tgv_cognitive_match_rate', 'tgv_competency_match_rate', 'tgv_behavioral_match_rate', 'tgv_contextual_match_rate']
    
    summary_df = results_df.drop_duplicates(subset=['employee_id']).copy() # Gunakan .copy() untuk menghindari SettingWithCopyWarning
    
    # Ambil skor TGV unik
    tgv_scores_unique = summary_df[['employee_id'] + [c for c in tgv_cols_for_summary if c in summary_df.columns]]
    
    if not tgv_scores_unique.empty:
        tgv_scores_unique = tgv_scores_unique.set_index('employee_id')
        tgv_scores_unique.columns = ['TGV Cognitive', 'TGV Competency', 'TGV Behavioral', 'TGV Contextual']
    
        # Gabungkan ke tabel ringkasan
        cols_to_select = [col for col in summary_cols if col in summary_df.columns]
        summary_df = summary_df[cols_to_select].set_index('employee_id')
        summary_df = summary_df.join(tgv_scores_unique, on='employee_id', how='left').reset_index()
        
        # Format & Tampilkan Top 10 Unik
        st.dataframe(summary_df.head(10).style.format({
            'final_match_rate': '{:.2f}%',
            'TGV Cognitive': '{:.2f}', 
            'TGV Competency': '{:.2f}', 
            'TGV Behavioral': '{:.2f}', 
            'TGV Contextual': '{:.2f}'
            }), use_container_width=True)
    else:
        st.warning("Kolom skor TGV tidak ditemukan dalam hasil query. Tidak dapat menampilkan ringkasan TGV.")
        # Tetap tampilkan ringkasan tanpa TGV jika kolom skor hilang, tetapi ada data lain
        summary_df_minimal = summary_df[[c for c in summary_cols if c in summary_df.columns]]
        st.dataframe(summary_df_minimal.head(10).style.format({'final_match_rate': '{:.2f}%'}), use_container_width=True)


    st.markdown("---")

    # 2. Visualisasi Radar Chart & Breakdown Detail TV
    st.subheader("📊 Perbandingan Kandidat vs Benchmark & Detail Skor TV")
    
    unique_candidates = results_df['employee_id'].unique()
    candidate_labels = {}
    if not df_employees.empty:
         candidate_labels = df_employees[df_employees['employee_id'].isin(unique_candidates)]\
            .set_index('employee_id')['label'].to_dict()

    # Dapatkan label untuk kandidat teratas sebagai default
    top_candidate_id = results_df.iloc[0]['employee_id']
    default_candidate_label = candidate_labels.get(top_candidate_id, top_candidate_id)

    selected_candidate_label = st.selectbox(
        "Pilih Kandidat untuk Detail:",
        options=[candidate_labels.get(eid, eid) for eid in unique_candidates],
        index=([candidate_labels.get(eid, eid) for eid in unique_candidates].index(default_candidate_label) if default_candidate_label in [candidate_labels.get(eid, eid) for eid in unique_candidates] else 0)
    )
    
    # Dapatkan ID kandidat yang dipilih
    selected_candidate_id = next((eid for eid, label in candidate_labels.items() if label == selected_candidate_label), 
                                 next((eid for eid in unique_candidates if eid == selected_candidate_label), None))
    
    if selected_candidate_id:
        candidate_data = results_df[results_df['employee_id'] == selected_candidate_id].copy()
        
        col1, col2 = st.columns([2, 3]) 
        
        with col1:
            st.markdown(f"#### Radar Chart: {selected_candidate_label}")
            tgv_data_radar = candidate_data.drop_duplicates(subset=['tgv_name'])
            categories = ['Cognitive', 'Competency', 'Behavioral', 'Contextual'] 
            
            candidate_scores_radar = []
            
            # Perbaikan: Ambil skor TGV match rate dari baris yang unik
            for cat in categories:
                # Kolom TGV match rate sudah ada di df hasil query dengan nama tgv_match_rate
                 score_row = tgv_data_radar[tgv_data_radar['tgv_name'].str.lower() == cat.lower()]
                 # Pastikan kolom tgv_match_rate ada dan ambil nilai pertama
                 score = score_row['tgv_match_rate'].iloc[0] if not score_row.empty and 'tgv_match_rate' in score_row.columns else 0
                 candidate_scores_radar.append(score)

            # Jika skor kosong atau kurang dari 4, set default ke 0
            if len(candidate_scores_radar) < len(categories):
                candidate_scores_radar = [0] * len(categories)
                
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=candidate_scores_radar, theta=categories, fill='toself', name=f'Kandidat', line=dict(color='blue')))
            fig_radar.add_trace(go.Scatterpolar(r=[100] * len(categories), theta=categories, fill=None, name='Benchmark (100%)', line=dict(color='grey', dash='dash')))
    
            max_val = 100 # Maksimum 100 karena match rate
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max_val * 1.05])), # Range sedikit di atas 100
                showlegend=True, height=400, margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(yanchor="bottom", y= -0.2, xanchor="center", x=0.5) 
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("Membandingkan skor TGV kandidat (area biru) dengan skor benchmark ideal (garis putus-putus 100%).")

        with col2:
            st.markdown(f"#### Detail Skor TV: {selected_candidate_label}")
            detail_cols = ['tgv_name', 'tv_name', 'baseline_score', 'user_score', 'tv_match_rate']
            cols_to_display = [c for c in detail_cols if c in candidate_data.columns]
            
            # Jika kolom yang dibutuhkan ada, tampilkan dataframe
            if cols_to_display:
                st.dataframe(candidate_data[cols_to_display].style.format({
                    'baseline_score': '{:.2f}',
                    'user_score': '{:.2f}',
                    'tv_match_rate': '{:.2f}%'
                }), height=410, use_container_width=True)
            else:
                 st.warning("Kolom detail TV tidak ditemukan dalam hasil query.")
            
        st.markdown("---")

        # 4. Ringkasan Insight (Placeholder)
        st.subheader("💡 Ringkasan Insight")

        # Mengambil skor dari summary_df yang sudah di filter di atas
        top_candidate_name_insight = summary_df.iloc[0]['fullname'] if not summary_df.empty else "Kandidat Teratas"
        top_score_insight = summary_df.iloc[0]['final_match_rate'] if not summary_df.empty else 0.0
        st.markdown(f"Kandidat teratas, **{top_candidate_name_insight}**, mencapai skor **{top_score_insight:.2f}%**. "
                    f"Ini menunjukkan keselarasan yang kuat dengan profil benchmark. Analisis lebih lanjut pada Radar Chart dan Detail Skor TV dapat mengungkapkan area kekuatan dan pengembangan spesifik.")
     

    # 3. Distribusi Skor 
    st.subheader("Distribusi Skor Kecocokan (Top 10)")
    if not summary_df.empty:
        fig_hist = px.histogram(summary_df.head(10), x="final_match_rate", nbins=5, title="Distribusi Final Match Rate (Top 10)")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Tidak dapat menampilkan histogram karena data ringkasan kosong.")


# Kondisi jika tombol ditekan tapi tidak ada hasil
elif run_button and results_df.empty:
     st.info("Tidak ada hasil yang ditemukan. Periksa kembali ID benchmark yang dipilih atau konfigurasi bobot. Pastikan juga benchmark memiliki data yang cukup.")

# --- Bagian AI Job Profile Generator ---
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Job Profile Generator")
generate_profile = st.sidebar.button("Buat Profil Pekerjaan (AI)")

if generate_profile:
    if not role_name or not role_purpose:
         st.sidebar.error("Nama peran (**Role Name**) dan Tujuan Peran (**Role Purpose**) tidak boleh kosong.")
    else:
        with st.spinner("Membuat draf profil pekerjaan dengan AI..."):
            profile_text = generate_job_profile_ai(role_name, job_level, role_purpose)
            
            # Perbaikan: Hanya simpan jika tidak ada string "Error:"
            if profile_text and not profile_text.startswith("Error:"):
                 st.session_state['generated_profile'] = profile_text
            elif profile_text and profile_text.startswith("Error:"):
                 st.session_state['generated_profile'] = None 
                 # Tampilkan error yang spesifik dari fungsi AI
                 st.error(profile_text.replace("Error: ", "")) 
            else:
                 st.session_state['generated_profile'] = None 

        st.rerun() # Rerun agar konten utama di bawah sidebar terupdate

# Tampilkan Profil AI jika sudah ada di session state
if st.session_state.get('generated_profile'):
    st.markdown("---")
    st.header("🤖 Draf Profil Pekerjaan (Dibuat oleh AI)")
    st.markdown(st.session_state['generated_profile'])
    
    # Tambahkan tombol untuk menghapus/membersihkan hasil
    if st.button("Bersihkan Draf Profil"):
        st.session_state['generated_profile'] = None
        st.rerun() 

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("Talent Match App v1.0 ")