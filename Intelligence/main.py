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

# --- Inisialisasi Session State ---
# Menyimpan hasil query agar tidak hilang saat widget berinteraksi
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
# Menyimpan profil AI
if 'generated_profile' not in st.session_state:
    st.session_state['generated_profile'] = None
# Menyimpan state pilihan kandidat di dropdown
if 'selected_candidate_detail' not in st.session_state:
    st.session_state['selected_candidate_detail'] = None

# --- Fungsi Utility Pengecekan Secrets ---
def check_secrets_key(key_path, description):
    """Memeriksa keberadaan key di st.secrets dan menampilkan error jika hilang."""
    keys = key_path.split('.')
    current = st.secrets
    for key in keys:
        if hasattr(current, 'get'):
            current = current.get(key)
        else:
            return None
        if current is None:
            st.error(f"Error: Kunci '{key_path}' ({description}) tidak ditemukan di Streamlit Secrets.")
            return None
    return current

# --- Fungsi AI Generated Profile (Menggunakan OpenRouter Llama 3.1) ---
@st.cache_data(show_spinner=False) 
def generate_job_profile_ai(role, level, purpose):
    """
    Menghasilkan draf profil pekerjaan terstruktur menggunakan OpenRouter (Llama 3.1).
    Meminta output JSON secara eksplisit.
    """
    if not role or not purpose:
         return "Error: Nama Peran dan Tujuan Peran harus diisi untuk membuat profil AI."

    # 1. Ambil API Key dari Streamlit Secrets
    api_key_or = check_secrets_key("openrouter.api_key", "OpenRouter API Key")
    if not api_key_or:
        st.error("Error: Kunci 'openrouter.api_key' tidak ditemukan di Streamlit Secrets.")
        return "Error: OpenRouter API Key tidak ditemukan."

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key_or,
        )

        # 2. Definisikan Prompt
        system_prompt = """Anda adalah spesialis HR terkemuka yang ahli dalam membuat draf profil pekerjaan yang komprehensif. 
        Tugas Anda adalah menghasilkan output HANYA dalam format JSON yang valid, berdasarkan permintaan pengguna.
        Pastikan JSON yang Anda hasilkan ketat (strict) dan tidak ada teks lain di luar blok JSON."""
        
        user_prompt = f"""
        Buat draf profil pekerjaan dalam Bahasa Indonesia untuk peran: '{role}' dengan level '{level}'. 
        Tujuan peran (role purpose) utama adalah: '{purpose}'. 
        
        Harap hasilkan JSON dengan struktur berikut:
        {{
          "summary": "Ringkasan Peran yang menarik (1-2 kalimat).",
          "responsibilities": [
            "Tanggung jawab utama 1",
            "Tanggung jawab utama 2",
            "Tanggung jawab utama 3",
            "Tanggung jawab utama 4",
            "Tanggung jawab utama 5"
          ],
          "qualifications": [
            "Kualifikasi minimum 1 (misal: S1 di bidang terkait)",
            "Kualifikasi minimum 2 (misal: 3+ tahun pengalaman)",
            "Kualifikasi minimum 3"
          ],
          "skills": [
            "Keterampilan teknis/soft skill 1",
            "Keterampilan teknis/soft skill 2",
            "Keterampilan teknis/soft skill 3",
            "Keterampilan teknis/soft skill 4",
            "Keterampilan teknis/soft skill 5"
          ]
        }}
        """

        # 3. Panggilan API
        completion = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"}, 
            max_tokens=2048,
            temperature=0.7,
        )

        # 4. Parse Respons
        response_text = completion.choices[0].message.content
        
        if not response_text:
            return "Error: AI mengembalikan respons kosong."

        return json.loads(response_text)

    except Exception as e:
        st.error(f"Error memanggil OpenRouter API: {e}")
        st.error("Pastikan API Key OpenRouter Anda valid, memiliki kuota, dan model 'meta-llama/llama-3.1-70b-instruct:free' tersedia.")
        return f"Error: Gagal terhubung ke AI: {e}"
    except json.JSONDecodeError as e:
        # Ini seharusnya jarang terjadi dengan response_format="json_object"
        st.error(f"Error: AI tidak mengembalikan JSON yang valid: {e}")
        st.error(f"Raw output from AI: {response_text}")
        return "Error: AI tidak mengembalikan JSON yang valid. Coba ulangi."


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
            st.error("Detail koneksi PostgreSQL (host, dbname, user, password, port) tidak ditemukan di Streamlit Secrets.")
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
@st.cache_data(ttl=600) # Refresh setiap 10 menit
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

        if 'fullname' in df.columns and 'employee_id' in df.columns:
            df['label'] = df['fullname'].str.strip() + " (" + df['employee_id'].astype(str).str.strip() + ")"
            return df[['employee_id', 'label']]
        else:
            st.error("Tabel 'employees' tidak memiliki kolom 'fullname' atau 'employee_id'.")
            return empty_df
    except Exception as e:
        st.error(f"Error fetching employee list: {e}")
        return empty_df

# Fungsi utama untuk menjalankan query match
def run_talent_match_query(conn, bench_ids, weights):
    """Menjalankan fungsi SQL fn_talent_management.""" 
    if not conn:
        return pd.DataFrame() 
    
    # 1. Validasi Bobot
    required_weights = {"cognitive", "competency", "behavioral", "contextual"}
    if not all(key in weights for key in required_weights):
         st.warning(f"Konfigurasi bobot tidak valid. Diperlukan kunci: {', '.join(required_weights)}.")
         return pd.DataFrame()

    weights_json = json.dumps(weights) 
    
    # 2. Eksekusi Query
    try:
        query = "SELECT * FROM fn_talent_management(%s::TEXT[], %s::JSONB);" 
        df_results = pd.read_sql(query, conn, params=(bench_ids, weights_json))
        
        df_results.columns = [col.replace('o_', '') for col in df_results.columns]
        
        rate_cols = [col for col in df_results.columns if 'match_rate' in col]
        for col in rate_cols:
            if pd.api.types.is_numeric_dtype(df_results[col]):
                df_results[col] = df_results[col].clip(0, 100)

        # Tambahkan kolom fullname untuk tampilan yang lebih baik (jika belum ada)
        df_employees = get_employee_list(conn)
        if not df_employees.empty and 'fullname' not in df_results.columns:
            employee_name_map = df_employees.set_index('employee_id')['label'].apply(lambda x: x.split(' (')[0]).rename('fullname')
            df_results = df_results.join(employee_name_map, on='employee_id', how='left')
        
        return df_results
    except Exception as e:
        st.error(f"Error running talent match function (fn_talent_management): {e}") 
        return pd.DataFrame()

# --- Fungsi Visualisasi ---

def create_radar_chart(df, candidate_id, candidate_name):
    """Membuat Radar Chart perbandingan TGV (Dinamis)."""
    # Ambil data unik TGV untuk kandidat
    candidate_data = df[df['employee_id'] == candidate_id].drop_duplicates(subset=['tgv_name'])
    
    if candidate_data.empty or 'tgv_name' not in candidate_data.columns:
        st.warning("Data TGV tidak ditemukan untuk Radar Chart.")
        return go.Figure()

    # Ambil kategori TGV secara dinamis dari data
    categories = candidate_data['tgv_name'].unique().tolist()
    
    candidate_scores = []
    
    # Ambil skor dari baris yang sesuai
    for cat in categories:
        score_row = candidate_data[candidate_data['tgv_name'] == cat]
        score = score_row['tgv_match_rate'].iloc[0] if not score_row.empty and 'tgv_match_rate' in score_row.columns else 0
        candidate_scores.append(max(0, min(score, 100))) 

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=candidate_scores, 
        theta=categories, 
        fill='toself', 
        name=f'{candidate_name}', 
        line=dict(color='#1067b3')
    ))
    fig.add_trace(go.Scatterpolar(
        r=[100] * len(categories), 
        theta=categories, 
        fill=None, 
        name='Benchmark (100%)', 
        line=dict(color='grey', dash='dot')
    ))

    fig.update_layout(
        title=f"Skor Kecocokan TGV untuk {candidate_name}",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 105], ticksuffix="%"),
            angularaxis=dict(direction="clockwise")
        ),
        showlegend=True, 
        height=400, 
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

# (BARU) Fungsi Heatmap TV
def create_tv_heatmap(df, candidate_id, candidate_name):
    """
    (BARU) Membuat Heatmap skor tv_match_rate untuk kandidat yang dipilih.
    """
    candidate_data = df[df['employee_id'] == candidate_id].copy()
    
    if 'tgv_name' not in candidate_data.columns or 'tv_name' not in candidate_data.columns or 'tv_match_rate' not in candidate_data.columns:
        st.warning("Data tidak lengkap untuk membuat Heatmap TV (memerlukan tgv_name, tv_name, tv_match_rate).")
        return go.Figure()

    # Buat pivot table untuk heatmap
    try:
        df_pivot = candidate_data.pivot(
            index='tgv_name', 
            columns='tv_name', 
            values='tv_match_rate'
        )
    except Exception as e:
        st.warning(f"Gagal membuat pivot data untuk heatmap: {e}")
        return go.Figure()

    fig = px.imshow(
        df_pivot,
        text_auto='.0f',
        aspect="auto",
        color_continuous_scale='RdYlGn', 
        range_color=[0, 100], 
        title=f"Heatmap Kecocokan Talent Variable (TV) untuk {candidate_name}"
    )
    
    fig.update_layout(
        xaxis_title="Talent Variable (TV)",
        yaxis_title="Talent Group Variable (TGV)",
        height=400, 
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    fig.update_xaxes(side="bottom")
    
    return fig

def create_strengths_gaps_charts(df, candidate_id):
    """
    Membuat bar chart horizontal untuk Top 5 Kekuatan dan Kesenjangan TV.
    """
    candidate_data = df[df['employee_id'] == candidate_id].copy()
    
    if 'tv_match_rate' not in candidate_data.columns or 'tv_name' not in candidate_data.columns:
        return None, None # Kembalikan None jika data tidak ada

    # Pastikan tv_match_rate adalah numerik
    candidate_data['tv_match_rate'] = pd.to_numeric(candidate_data['tv_match_rate'], errors='coerce')
    candidate_data['tv_match_rate'] = candidate_data['tv_match_rate'].clip(0, 100) # (BARU) Clip data TV
    candidate_data = candidate_data.dropna(subset=['tv_match_rate'])

    # Urutkan berdasarkan TV Match Rate
    candidate_data = candidate_data.sort_values(by='tv_match_rate', ascending=False)

    # Ambil Top 5 Kekuatan (Skor Tertinggi)
    df_strengths = candidate_data.head(5)
    
    # Ambil Top 5 Kesenjangan (Skor Terendah)
    df_gaps = candidate_data.tail(5).sort_values(by='tv_match_rate', ascending=True)

    # Visualisasi Kekuatan
    fig_strengths = px.bar(
        df_strengths,
        x='tv_match_rate',
        y='tv_name',
        orientation='h',
        title='Top 5 Kekuatan (Talent Variables)',
        labels={'tv_match_rate': 'Match Rate (%)', 'tv_name': 'Talent Variable'},
        color='tv_match_rate',
        color_continuous_scale=px.colors.sequential.Greens,
        range_color=[50, 100],
        text='tv_match_rate'
    )
    fig_strengths.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_strengths.update_layout(yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(l=40, r=40, t=50, b=40))

    # Visualisasi Kesenjangan
    fig_gaps = px.bar(
        df_gaps,
        x='tv_match_rate',
        y='tv_name',
        orientation='h',
        title='Top 5 Kesenjangan (Talent Variables)',
        labels={'tv_match_rate': 'Match Rate (%)', 'tv_name': 'Talent Variable'},
        color='tv_match_rate',
        color_continuous_scale=px.colors.sequential.Reds_r, # Reverse Reds
        range_color=[0, 50],
        text='tv_match_rate'
    )
    fig_gaps.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_gaps.update_layout(yaxis={'categoryorder':'total descending'}, height=350, margin=dict(l=40, r=40, t=50, b=40))

    return fig_strengths, fig_gaps

# Fungsi Ringkasan Wawasan Kandidat Dinamis
def generate_candidate_summary(selected_candidate_id, candidate_name, summary_df, results_df):
    """
    (BARU) Menghasilkan ringkasan insight dinamis (menggunakan st.metric)
    untuk kandidat yang dipilih.
    """
    try:
        # 1. Ambil Data Ringkasan (Peringkat, Skor Akhir)
        candidate_summary = summary_df[summary_df['employee_id'] == selected_candidate_id].iloc[0]
        
        # 2. Ambil Data Detail TV/TGV
        candidate_data_tv = results_df[results_df['employee_id'] == selected_candidate_id]

        # 3. Hitung TGV Terbaik/Terburuk
        tgv_data = candidate_data_tv.drop_duplicates(subset=['tgv_name']).sort_values('tgv_match_rate', ascending=False)
        best_tgv = tgv_data.iloc[0]
        worst_tgv = tgv_data.iloc[-1]

        # 4. Hitung TV Terbaik (Kekuatan) / Terburuk (Kesenjangan)
        tv_data = candidate_data_tv.sort_values('tv_match_rate', ascending=False)
        best_tv = tv_data.iloc[0]
        worst_tv = tv_data.iloc[-1]
        
        # 5. Tampilkan Metrik
        st.subheader(f"💡 Ringkasan Wawasan untuk {candidate_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Kolom 1: Peringkat & Skor
        col1.metric(
            label="Peringkat Keseluruhan",
            value=f"#{int(candidate_summary['rank'])}",
            delta=f"{candidate_summary['final_match_rate']:.2f}% Match",
            delta_color="off"
        )
        
        # Kolom 2: TGV Terbaik
        col2.metric(
            label="TGV Terbaik",
            value=best_tgv['tgv_name'],
            delta=f"{best_tgv['tgv_match_rate']:.2f}%",
            delta_color="normal" # "normal" = hijau
        )
        
        # Kolom 3: TGV Terburuk
        col3.metric(
            label="TGV Terendah",
            value=worst_tgv['tgv_name'],
            delta=f"{worst_tgv['tgv_match_rate']:.2f}%",
            delta_color="inverse" # "inverse" = merah
        )
        
        # Kolom 4: Info Tambahan
        col4.metric(
            label="Direktorat",
            value=candidate_summary['directorate'],
            delta=candidate_summary['grade'],
            delta_color="off"
        )

        st.markdown("---")
        
        col_tv1, col_tv2 = st.columns(2)
        
        with col_tv1:
            st.success(f"""
            **Kekuatan TV Teratas:**
            - **{best_tv['tv_name']}** (dari {best_tv['tgv_name']})
            - Skor Kecocokan: **{best_tv['tv_match_rate']:.2f}%**
            """)

        with col_tv2:
            st.warning(f"""
            **Kesenjangan TV Terbesar:**
            - **{worst_tv['tv_name']}** (dari {worst_tv['tgv_name']})
            - Skor Kecocokan: **{worst_tv['tv_match_rate']:.2f}%**
            """)

    except Exception as e:
        st.error(f"Gagal membuat ringkasan wawasan: {e}")


# --- Sidebar Input Form ---
st.sidebar.header("🔍 Konfigurasi Pencocokan")

# Inisialisasi koneksi dan ambil data karyawan
conn = init_connection()
df_employees = get_employee_list(conn)

# 1. Input Metadata Lowongan
job_vacancy_id = st.sidebar.text_input("Job Vacancy ID (Optional)", placeholder="E.g., DV-2025-01")
role_name = st.sidebar.text_input("Role Name", placeholder=" E.g. Data Analyst", key="role_name_input") 
job_level = st.sidebar.selectbox("Job Level / Grade", ["I", "II", "III", "IV", "V", "VI"], index=0, key="job_level_input") 
role_purpose = st.sidebar.text_area("Role Purpose (1-2 sentences)", placeholder=" E.g. Menyediakan wawasan data yang memimpin keputusan strategis dan mengelola dashboard kinerja.", key="role_purpose_input") 

# 2. Input Benchmark Talenta
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Pilih Benchmark Talenta (Rating 5)")
selected_labels = st.sidebar.multiselect(
    "Pilih Karyawan Benchmark:",
    options=df_employees['label'].tolist(),
    default=df_employees['label'].head(3).tolist() if not df_employees.empty else [],
    max_selections=3
)
selected_talent_ids = df_employees[df_employees['label'].isin(selected_labels)]['employee_id'].tolist()

# 3. Input Bobot TGV
st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Konfigurasi Bobot TGV")
default_weights = {"cognitive": 0.35, "competency": 0.30, "behavioral": 0.25, "contextual": 0.10}
current_total = sum(default_weights.values())

# Tampilkan input bobot dengan gaya yang rapi
col_w1, col_w2 = st.sidebar.columns(2)
weights_config = {}

weights_config["cognitive"] = col_w1.number_input("Cognitive", 0.0, 1.0, default_weights["cognitive"], 0.05, key="num_cog")
weights_config["competency"] = col_w2.number_input("Competency", 0.0, 1.0, default_weights["competency"], 0.05, key="num_comp")
weights_config["behavioral"] = col_w1.number_input("Behavioral", 0.0, 1.0, default_weights["behavioral"], 0.05, key="num_beh")
weights_config["contextual"] = col_w2.number_input("Contextual", 0.0, 1.0, default_weights["contextual"], 0.05, key="num_cont")

current_total = sum(weights_config.values())
st.sidebar.caption(f"Total Bobot Saat Ini: **{current_total:.2f}**")
is_weights_valid = abs(current_total - 1.0) < 0.01

if not is_weights_valid:
    st.sidebar.error("Total bobot harus mendekati 1.0.")

run_button = st.sidebar.button("🚀 Jalankan Pencocokan Talenta", disabled=not conn or not selected_talent_ids or not is_weights_valid)

# --- Area Tampilan Hasil ---

if run_button:
    if not conn:
        st.error("Koneksi database gagal. Harap periksa `secrets.toml` Anda.")
    elif not selected_talent_ids:
        st.sidebar.error("Harap pilih minimal satu karyawan benchmark.")
    elif not is_weights_valid:
         st.sidebar.error("Konfigurasi bobot tidak valid. Harap periksa input Anda.")
    else:
        with st.spinner(f"Menjalankan SQL Talent Match untuk {st.session_state['role_name_input']}..."):
            # (BARU) Simpan hasil ke session state
            st.session_state.results_df = run_talent_match_query(conn, selected_talent_ids, weights_config)
            # (BARU) Reset pilihan kandidat detail saat query baru dijalankan
            if "selected_candidate_detail" in st.session_state:
                st.session_state.selected_candidate_detail = None 

if not st.session_state.results_df.empty:
    
    results_df = st.session_state.results_df
    
    # Dapatkan ringkasan unik per karyawan
    summary_df = results_df.drop_duplicates(subset=['employee_id']).copy()
    
    # Penambahan kolom peringkat
    summary_df = summary_df.sort_values(by='final_match_rate', ascending=False).reset_index(drop=True)
    summary_df['rank'] = summary_df.index + 1
    
    # Ambil kolom skor TGV yang relevan (kolom match rate TGV diasumsikan dari SQL)
    # Pastikan kolom-kolom ini ada
    tgv_cols_to_check = [
        'tgv_cognitive_match_rate', 'tgv_competency_match_rate', 
        'tgv_behavioral_match_rate', 'tgv_contextual_match_rate'
    ]
    
    tgv_cols_map = {}
    summary_cols = ['rank', 'employee_id', 'fullname', 'directorate', 'role', 'grade', 'final_match_rate']
    
    # Cek apakah kolom TGV ada
    if all(col in summary_df.columns for col in tgv_cols_to_check):
        tgv_cols_map = {
            'TGV Cognitive Match Rate': summary_df['tgv_cognitive_match_rate'].round(2),
            'TGV Competency Match Rate': summary_df['tgv_competency_match_rate'].round(2),
            'TGV Behavioral Match Rate': summary_df['tgv_behavioral_match_rate'].round(2),
            'TGV Contextual Match Rate': summary_df['tgv_contextual_match_rate'].round(2)
        }
        # Ambil TGV match rate dari summary_df
        for name, series in tgv_cols_map.items():
            summary_df[name] = series
        
        summary_cols.extend(list(tgv_cols_map.keys()))
    
    # 1. Tampilkan Ringkasan & Tabel Peringkat
    st.header(f"🏆 Peringkat {len(summary_df)} Kandidat untuk {st.session_state['role_name_input']}")
    
    st.dataframe(
        summary_df[summary_cols].head(10).style.format({
            'final_match_rate': '{:.2f}%',
            **{k: '{:.2f}' for k in tgv_cols_map.keys()}
        }), 
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")

    # 2. Visualisasi Detail
    st.header("📊 Detail Analisis TGV dan TV")

    # Logika selectbox untuk persistensi
    unique_candidates = summary_df['employee_id'].unique()
    candidate_labels = {}
    if not df_employees.empty:
         candidate_labels = df_employees[df_employees['employee_id'].isin(unique_candidates)]\
            .set_index('employee_id')['label'].to_dict()

    current_options = [candidate_labels.get(eid, eid) for eid in unique_candidates]
    state_key = "selected_candidate_detail"

    # Set default jika state belum ada 
    if state_key not in st.session_state or st.session_state[state_key] is None or st.session_state[state_key] not in current_options:
        top_candidate_id = summary_df.iloc[0]['employee_id']
        st.session_state[state_key] = candidate_labels.get(top_candidate_id, top_candidate_id)

    selected_candidate_label = st.selectbox(
        "Pilih Kandidat untuk Perbandingan Detail:",
        options=current_options,
        key=state_key 
    )
    
    selected_candidate_id = next((eid for eid, label in candidate_labels.items() if label == selected_candidate_label), 
                                 next((eid for eid in unique_candidates if eid == selected_candidate_label), None))

    if selected_candidate_id:
        
        generate_candidate_summary(
            selected_candidate_id, 
            selected_candidate_label.split(' (')[0], 
            summary_df, 
            results_df
        )
        
        col_viz_1, col_viz_2 = st.columns([2, 3]) 
        
        with col_viz_1:
            # Radar Chart
            fig_radar = create_radar_chart(results_df, selected_candidate_id, selected_candidate_label.split(' (')[0])
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_viz_2:
            st.markdown(f"#### Heatmap Skor TV: {selected_candidate_label.split(' (')[0]}")
            fig_heatmap = create_tv_heatmap(results_df, selected_candidate_id, selected_candidate_label.split(' (')[0])
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
        
        # Visualisasi Kekuatan dan Kesenjangan
        st.markdown("---")
        st.subheader(f"Analisis Kekuatan & Kesenjangan TV untuk {selected_candidate_label.split(' (')[0]}")
        
        fig_strengths, fig_gaps = create_strengths_gaps_charts(results_df, selected_candidate_id)
        
        if fig_strengths is not None and fig_gaps is not None:
            col_gap_1, col_gap_2 = st.columns(2)
            with col_gap_1:
                st.plotly_chart(fig_strengths, use_container_width=True)
            with col_gap_2:
                st.plotly_chart(fig_gaps, use_container_width=True)
        else:
            st.warning("Tidak dapat membuat visualisasi Kekuatan/Kesenjangan. Kolom 'tv_match_rate' atau 'tv_name' mungkin hilang dari hasil query.")


elif run_button and st.session_state.results_df.empty:
     st.warning("Tidak ada hasil yang ditemukan. Pastikan koneksi database aktif, dan fungsi SQL 'fn_talent_management' tersedia dengan parameter yang benar, serta ID benchmark memiliki data yang memadai.")

# --- Bagian AI Job Profile Generator ---
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Job Profile Generator")
generate_profile = st.sidebar.button("Buat Profil Pekerjaan (AI)", key="ai_button")

if generate_profile:
    role = st.session_state['role_name_input']
    level = st.session_state['job_level_input']
    purpose = st.session_state['role_purpose_input']
    
    if not role or not purpose:
         st.sidebar.error("Nama peran dan Tujuan Peran tidak boleh kosong.")
    else:
        with st.spinner("Membuat draf profil pekerjaan dengan AI (Llama 3.1)..."):
            profile_json = generate_job_profile_ai(role, level, purpose)
            
            if isinstance(profile_json, dict):
                st.session_state['generated_profile'] = profile_json
            elif isinstance(profile_json, str) and profile_json.startswith("Error:"):
                 st.session_state['generated_profile'] = None 
                 st.error(profile_json.replace("Error: ", "")) 
            else:
                 st.session_state['generated_profile'] = None 

# Tampilkan Profil AI jika sudah ada di session state
if st.session_state.get('generated_profile'):
    st.markdown("---")
    st.header(f"🤖 Draf Profil Pekerjaan untuk {st.session_state['role_name_input']} (Dibuat oleh AI)")
    
    profile = st.session_state['generated_profile']
    
    # Tampilkan Ringkasan
    st.subheader("Ringkasan Peran")
    st.markdown(f"***{profile.get('summary', 'Tidak Ada Ringkasan')}***")

    col_ai_1, col_ai_2 = st.columns(2)
    
    with col_ai_1:
        st.subheader("Tanggung Jawab Utama")
        responsibilities = profile.get('responsibilities', [])
        if responsibilities:
            for item in responsibilities:
                st.markdown(f"- {item}")
        else:
            st.info("Tidak ada data tanggung jawab.")
            
        st.subheader("Kualifikasi Minimum")
        qualifications = profile.get('qualifications', [])
        if qualifications:
            for item in qualifications:
                st.markdown(f"- {item}")
        else:
            st.info("Tidak ada data kualifikasi.")

    with col_ai_2:
        st.subheader("Keterampilan yang Diutamakan")
        skills = profile.get('skills', [])
        if skills:
            for item in skills:
                st.markdown(f"- {item}")
        else:
            st.info("Tidak ada data keterampilan.")
    
    st.markdown("---")
    if st.button("Bersihkan Draf Profil", key="clear_ai_button"):
        st.session_state['generated_profile'] = None
        st.rerun() 

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("Talent Match App v1.0 | OpenRouter Llama 3.3")

