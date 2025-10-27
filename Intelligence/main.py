import streamlit as st
import pandas as pd
import psycopg2 
import json
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI 

# --- Page Config & Title ---
st.set_page_config(layout="wide", page_title="AI Talent Match Intelligence")
st.title("🚀 AI Talent Match Intelligence Dashboard")
st.markdown("Menemukan kandidat internal terbaik berdasarkan profil benchmark.")

# --- Helper Functions ---

# Cache database connection
@st.cache_resource
def init_connection():
    """Initializes connection to PostgreSQL database using Streamlit secrets."""
    try:
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

# Cache employee list data
@st.cache_data
def get_employee_list(_conn):
    """Fetches employee_id and fullname from the database."""
    if _conn is None:
        return pd.DataFrame({'employee_id': [], 'label': []})
    try:
        query = "SELECT employee_id, fullname FROM employees ORDER BY fullname;"
        df = pd.read_sql(query, _conn)
        # Create a more informative label for the multiselect widget
        df['label'] = df['fullname'] + " (" + df['employee_id'] + ")"
        return df[['employee_id', 'label']]
    except Exception as e:
        st.error(f"Error fetching employee list: {e}")
        return pd.DataFrame({'employee_id': [], 'label': []})

def run_talent_match_query(conn, bench_ids, weights):
    """Executes the fn_talent_management SQL function."""
    if not conn or not bench_ids:
        st.warning("Koneksi database gagal atau tidak ada benchmark ID yang dipilih.")
        return pd.DataFrame()

    try:
        weights_json = json.dumps(weights)
        query = "SELECT * FROM fn_talent_management(%s::TEXT[], %s::JSONB);"
        df_results = pd.read_sql(query, conn, params=(bench_ids, weights_json))
        df_results.columns = [col.replace('o_', '') for col in df_results.columns] # Rename output columns
        return df_results
    except Exception as e:
        st.error(f"Error running talent match function (fn_talent_management): {e}")
        st.error(f"Query parameters: bench_ids={bench_ids}, weights={weights_json}")
        if hasattr(e, 'pgcode'): st.error(f"PostgreSQL Error Code: {e.pgcode}")
        if hasattr(e, 'pgerror'): st.error(f"PostgreSQL Error Message: {e.pgerror}")
        return pd.DataFrame()

# --- LLM Function ---
def call_llm_api(prompt):
    """
    Memanggil API OpenRouter untuk menghasilkan teks berdasarkan prompt.
    """
    try:
        # Inisialisasi Klien OpenAI dengan base_url OpenRouter
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1", 
            api_key=st.secrets["openrouter"]["api_key"], 
        )
        selected_model = "deepseek/deepseek-chat-v3.1:free" 
        st.info(f"Melakukan Fetching pada Openrouter")

        response = client.chat.completions.create(
            model=selected_model,
            max_tokens=1000,
            temperature=0.7,
        )

        # Ekstrak teks respons
        if response.choices:
            return response.choices[0].message.content.strip()
        else:
            st.warning("Respons dari OpenRouter tidak berisi pilihan (choices).")
            return "Maaf, AI tidak dapat menghasilkan respons saat ini."

    except Exception as e:
        st.error(f"Error calling OpenRouter API: {e}")
        if "Incorrect API key provided" in str(e) or "authentication" in str(e).lower():
            st.error("API Key OpenRouter tidak valid atau hilang. Periksa file .streamlit/secrets.toml.")
        elif "rate limit" in str(e).lower() or "429" in str(e):
            st.error("Anda telah mencapai batas permintaan OpenRouter. Coba lagi nanti atau cek akun Anda.")
        return "Terjadi kesalahan saat menghubungi OpenRouter AI."


# --- Inisialisasi Session State ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# --- Sidebar Input Form ---
st.sidebar.header("🔍 Konfigurasi Pencocokan")

conn = init_connection()
# Hanya lanjutkan jika koneksi berhasil
if conn:
    df_employees = get_employee_list(conn)

    # 1. Input Metadata Lowongan
    job_vacancy_id = st.sidebar.text_input("Job Vacancy ID")
    role_name = st.sidebar.text_input("Role Name")
    job_level = st.sidebar.selectbox("Job Level / Grade", ["I", "II", "III", "IV", "V", "VI"], index=0)
    role_purpose = st.sidebar.text_area("Role Purpose (1-2 sentences)")

    # Store inputs in session state for the LLM
    st.session_state['role_name'] = role_name
    st.session_state['job_level'] = job_level
    st.session_state['role_purpose'] = role_purpose

    # 2. Input Benchmark Talenta
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Pilih Benchmark Talenta (Rating 5)")
    selected_labels = st.sidebar.multiselect(
        "Pilih Karyawan Benchmark:",
        options=df_employees['label'].tolist(),
        default=df_employees['label'].head(3).tolist() if not df_employees.empty else []
    )
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
    else: # JSON Input
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

    # --- Logika Pemrosesan Tombol & Penyimpanan State ---
    if run_button:
        if not selected_talent_ids:
            st.sidebar.error("Harap pilih minimal satu karyawan benchmark.")
        elif not weights_config:
            st.sidebar.error("Konfigurasi bobot tidak valid.")
        else:
            with st.spinner("Mencari kandidat terbaik..."):
                # Panggil fungsi query
                temp_results_df = run_talent_match_query(conn, selected_talent_ids, weights_config)

                # Join nama karyawan
                if not temp_results_df.empty and 'fullname' not in temp_results_df.columns:
                    if not df_employees.empty:
                        employee_names = df_employees.set_index('employee_id')['label'].str.split(' \(').str[0]
                        temp_results_df = temp_results_df.join(employee_names.rename('fullname'), on='employee_id')
                        if 'fullname' in temp_results_df.columns:
                            desired_cols_start = ['employee_id', 'fullname']
                            existing_cols_end = [col for col in temp_results_df.columns if col not in desired_cols_start]
                            final_cols_order = desired_cols_start + existing_cols_end
                            # Pastikan semua kolom ada sebelum reorder
                            if all(c in temp_results_df.columns for c in final_cols_order):
                                temp_results_df = temp_results_df[final_cols_order]

                    else:
                        st.warning("Data karyawan tidak dapat dimuat untuk menampilkan nama lengkap.")

                # Simpan hasil ke session state
                st.session_state.results_df = temp_results_df
                # Beri pesan sukses/warning
                if not st.session_state.results_df.empty:
                    st.success("Pencocokan selesai!")
                else:
                    st.warning("Pencocokan selesai, namun tidak ada hasil ditemukan.")


    # --- Area Tampilan Hasil (Menggunakan Session State) ---
    # Cek apakah ada data di session state untuk ditampilkan
    if not st.session_state.results_df.empty:
        results_df_display = st.session_state.results_df # Alias

        st.header(f"🏆 Top 10 Kandidat untuk {st.session_state.get('role_name', 'Peran Dipilih')}")
        st.markdown(f"**Vacancy ID:** {job_vacancy_id} | **Level:** {st.session_state.get('job_level', 'N/A')}")
        st.markdown("---")

        # 1. Tabel Hasil Peringkat Ringkas
        st.subheader("🥇 Top 10 Kandidat Internal (Ringkasan)")
        summary_cols_base = ['employee_id', 'fullname', 'directorate', 'role', 'grade', 'final_match_rate']

        summary_df = results_df_display.drop_duplicates(subset=['employee_id'])
        cols_to_select_summary = [col for col in summary_cols_base if col in summary_df.columns]
        summary_df = summary_df[cols_to_select_summary].copy() 
        
        # Bersihkan spasi dari 'fullname' di summary_df
        if 'fullname' in summary_df.columns:
            summary_df['fullname'] = summary_df['fullname'].str.strip()

        tgv_scores_wide = pd.DataFrame()
        try:
            if all(c in results_df_display.columns for c in ['employee_id', 'tgv_name', 'tgv_match_rate']):
                tgv_data_pivot = results_df_display[['employee_id', 'tgv_name', 'tgv_match_rate']].dropna().drop_duplicates()
                if not tgv_data_pivot.empty:
                    tgv_scores_wide = tgv_data_pivot.pivot(index='employee_id', columns='tgv_name', values='tgv_match_rate')
                    tgv_scores_wide = tgv_scores_wide.rename(columns={
                        'Cognitive': 'TGV Cognitive', 'Competency': 'TGV Competency',
                        'Behavioral': 'TGV Behavioral', 'Contextual': 'TGV Contextual'
                    })
            else:
                st.warning("Kolom 'employee_id', 'tgv_name', atau 'tgv_match_rate' tidak ditemukan untuk pivot.")

        except Exception as e:
            st.error(f"Gagal melakukan pivot skor TGV: {e}")

        if 'employee_id' in summary_df.columns and not tgv_scores_wide.empty:
            if not summary_df.index.name == 'employee_id':
                summary_df = summary_df.set_index('employee_id')
            summary_df = summary_df.join(tgv_scores_wide).reset_index()
        elif 'employee_id' not in summary_df.columns:
            st.warning("Kolom 'employee_id' tidak ditemukan untuk join skor TGV.")

        for col_name in ['TGV Cognitive', 'TGV Competency', 'TGV Behavioral', 'TGV Contextual']:
            if col_name not in summary_df.columns:
                summary_df[col_name] = pd.NA
        
        MAX_SUMMARY_CAP = 100.0
        cols_to_cap = ['final_match_rate', 'TGV Cognitive', 'TGV Competency', 'TGV Behavioral', 'TGV Contextual']
        
        for col in cols_to_cap:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].clip(upper=MAX_SUMMARY_CAP)

        format_dict = {}
        if 'final_match_rate' in summary_df.columns: format_dict['final_match_rate'] = '{:.2f}%'
        if 'TGV Cognitive' in summary_df.columns: format_dict['TGV Cognitive'] = '{:.2f}'
        if 'TGV Competency' in summary_df.columns: format_dict['TGV Competency'] = '{:.2f}'
        if 'TGV Behavioral' in summary_df.columns: format_dict['TGV Behavioral'] = '{:.2f}'
        if 'TGV Contextual' in summary_df.columns: format_dict['TGV Contextual'] = '{:.2f}'

        st.dataframe(summary_df.head(10).style.format(format_dict), use_container_width=True)


        # --- [KODE VISUALISASI] ---
        st.markdown("---")
        st.subheader("📊 Analisis Kumpulan Kandidat")
        
        # Ambil data unik per karyawan untuk visualisasi ini
        unique_candidate_df = results_df_display.drop_duplicates(subset=['employee_id'])

        col1_macro, col2_macro = st.columns(2)

        with col1_macro:
            # 1. VISUALISASI: Distribusi Match Rate (Histogram)
            st.markdown("**Distribusi Match Rate (Semua Kandidat)**")
            if 'final_match_rate' in unique_candidate_df.columns:
                fig_hist = px.histogram(
                    unique_candidate_df, 
                    x='final_match_rate', 
                    nbins=20, 
                    title="Sebaran Skor Kandidat",
                    labels={'final_match_rate': 'Skor Match (%)'}
                )
                fig_hist.update_layout(height=350, margin=dict(l=30, r=30, t=60, b=30))
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("Kolom 'final_match_rate' tidak ditemukan untuk histogram.")

        with col2_macro:
            # 2. VISUALISASI: Kekuatan & Kelemahan Umum (Bar Charts)
            st.markdown("**Kekuatan & Kelemahan Umum (Rata-rata TV)**")
            if 'tv_name' in results_df_display.columns and 'tv_match_rate' in results_df_display.columns:
                # Hitung rata-rata skor untuk setiap TV di semua kandidat
                avg_tv_scores = results_df_display.groupby('tv_name')['tv_match_rate'].mean().sort_values()
                
                # Ambil 5 teratas dan 5 terbawah
                top_gaps = avg_tv_scores.head(5).reset_index()
                top_strengths = avg_tv_scores.tail(5).sort_values(ascending=False).reset_index()

                # Buat tab untuk beralih
                tab1, tab2 = st.tabs(["Top 5 Gaps (Kelemahan)", "Top 5 Strengths (Kekuatan)"])
                
                with tab1:
                    fig_gaps = px.bar(
                        top_gaps, 
                        x='tv_match_rate', 
                        y='tv_name', 
                        orientation='h', 
                        title="Top 5 Gaps (Rata-rata Terendah)",
                        labels={'tv_name': 'Talent Variable', 'tv_match_rate': 'Rata-rata Skor Match (%)'},
                        color_discrete_sequence=['#FF6B6B'] # Merah
                    )
                    fig_gaps.update_yaxes(categoryorder="total ascending")
                    fig_gaps.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=30))
                    st.plotly_chart(fig_gaps, use_container_width=True)

                with tab2:
                    fig_strengths = px.bar(
                        top_strengths, 
                        x='tv_match_rate', 
                        y='tv_name', 
                        orientation='h', 
                        title="Top 5 Strengths (Rata-rata Tertinggi)",
                        labels={'tv_name': 'Talent Variable', 'tv_match_rate': 'Rata-rata Skor Match (%)'},
                        color_discrete_sequence=['#6BDB6B'] # Hijau
                    )
                    fig_strengths.update_yaxes(categoryorder="total descending")
                    fig_strengths.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=30))
                    st.plotly_chart(fig_strengths, use_container_width=True)
            else:
                st.warning("Kolom 'tv_name' atau 'tv_match_rate' tidak ditemukan untuk analisis kekuatan/kelemahan.")

        # 3. VISUALISASI: Perbandingan Kandidat (Heatmap)
        st.markdown("**Perbandingan Detail TV (Heatmap Top 10)**")
        
        if 'employee_id' in summary_df.columns and 'fullname' in summary_df.columns:
            
            top_10_names_ordered = summary_df.head(10)['fullname'].drop_duplicates().tolist()
            top_10_ids = summary_df.head(10)['employee_id'].tolist()

            heatmap_data_raw = results_df_display[results_df_display['employee_id'].isin(top_10_ids)].copy()
            
            names_map = summary_df.set_index('employee_id')['fullname']
            heatmap_data_raw['fullname'] = heatmap_data_raw['employee_id'].map(names_map)
            
            heatmap_data_raw['fullname'] = heatmap_data_raw['fullname'].str.strip()
            heatmap_data_raw_clean = heatmap_data_raw.dropna(subset=['fullname', 'tv_name', 'tv_match_rate'])

            if not heatmap_data_raw_clean.empty:
                try:
                    heatmap_pivot = heatmap_data_raw_clean.pivot_table(
                        index='fullname', 
                        columns='tv_name', 
                        values='tv_match_rate',
                        aggfunc='mean'
                    )

                    heatmap_pivot.index = pd.Categorical(heatmap_pivot.index, categories=top_10_names_ordered, ordered=True)
                    heatmap_pivot = heatmap_pivot.sort_index()
                    
                    heatmap_pivot_final = heatmap_pivot[heatmap_pivot.index.isin(top_10_names_ordered)]

                    if heatmap_pivot_final.empty or heatmap_pivot_final.isnull().all().all():
                        st.warning("Heatmap kosong. Data pivot ada tapi tidak ada yang cocok dengan 10 nama teratas.")
                    else:
                        
                        # Tentukan batas maksimal
                        MAX_HEATMAP_CAP = 100.0

                        # Buat DataFrame untuk WARNA 
                        heatmap_pivot_color = heatmap_pivot_final.clip(upper=MAX_HEATMAP_CAP)

                        # Buat DataFrame untuk TEKS 
                        text_labels = heatmap_pivot_final.applymap(
                            lambda x: f'{min(x, MAX_HEATMAP_CAP):.0f}' if pd.notnull(x) else ''
                        )

                        # Tampilkan heatmap
                        fig_heatmap = px.imshow(
                            heatmap_pivot_color, # <-- Data untuk WARNA (sudah dipotong)
                            text_auto=False,     # <-- Matikan teks otomatis
                            aspect="auto",
                            title="Perbandingan Skor TV (Top 10 Kandidat vs Benchmark)",
                            labels=dict(x="Talent Variable (TV)", y="Kandidat", color="Skor Match (%)"),
                            color_continuous_scale='RdYlGn',
                            range_color=[0, 100] # Atur skala warna
                        )

                        # Tambahkan TEKS yang sudah kita siapkan secara manual
                        fig_heatmap.update_traces(
                            text=text_labels, 
                            texttemplate="%{text}"
                        )
                        
                        fig_heatmap.update_xaxes(side="top")
                        fig_heatmap.update_layout(height=400, margin=dict(t=100))
                        st.plotly_chart(fig_heatmap, use_container_width=True)

                        # --- NARASI INSIGHT---
                        st.markdown("---")
                        st.markdown("##### 💡 Ringkasan Insight (Top 10)")
                        
                        try:
                            # === 1. INSIGHT KANDIDAT TERATAS ===
                            best_candidate_name = summary_df.iloc[0]['fullname']
                            best_candidate_score = summary_df.iloc[0]['final_match_rate']
                            st.write(f"🥇 **Kandidat Teratas:** **{best_candidate_name}** memimpin dengan skor kecocokan final **{best_candidate_score:.2f}%**.")

                            # === 2. INSIGHT DARI TGV (DATA RADAR) ===
                            # Menganalisis rata-rata TGV (skor grup) dari Top 10 kandidat
                            tgv_cols = ['TGV Cognitive', 'TGV Competency', 'TGV Behavioral', 'TGV Contextual']
                            # Pastikan kolom TGV ada sebelum dianalisis
                            valid_tgv_cols = [col for col in tgv_cols if col in summary_df.columns]
                            
                            if valid_tgv_cols:
                                # Hitung rata-rata skor TGV untuk 10 kandidat teratas
                                avg_tgv_scores = summary_df.head(10)[valid_tgv_cols].mean().sort_values(ascending=False)
                                
                                highest_tgv_name = avg_tgv_scores.index[0].replace('TGV ', '')
                                highest_tgv_score = avg_tgv_scores.iloc[0]
                                lowest_tgv_name = avg_tgv_scores.index[-1].replace('TGV ', '')
                                lowest_tgv_score = avg_tgv_scores.iloc[-1]
                                
                                st.write(f"📈 **Profil Grup (TGV):** Rata-rata terkuat dari Top 10 kandidat adalah di area **{highest_tgv_name}** (rata-rata skor: {highest_tgv_score:.2f}). Area terendah mereka adalah **{lowest_tgv_name}** (rata-rata skor: {lowest_tgv_score:.2f}).")
                            else:
                                st.warning("Kolom TGV tidak ditemukan di summary_df untuk insight Profil Grup.")


                            # === 3. INSIGHT DARI TV (DATA HEATMAP) ===
                            # Menganalisis data mentah heatmap 
                            num_candidates = len(heatmap_pivot_final)
                            
                            # A. Kekuatan Umum (Skor == 100)
                            strengths_count = (heatmap_pivot_final == 100).sum().sort_values(ascending=False)

                            # Temukan TV di mana >= 80% kandidat Top 10 memenuhi baseline (== 100)
                            common_strengths = strengths_count[strengths_count >= (num_candidates * 0.8)].index.tolist() 
                            
                            if common_strengths:
                                st.write(f"✅ **Kekuatan Spesifik (TV):** Sebagian besar (>=80%) kandidat Top 10 sangat kuat (skor == 100) di area: **{', '.join(common_strengths)}**.")
                            else:
                                st.write("✅ **Kekuatan Spesifik (TV):** Tidak ada kekuatan umum yang menonjol (di mana >= 80% kandidat memiliki skor 100) di antara Top 10.")

                            # B. Area Pengembangan (Skor < 50)
                            gaps_count = (heatmap_pivot_final < 50).sum().sort_values(ascending=False)
                            # Temukan TV di mana >= 30% kandidat Top 10 memiliki skor < 50
                            significant_gaps = gaps_count[gaps_count >= (num_candidates * 0.3)].head(3).index.tolist() 

                            if significant_gaps:
                                st.write(f"⚠️ **Area Pengembangan (TV):** Terdapat gap yang signifikan (skor < 50) untuk beberapa kandidat di area: **{', '.join(significant_gaps)}**.")
                            else:
                                st.write("⚠️ **Area Pengembangan (TV):** Tidak ada gap signifikan yang teridentifikasi (di mana >= 30% kandidat memiliki skor < 50).")
                            
                        except Exception as insight_e:
                            st.error(f"Tidak dapat menghasilkan ringkasan insight: {insight_e}")
                
                except Exception as e:
                    st.error(f"Error saat membuat heatmap: {e}")

            else:
                st.warning("Data tidak cukup untuk membuat heatmap (data mentah kosong setelah dibersihkan).")
        else:
            st.warning("Kolom 'employee_id' atau 'fullname' tidak ada di summary_df untuk membuat heatmap.")

        # 2. Detail Kandidat Terpilih (Radar & TV Breakdown)
        st.markdown("---")
        st.subheader("📊 Analisis Detail Kandidat")
        unique_candidates = results_df_display['employee_id'].unique()
        candidate_labels = {}
        if not df_employees.empty and 'employee_id' in df_employees.columns:
            candidate_labels = df_employees[df_employees['employee_id'].isin(unique_candidates)]\
                .set_index('employee_id')['label'].to_dict()

        selectbox_options = [candidate_labels.get(eid, eid) for eid in unique_candidates]

        if not selectbox_options:
            st.warning("Tidak ada kandidat yang ditemukan untuk ditampilkan detailnya.")
        else:
            selected_candidate_label = st.selectbox("Pilih Kandidat untuk Detail:", options=selectbox_options)
            selected_candidate_id = next((eid for eid, label in candidate_labels.items() if label == selected_candidate_label), unique_candidates[0] if len(unique_candidates)>0 else None)

            if selected_candidate_id:
                candidate_data = results_df_display[results_df_display['employee_id'] == selected_candidate_id]
                col1_detail, col2_detail = st.columns([2, 3])

                # Radar Chart
                with col1_detail: 
                    st.markdown(f"**Radar Chart:** {selected_candidate_label}")
                    tgv_data_radar = candidate_data[['tgv_name', 'tgv_match_rate']].drop_duplicates()
                    categories = ['Cognitive', 'Competency', 'Behavioral', 'Contextual']
                    candidate_scores_dict = tgv_data_radar.set_index('tgv_name')['tgv_match_rate'].to_dict()
                    
                    # Ambil skor mentah
                    candidate_scores_radar_raw = [candidate_scores_dict.get(cat, 0) for cat in categories]
                    # Bersihkan data NaN/None
                    candidate_scores_radar_clean = [s if pd.notna(s) else 0 for s in candidate_scores_radar_raw]

                    # Batasi skor di 100
                    candidate_scores_radar = [min(s, 100.0) for s in candidate_scores_radar_clean]
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(r=candidate_scores_radar, theta=categories, fill='toself', name=f'Kandidat', line=dict(color='royalblue')))
                    fig_radar.add_trace(go.Scatterpolar(r=[100] * len(categories), theta=categories, fill=None, name='Benchmark (100%)', line=dict(color='grey', dash='dash')))
                    
                    #Atur sumbu maksimal 100
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
                        showlegend=True, 
                        height=350, 
                        margin=dict(l=30, r=30, t=30, b=30), 
                        legend=dict(yanchor="bottom", y= -0.3, xanchor="center", x=0.5)
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)

                # TV Breakdown Table
                with col2_detail: 
                    st.markdown(f"**Detail Skor TV:** {selected_candidate_label}")
                    detail_cols = ['tgv_name', 'tv_name', 'baseline_score', 'user_score', 'tv_match_rate']
                    cols_to_display = [col for col in detail_cols if col in candidate_data.columns]

                    # --- [Filter baris 'None' yang tidak valid] ---
                    tv_detail_data = candidate_data.dropna(subset=['tv_name', 'baseline_score', 'user_score'])
                    
                    # --- [Batasi match rate yang ekstrem] ---
                    tv_detail_data_display = tv_detail_data.copy()
                    if 'tv_match_rate' in tv_detail_data_display.columns:
                        MAX_RATE_CAP = 100.0
                        tv_detail_data_display['tv_match_rate'] = tv_detail_data_display['tv_match_rate'].apply(lambda x: min(x, MAX_RATE_CAP))


                    st.dataframe(
                        tv_detail_data_display[cols_to_display].style.format({
                            'baseline_score': '{:.2f}', 
                            'user_score': '{:.2f}', 
                            'tv_match_rate': '{:.2f}%'
                        }).hide(axis='index'), 
                        height=360
                    )

    # Pesan jika tombol belum ditekan dan session state kosong
    elif not run_button and st.session_state.results_df.empty:
        st.info("Masukkan konfigurasi di sidebar dan klik 'Jalankan Pencocokan Talenta' untuk melihat hasil.")
    # Pesan jika tombol ditekan tapi hasil memang kosong (dari session state)
    elif run_button and st.session_state.results_df.empty:
        st.warning("Tidak ada hasil yang cocok ditemukan. Coba periksa ID benchmark atau bobot. Pastikan benchmark memiliki data yang cukup.")


    # --- AI Job Profile Generator (Sidebar - Menggunakan OpenRouter) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Job Profile Generator")
    generate_profile = st.sidebar.button("Buat Profil Pekerjaan (AI)")

    if generate_profile:
        prompt = f"Anda adalah asisten HR. Buat draf deskripsi pekerjaan (Job Description) dan profil kandidat ideal (Ideal Candidate Profile) untuk posisi:\n\n" \
            f"**Nama Peran:** {st.session_state.get('role_name', 'N/A')}\n\n" \
            f"**Level Jabatan:** {st.session_state.get('job_level', 'N/A')}\n\n" \
            f"**Tujuan Peran:** {st.session_state.get('role_purpose', 'N/A')}\n\n" \
            f"Gunakan format standar (Ringkasan, Tanggung Jawab, Kualifikasi, Keterampilan). Profil ideal harus mencerminkan karakteristik umum untuk peran ini."

        with st.spinner("Menghubungi OpenRouter AI..."):
            ai_response = call_llm_api(prompt) # Memanggil fungsi OpenRouter

        # Tampilkan hasil AI di MAIN AREA
        st.markdown("---")
        st.subheader("📄 AI Generated Job Description")
        st.markdown(ai_response)
        if "error" not in ai_response.lower() and "maaf" not in ai_response.lower():
            st.success("Profil pekerjaan berhasil dibuat oleh AI.")


    # --- Footer ---
    st.sidebar.markdown("---")
    st.sidebar.caption("Talent Match App v1.0")

# Pesan jika koneksi DB gagal di awal
else:
    st.error("Gagal terhubung ke database. Aplikasi tidak dapat dimulai.")
    st.info("Pastikan detail koneksi di file .streamlit/secrets.toml sudah benar dan server database berjalan.")
