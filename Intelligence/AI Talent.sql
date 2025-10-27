CREATE OR REPLACE FUNCTION public.fn_talent_management(
    p_bench_ids text[],
    p_weights jsonb)
    RETURNS TABLE(
        o_employee_id text, 
        o_directorate text, 
        o_role text, 
        o_grade text, 
        o_tgv_name text, 
        o_tv_name text, 
        o_baseline_score numeric, 
        o_user_score numeric, 
        o_tv_match_rate numeric, 
        o_tgv_match_rate numeric, 
        o_final_match_rate numeric
    ) 
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000
AS $BODY$
DECLARE
    latest_competency_year INTEGER;
BEGIN
    SELECT COALESCE(MAX(year), 9999) INTO latest_competency_year FROM competencies_yearly;
    RETURN QUERY
    WITH 
    -- 1. BASELINE: BENCHMARKS
    bench_employees AS (
        SELECT unnest(p_bench_ids) AS bench_employee_id
    ),

    -- 2. BASELINE: COGNITIVE
    bench_profiles_numeric AS (
        SELECT
            percentile_cont(0.5) WITHIN GROUP (ORDER BY bp.iq) AS iq_med,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY bp.pauli) AS pauli_med,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY bp.faxtor) AS faxtor_med,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY bp.gtq) AS gtq_med,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY bp.tiki) AS tiki_med
        FROM profiles_psych bp
        JOIN bench_employees be ON bp.employee_id = be.bench_employee_id
    ),

    -- 3. BASELINE: COMPETENCY (MENGGUNAKAN VARIABEL LATEST YEAR)
    bench_competencies AS (
        SELECT c.pillar_code,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY c.score) AS baseline_score
        FROM competencies_yearly c
        JOIN bench_employees be ON c.employee_id = be.bench_employee_id
        -- Menggunakan variabel yang sudah dihitung
        WHERE c.year = latest_competency_year 
        GROUP BY c.pillar_code
    ),

    -- 4. BASELINE: PAPI
    bench_papi AS (
        SELECT p.scale_code,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY p.score) AS baseline_score
        FROM papi_scores p
        JOIN bench_employees be ON p.employee_id = be.bench_employee_id
        WHERE p.scale_code = 'Papi_T'
        GROUP BY p.scale_code
    ),

    -- 5. BASELINE: STRENGTHS
    bench_strengths AS (
        SELECT array_agg(DISTINCT s.theme) AS themes
        FROM strengths s
        JOIN bench_employees be ON s.employee_id = be.bench_employee_id
        WHERE s.rank <= 5 
    ),

    -- 6. BASELINE: CONTEXTUAL (YOS)
    bench_years_median AS (
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY e.years_of_service_months) AS yrs_median
        FROM employees e
        JOIN bench_employees be ON e.employee_id = be.bench_employee_id
    ),

    -- 7. CANDIDATE FEATURES
    candidate_features AS (
        SELECT e.employee_id, dd.name AS directorate, dp.name AS role, dg.name AS grade, e.years_of_service_months,
                pp.iq, pp.pauli, pp.faxtor, pp.gtq, pp.tiki, 
                (SELECT score FROM papi_scores WHERE employee_id = e.employee_id AND scale_code = 'Papi_T') AS papi_t,
                (SELECT score FROM competencies_yearly WHERE employee_id = e.employee_id AND pillar_code = 'IDS' AND year = latest_competency_year) AS ids,
                (SELECT score FROM competencies_yearly WHERE employee_id = e.employee_id AND pillar_code = 'QDD' AND year = latest_competency_year) AS qdd,
                (SELECT score FROM competencies_yearly WHERE employee_id = e.employee_id AND pillar_code = 'GDR' AND year = latest_competency_year) AS gdr,
                (SELECT array_agg(s.theme ORDER BY s.rank) FROM strengths s WHERE s.employee_id = e.employee_id AND s.rank <= 5) AS strengths_top
        FROM employees e
        LEFT JOIN dim_directorates dd ON e.directorate_id = dd.directorate_id
        LEFT JOIN dim_positions dp ON e.position_id = dp.position_id
        LEFT JOIN dim_grades dg ON e.grade_id = dg.grade_id
        LEFT JOIN profiles_psych pp ON e.employee_id = pp.employee_id
    ),

    -- 8. TV MATCH RATE
    tv_matches AS (
        SELECT 
            cf.*,
            CASE WHEN bpn.iq_med IS NOT NULL AND cf.iq IS NOT NULL THEN LEAST(100.0, (cf.iq::numeric / bpn.iq_med::numeric) * 100.0) END AS tv_iq_match,
            CASE WHEN bpn.pauli_med IS NOT NULL AND cf.pauli IS NOT NULL THEN LEAST(100.0, (cf.pauli::numeric / bpn.pauli_med::numeric) * 100.0) END AS tv_pauli_match,
            CASE WHEN bpn.faxtor_med IS NOT NULL AND cf.faxtor IS NOT NULL THEN LEAST(100.0, (cf.faxtor::numeric / bpn.faxtor_med::numeric) * 100.0) END AS tv_faxtor_match,
            CASE WHEN bpn.gtq_med IS NOT NULL AND cf.gtq IS NOT NULL THEN LEAST(100.0, (cf.gtq::numeric / bpn.gtq_med::numeric) * 100.0) END AS tv_gtq_match,
            CASE WHEN bpn.tiki_med IS NOT NULL AND cf.tiki IS NOT NULL THEN LEAST(100.0, (cf.tiki::numeric / bpn.tiki_med::numeric) * 100.0) END AS tv_tiki_match,
            -- Skor Kompetensi dan Behavioral/Strengths tidak dibatasi 100%, biarkan data mentah
            CASE WHEN bc_ids.baseline_score IS NOT NULL AND cf.ids IS NOT NULL THEN (cf.ids::numeric / bc_ids.baseline_score::numeric) * 100.0 END AS tv_ids_match,
            CASE WHEN bc_qdd.baseline_score IS NOT NULL AND cf.qdd IS NOT NULL THEN (cf.qdd::numeric / bc_qdd.baseline_score::numeric) * 100.0 END AS tv_qdd_match,
            CASE WHEN bc_gdr.baseline_score IS NOT NULL AND cf.gdr IS NOT NULL THEN (cf.gdr::numeric / bc_gdr.baseline_score::numeric) * 100.0 END AS tv_gdr_match,
            CASE WHEN bpp_t.baseline_score IS NOT NULL AND cf.papi_t IS NOT NULL THEN (cf.papi_t::numeric / bpp_t.baseline_score::numeric) * 100.0 END AS tv_papi_t_match,
            CASE WHEN cf.strengths_top IS NULL THEN NULL ELSE (
                SELECT (COUNT(*)::numeric / LEAST(5, CARDINALITY(cf.strengths_top))::numeric * 100.0)
                FROM unnest(cf.strengths_top) s(theme)
                JOIN bench_strengths bs ON s.theme = ANY (bs.themes)
            ) END AS tv_strengths_match
        FROM candidate_features cf
        CROSS JOIN bench_profiles_numeric bpn
        LEFT JOIN bench_competencies bc_ids ON bc_ids.pillar_code = 'IDS'
        LEFT JOIN bench_competencies bc_qdd ON bc_qdd.pillar_code = 'QDD'
        LEFT JOIN bench_competencies bc_gdr ON bc_gdr.pillar_code = 'GDR'
        LEFT JOIN bench_papi bpp_t ON bpp_t.scale_code = 'Papi_T'
    ),

    -- 9. TGV MATCH RATE
    tgv_agg AS (
        SELECT
            t.*, bym.yrs_median, 
            (COALESCE(tv_iq_match,0) + COALESCE(tv_pauli_match,0) + COALESCE(tv_faxtor_match,0) + COALESCE(tv_gtq_match,0) + COALESCE(tv_tiki_match,0))
             / NULLIF((CASE WHEN tv_iq_match IS NOT NULL THEN 1 ELSE 0 END) + (CASE WHEN tv_pauli_match IS NOT NULL THEN 1 ELSE 0 END) + (CASE WHEN tv_faxtor_match IS NOT NULL THEN 1 ELSE 0 END) + (CASE WHEN tv_gtq_match IS NOT NULL THEN 1 ELSE 0 END) + (CASE WHEN tv_tiki_match IS NOT NULL THEN 1 ELSE 0 END),0)::numeric
             AS tgv_cognitive_match_rate,
            (COALESCE(tv_ids_match,0) + COALESCE(tv_qdd_match,0) + COALESCE(tv_gdr_match,0))
             / NULLIF((CASE WHEN tv_ids_match IS NOT NULL THEN 1 ELSE 0 END) + (CASE WHEN tv_qdd_match IS NOT NULL THEN 1 ELSE 0 END) + (CASE WHEN tv_gdr_match IS NOT NULL THEN 1 ELSE 0 END),0)::numeric
             AS tgv_competency_match_rate,
            (COALESCE(tv_papi_t_match,0) + COALESCE(tv_strengths_match,0))
             / NULLIF((CASE WHEN tv_papi_t_match IS NOT NULL THEN 1 ELSE 0 END) + (CASE WHEN tv_strengths_match IS NOT NULL THEN 1 ELSE 0 END),0)::numeric
             AS tgv_behavioral_match_rate
        FROM tv_matches t
        CROSS JOIN bench_years_median bym
    ),

    -- 10. FINAL SCORES
    final_scores AS (
        SELECT
            t.*, 
            GREATEST(0, 100 - ABS(t.years_of_service_months - t.yrs_median) * 1.5)::numeric AS tgv_contextual_match_rate,
            p_weights AS weights_config
        FROM tgv_agg t
    ),

    -- 11. Temukan TOP 10 Karyawan
    top_10_employees AS (
        SELECT
            employee_id,
            ROUND( (
                COALESCE(fs.tgv_cognitive_match_rate,0) * (fs.weights_config->>'cognitive')::numeric +
                COALESCE(fs.tgv_competency_match_rate,0) * (fs.weights_config->>'competency')::numeric +
                COALESCE(fs.tgv_behavioral_match_rate,0) * (fs.weights_config->>'behavioral')::numeric +
                COALESCE(fs.tgv_contextual_match_rate,0) * (fs.weights_config->>'contextual')::numeric
            )::numeric / (
                (fs.weights_config->>'cognitive')::numeric +
                (fs.weights_config->>'competency')::numeric +
                (fs.weights_config->>'behavioral')::numeric +
                (fs.weights_config->>'contextual')::numeric
            )::numeric, 2) AS final_match_rate
        FROM final_scores fs
        ORDER BY final_match_rate DESC
        LIMIT 10
    ),

    -- 12. FINAL BREAKDOWN (Hanya untuk 10 Karyawan Teratas)
    final_breakdown AS (
        SELECT 
            fs.employee_id, fs.directorate, fs.role, fs.grade,
            t10.final_match_rate, 
            CASE tv.tv_name
              WHEN 'IQ' THEN 'Cognitive' WHEN 'Pauli' THEN 'Cognitive' WHEN 'Faxtor' THEN 'Cognitive' WHEN 'GTQ' THEN 'Cognitive'
              WHEN 'Tiki' THEN 'Cognitive'
              WHEN 'IDS' THEN 'Competency' WHEN 'QDD' THEN 'Competency' WHEN 'GDR' THEN 'Competency'
              WHEN 'PAPI_T' THEN 'Behavioral' WHEN 'Strengths' THEN 'Behavioral'
              WHEN 'Years_of_Service' THEN 'Contextual'
            END AS tgv_name,
            tv.tv_name,
            ROUND(tv.baseline_score::numeric, 2) AS baseline_score,
            ROUND(tv.user_score::numeric, 2) AS user_score,
            ROUND(tv.tv_match_rate::numeric, 2) AS tv_match_rate,
            ROUND(tv.tgv_match_rate::numeric, 2) AS tgv_match_rate

        FROM final_scores fs
        JOIN top_10_employees t10 ON fs.employee_id = t10.employee_id
        
        CROSS JOIN LATERAL (VALUES
            ('IQ', (SELECT iq_med FROM bench_profiles_numeric), fs.iq::numeric, fs.tv_iq_match::numeric, fs.tgv_cognitive_match_rate),
            ('Pauli', (SELECT pauli_med FROM bench_profiles_numeric), fs.pauli::numeric, fs.tv_pauli_match::numeric, fs.tgv_cognitive_match_rate),
            ('Faxtor', (SELECT faxtor_med FROM bench_profiles_numeric), fs.faxtor::numeric, fs.tv_faxtor_match::numeric, fs.tgv_cognitive_match_rate),
            ('GTQ', (SELECT gtq_med FROM bench_profiles_numeric), fs.gtq::numeric, fs.tv_gtq_match::numeric, fs.tgv_cognitive_match_rate),
            ('Tiki', (SELECT tiki_med FROM bench_profiles_numeric), fs.tiki::numeric, fs.tv_tiki_match::numeric, fs.tgv_cognitive_match_rate),
            ('IDS', (SELECT baseline_score FROM bench_competencies WHERE pillar_code='IDS'), fs.ids::numeric, fs.tv_ids_match::numeric, fs.tgv_competency_match_rate),
            ('QDD', (SELECT baseline_score FROM bench_competencies WHERE pillar_code='QDD'), fs.qdd::numeric, fs.tv_qdd_match::numeric, fs.tgv_competency_match_rate),
            ('GDR', (SELECT baseline_score FROM bench_competencies WHERE pillar_code='GDR'), fs.gdr::numeric, fs.tv_gdr_match::numeric, fs.tgv_competency_match_rate),
            ('PAPI_T', (SELECT baseline_score FROM bench_papi WHERE scale_code='Papi_T'), fs.papi_t::numeric, fs.tv_papi_t_match::numeric, fs.tgv_behavioral_match_rate),
            ('Strengths', NULL, NULL, fs.tv_strengths_match::numeric, fs.tgv_behavioral_match_rate),
            ('Years_of_Service', fs.yrs_median, fs.years_of_service_months::numeric, fs.tgv_contextual_match_rate, fs.tgv_contextual_match_rate)
        ) AS tv(tv_name, baseline_score, user_score, tv_match_rate, tgv_match_rate)
    )
    -- FINAL SELECT
    SELECT 
        fb.employee_id, fb.directorate, fb.role, fb.grade, fb.tgv_name, fb.tv_name,
        fb.baseline_score, fb.user_score, fb.tv_match_rate, fb.tgv_match_rate, fb.final_match_rate
    FROM final_breakdown fb
    WHERE fb.tgv_name IS NOT NULL AND fb.tv_match_rate IS NOT NULL
    ORDER BY fb.final_match_rate DESC, fb.employee_id, fb.tgv_name, fb.tv_name;

END;
$BODY$;

