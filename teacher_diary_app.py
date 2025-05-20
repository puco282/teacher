# --- 세션 상태 초기화 ---
session_defaults = {
    "teacher_logged_in": False,
    "all_students_today_data_loaded": False,
    "all_students_today_data": [],
    "detail_view_selected_student": "",
    "tab3_student_data_cache": {}  # 탭3 학생 데이터 캐시용 딕셔너리 추가
}
for k, v_init in session_defaults.items():
    if k not in st.session_state: st.session_state[k] = v_init

# --- 교사용 로그인 페이지 ---
if not st.session_state.teacher_logged_in:
    # ... (로그인 로직) ...
    if st.button("로그인", key="admin_login_btn_vfinal_login_cache"):
        if admin_pw == st.secrets.get("ADMIN_TEACHER_PASSWORD", "silverline"):
            st.session_state.teacher_logged_in = True
            # 로그인 시 모든 캐시 및 관련 세션 상태 초기화
            for k_reset, v_reset in session_defaults.items():
                 if k_reset != "teacher_logged_in": st.session_state[k_reset] = v_reset
            st.cache_data.clear() # Streamlit의 기본 데이터 캐시도 클리어
            st.rerun()
        # ...
else: # --- 교사용 기능 페이지 ---
    # ... (사이드바 로그아웃 및 새로고침 버튼 로직에 tab3_student_data_cache 초기화 추가) ...
    if st.sidebar.button("로그아웃", key="logout_vfinal_consolidated_cache"):
        for k_reset, v_reset in session_defaults.items(): st.session_state[k_reset] = v_reset
        # st.session_state.tab3_student_data_cache = {} # session_defaults에 포함되어 이미 처리됨
        st.cache_data.clear()
        st.rerun()
    
    if st.sidebar.button("오늘 학생 데이터 새로고침 ♻️", key="refresh_data_vfinal_consolidated_cache"):
        st.session_state.all_students_today_data_loaded = False
        st.session_state.tab3_student_data_cache = {} # 탭3 상세 보기 캐시도 초기화
        st.cache_data.clear()
        st.rerun()

    # ... (탭1, 탭2 로직은 이전과 동일) ...

    with tab3: # 학생별 일기 상세 보기
        st.header(tab_titles_list[2])
        if students_df.empty: st.warning("학생 목록을 먼저 불러오세요 (오류 시 '학생목록' 시트 점검).")
        else:
            # ... (학생 선택 드롭다운 로직은 이전과 동일, st.session_state.detail_view_selected_student 사용) ...
            student_options_tab3 = [""] + students_df["이름"].tolist()
            sel_student_idx_tab3 = 0
            if st.session_state.detail_view_selected_student in student_options_tab3:
                sel_student_idx_tab3 = student_options_tab3.index(st.session_state.detail_view_selected_student)
            
            st.session_state.detail_view_selected_student = st.selectbox(
                "학생 선택", options=student_options_tab3, index=sel_student_idx_tab3, 
                key="sel_student_tab3_final_v3_cache"
            )
            selected_student_name_tab3 = st.session_state.detail_view_selected_student

            if selected_student_name_tab3:
                student_info_tab3 = students_df[students_df["이름"] == selected_student_name_tab3].iloc[0]
                s_name_tab3 = student_info_tab3["이름"]
                s_url_tab3 = student_info_tab3["시트URL"]
                
                # ... ("다른 학생 선택" 버튼 및 날짜 선택 input 로직은 이전과 동일) ...
                back_btn_col_tab3, date_input_col_tab3 = st.columns([1,3])
                with back_btn_col_tab3:
                    if st.button(f"다른 학생 선택", key=f"back_btn_tab3_final_cache_{s_name_tab3}"):
                        st.session_state.detail_view_selected_student = ""
                        # 현재 학생 캐시는 유지해도 되지만, 원한다면 여기서 삭제 가능
                        # st.session_state.tab3_student_data_cache.pop(s_name_tab3, None) 
                        st.rerun()
                with date_input_col_tab3:
                    date_selected_tab3 = st.date_input("날짜 선택", value=datetime.today(), 
                                                        key=f"date_pick_tab3_final_cache_{s_name_tab3}")
                date_str_selected_tab3 = date_selected_tab3.strftime("%Y-%m-%d")


                if not s_url_tab3 or not isinstance(s_url_tab3, str) or not s_url_tab3.startswith("http"):
                    st.error(f"'{s_name_tab3}' 학생의 시트 URL이 올바르지 않습니다.")
                else:
                    df_student_all_records_tab3 = None
                    ws_student_for_detail_tab3 = None # 쪽지 저장 시점에만 필요할 수 있음
                    all_records_for_student_tab3 = [] # 쪽지 저장 시 행 인덱싱을 위해 사용 가능

                    # 1. 캐시에서 데이터 가져오기 시도
                    if s_name_tab3 in st.session_state.tab3_student_data_cache:
                        df_student_all_records_tab3 = st.session_state.tab3_student_data_cache[s_name_tab3]['df']
                        all_records_for_student_tab3 = st.session_state.tab3_student_data_cache[s_name_tab3]['list']
                        # st.info(f"'{s_name_tab3}' 학생의 캐시된 데이터를 사용합니다.") # 디버깅용 메시지
                    
                    # 2. 캐시에 없으면 새로 로드
                    if df_student_all_records_tab3 is None:
                        try:
                            with st.spinner(f"'{s_name_tab3}' 학생의 전체 일기 기록 로딩 중... (API 호출)"):
                                ws_temp = g_client_main.open_by_url(s_url_tab3).sheet1
                                all_records_for_student_tab3 = get_records_from_row2_header(ws_temp, EXPECTED_STUDENT_SHEET_HEADER)
                                df_student_all_records_tab3 = pd.DataFrame(all_records_for_student_tab3)
                                # 데이터프레임과 원본 리스트(행 인덱싱용) 함께 캐시
                                st.session_state.tab3_student_data_cache[s_name_tab3] = {
                                    'df': df_student_all_records_tab3,
                                    'list': all_records_for_student_tab3 
                                }
                        except Exception as e_load_detail:
                            st.error(f"'{s_name_tab3}' 학생 데이터 로딩 중 오류: {e_load_detail}")
                            df_student_all_records_tab3 = pd.DataFrame() # 오류 시 빈 데이터프레임
                            all_records_for_student_tab3 = []


                    # --- 이제 df_student_all_records_tab3 (캐시되었거나 새로 로드된)를 사용하여 UI 구성 ---
                    if df_student_all_records_tab3.empty or "날짜" not in df_student_all_records_tab3.columns:
                        st.warning(f"'{s_name_tab3}' 학생 시트에 데이터가 없거나 '날짜' 열이 없습니다.")
                    else:
                        # (선택한 날짜의 일기 표시, 누적 분석 버튼, GPT 분석 버튼, 선생님 쪽지 로직은 이전과 동일)
                        # 단, 쪽지 저장 시에는 캐시를 무효화해야 함.
                        # 예시: 쪽지 저장 로직
                        entry_df_for_date_tab3 = df_student_all_records_tab3[df_student_all_records_tab3["날짜"] == date_str_selected_tab3]
                        if not entry_df_for_date_tab3.empty:
                            diary_entry_to_show_tab3 = entry_df_for_date_tab3.iloc[0]
                            # ... (일기 내용 표시) ...
                            teacher_note_current_val_tab3 = diary_entry_to_show_tab3.get('선생님 쪽지', '')
                            teacher_note_input_val_tab3 = st.text_area(f"✏️ 쪽지 작성/수정", 
                                                                       value=teacher_note_current_val_tab3, 
                                                                       key=f"note_input_final_key_cache_{s_name_tab3}_{date_str_selected_tab3}")
                            
                            if st.button(f"💾 쪽지 저장", key=f"save_note_btn_final_key_cache_{s_name_tab3}_{date_str_selected_tab3}"):
                                if not teacher_note_input_val_tab3.strip() and not teacher_note_current_val_tab3:
                                    st.warning("쪽지 내용이 비어있습니다.")
                                else:
                                    try:
                                        # 쪽지 저장을 위해 워크시트 객체 다시 열기 (안전하게)
                                        with st.spinner("쪽지 저장 중..."):
                                            ws_to_update = g_client_main.open_by_url(s_url_tab3).sheet1
                                            # 원본 리스트(all_records_for_student_tab3)에서 행 인덱스 찾기
                                            row_idx_for_update_tab3, sheet_headers_tab3 = -1, ws_to_update.row_values(2)
                                            for i_entry, r_entry_data in enumerate(all_records_for_student_tab3): # 캐시된 리스트 사용
                                                if r_entry_data.get("날짜") == date_str_selected_tab3:
                                                    row_idx_for_update_tab3 = i_entry + 3; break
                                            
                                            if row_idx_for_update_tab3 != -1:
                                                note_col_idx_update_tab3 = sheet_headers_tab3.index("선생님 쪽지") + 1 if "선생님 쪽지" in sheet_headers_tab3 else 5
                                                ws_to_update.update_cell(row_idx_for_update_tab3, note_col_idx_update_tab3, teacher_note_input_val_tab3)
                                                
                                                # 캐시 무효화
                                                if s_name_tab3 in st.session_state.tab3_student_data_cache:
                                                    del st.session_state.tab3_student_data_cache[s_name_tab3]
                                                st.success(f"쪽지를 저장했습니다! 데이터가 새로고침됩니다.")
                                                st.cache_data.clear() # 전체 데이터 캐시도 한번 클리어 (선택적)
                                                st.rerun()
                                            else:
                                                st.error("쪽지를 저장할 해당 날짜의 일기 항목을 찾지 못했습니다.")
                                        except Exception as e_save_note_final_cache:
                                            st.error(f"쪽지 저장 중 오류: {e_save_note_final_cache}")
                        # ... (이하 GPT 분석, 누적 분석 버튼 로직은 이전과 동일하게 df_student_all_records_tab3 사용) ...

            # ... (학생 미선택 시 로직은 이전과 동일) ...
