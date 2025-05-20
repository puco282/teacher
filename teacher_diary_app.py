# (이전 전체 코드에서 필요한 부분만 발췌 및 수정 제안)
# ... (상단 코드 및 다른 탭 로직은 이전과 동일) ...

# --- 세션 상태 초기화 ---
# ... (기존 세션 상태 초기화 코드) ...
if "detail_view_selected_student" not in st.session_state: # 상세 보기 탭에서 선택된 학생
    st.session_state.detail_view_selected_student = ""
# ... (나머지 세션 상태 초기화) ...


# --- 교사용 기능 페이지 (로그인 완료 후) ---
# ... (로그인 후 페이지 상단 로직) ...
    with tab_detail_view: # "학생별 일기 상세 보기 📖" 탭
        st.header(tab_titles_main[2])
        if students_df_main.empty:
            st.warning("학생 목록 데이터가 없습니다.")
        else:
            # 학생 선택 드롭다운 (세션 상태와 연동)
            options_students = [""] + students_df_main["이름"].tolist()
            # 현재 선택된 학생으로 index 설정
            current_selection_index = 0
            if st.session_state.detail_view_selected_student in options_students:
                current_selection_index = options_students.index(st.session_state.detail_view_selected_student)
            
            st.session_state.detail_view_selected_student = st.selectbox(
                "학생 선택", 
                options=options_students,
                index=current_selection_index, # 이전에 선택한 학생 유지 또는 초기값
                key="selectbox_student_detail_view_final" 
            )
            
            name_detail_final = st.session_state.detail_view_selected_student

            if name_detail_final: # 학생이 선택된 경우
                student_info_detail_final = students_df_main[students_df_main["이름"] == name_detail_final].iloc[0]
                sheet_url_detail_final = student_info_detail_final["시트URL"]
                
                # "뒤로가기" (다른 학생 선택) 버튼
                if st.button(f"다른 학생 선택하기 (목록으로)", key=f"back_to_list_btn_{name_detail_final}"):
                    st.session_state.detail_view_selected_student = "" # 선택된 학생 초기화
                    # 필요하다면 이 학생과 관련된 다른 세션 상태 값들도 여기서 초기화
                    # 예: st.session_state.pop(f'gpt_analysis_result_{name_detail_final}', None)
                    st.rerun() # 페이지를 새로고침하여 변경사항 반영

                selected_diary_date_detail_final = st.date_input(
                    "확인할 날짜 선택", value=datetime.today(), 
                    key=f"date_select_detail_final_{name_detail_final}"
                )
                date_str_detail_final = selected_diary_date_detail_final.strftime("%Y-%m-%d")

                # ... (이하 특정 학생의 일기 로딩, 표시, 분석 버튼, 쪽지 기능 등은 이전과 동일) ...
                # (이 부분은 이전 답변의 상세 코드 내용을 그대로 사용합니다)
                # 예시:
                if not sheet_url_detail_final or not isinstance(sheet_url_detail_final, str) or not sheet_url_detail_final.startswith("http"):
                    st.error(f"'{name_detail_final}' 학생의 시트 URL이 올바르지 않습니다.")
                else:
                    try:
                        # ... (학생 데이터 로딩: student_detail_ws_final, all_entries_final, df_student_detail_final) ...
                        # ... (선택한 날짜 일기 표시 및 쪽지 로직) ...
                        # ... (GPT 분석 버튼 로직) ...
                        # ... (누적 분석 버튼 로직) ...
                        # (이전 답변의 상세 코드와 동일하게 이 부분들이 채워져야 함)
                        
                        # 임시로 상세 로직 부분은 주석 처리 (이전 코드에서 가져와야 함)
                        st.markdown(f"_{name_detail_final} 학생의 {date_str_detail_final} 일기 내용 및 분석 기능 표시 영역_")
                        st.markdown("_(이전 코드에서 이 부분의 상세 내용을 가져와 통합해야 합니다.)_")


                    except gspread.exceptions.SpreadsheetNotFound:
                        st.error(f"'{name_detail_final}' 학생의 시트 URL({sheet_url_detail_final})이 잘못되었거나 시트를 찾을 수 없습니다.")
                    except Exception as e_detail_final:
                        st.error(f"'{name_detail_final}' 학생 데이터 로딩 중 오류: {type(e_detail_final).__name__} - {e_detail_final}")
            else: # 학생이 선택되지 않은 경우 (초기 상태 또는 "뒤로가기" 이후)
                st.info("위에서 학생을 선택하여 상세 일기를 확인하세요.")
