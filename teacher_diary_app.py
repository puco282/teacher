import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from openai import OpenAI

# --- 상수 정의 ---
EXPECTED_STUDENT_SHEET_HEADER = ["날짜", "감정", "감사한 일", "하고 싶은 말", "선생님 쪽지"]
EMOTION_GROUPS = ["😀 긍정", "😐 보통", "😢 부정"]
FONT_PATH = "NanumGothic.ttf"

GPT_CUMULATIVE_SYSTEM_PROMPT = """
당신은 초등학교 학생들의 심리 및 상담 분야에서 깊은 전문성을 가진 AI 상담 보조입니다. 
당신의 분석은 인간 중심 상담 이론, 인지 행동 이론 등 실제 상담 이론에 기반해야 합니다. 
제공되는 한 학생의 **누적된 익명 일기 기록 전체 (감정, 감사한 일, 하고 싶은 말)**를 바탕으로, 다음 6가지 항목에 대해 구체적이고 통찰력 있는 리포트를 선생님께 제공해주세요. 
학생의 이름이나 개인 식별 정보는 절대 언급하지 마세요. 답변은 명확한 항목 구분을 위해 마크다운 헤더를 사용해주세요.

제공될 누적 일기 내용 형식 (실제 내용은 아래 데이터 형식으로 전달됩니다):
- 전체 감정 기록: [날짜1: 감정1, 날짜2: 감정2, ...]
- 전체 감사한 일 기록: [날짜1: 감사한 일1, 날짜2: 감사한 일2, ...]
- 전체 하고 싶은 말 기록: [날짜1: 하고 싶은 말1, 날짜2: 하고 싶은 말2, ...]

학생의 누적 기록 데이터:
{cumulative_diary_data_for_gpt}

리포트 항목:
1.  **누적된 감정 기록 분석 및 가장 보편적인 감정 상태**: 제공된 모든 '감정' 기록을 바탕으로 학생이 가장 자주 표현하는 감정(들)은 무엇인지, 긍정/부정/중립 감정의 전반적인 비율이나 경향성은 어떠한지, 그리고 이를 통해 파악할 수 있는 학생의 가장 보편적인 감정 상태에 대해 분석해주세요.
2.  **문체 및 표현 특성 (누적 기록 기반)**: 누적된 글 전체에서 나타나는 학생의 언어 수준(초등학생 수준 고려), 문장 구조의 복잡성, 자기 생각이나 감정을 얼마나 명확하고 풍부하게 표현하고 있는지, 주관적 감정 표현의 강도는 어떠한지 등을 평가해주세요.
3.  **주요 키워드 및 주제 추출 (누적 기록 기반)**: 누적된 글 전체에서 반복적으로 나타나거나 중요하다고 판단되는 핵심 단어(키워드)를 3-5개 추출하고, 이를 통해 학생의 주요 관심사, 자주 언급되는 대상(예: 친구, 가족, 특정 활동 등), 또는 반복되는 상황이나 사건을 파악해주세요.
4.  **누적된 기록에 대한 종합 요약**: 위 분석들을 바탕으로 누적된 기록 전체를 통해 파악할 수 있는 학생의 생각, 경험, 감정의 핵심적인 패턴이나 내용을 간결하게 요약해주세요.
5.  **관찰 및 변화 추이 (누적 기록 기반)**: 누적된 기록을 통해 학생의 감정, 생각, 표현 방식 등에서 시간에 따른 변화나 일관된 패턴이 있다면 언급해주세요. 선생님께서 앞으로 이 학생의 어떤 면을 좀 더 관심 있게 지켜보면 좋을지, 또는 어떤 긍정적/부정적 변화의 가능성이 엿보이는지 구체적으로 언급해주세요.
6.  **선생님을 위한 상담적 조언 (누적 기록 기반)**: 누적된 기록에서 파악된 학생의 특성을 바탕으로, 선생님께서 이 학생을 지지하고 돕기 위해 활용할 수 있는 인간 중심적 또는 인지 행동적 접근 방식에 기반한 구체적인 상담 전략이나 소통 방법을 1-2가지 제안해주세요.
"""

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="감정 일기장 (교사용)", page_icon="🧑‍🏫", layout="wide")

# --- Helper Functions ---
@st.cache_resource
def authorize_gspread():
    try:
        google_creds_dict = st.secrets["GOOGLE_CREDENTIALS"]
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(google_creds_dict, scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Google API 인증 중 오류: {e}. '.streamlit/secrets.toml' 설정을 확인하세요."); st.stop(); return None

@st.cache_data(ttl=600)
def get_students_df(_client_gspread):
    if not _client_gspread: return pd.DataFrame()
    try:
        ws = _client_gspread.open("학생목록").sheet1
        df = pd.DataFrame(ws.get_all_records())
        if not df.empty and ("이름" not in df.columns or "시트URL" not in df.columns):
            st.error("'학생목록' 시트에 '이름'/'시트URL' 열이 없습니다."); return pd.DataFrame()
        return df
    except Exception as e: st.error(f"학생 목록 로딩 오류: {e}"); return pd.DataFrame()

def get_records_from_row2_header(worksheet, expected_headers):
    all_values = worksheet.get_all_values()
    if len(all_values) < 2: return []
    header_row_actual = all_values[1] # 실제 시트의 2번째 행 (디버깅용)
    data_rows = all_values[2:]
    records = []
    num_expected_headers = len(expected_headers)
    for r_vals in data_rows:
        rec = {}
        for i, header_name in enumerate(expected_headers):
            if i < len(r_vals):
                rec[header_name] = r_vals[i]
            else:
                rec[header_name] = None # 데이터 행이 예상 헤더보다 짧으면 None으로 채움
        records.append(rec)
    return records

@st.cache_data(ttl=300)
def fetch_all_students_today_data(_students_df, today_str, _client_gspread, headers_list):
    all_data = []
    if _students_df.empty: return all_data
    prog_bar = st.progress(0, text="전체 학생 오늘 자 요약 정보 로딩 중... (0%)")
    total = len(_students_df)
    for i, (_, row) in enumerate(_students_df.iterrows()):
        name, url = row["이름"], row["시트URL"]
        entry = {"name": name, "emotion_today": None, "message_today": None, "error": None}
        prog_val = (i + 1) / total
        prog_bar.progress(prog_val, text=f"'{name}' 학생 정보 확인 중... ({int(prog_val*100)}%)")
        if not url or not isinstance(url, str) or not url.startswith("http"):
            entry["error"] = "시트 URL 형식 오류"; all_data.append(entry); continue
        try:
            ws = _client_gspread.open_by_url(url).sheet1
            recs = get_records_from_row2_header(ws, headers_list)
            found = False
            for r in recs:
                if r.get("날짜") == today_str:
                    entry["emotion_today"], entry["message_today"] = r.get("감정"), r.get("하고 싶은 말")
                    found = True; break
            if not found: entry["error"] = "오늘 일기 없음"
        except gspread.exceptions.GSpreadException as ge: # 구체적인 gspread 오류 처리
            entry["error"] = f"시트 접근/파싱 오류 ({type(ge).__name__})"
        except Exception as e: entry["error"] = f"알 수 없는 오류 ({type(e).__name__})"
        all_data.append(entry)
    prog_bar.empty(); return all_data

# --- OpenAI API 클라이언트 초기화 ---
client_openai = None
openai_api_key_value = st.secrets.get("OPENAI_API_KEY")
if openai_api_key_value:
    try: client_openai = OpenAI(api_key=openai_api_key_value)
    except Exception as e: st.warning(f"OpenAI 클라이언트 초기화 오류: {e} (GPT 기능 사용 불가)")

# --- 세션 상태 초기화 ---
session_defaults = {
    "teacher_logged_in": False, "all_students_today_data_loaded": False,
    "all_students_today_data": [], "detail_view_selected_student": ""
}
for k, v_init in session_defaults.items():
    if k not in st.session_state: st.session_state[k] = v_init

# --- 교사용 로그인 ---
if not st.session_state.teacher_logged_in:
    st.title("🧑‍🏫 감정일기 로그인 (교사용)")
    admin_pw = st.text_input("관리자 비밀번호", type="password", key="admin_pw_vfinal_login")
    if st.button("로그인", key="admin_login_btn_vfinal_login"):
        if admin_pw == st.secrets.get("ADMIN_TEACHER_PASSWORD", "silverline"):
            st.session_state.teacher_logged_in = True
            for k_reset, v_reset in session_defaults.items(): # 로그인 시 관련 세션 상태 초기화
                 if k_reset != "teacher_logged_in": st.session_state[k_reset] = v_reset
            st.rerun()
        else: st.error("비밀번호가 올바르지 않습니다.")
else: # --- 교사용 기능 페이지 ---
    g_client_main = authorize_gspread()
    students_df = get_students_df(g_client_main)

    st.sidebar.title("🧑‍🏫 교사 메뉴")
    if st.sidebar.button("로그아웃", key="logout_vfinal_consolidated"):
        for k_reset, v_reset in session_defaults.items(): st.session_state[k_reset] = v_reset
        st.rerun()
    if st.sidebar.button("오늘 학생 데이터 새로고침 ♻️", key="refresh_data_vfinal_consolidated"):
        st.session_state.all_students_today_data_loaded = False
        st.cache_data.clear(); st.rerun()

    st.title("🧑‍🏫 교사용 대시보드")

    if not st.session_state.all_students_today_data_loaded:
        if students_df.empty and g_client_main: # 학생 목록 로드 시도 후 비었는지 확인
            st.warning("'학생목록' 시트가 비었거나, 접근 권한 또는 내용을 확인해주세요. 문제가 지속되면 새로고침 버튼을 사용하세요.")
            st.session_state.all_students_today_data = []
            st.session_state.all_students_today_data_loaded = True
        elif not g_client_main: # gspread 클라이언트 인증 실패 시
             pass # authorize_gspread 함수에서 이미 st.error와 st.stop() 호출
        else: # 학생 목록은 있는데, 아직 요약 데이터 로드 안한 경우
            today_str_main = datetime.today().strftime("%Y-%m-%d")
            st.session_state.all_students_today_data = fetch_all_students_today_data(
                students_df, today_str_main, g_client_main, EXPECTED_STUDENT_SHEET_HEADER)
            st.session_state.all_students_today_data_loaded = True
            if st.session_state.all_students_today_data or students_df.empty: # 데이터가 있거나, 학생 자체가 없어서 빈 경우
                 st.success("오늘 자 학생 요약 정보 로드가 완료되었습니다!")

    current_summary_data = st.session_state.get("all_students_today_data", [])
    
    tab_titles_list = ["오늘의 학급 감정 분포 📊", "학생들이 전달하는 메시지 💌", "학생별 일기 상세 보기 📖"]
    tab1, tab2, tab3 = st.tabs(tab_titles_list)

    with tab1: # 오늘의 학급 감정 분포
        st.header(tab_titles_list[0])
        st.markdown(f"**조회 날짜:** {datetime.today().strftime('%Y-%m-%d')}")
        if not current_summary_data: st.info("요약할 학생 데이터가 없습니다. (학생목록 확인 또는 데이터 새로고침)")
        else:
            emotion_cats_tab1 = {g: [] for g in EMOTION_GROUPS}
            emotion_cats_tab1.update({"감정 미분류": [], "일기 미제출 또는 오류": []})
            for data_item in current_summary_data:
                s_name = data_item["name"]
                if data_item["error"] and data_item["error"] != "오늘 일기 없음": emotion_cats_tab1["일기 미제출 또는 오류"].append(f"{s_name} ({data_item['error']})")
                elif data_item["error"] == "오늘 일기 없음" or not data_item["emotion_today"]: emotion_cats_tab1["일기 미제출 또는 오류"].append(s_name)
                elif data_item["emotion_today"] and isinstance(data_item["emotion_today"], str) and " - " in data_item["emotion_today"]:
                    main_emotion_cat = data_item["emotion_today"].split(" - ")[0].strip()
                    if main_emotion_cat in EMOTION_GROUPS: emotion_cats_tab1[main_emotion_cat].append(s_name)
                    else: emotion_cats_tab1["감정 미분류"].append(f"{s_name} (감정: {data_item['emotion_today']})")
                else: emotion_cats_tab1["일기 미제출 또는 오류"].append(f"{s_name} (감정 형식 오류: {data_item['emotion_today']})")
            
            cols_tab1_overview = st.columns(len(EMOTION_GROUPS))
            for i, grp_name in enumerate(EMOTION_GROUPS):
                with cols_tab1_overview[i]:
                    st.subheader(f"{grp_name} ({len(emotion_cats_tab1[grp_name])}명)")
                    if emotion_cats_tab1[grp_name]: st.markdown("\n".join([f"- {n}" for n in sorted(emotion_cats_tab1[grp_name])]))
                    else: st.info("이 감정을 느낀 학생이 없습니다.")
            
            exp_col1_tab1, exp_col2_tab1 = st.columns(2)
            with exp_col1_tab1:
                if emotion_cats_tab1["일기 미제출 또는 오류"]:
                    with st.expander(f"📝 일기 미제출/오류 ({len(emotion_cats_tab1['일기 미제출 또는 오류'])}명)"):
                        st.markdown("\n".join([f"- {s}" for s in sorted(emotion_cats_tab1["일기 미제출 또는 오류"])]))
            with exp_col2_tab1:
                if emotion_cats_tab1["감정 미분류"]:
                    with st.expander(f"🤔 감정 미분류 ({len(emotion_cats_tab1['감정 미분류'])}명)"):
                        st.markdown("\n".join([f"- {s}" for s in sorted(emotion_cats_tab1["감정 미분류"])]))
    
    with tab2: # 학생들이 전달하는 메시지
        st.header(tab_titles_list[1])
        if not current_summary_data: st.info("요약할 학생 데이터가 없습니다.")
        else:
            neg_feedback, other_feedback = [], []
            for data_item in current_summary_data:
                if data_item["error"] or not data_item["emotion_today"] or not data_item["message_today"] or not data_item["message_today"].strip(): continue
                emo_full_str = data_item["emotion_today"]
                if not isinstance(emo_full_str, str) or " - " not in emo_full_str: continue
                item_data = {"name": data_item["name"], "emotion": emo_full_str, "message": data_item["message_today"].strip()}
                if emo_full_str.split(" - ")[0].strip() == "😢 부정": neg_feedback.append(item_data)
                elif emo_full_str.split(" - ")[0].strip() in ["😀 긍정", "😐 보통"]: other_feedback.append(item_data)
            
            if not neg_feedback and not other_feedback: st.success("오늘 선생님이나 친구들에게 하고 싶은 말을 적은 학생이 없습니다. 😊")
            else:
                st.subheader("😥 부정적 감정 학생들의 메시지")
                if neg_feedback:
                    for item_msg in sorted(neg_feedback, key=lambda x: x['name']):
                        with st.container(border=True): st.markdown(f"**학생명:** {item_msg['name']} (<span style='color:red;'>{item_msg['emotion']}</span>)\n\n**메시지:**\n> {item_msg['message']}", unsafe_allow_html=True)
                else: st.info("오늘, 부정적인 감정과 함께 메시지를 남긴 학생은 없습니다.")
                st.markdown("---")
                st.subheader("😊 그 외 감정 학생들의 메시지")
                if other_feedback:
                    for item_msg in sorted(other_feedback, key=lambda x: x['name']):
                        with st.container(border=True): st.markdown(f"**학생명:** {item_msg['name']} ({item_msg['emotion']})\n\n**메시지:**\n> {item_msg['message']}")
                else: st.info("오늘, 긍정적 또는 보통 감정과 함께 메시지를 남긴 학생은 없습니다.")

    with tab3: # 학생별 일기 상세 보기
        st.header(tab_titles_list[2])
        if students_df.empty: st.warning("학생 목록을 먼저 불러오세요 (오류 시 '학생목록' 시트 점검).")
        else:
            student_options = [""] + students_df["이름"].tolist()
            sel_student_idx = 0
            if st.session_state.detail_view_selected_student in student_options:
                sel_student_idx = student_options.index(st.session_state.detail_view_selected_student)
            
            st.session_state.detail_view_selected_student = st.selectbox("학생 선택", options=student_options, index=sel_student_idx, key="sel_student_tab3_final_v2")
            student_name_selected = st.session_state.detail_view_selected_student

            if student_name_selected:
                student_info_selected = students_df[students_df["이름"] == student_name_selected].iloc[0]
                s_name_detail = student_info_selected["이름"]
                s_url_detail = student_info_selected["시트URL"]
                
                back_btn_col, date_input_col = st.columns([1,3])
                with back_btn_col:
                    if st.button(f"다른 학생 선택", key=f"back_btn_tab3_final_{s_name_detail}"):
                        st.session_state.detail_view_selected_student = ""; st.rerun()
                with date_input_col:
                    date_selected_detail = st.date_input("날짜 선택", value=datetime.today(), key=f"date_pick_tab3_final_{s_name_detail}")
                date_str_selected_detail = date_selected_detail.strftime("%Y-%m-%d")

                if not s_url_detail or not isinstance(s_url_detail, str) or not s_url_detail.startswith("http"):
                    st.error(f"'{s_name_detail}' 학생의 시트 URL이 올바르지 않습니다.")
                else:
                    try:
                        df_student_all_records = pd.DataFrame()
                        ws_student_for_detail = None

                        with st.spinner(f"'{s_name_detail}' 학생의 전체 일기 기록 로딩 중..."):
                            ws_student_for_detail = g_client_main.open_by_url(s_url_detail).sheet1
                            all_records_for_student = get_records_from_row2_header(ws_student_for_detail, EXPECTED_STUDENT_SHEET_HEADER)
                            df_student_all_records = pd.DataFrame(all_records_for_student)

                        if df_student_all_records.empty or "날짜" not in df_student_all_records.columns:
                            st.warning(f"'{s_name_detail}' 학생 시트에 데이터가 없거나 '날짜' 열이 없습니다.")
                        else:
                            # 1. 선택한 날짜의 일기 표시
                            entry_df_for_date = df_student_all_records[df_student_all_records["날짜"] == date_str_selected_detail]
                            if not entry_df_for_date.empty:
                                diary_entry_to_show = entry_df_for_date.iloc[0]
                                st.subheader(f"📘 {s_name_detail} ({date_str_selected_detail}) 일기"); st.divider()
                                st.write(f"**감정:** {diary_entry_to_show.get('감정', 'N/A')}")
                                st.write(f"**감사한 일:** {diary_entry_to_show.get('감사한 일', 'N/A')}")
                                st.write(f"**하고 싶은 말:** {diary_entry_to_show.get('하고 싶은 말', 'N/A')}")
                                teacher_note_current_val = diary_entry_to_show.get('선생님 쪽지', '')
                                st.write(f"**선생님 쪽지:** {teacher_note_current_val}")

                                teacher_note_input_val = st.text_area(f"✏️ 쪽지 작성/수정", value=teacher_note_current_val, key=f"note_input_final_key_{s_name_detail}_{date_str_selected_detail}")
                                if st.button(f"💾 쪽지 저장", key=f"save_note_btn_final_key_{s_name_detail}_{date_str_selected_detail}"):
                                    if not teacher_note_input_val.strip() and not teacher_note_current_val: st.warning("쪽지 내용이 비어있습니다.")
                                    else:
                                        try:
                                            row_idx_for_update, sheet_headers = -1, ws_student_for_detail.row_values(2)
                                            for i_entry, r_entry_data in enumerate(all_records_for_student):
                                                if r_entry_data.get("날짜") == date_str_selected_detail: row_idx_for_update = i_entry + 3; break
                                            if row_idx_for_update != -1:
                                                note_col_idx_update = sheet_headers.index("선생님 쪽지") + 1 if "선생님 쪽지" in sheet_headers else 5
                                                ws_student_for_detail.update_cell(row_idx_for_update, note_col_idx_update, teacher_note_input_val)
                                                st.success(f"쪽지를 저장했습니다!"); st.cache_data.clear(); st.rerun()
                                            else: st.error("쪽지 저장 대상 일기 항목을 찾지 못했습니다.")
                                        except Exception as e_save_note_final: st.error(f"쪽지 저장 중 오류: {e_save_note_final}")
                            else: 
                                st.info(f"'{s_name_detail}' 학생은 {date_str_selected_detail}에 작성한 일기가 없습니다.")

                        # 2. 학생 전체 기록 기반 분석 (누적 & GPT)
                        if not df_student_all_records.empty: # 학생의 과거 기록이 있어야 분석 가능
                            st.markdown("---"); st.subheader("📊 학생 전체 기록 기반 분석")
                            
                            if st.button(f"{s_name_detail} 학생 전체 기록 누적 분석 (워드클라우드, 감정 통계)", key=f"cumulative_analysis_btn_final_key_{s_name_detail}"):
                                st.write("##### 학생 전체 감정 대분류 통계 (긍정, 보통, 부정)")
                                if "감정" in df_student_all_records.columns and not df_student_all_records["감정"].empty:
                                    def get_main_emotion_group(emotion_str_func):
                                        if isinstance(emotion_str_func, str) and " - " in emotion_str_func:
                                            main_group_func = emotion_str_func.split(" - ")[0].strip()
                                            if main_group_func in EMOTION_GROUPS: return main_group_func
                                        return None
                                    df_student_all_records['감정 대분류 필터링됨'] = df_student_all_records['감정'].apply(get_main_emotion_group)
                                    valid_emotion_counts_hist = df_student_all_records['감정 대분류 필터링됨'].dropna().value_counts()
                                    
                                    final_emotion_counts_chart = pd.Series(index=EMOTION_GROUPS, dtype='int64').fillna(0)
                                    for grp_chart, cnt_chart in valid_emotion_counts_hist.items():
                                        if grp_chart in final_emotion_counts_chart.index: final_emotion_counts_chart[grp_chart] = cnt_chart
                                    
                                    if not final_emotion_counts_chart.empty and final_emotion_counts_chart.sum() > 0:
                                        st.bar_chart(final_emotion_counts_chart)
                                    else: st.info("차트에 표시할 유효한 감정 기록(긍정, 보통, 부정)이 없습니다.")
                                else: st.info("감정 데이터가 없어 통계 표시 불가.")
                                
                                st.write("##### 학생 전체 '감사한 일' & '하고 싶은 말' 단어 분석 (워드클라우드)")
                                wc_text_list_final = []
                                for col_name_for_wc in ["감사한 일", "하고 싶은 말"]:
                                    if col_name_for_wc in df_student_all_records.columns:
                                        wc_text_list_final.extend(df_student_all_records[col_name_for_wc].dropna().astype(str).tolist())
                                wc_text_data_final = " ".join(wc_text_list_final)
                                if wc_text_data_final.strip():
                                    try:
                                        wc_generated = WordCloud(font_path=FONT_PATH, width=700, height=350, background_color="white").generate(wc_text_data_final)
                                        fig_wc_generated, ax_wc_generated = plt.subplots(); ax_wc_generated.imshow(wc_generated, interpolation='bilinear'); ax_wc_generated.axis("off"); st.pyplot(fig_wc_generated)
                                    except RuntimeError as e_font_final: st.error(f"워드클라우드 폰트('{FONT_PATH}') 오류: {e_font_final}. 폰트 파일을 앱 폴더에 추가하거나 경로를 확인하세요.")
                                    except Exception as e_wc_final: st.error(f"워드클라우드 생성 오류: {e_wc_final}")
                                else: st.info("워드클라우드를 생성할 단어가 부족합니다.")
                            
                            st.markdown("---") # GPT 분석 섹션 구분
                            st.subheader(f"🤖 {s_name_detail} 학생 전체 기록 GPT 심층 분석")
                            if st.button(f"GPT로 전체 기록 심층 분석 실행 📝", key=f"gpt_cumulative_btn_final_key_{s_name_detail}"):
                                if not openai_api_key_value or not client_openai:
                                    st.error("OpenAI API 키 또는 클라이언트가 설정되지 않았습니다. secrets 설정을 확인하세요.")
                                else:
                                    with st.spinner(f"GPT가 {s_name_detail} 학생의 전체 누적 기록을 심층 분석 중입니다... (시간 소요)"):
                                        try:
                                            cum_emotions = [f"날짜 {idx+1}({r.get('날짜','미기재')}): {r.get('감정','')}" for idx,r in enumerate(all_records_for_student) if r.get('감정')]
                                            cum_gratitude = [f"날짜 {idx+1}({r.get('날짜','미기재')}): {r.get('감사한 일','')}" for idx,r in enumerate(all_records_for_student) if r.get('감사한 일','').strip()]
                                            cum_message = [f"날짜 {idx+1}({r.get('날짜','미기재')}): {r.get('하고 싶은 말','')}" for idx,r in enumerate(all_records_for_student) if r.get('하고 싶은 말','').strip()]
                                            
                                            data_for_gpt_prompt = (
                                                f"### 전체 감정 기록 모음:\n" + "\n".join(cum_emotions) + "\n\n"
                                                f"### 전체 감사한 일 기록 모음:\n" + "\n".join(cum_gratitude) + "\n\n"
                                                f"### 전체 하고 싶은 말 기록 모음:\n" + "\n".join(cum_message)
                                            )
                                            
                                            prompt_parts_gpt = GPT_CUMULATIVE_SYSTEM_PROMPT.split("학생의 누적 기록 데이터:")
                                            sys_instructions_gpt = prompt_parts_gpt[0].strip()
                                            user_req_template_gpt = "학생의 누적 기록 데이터:" + prompt_parts_gpt[1]
                                            
                                            formatted_user_req_gpt = user_req_template_gpt.format(cumulative_diary_data_for_gpt=data_for_gpt_prompt)

                                            gpt_response_obj_final = client_openai.chat.completions.create(
                                                model="gpt-4o",
                                                messages=[{"role": "system", "content": sys_instructions_gpt}, {"role": "user", "content": formatted_user_req_gpt}],
                                                temperature=0.7, max_tokens=3000 
                                            )
                                            gpt_analysis_text = gpt_response_obj_final.choices[0].message.content
                                            st.markdown("##### 💡 GPT 누적 기록 심층 분석 리포트:")
                                            with st.expander("분석 결과 보기", expanded=True): st.markdown(gpt_analysis_text)
                                        except Exception as e_gpt_call_final: st.error(f"GPT 누적 분석 중 오류 발생: {e_gpt_call_final}")
                    except gspread.exceptions.SpreadsheetNotFound:
                        st.error(f"'{s_name_detail}' 학생 시트 URL({s_url_detail})을 찾을 수 없습니다.")
                    except Exception as e_detail_processing:
                        st.error(f"'{s_name_detail}' 학생 데이터 처리 중 오류: {type(e_detail_processing).__name__} - {e_detail_processing}")
            else: # 학생 미선택 시
                st.info("상단에서 학생을 선택하여 상세 내용을 확인하고 분석 기능을 사용하세요.")
