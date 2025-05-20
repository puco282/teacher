import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from openai import OpenAI
import time # API 호출 사이 지연을 위해 (선택적)

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
@st.cache_resource # 리소스 캐싱 (gspread 클라이언트 객체)
def authorize_gspread():
    try:
        google_creds_dict = st.secrets["GOOGLE_CREDENTIALS"]
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(google_creds_dict, scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Google API 인증 오류: {e}. '.streamlit/secrets.toml' 설정을 확인하세요."); st.stop(); return None

@st.cache_data(ttl=600) # 데이터 캐싱 (10분)
def get_students_df(_client_gspread):
    if not _client_gspread: return pd.DataFrame()
    try:
        ws = _client_gspread.open("학생목록").sheet1
        df = pd.DataFrame(ws.get_all_records(head=1)) # 첫 행을 헤더로 사용 명시
        if not df.empty and ("이름" not in df.columns or "시트URL" not in df.columns):
            st.error("'학생목록' 시트에 '이름' 또는 '시트URL' 열이 없습니다."); return pd.DataFrame()
        return df
    except Exception as e: st.error(f"학생 목록 로딩 오류: {e}"); return pd.DataFrame()

def get_records_from_row2_header(worksheet, expected_headers):
    all_values = worksheet.get_all_values()
    if len(all_values) < 2: return [] # 설정행(1행), 헤더행(2행) 필수
    # 실제 시트의 2번째 행을 헤더로 사용
    # header_row_actual = all_values[1] # 디버깅 시 실제 헤더 확인용
    data_rows = all_values[2:]
    records = []
    num_expected_headers = len(expected_headers)
    for r_vals in data_rows:
        rec = {}
        for i, header_name in enumerate(expected_headers):
            if i < len(r_vals): rec[header_name] = r_vals[i]
            else: rec[header_name] = None 
        records.append(rec)
    return records

@st.cache_data(ttl=300) # 데이터 캐싱 (5분)
def fetch_all_students_today_data(_students_df, today_str, _client_gspread, headers_list):
    all_data = []
    if _students_df.empty: return all_data
    
    total_students = len(_students_df)
    # st.progress는 메인 스레드에서만 사용 권장, 여기서는 로딩 중임을 알리는 메시지로 대체
    loading_message = st.empty() # 메시지 영역 확보
    loading_message.info("전체 학생의 오늘 자 요약 정보 로딩 중...")

    for i, (_, student_row) in enumerate(_students_df.iterrows()):
        name, sheet_url = student_row["이름"], student_row["시트URL"]
        student_entry = {"name": name, "emotion_today": None, "message_today": None, "error": None}
        
        # st.progress 사용 대신 간단한 텍스트 업데이트
        # loading_message.info(f"'{name}' 학생 정보 확인 중... ({int(((i+1)/total_students)*100)}%)")
        # time.sleep(0.1) # API 호출 부담 줄이기 위한 아주 짧은 지연 (선택적)

        if not sheet_url or not isinstance(sheet_url, str) or not sheet_url.startswith("http"):
            student_entry["error"] = "시트 URL 형식 오류"; all_data.append(student_entry); continue
        try:
            student_ws = _client_gspread.open_by_url(sheet_url).sheet1
            records = get_records_from_row2_header(student_ws, headers_list)
            todays_record_found = False
            for record in records:
                if record.get("날짜") == today_str:
                    student_entry["emotion_today"] = record.get("감정")
                    student_entry["message_today"] = record.get("하고 싶은 말")
                    todays_record_found = True; break
            if not todays_record_found: student_entry["error"] = "오늘 일기 없음"
        except gspread.exceptions.APIError as ge:
            student_entry["error"] = f"API 할당량 초과 가능성 ({ge.response.status_code})" # 429 오류 등
        except gspread.exceptions.SpreadsheetNotFound:
            student_entry["error"] = "시트를 찾을 수 없음"
        except Exception as e: student_entry["error"] = f"데이터 로딩 중 알 수 없는 오류 ({type(e).__name__})"
        all_data.append(student_entry)
    
    loading_message.empty() # 로딩 완료 후 메시지 제거
    return all_data

# --- OpenAI API 클라이언트 초기화 ---
client_openai = None
openai_api_key_value = st.secrets.get("OPENAI_API_KEY")
if openai_api_key_value:
    try: client_openai = OpenAI(api_key=openai_api_key_value)
    except Exception as e: st.warning(f"OpenAI 클라이언트 초기화 오류: {e} (GPT 기능 사용이 제한될 수 있습니다.)")

# --- 세션 상태 초기화 ---
session_defaults = {
    "teacher_logged_in": False, "all_students_today_data_loaded": False,
    "all_students_today_data": [], "detail_view_selected_student": "",
    "tab3_student_data_cache": {} # 학생별 상세 데이터 캐시
}
for k, v_init in session_defaults.items():
    if k not in st.session_state: st.session_state[k] = v_init

# --- MAIN APP LOGIC ---
if not st.session_state.teacher_logged_in: # --- 교사용 로그인 페이지 ---
    st.title("🧑‍🏫 감정일기 로그인 (교사용)")
    admin_pw_input = st.text_input("관리자 비밀번호", type="password", key="admin_pw_vfinal_consolidated_login")
    if st.button("로그인", key="admin_login_btn_vfinal_consolidated_login"):
        if admin_pw_input == st.secrets.get("ADMIN_TEACHER_PASSWORD", "silverline"): # 예시 비밀번호
            st.session_state.teacher_logged_in = True
            # 로그인 성공 시 모든 관련 세션 상태 초기화
            for key_to_reset in session_defaults.keys():
                 if key_to_reset != "teacher_logged_in": 
                     st.session_state[key_to_reset] = session_defaults[key_to_reset]
            st.cache_data.clear() # Streamlit의 내부 데이터 캐시도 클리어
            st.rerun()
        else: st.error("비밀번호가 올바르지 않습니다.")
else: # --- 교사용 기능 페이지 (로그인 성공 후) ---
    g_client = authorize_gspread() # gspread 클라이언트 가져오기
    students_df = get_students_df(g_client) # 학생 목록 가져오기

    st.sidebar.title("🧑‍🏫 교사 메뉴")
    if st.sidebar.button("로그아웃", key="logout_vfinal_consolidated_app"):
        for key_to_reset in session_defaults.keys(): st.session_state[key_to_reset] = session_defaults[key_to_reset]
        st.cache_data.clear() # 로그아웃 시 모든 데이터 캐시 클리어
        st.rerun()
    
    if st.sidebar.button("오늘 학생 데이터 새로고침 ♻️", key="refresh_data_vfinal_consolidated_app"):
        st.session_state.all_students_today_data_loaded = False # 리로드 플래그
        st.session_state.tab3_student_data_cache = {} # 상세 보기 탭 캐시도 초기화
        st.cache_data.clear() # 모든 st.cache_data 클리어
        st.rerun()

    st.title("🧑‍🏫 교사용 대시보드")

    # 전체 학생 오늘 자 요약 데이터 로드 (필요시)
    if not st.session_state.all_students_today_data_loaded:
        if students_df.empty:
            if g_client: # g_client 인증은 성공했으나 학생목록이 비었거나 접근불가한 경우
                 st.warning("'학생목록' 시트가 비어있거나, 시트 접근 권한 또는 내용을 확인해주세요.")
            st.session_state.all_students_today_data = []
            st.session_state.all_students_today_data_loaded = True 
        else:
            today_str = datetime.today().strftime("%Y-%m-%d")
            st.session_state.all_students_today_data = fetch_all_students_today_data(
                students_df, today_str, g_client, EXPECTED_STUDENT_SHEET_HEADER)
            st.session_state.all_students_today_data_loaded = True
            # 로드 완료 메시지는 fetch_all_students_today_data 내에서 처리되거나, 여기서 조건부 표시 가능
            if st.session_state.all_students_today_data or students_df.empty:
                 pass # fetch_all_students_today_data 함수 내부에서 st.empty()로 관리

    summary_data_for_tabs = st.session_state.get("all_students_today_data", [])
    
    tab_titles = ["오늘의 학급 감정 분포 📊", "학생들이 전달하는 메시지 💌", "학생별 일기 상세 보기 📖"]
    tab1, tab2, tab3 = st.tabs(tab_titles)

    with tab1: # 오늘의 학급 감정 분포
        st.header(tab_titles[0])
        st.markdown(f"**조회 날짜:** {datetime.today().strftime('%Y-%m-%d')}")
        if not summary_data_for_tabs and not students_df.empty :
            st.info("오늘 자 학생 요약 정보를 아직 불러오지 못했습니다. '데이터 새로고침'을 시도하거나 잠시 후 다시 확인해주세요.")
        elif not summary_data_for_tabs and students_df.empty :
             st.warning("'학생목록'이 비어있어 표시할 내용이 없습니다.")
        else:
            emotion_categories = {group: [] for group in EMOTION_GROUPS}
            emotion_categories.update({"감정 미분류": [], "일기 미제출 또는 오류": []})
            for data in summary_data_for_tabs:
                s_name = data["name"]
                if data["error"] and data["error"] != "오늘 일기 없음": emotion_categories["일기 미제출 또는 오류"].append(f"{s_name} ({data['error']})")
                elif data["error"] == "오늘 일기 없음" or not data["emotion_today"]: emotion_categories["일기 미제출 또는 오류"].append(s_name)
                elif data["emotion_today"] and isinstance(data["emotion_today"], str) and " - " in data["emotion_today"]:
                    main_emotion = data["emotion_today"].split(" - ")[0].strip()
                    if main_emotion in EMOTION_GROUPS: emotion_categories[main_emotion].append(s_name)
                    else: emotion_categories["감정 미분류"].append(f"{s_name} (감정: {data['emotion_today']})")
                else: emotion_categories["일기 미제출 또는 오류"].append(f"{s_name} (감정 형식 오류: {data['emotion_today']})")
            
            overview_cols = st.columns(len(EMOTION_GROUPS))
            for i, group_name in enumerate(EMOTION_GROUPS):
                with overview_cols[i]:
                    st.subheader(f"{group_name} ({len(emotion_categories[group_name])}명)")
                    if emotion_categories[group_name]: st.markdown("\n".join([f"- {n}" for n in sorted(emotion_categories[group_name])]))
                    else: st.info("이 감정을 느낀 학생이 없습니다.")
            
            expander_col1, expander_col2 = st.columns(2)
            with expander_col1:
                if emotion_categories["일기 미제출 또는 오류"]:
                    with st.expander(f"📝 일기 미제출/오류 ({len(emotion_categories['일기 미제출 또는 오류'])}명)", expanded=False):
                        st.markdown("\n".join([f"- {s}" for s in sorted(emotion_categories["일기 미제출 또는 오류"])]))
            with expander_col2:
                if emotion_categories["감정 미분류"]:
                    with st.expander(f"🤔 감정 미분류 ({len(emotion_categories['감정 미분류'])}명)", expanded=False):
                        st.markdown("\n".join([f"- {s}" for s in sorted(emotion_categories["감정 미분류"])]))
    
    with tab2: # 학생들이 전달하는 메시지
        st.header(tab_titles[1])
        if not summary_data_for_tabs and not students_df.empty: st.info("오늘 자 학생 요약 정보를 아직 불러오지 못했습니다.")
        elif not summary_data_for_tabs and students_df.empty : st.warning("'학생목록'이 비어있어 표시할 내용이 없습니다.")
        else:
            negative_feedback_list, other_feedback_list = [], []
            for data_item in summary_data_for_tabs: # 변수명 변경
                if data_item["error"] or not data_item["emotion_today"] or \
                   not data_item["message_today"] or not data_item["message_today"].strip(): continue
                emotion_full_str_item = data_item["emotion_today"]
                if not isinstance(emotion_full_str_item, str) or " - " not in emotion_full_str_item: continue
                
                item_details_msg_tab2 = {"name": data_item["name"], "emotion": emotion_full_str_item, "message": data_item["message_today"].strip()}
                if emotion_full_str_item.split(" - ")[0].strip() == "😢 부정": negative_feedback_list.append(item_details_msg_tab2)
                elif emotion_full_str_item.split(" - ")[0].strip() in ["😀 긍정", "😐 보통"]: other_feedback_list.append(item_details_msg_tab2)
            
            if not negative_feedback_list and not other_feedback_list: 
                st.success("오늘 선생님이나 친구들에게 하고 싶은 말을 적은 학생이 없습니다. 😊")
            else:
                st.subheader("😥 부정적 감정 학생들의 메시지")
                if negative_feedback_list:
                    for item_neg_fb in sorted(negative_feedback_list, key=lambda x: x['name']):
                        with st.container(border=True): st.markdown(f"**학생명:** {item_neg_fb['name']} (<span style='color:red;'>{item_neg_fb['emotion']}</span>)\n\n**메시지:**\n> {item_neg_fb['message']}", unsafe_allow_html=True)
                else: st.info("오늘, 부정적인 감정과 함께 메시지를 남긴 학생은 없습니다.")
                st.markdown("---")
                st.subheader("😊 그 외 감정 학생들의 메시지")
                if other_feedback_list:
                    for item_oth_fb in sorted(other_feedback_list, key=lambda x: x['name']):
                        with st.container(border=True): st.markdown(f"**학생명:** {item_oth_fb['name']} ({item_oth_fb['emotion']})\n\n**메시지:**\n> {item_oth_fb['message']}")
                else: st.info("오늘, 긍정적 또는 보통 감정과 함께 메시지를 남긴 학생은 없습니다.")

    with tab3: # 학생별 일기 상세 보기
        st.header(tab_titles[2])
        if students_df.empty: st.warning("학생 목록을 먼저 불러오세요.")
        else:
            student_options_tab3_final = [""] + students_df["이름"].tolist()
            sel_student_idx_tab3_final = 0
            if st.session_state.detail_view_selected_student in student_options_tab3_final:
                sel_student_idx_tab3_final = student_options_tab3_final.index(st.session_state.detail_view_selected_student)
            
            st.session_state.detail_view_selected_student = st.selectbox("학생 선택", options=student_options_tab3_final, 
                                                                        index=sel_student_idx_tab3_final, 
                                                                        key="selectbox_student_detail_final_version")
            selected_student_name_final = st.session_state.detail_view_selected_student

            if selected_student_name_final:
                student_info_final = students_df[students_df["이름"] == selected_student_name_final].iloc[0]
                s_name_final = student_info_final["이름"]
                s_url_final = student_info_final["시트URL"]
                
                back_btn_col_final, date_input_col_final = st.columns([0.25, 0.75]) # 버튼 크기 조절
                with back_btn_col_final:
                    if st.button(f"다른 학생 선택", key=f"back_btn_tab3_final_version_{s_name_final}"):
                        st.session_state.detail_view_selected_student = ""
                        st.rerun()
                with date_input_col_final:
                    date_selected_final = st.date_input("날짜 선택", value=datetime.today(), key=f"date_pick_tab3_final_version_{s_name_final}", label_visibility="collapsed")
                
                date_str_selected_final = date_selected_final.strftime("%Y-%m-%d")

                if not s_url_final or not isinstance(s_url_final, str) or not s_url_final.startswith("http"):
                    st.error(f"'{s_name_final}' 학생의 시트 URL이 올바르지 않습니다: {s_url_final}")
                else:
                    df_student_all_entries_final = None
                    all_entries_list_final = [] # 쪽지 저장 시 행 인덱싱에 사용

                    # 캐시 확인 또는 데이터 로드
                    if s_name_final in st.session_state.tab3_student_data_cache:
                        cached_data = st.session_state.tab3_student_data_cache[s_name_final]
                        df_student_all_entries_final = cached_data['df']
                        all_entries_list_final = cached_data['list']
                        # st.caption(f"'{s_name_final}' 학생 캐시된 데이터 사용됨.") # 디버깅용
                    else:
                        try:
                            with st.spinner(f"'{s_name_final}' 학생의 전체 일기 기록 로딩 중... (API 호출)"):
                                ws_temp_final = g_client.open_by_url(s_url_final).sheet1
                                all_entries_list_final = get_records_from_row2_header(ws_temp_final, EXPECTED_STUDENT_SHEET_HEADER)
                                df_student_all_entries_final = pd.DataFrame(all_entries_list_final)
                                st.session_state.tab3_student_data_cache[s_name_final] = {'df': df_student_all_entries_final, 'list': all_entries_list_final}
                        except Exception as e:
                            st.error(f"'{s_name_final}' 학생 데이터 로딩 중 오류: {e}")
                            df_student_all_entries_final = pd.DataFrame() # 오류 시 빈 DF
                            all_entries_list_final = []
                    
                    # --- 데이터 로드 후 UI 표시 ---
                    if df_student_all_entries_final.empty or "날짜" not in df_student_all_entries_final.columns:
                        st.warning(f"'{s_name_final}' 학생 시트에 유효한 데이터가 없습니다 (헤더 또는 내용 점검 필요).")
                    else:
                        entry_df_selected = df_student_all_entries_final[df_student_all_entries_final["날짜"] == date_str_selected_final]
                        if not entry_df_selected.empty:
                            diary_entry_display = entry_df_selected.iloc[0]
                            st.subheader(f"📘 {s_name_final} ({date_str_selected_final}) 일기"); st.divider()
                            st.write(f"**감정:** {diary_entry_display.get('감정', 'N/A')}")
                            st.write(f"**감사한 일:** {diary_entry_display.get('감사한 일', 'N/A')}")
                            st.write(f"**하고 싶은 말:** {diary_entry_display.get('하고 싶은 말', 'N/A')}")
                            teacher_note_val_display = diary_entry_display.get('선생님 쪽지', '')
                            st.write(f"**선생님 쪽지:** {teacher_note_val_display}")

                            note_input_val = st.text_area(f"✏️ 쪽지 작성/수정", value=teacher_note_val_display, key=f"note_input_key_{s_name_final}_{date_str_selected_final}")
                            if st.button(f"💾 쪽지 저장", key=f"save_note_key_{s_name_final}_{date_str_selected_final}"):
                                if not note_input_val.strip() and not teacher_note_val_display: st.warning("쪽지 내용이 비어있습니다.")
                                else:
                                    try:
                                        with st.spinner("쪽지 저장 중..."):
                                            ws_for_save = g_client.open_by_url(s_url_final).sheet1 # 저장 시점의 워크시트 객체
                                            row_idx_save, headers_save = -1, ws_for_save.row_values(2)
                                            # all_entries_list_final 사용 (캐시 또는 로드된 리스트)
                                            for i_save, r_save in enumerate(all_entries_list_final):
                                                if r_save.get("날짜") == date_str_selected_final: row_idx_save = i_save + 3; break
                                            if row_idx_save != -1:
                                                note_col_idx_save = headers_save.index("선생님 쪽지") + 1 if "선생님 쪽지" in headers_save else 5
                                                ws_for_save.update_cell(row_idx_save, note_col_idx_save, note_input_val)
                                                if s_name_final in st.session_state.tab3_student_data_cache: # 캐시 무효화
                                                    del st.session_state.tab3_student_data_cache[s_name_final]
                                                st.success(f"쪽지를 저장했습니다!"); st.rerun()
                                            else: st.error("쪽지 저장 대상 일기 항목을 찾지 못했습니다.")
                                    except Exception as e_save_final: st.error(f"쪽지 저장 오류: {e_save_final}")
                        else: 
                            st.info(f"'{s_name_final}' 학생은 {date_str_selected_final}에 작성한 일기가 없습니다.")

                        # --- 학생 전체 기록 기반 분석 섹션 ---
                        if not df_student_all_entries_final.empty:
                            st.markdown("---"); st.subheader("📊 학생 전체 기록 기반 분석")
                            
                            # 누적 분석 (워드클라우드, 감정통계) 버튼
                            if st.button(f"{s_name_final} 학생 전체 기록 누적 분석", key=f"cumulative_btn_key_{s_name_final}"):
                                st.write("##### 학생 전체 감정 대분류 통계 (긍정, 보통, 부정)")
                                if "감정" in df_student_all_entries_final.columns and not df_student_all_entries_final["감정"].empty:
                                    def get_main_emotion_group_chart(emotion_str_chart):
                                        if isinstance(emotion_str_chart, str) and " - " in emotion_str_chart:
                                            main_grp_chart = emotion_str_chart.split(" - ")[0].strip()
                                            if main_grp_chart in EMOTION_GROUPS: return main_grp_chart
                                        return None
                                    df_student_all_entries_final['감정 대분류 차트용'] = df_student_all_entries_final['감정'].apply(get_main_emotion_group_chart)
                                    valid_emotion_counts_chart = df_student_all_entries_final['감정 대분류 차트용'].dropna().value_counts()
                                    
                                    chart_data = pd.Series(index=EMOTION_GROUPS, dtype='int64').fillna(0)
                                    for grp, cnt in valid_emotion_counts_chart.items():
                                        if grp in chart_data.index: chart_data[grp] = cnt
                                    
                                    if not chart_data.empty and chart_data.sum() > 0: st.bar_chart(chart_data)
                                    else: st.info("차트에 표시할 유효한 감정 기록(긍정, 보통, 부정)이 없습니다.")
                                else: st.info("감정 데이터가 없어 통계 표시 불가.")
                                
                                st.write("##### 학생 전체 '감사한 일' & '하고 싶은 말' 단어 분석 (워드클라우드)")
                                wc_text_list = []
                                for col_wc_name in ["감사한 일", "하고 싶은 말"]:
                                    if col_wc_name in df_student_all_entries_final.columns:
                                        wc_text_list.extend(df_student_all_entries_final[col_wc_name].dropna().astype(str).tolist())
                                wc_text_str = " ".join(wc_text_list)
                                if wc_text_str.strip():
                                    try:
                                        wc_img_obj = WordCloud(font_path=FONT_PATH, width=700, height=350, background_color="white").generate(wc_text_str)
                                        fig_wc, ax_wc = plt.subplots(); ax_wc.imshow(wc_img_obj, interpolation='bilinear'); ax_wc.axis("off"); st.pyplot(fig_wc)
                                    except RuntimeError as e_font_wc: st.error(f"워드클라우드 폰트('{FONT_PATH}') 오류: {e_font_wc}.")
                                    except Exception as e_wc_gen: st.error(f"워드클라우드 생성 오류: {e_wc_gen}")
                                else: st.info("워드클라우드를 생성할 단어가 부족합니다.")
                            
                            # GPT 누적 분석 버튼
                            st.markdown("---") 
                            st.subheader(f"🤖 {s_name_final} 학생 전체 기록 GPT 심층 분석") 
                            if st.button(f"GPT로 전체 기록 심층 분석 실행 📝", key=f"gpt_cumulative_btn_key_{s_name_final}"):
                                if not openai_api_key_value or not client_openai:
                                    st.error("OpenAI API 키 또는 클라이언트가 설정되지 않았습니다. secrets 설정을 확인하세요.")
                                else:
                                    with st.spinner(f"GPT가 {s_name_final} 학생의 전체 누적 기록을 심층 분석 중입니다... (시간 소요)"):
                                        try:
                                            cum_emotions_gpt_list = [f"일자({r.get('날짜','미기재')}): {r.get('감정','')}" for r in all_entries_list_final if r.get('감정')]
                                            cum_gratitude_gpt_list = [f"일자({r.get('날짜','미기재')}): {r.get('감사한 일','')}" for r in all_entries_list_final if r.get('감사한 일','').strip()]
                                            cum_message_gpt_list = [f"일자({r.get('날짜','미기재')}): {r.get('하고 싶은 말','')}" for r in all_entries_list_final if r.get('하고 싶은 말','').strip()]
                                            
                                            cumulative_data_str_for_gpt = (
                                                f"### 전체 감정 기록 모음:\n" + ("\n".join(cum_emotions_gpt_list) if cum_emotions_gpt_list else "기록 없음") + "\n\n"
                                                f"### 전체 감사한 일 기록 모음:\n" + ("\n".join(cum_gratitude_gpt_list) if cum_gratitude_gpt_list else "기록 없음") + "\n\n"
                                                f"### 전체 하고 싶은 말 기록 모음:\n" + ("\n".join(cum_message_gpt_list) if cum_message_gpt_list else "기록 없음")
                                            )
                                            
                                            prompt_parts_gpt_final = GPT_CUMULATIVE_SYSTEM_PROMPT.split("학생의 누적 기록 데이터:")
                                            system_instructions_gpt_final = prompt_parts_gpt_final[0].strip()
                                            user_request_template_final_gpt = "학생의 누적 기록 데이터:" + prompt_parts_gpt_final[1]
                                            
                                            formatted_user_request_final_gpt = user_request_template_final_gpt.format(
                                                 cumulative_diary_data_for_gpt=cumulative_data_str_for_gpt
                                            )

                                            gpt_response_obj = client_openai.chat.completions.create(
                                                model="gpt-4o",
                                                messages=[
                                                    {"role": "system", "content": system_instructions_gpt_final},
                                                    {"role": "user", "content": formatted_user_request_final_gpt}
                                                ],
                                                temperature=0.7, max_tokens=3500 # 응답 길이 늘림
                                            )
                                            gpt_analysis_result_text = gpt_response_obj.choices[0].message.content
                                            st.markdown("##### 💡 GPT 누적 기록 심층 분석 리포트:")
                                            with st.expander("분석 결과 보기", expanded=True): st.markdown(gpt_analysis_result_text)
                                        except Exception as e_gpt_call: st.error(f"GPT 누적 분석 중 오류 발생: {e_gpt_call}")
                    except Exception as e_detail_main: st.error(f"'{s_name_final}' 학생 데이터 처리 중 주요 오류: {e_detail_main}")
            else: st.info("상단에서 학생을 선택하여 상세 내용을 확인하고 분석 기능을 사용하세요.")
