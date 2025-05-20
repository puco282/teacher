import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from openai import OpenAI # 최신 OpenAI 라이브러리

# --- 상수 정의 ---
EXPECTED_STUDENT_SHEET_HEADER = ["날짜", "감정", "감사한 일", "하고 싶은 말", "선생님 쪽지"]
EMOTION_GROUPS = ["😀 긍정", "😐 보통", "😢 부정"] # 학생용 앱의 감정 대분류
FONT_PATH = "NanumGothic.ttf"  # 워드클라우드용 폰트 경로 (앱 위치에 파일이 있거나, 시스템 경로 지정)

GPT_SYSTEM_PROMPT = """
당신은 초등학교 학생들의 심리 및 상담 분야에서 깊은 전문성을 가진 AI 상담 보조입니다. 
당신의 분석은 인간 중심 상담 이론, 인지 행동 이론 등 실제 상담 이론에 기반해야 합니다. 
제공되는 학생의 익명 단일 일기 내용을 바탕으로, 다음 6가지 항목에 대해 구체적이고 통찰력 있는 리포트를 선생님께 제공해주세요. 
학생의 이름이나 개인 식별 정보는 절대 언급하지 마세요. 답변은 명확한 항목 구분을 위해 마크다운 헤더를 사용해주세요.

제공된 오늘의 일기 내용:
- 오늘 표현된 감정: {emotion_data}
- 오늘의 감사한 일: {gratitude_data}
- 오늘의 하고 싶은 말/일기 내용: {message_data}

리포트 항목:
1.  **오늘 표현된 감정 상태 심층 분석**: 학생이 표현한 감정의 명확성, 감정의 이면에 있을 수 있는 생각이나 느낌, 그리고 이 감정이 학생에게 어떤 의미를 가질 수 있는지 분석해주세요. (단일 일기이므로 '통계'가 아닌 심층 분석입니다.)
2.  **문체 및 표현 특성**: 사용된 어휘의 수준(초등학생 수준 고려), 문장 구조의 복잡성, 자기 생각이나 감정을 얼마나 명확하고 풍부하게 표현하고 있는지, 주관적 감정 표현의 강도는 어떠한지 등을 평가해주세요.
3.  **주요 키워드 및 주제 추출**: 오늘 일기 내용에서 반복적으로 나타나거나 중요하다고 판단되는 핵심 단어(키워드)를 3~5개 추출하고, 이를 통해 학생의 현재 주요 관심사, 자주 언급되는 대상(예: 친구, 가족, 특정 활동 등), 또는 반복되는 상황이나 사건을 파악해주세요.
4.  **오늘 일기에 대한 종합 요약**: 위 분석들을 바탕으로 오늘 학생이 일기를 통해 전달하고자 하는 핵심적인 내용이나 전반적인 상태를 간결하게 요약해주세요.
5.  **관찰 및 변화 고려 지점**: 오늘 일기 내용을 바탕으로 선생님께서 앞으로 이 학생의 어떤 면을 좀 더 관심 있게 지켜보면 좋을지, 또는 어떤 긍정적/부정적 변화의 가능성이 엿보이는지 구체적으로 언급해주세요. (단일 일기이므로 '변화 추적'이 아닌 미래 관찰 지점 제안입니다.)
6.  **선생님을 위한 상담적 조언**: 학생의 현재 상태를 고려하여, 선생님께서 이 학생을 지지하고 돕기 위해 활용할 수 있는 인간 중심적 또는 인지 행동적 접근 방식에 기반한 구체적인 상담 전략이나 소통 방법을 1~2가지 제안해주세요. 예를 들어, 어떤 질문을 해볼 수 있는지, 어떤 공감적 반응을 보일 수 있는지 등을 포함할 수 있습니다.
"""

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="감정 일기장 (교사용)", page_icon="🧑‍🏫", layout="wide")

# --- Helper Functions ---
@st.cache_resource # gspread 클라이언트 캐싱
def authorize_gspread():
    try:
        google_creds_dict = st.secrets["GOOGLE_CREDENTIALS"]
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(google_creds_dict, scope)
        client_gspread = gspread.authorize(creds)
        return client_gspread
    except Exception as e:
        st.error(f"Google API 인증 중 오류 발생: {e}. '.streamlit/secrets.toml' 파일 설정을 확인해주세요.")
        st.stop()
        return None

@st.cache_data(ttl=600) # 10분마다 학생 목록 캐시 갱신
def get_students_df(_client_gspread):
    if not _client_gspread: return pd.DataFrame()
    try:
        student_list_ws = _client_gspread.open("학생목록").sheet1 # 학생목록 시트는 첫 행을 헤더로 사용
        df = pd.DataFrame(student_list_ws.get_all_records())
        if "이름" not in df.columns or "시트URL" not in df.columns:
            st.error("'학생목록' 시트에 '이름' 또는 '시트URL' 열이 없습니다. 확인해주세요.")
            return pd.DataFrame()
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Google Sheets에서 '학생목록' 시트를 찾을 수 없습니다. 시트 이름과 공유 설정을 확인해주세요.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"학생 목록 로딩 중 오류: {e}")
        return pd.DataFrame()

def get_records_from_row2_header(worksheet, expected_header):
    """학생 시트 읽기 (Row1:설정, Row2:헤더, Row3~:데이터)"""
    all_values = worksheet.get_all_values()
    if len(all_values) < 2: return [] # 설정행, 헤더행 최소 2줄 필요
    # header_row_from_sheet = all_values[1] # 실제 헤더와 비교는 현재 생략
    data_rows = all_values[2:]
    records = []
    for row_values in data_rows:
        padded_row_values = row_values + [None] * (len(expected_header) - len(row_values))
        record = dict(zip(expected_header, padded_row_values))
        records.append(record)
    return records

@st.cache_data(ttl=300) # 5분마다 전체 학생의 오늘 데이터 캐시 갱신
def fetch_all_students_today_data(_students_df, today_date_str, _client_gspread, header):
    """오늘 날짜 기준, 모든 학생의 감정 및 메시지 데이터 로드 (API 최소화용)"""
    all_data = []
    if _students_df.empty: return all_data

    progress_bar = st.progress(0, text="학생 데이터 로딩 중...")
    total_students = len(_students_df)

    for i, (_, student_row) in enumerate(_students_df.iterrows()):
        name = student_row["이름"]
        sheet_url = student_row["시트URL"]
        student_entry = {"name": name, "emotion_today": None, "message_today": None, "error": None}

        if not sheet_url or not isinstance(sheet_url, str) or not sheet_url.startswith("http"):
            student_entry["error"] = f"시트 URL 형식 오류 ({sheet_url})"
            all_data.append(student_entry)
            progress_bar.progress((i + 1) / total_students, text=f"{name} 학생 데이터 로딩 중 (URL 오류)...")
            continue
        
        try:
            student_ws = _client_gspread.open_by_url(sheet_url).sheet1
            records = get_records_from_row2_header(student_ws, header)
            
            todays_record_found = False
            for record in records:
                if record.get("날짜") == today_date_str:
                    student_entry["emotion_today"] = record.get("감정")
                    student_entry["message_today"] = record.get("하고 싶은 말")
                    todays_record_found = True
                    break
            if not todays_record_found:
                student_entry["error"] = "오늘 일기 없음"
        except gspread.exceptions.SpreadsheetNotFound:
            student_entry["error"] = "시트 찾을 수 없음"
        except Exception as e:
            student_entry["error"] = f"데이터 로딩 오류 ({type(e).__name__})"
        
        all_data.append(student_entry)
        progress_bar.progress((i + 1) / total_students, text=f"{name} 학생 데이터 로딩 완료.")
    
    progress_bar.empty() # 로딩 완료 후 프로그레스 바 제거
    return all_data

# --- OpenAI API 클라이언트 초기화 ---
try:
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        # 이 오류는 앱의 모든 부분에서 키가 필요할 때 발생할 수 있으므로, 로그인 후나 특정 기능 사용 시점에 표시
        pass # st.error("OpenAI API 키가 secrets에 설정되지 않았습니다.") -> 버튼 클릭 시점에 다시 확인
    client_openai = OpenAI(api_key=openai_api_key)
except Exception as e:
    # st.error(f"OpenAI 클라이언트 초기화 중 오류: {e}")
    client_openai = None # 오류 발생 시 None으로 설정


# --- 세션 상태 초기화 ---
if "teacher_logged_in" not in st.session_state:
    st.session_state.teacher_logged_in = False
if "all_students_today_data_loaded" not in st.session_state: # 데이터 로드 상태 플래그
    st.session_state.all_students_today_data_loaded = False
if "all_students_today_data" not in st.session_state: # 실제 데이터 저장
    st.session_state.all_students_today_data = []


# --- 교사용 로그인 페이지 ---
if not st.session_state.teacher_logged_in:
    st.title("🧑‍🏫 감정일기 로그인 (교사용)")
    admin_password = st.text_input("관리자 비밀번호를 입력하세요", type="password", key="admin_pw_input_consolidated")
    
    if st.button("로그인", key="admin_login_btn_consolidated"):
        # 실제 운영 시에는 st.secrets["ADMIN_PASSWORD"] 등으로 비밀번호를 관리
        if admin_password == st.secrets.get("ADMIN_TEACHER_PASSWORD", "silverline"): # 예시 비밀번호
            st.session_state.teacher_logged_in = True
            st.session_state.all_students_today_data_loaded = False # 로그인 시 데이터 새로 로드하도록 플래그 초기화
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")
else: # --- 교사용 기능 페이지 (로그인 완료 후) ---
    client_gspread = authorize_gspread() # 로그인 후에 gspread 클라이언트 초기화
    students_df = get_students_df(client_gspread)

    st.sidebar.title(f"🧑‍🏫 교사 메뉴")
    if st.sidebar.button("로그아웃", key="teacher_logout_sidebar_consolidated"):
        st.session_state.teacher_logged_in = False
        st.session_state.all_students_today_data_loaded = False
        st.session_state.pop('all_students_today_data', None)
        # 필요한 다른 세션 상태도 초기화
        st.rerun()
    
    if st.sidebar.button("오늘 학생 데이터 새로고침 ♻️", key="refresh_all_student_data"):
        st.session_state.all_students_today_data_loaded = False # 리로드 플래그 설정
        st.cache_data.clear() # 관련된 모든 st.cache_data 함수 캐시 지우기
        st.rerun()

    st.title("🧑‍🏫 교사용 대시보드")

    # --- 데이터 로딩 (로그인 후 또는 새로고침 시 한 번 실행) ---
    if not st.session_state.all_students_today_data_loaded:
        if students_df.empty:
            st.warning("'학생목록' 시트에 학생이 없거나 데이터를 불러올 수 없습니다.")
            # st.stop() # 필요에 따라 중단
        else:
            today_date_str = datetime.today().strftime("%Y-%m-%d")
            st.session_state.all_students_today_data = fetch_all_students_today_data(
                students_df, today_date_str, client_gspread, EXPECTED_STUDENT_SHEET_HEADER
            )
            st.session_state.all_students_today_data_loaded = True
            st.success("모든 학생의 오늘 자 요약 정보를 로드했습니다!") # 로드 완료 메시지
            # st.rerun() # 데이터 로드 후 UI 즉시 업데이트 원할 시 (프로그레스 바 사용 시 자동 업데이트됨)

    all_students_today_summary_data = st.session_state.get("all_students_today_data", [])

    # --- 탭 인터페이스 ---
    tab_titles = ["오늘의 학급 감정 분포 📊", "학생들이 전달하는 메시지 💌", "학생별 일기 상세 보기 📖"]
    tab_emotion_overview, tab_student_messages, tab_detail_view = st.tabs(tab_titles)

    # --- Tab 1: 오늘의 학급 감정 분포 ---
    with tab_emotion_overview:
        st.header(tab_titles[0])
        today_display_date_str = datetime.today().strftime("%Y-%m-%d")
        st.markdown(f"**조회 날짜:** {today_display_date_str}")

        if not all_students_today_summary_data:
            st.info("요약할 학생 데이터가 없습니다. '오늘 학생 데이터 새로고침'을 시도해보세요.")
        else:
            emotion_summary_cats = {group: [] for group in EMOTION_GROUPS}
            emotion_summary_cats["감정 미분류"] = []
            emotion_summary_cats["일기 미제출 또는 오류"] = []

            for student_data in all_students_today_summary_data:
                name = student_data["name"]
                if student_data["error"] and student_data["error"] != "오늘 일기 없음":
                    emotion_summary_cats["일기 미제출 또는 오류"].append(f"{name} ({student_data['error']})")
                    continue
                if student_data["error"] == "오늘 일기 없음" or not student_data["emotion_today"]:
                     emotion_summary_cats["일기 미제출 또는 오류"].append(name)
                     continue

                emotion_str = student_data["emotion_today"]
                if emotion_str and isinstance(emotion_str, str) and " - " in emotion_str:
                    main_emotion_part = emotion_str.split(" - ")[0].strip()
                    if main_emotion_part in EMOTION_GROUPS:
                        emotion_summary_cats[main_emotion_part].append(name)
                    else:
                        emotion_summary_cats["감정 미분류"].append(f"{name} (감정: {emotion_str})")
                else:
                     emotion_summary_cats["일기 미제출 또는 오류"].append(f"{name} (감정 항목 형식 오류: {emotion_str})")
            
            overview_cols = st.columns(len(EMOTION_GROUPS))
            for i, group in enumerate(EMOTION_GROUPS):
                with overview_cols[i]:
                    st.subheader(f"{group} ({len(emotion_summary_cats[group])}명)")
                    if emotion_summary_cats[group]:
                        st.text_area(f"{group} 학생 명단", value="\n".join(sorted(emotion_summary_cats[group])), height=150, disabled=True, key=f"overview_text_{group.replace(' ', '_')}")
                    else:
                        st.caption("해당 없음")
            
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                if emotion_summary_cats["일기 미제출 또는 오류"]:
                    with st.expander(f"📝 일기 미제출 또는 오류 ({len(emotion_summary_cats['일기 미제출 또는 오류'])}명)", expanded=False):
                        st.markdown("\n".join([f"- {s}" for s in sorted(emotion_summary_cats["일기 미제출 또는 오류"])]))
            with exp_col2:
                if emotion_summary_cats["감정 미분류"]:
                    with st.expander(f"🤔 감정 미분류 ({len(emotion_summary_cats['감정 미분류'])}명)", expanded=False):
                        st.markdown("\n".join([f"- {s}" for s in sorted(emotion_summary_cats["감정 미분류"])]))
    
    # --- Tab 2: 학생들이 전달하는 메시지 ---
    with tab_student_messages:
        st.header(tab_titles[1])
        if not all_students_today_summary_data:
            st.info("요약할 학생 데이터가 없습니다.")
        else:
            negative_fb_students = []
            other_fb_students = []

            for student_data in all_students_today_summary_data:
                if student_data["error"] or \
                   not student_data["emotion_today"] or \
                   not student_data["message_today"] or \
                   not student_data["message_today"].strip():
                    continue 

                emotion_full_str = student_data["emotion_today"]
                if not isinstance(emotion_full_str, str) or " - " not in emotion_full_str: continue
                
                emotion_group = emotion_full_str.split(" - ")[0].strip()
                message_content = student_data["message_today"].strip()

                item_details = {"name": student_data["name"], "emotion": emotion_full_str, "message": message_content}
                if emotion_group == "😢 부정":
                    negative_fb_students.append(item_details)
                elif emotion_group in ["😀 긍정", "😐 보통"]:
                    other_fb_students.append(item_details)
            
            st.subheader("😥 부정적 감정 학생들의 메시지")
            if negative_fb_students:
                # st.info(f"총 {len(negative_fb_students)}명의 학생이 부정적인 감정과 함께 메시지를 남겼습니다.")
                for item in sorted(negative_fb_students, key=lambda x: x['name']):
                    with st.container(border=True):
                        st.markdown(f"**학생명:** {item['name']} (<span style='color:red;'>{item['emotion']}</span>)", unsafe_allow_html=True)
                        st.markdown(f"**메시지:**\n> {item['message']}")
            else:
                st.success("오늘, 부정적인 감정과 함께 메시지를 남긴 학생은 없습니다. 👍")

            st.markdown("---")
            st.subheader("😊 그 외 감정 학생들의 메시지")
            if other_fb_students:
                # st.info(f"총 {len(other_fb_students)}명의 학생이 긍정적 또는 보통 감정과 함께 메시지를 남겼습니다.")
                for item in sorted(other_fb_students, key=lambda x: x['name']):
                    with st.container(border=True):
                        st.markdown(f"**학생명:** {item['name']} ({item['emotion']})")
                        st.markdown(f"**메시지:**\n> {item['message']}")
            else:
                st.info("오늘, 긍정적 또는 보통 감정과 함께 메시지를 남긴 학생은 없습니다.")

    # --- Tab 3: 학생별 일기 상세 보기 ---
    with tab_detail_view:
        st.header(tab_titles[2])
        if students_df.empty:
            st.warning("학생 목록 데이터가 없습니다.")
        else:
            selected_student_name_detail = st.selectbox(
                "학생 선택", options=[""] + students_df["이름"].tolist(), 
                key="teacher_student_select_detail_consolidated",
                index=0 # 기본값 없음
            )

            if selected_student_name_detail:
                student_info_detail = students_df[students_df["이름"] == selected_student_name_detail].iloc[0]
                name_for_detail_view = student_info_detail["이름"]
                sheet_url_for_detail_view = student_info_detail["시트URL"]
                
                selected_diary_date_detail = st.date_input(
                    "확인할 날짜 선택", value=datetime.today(), 
                    key=f"teacher_date_select_detail_consolidated_{name_for_detail_view}"
                )
                date_str_for_detail_view = selected_diary_date_detail.strftime("%Y-%m-%d")

                if not sheet_url_for_detail_view or not isinstance(sheet_url_for_detail_view, str) or not sheet_url_for_detail_view.startswith("http"):
                    st.error(f"'{name_for_detail_view}' 학생의 시트 URL이 올바르지 않습니다.")
                else:
                    try:
                        with st.spinner(f"'{name_for_detail_view}' 학생의 전체 일기 기록 로딩 중..."):
                            student_detail_ws = client_gspread.open_by_url(sheet_url_for_detail_view).sheet1
                        all_student_entries_for_detail = get_records_from_row2_header(student_detail_ws, EXPECTED_STUDENT_SHEET_HEADER)
                        df_student_detail_view = pd.DataFrame(all_student_entries_for_detail)

                        # 1. 선택한 날짜의 일기 표시
                        if df_student_detail_view.empty or "날짜" not in df_student_detail_view.columns:
                            st.warning(f"'{name_for_detail_view}' 학생의 시트에 일기 데이터가 없거나 '날짜' 열이 없습니다.")
                        else:
                            entry_for_selected_date_df = df_student_detail_view[df_student_detail_view["날짜"] == date_str_for_detail_view]
                            
                            if not entry_for_selected_date_df.empty:
                                diary_entry_selected = entry_for_selected_date_df.iloc[0]
                                st.subheader(f"📘 {name_for_detail_view}의 {date_str_for_detail_view} 일기")
                                st.write(f"**감정:** {diary_entry_selected.get('감정', 'N/A')}")
                                st.write(f"**감사한 일:** {diary_entry_selected.get('감사한 일', 'N/A')}")
                                st.write(f"**하고 싶은 말:** {diary_entry_selected.get('하고 싶은 말', 'N/A')}")
                                current_teacher_note_val = diary_entry_selected.get('선생님 쪽지', '')
                                st.write(f"**선생님 쪽지:** {current_teacher_note_val}")

                                # 선생님 쪽지 작성 (선택한 날짜에 일기가 있을 때)
                                note_col_header = "선생님 쪽지"
                                teacher_note_key = f"teacher_note_area_{name_for_detail_view}_{date_str_for_detail_view}"
                                teacher_note_input = st.text_area(f"✏️ {name_for_detail_view}에게 ({date_str_for_detail_view}) 쪽지 작성/수정", 
                                                                  value=current_teacher_note_val, key=teacher_note_key)
                                
                                if st.button(f"💾 쪽지 저장", key=f"save_teacher_note_btn_{name_for_detail_view}_{date_str_for_detail_view}"):
                                    if not teacher_note_input.strip() and not current_teacher_note_val:
                                        st.warning("쪽지 내용이 비어있습니다.")
                                    else:
                                        try:
                                            row_idx_to_update = -1
                                            for i, r_entry in enumerate(all_student_entries_for_detail):
                                                if r_entry.get("날짜") == date_str_for_detail_view:
                                                    row_idx_to_update = i + 3 # 데이터는 3번째 행부터 (1-based)
                                                    break
                                            
                                            if row_idx_to_update != -1:
                                                headers_stud_sheet = student_detail_ws.row_values(2) # 헤더는 2번째 행
                                                note_col_idx = headers_stud_sheet.index(note_col_header) + 1 if note_col_header in headers_stud_sheet else 5
                                                
                                                student_detail_ws.update_cell(row_idx_to_update, note_col_idx, teacher_note_input)
                                                st.success(f"쪽지를 저장했습니다!")
                                                st.cache_data.clear() # 데이터 변경 후 캐시 클리어
                                                st.rerun()
                                            else:
                                                 st.error("쪽지를 저장할 해당 날짜의 일기 항목을 찾지 못했습니다.")
                                        except Exception as e_save_note:
                                            st.error(f"쪽지 저장 중 오류: {e_save_note}")
                                
                                # GPT 분석 버튼 (선택한 날짜에 일기가 있을 때)
                                st.markdown("---")
                                st.subheader("🤖 AI 기반 오늘 일기 심층 분석 (GPT)")
                                if st.button("오늘 일기 GPT로 분석하기 🔍", key=f"gpt_analyze_btn_consolidated_{name_for_detail_view}_{date_str_for_detail_view}"):
                                    if not openai_api_key: # secrets에서 키를 못가져왔다면
                                        st.error("OpenAI API 키가 설정되지 않았습니다. 앱 관리자에게 문의하세요.")
                                    elif not client_openai: # 클라이언트 초기화 실패 시
                                        st.error("OpenAI 클라이언트 초기화에 실패했습니다. API 키 설정을 확인하세요.")
                                    else:
                                        with st.spinner("GPT가 학생의 오늘 일기를 심층 분석 중입니다..."):
                                            try:
                                                gpt_emotion = diary_entry_selected.get('감정', '기록 없음')
                                                gpt_gratitude = diary_entry_selected.get('감사한 일', '기록 없음')
                                                gpt_message = diary_entry_selected.get('하고 싶은 말', '기록 없음')
                                                
                                                formatted_user_prompt = GPT_SYSTEM_PROMPT.split("리포트 항목:")[0].format(
                                                    emotion_data=gpt_emotion,
                                                    gratitude_data=gpt_gratitude,
                                                    message_data=gpt_message
                                                ) + "\n리포트 항목:" + GPT_SYSTEM_PROMPT.split("리포트 항목:")[1]


                                                gpt_response = client_openai.chat.completions.create(
                                                    model="gpt-4o",
                                                    messages=[
                                                        {"role": "system", "content": "당신은 초등학교 학생들의 심리 및 상담 분야에서 깊은 전문성을 가진 AI 상담 보조입니다. 당신의 분석은 인간 중심 상담 이론, 인지 행동 이론 등 실제 상담 이론에 기반해야 합니다. 제공되는 학생의 익명 단일 일기 내용을 바탕으로 요청된 항목에 대해 구체적이고 통찰력 있는 리포트를 선생님께 제공해주세요. 학생의 이름이나 개인 식별 정보는 절대 언급하지 마세요. 답변은 명확한 항목 구분을 위해 마크다운 헤더를 사용해주세요."},
                                                        {"role": "user", "content": formatted_user_prompt}
                                                    ],
                                                    temperature=0.7, max_tokens=2000
                                                )
                                                analysis_result_gpt = gpt_response.choices[0].message.content
                                                st.markdown("##### 💡 GPT 심층 분석 리포트:")
                                                with st.expander("분석 결과 보기", expanded=True):
                                                    st.markdown(analysis_result_gpt)
                                            except Exception as e_gpt:
                                                st.error(f"GPT 분석 중 오류 발생: {e_gpt}")
                            else: # 해당 날짜에 일기가 없는 경우
                                st.info(f"'{name_for_detail_view}' 학생은 {date_str_for_detail_view}에 작성한 일기가 없습니다.")

                        # 학생 전체 기록 누적 분석 (선택한 날짜 일기 유무와 관계 없이, 학생 기록만 있으면 가능)
                        if not df_student_detail_view.empty:
                            with st.expander("📊 이 학생의 전체 기록 누적 분석 (워드클라우드 등)", expanded=False):
                                if st.button(f"{name_for_detail_view} 학생 전체 기록 분석 실행", key=f"cumulative_analysis_btn_{name_for_detail_view}"):
                                    st.markdown("---")
                                    st.write("##### 학생 전체 감정 대분류 통계")
                                    if "감정" in df_student_detail_view.columns and not df_student_detail_view["감정"].empty:
                                        df_student_detail_view['감정 대분류'] = df_student_detail_view['감정'].astype(str).apply(
                                            lambda x: x.split(" - ")[0].strip() if isinstance(x, str) and " - " in x else "미분류"
                                        )
                                        emotion_group_counts_hist = df_student_detail_view['감정 대분류'].value_counts()
                                        st.bar_chart(emotion_group_counts_hist)
                                    else:
                                        st.info("이 학생의 감정 데이터가 없어 통계를 표시할 수 없습니다.")
                                    
                                    st.markdown("---")
                                    st.write("##### 학생 전체 '감사한 일' & '하고 싶은 말' 단어 분석 (워드 클라우드)")
                                    combined_wc_text = []
                                    for col in ["감사한 일", "하고 싶은 말"]:
                                        if col in df_student_detail_view.columns:
                                            combined_wc_text.extend(df_student_detail_view[col].dropna().astype(str).tolist())
                                    text_for_wc_hist = " ".join(combined_wc_text)

                                    if text_for_wc_hist.strip():
                                        try:
                                            wordcloud_obj = WordCloud(font_path=FONT_PATH, width=800, height=300, background_color="white").generate(text_for_wc_hist)
                                            fig_wc, ax_wc = plt.subplots()
                                            ax_wc.imshow(wordcloud_obj, interpolation='bilinear'); ax_wc.axis("off")
                                            st.pyplot(fig_wc)
                                        except RuntimeError as font_err:
                                            st.error(f"워드클라우드 폰트('{FONT_PATH}') 로드 오류: {font_err}. 폰트 파일 위치를 확인하세요.")
                                        except Exception as wc_e:
                                            st.error(f"워드클라우드 생성 중 오류: {wc_e}")
                                    else:
                                        st.info("단어 분석을 위한 텍스트 데이터가 없습니다.")
                        # else: # 학생 시트 자체가 비어있거나, "날짜" 열이 없는 경우 (위에서 st.warning으로 이미 처리)
                        #    pass

                    except gspread.exceptions.SpreadsheetNotFound:
                        st.error(f"'{name_for_detail_view}' 학생의 시트 URL({sheet_url_for_detail_view})이 잘못되었거나 시트를 찾을 수 없습니다.")
                    except Exception as e_detail:
                        st.error(f"'{name_for_detail_view}' 학생 데이터 로딩 중 오류: {type(e_detail).__name__} - {e_detail}")
            # else: # 학생 미선택 시 (selectbox 기본값)
                # st.info("위에서 학생을 선택하여 상세 일기를 확인하세요.")
