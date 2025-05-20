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
FONT_PATH = "NanumGothic.ttf"  # 예: "NanumGothic.ttf" (앱과 같은 폴더에 있을 경우)

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
        st.error(f"Google API 인증 중 오류: {e}. '.streamlit/secrets.toml' 설정을 확인하세요.")
        st.stop(); return None

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

def get_records_from_row2_header(worksheet, headers):
    all_values = worksheet.get_all_values()
    if len(all_values) < 2: return []
    data_rows, records = all_values[2:], []
    for r_vals in data_rows:
        rec, pad_len = {}, len(headers) - len(r_vals)
        padded_vals = r_vals + [None] * pad_len if pad_len > 0 else r_vals
        for i, h in enumerate(headers): rec[h] = padded_vals[i] if i < len(padded_vals) else None
        records.append(rec)
    return records

@st.cache_data(ttl=300)
def fetch_all_students_today_data(_students_df, today_str, _client_gspread, headers):
    all_data = []
    if _students_df.empty: return all_data
    prog_bar = st.progress(0, text="요약 정보 로딩 중... (0%)")
    total = len(_students_df)
    for i, (_, row) in enumerate(_students_df.iterrows()):
        name, url = row["이름"], row["시트URL"]
        entry = {"name": name, "emotion_today": None, "message_today": None, "error": None}
        prog_val = (i + 1) / total
        prog_bar.progress(prog_val, text=f"'{name}' 확인 중... ({int(prog_val*100)}%)")
        if not url or not isinstance(url, str) or not url.startswith("http"):
            entry["error"] = "시트 URL 형식 오류"; all_data.append(entry); continue
        try:
            ws = _client_gspread.open_by_url(url).sheet1
            recs = get_records_from_row2_header(ws, headers)
            found = False
            for r in recs:
                if r.get("날짜") == today_str:
                    entry["emotion_today"], entry["message_today"] = r.get("감정"), r.get("하고 싶은 말")
                    found = True; break
            if not found: entry["error"] = "오늘 일기 없음"
        except Exception as e: entry["error"] = f"오류 ({type(e).__name__})"
        all_data.append(entry)
    prog_bar.empty(); return all_data

# --- OpenAI API 클라이언트 초기화 ---
client_openai = None
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if openai_api_key:
    try: client_openai = OpenAI(api_key=openai_api_key)
    except Exception as e: st.warning(f"OpenAI 클라이언트 초기화 오류: {e} (GPT 기능 비활성화)")

# --- 세션 상태 초기화 ---
session_defaults = {
    "teacher_logged_in": False, "all_students_today_data_loaded": False,
    "all_students_today_data": [], "detail_view_selected_student": ""
}
for k, v in session_defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# --- 교사용 로그인 ---
if not st.session_state.teacher_logged_in:
    st.title("🧑‍🏫 감정일기 로그인 (교사용)")
    admin_pw = st.text_input("관리자 비밀번호", type="password", key="admin_pw_final_consolidated")
    if st.button("로그인", key="admin_login_btn_final_consolidated"):
        if admin_pw == st.secrets.get("ADMIN_TEACHER_PASSWORD", "silverline"):
            st.session_state.teacher_logged_in = True
            for k, v_init in session_defaults.items(): # 로그인 시 세션 상태 초기화 (로그인 상태 제외)
                 if k != "teacher_logged_in": st.session_state[k] = v_init
            st.rerun()
        else: st.error("비밀번호가 올바르지 않습니다.")
else: # --- 교사용 기능 페이지 ---
    g_client = authorize_gspread()
    students_main_df = get_students_df(g_client)

    st.sidebar.title("🧑‍🏫 교사 메뉴")
    if st.sidebar.button("로그아웃", key="logout_final_consolidated"):
        for k, v_init in session_defaults.items(): st.session_state[k] = v_init
        st.rerun()
    if st.sidebar.button("오늘 학생 데이터 새로고침 ♻️", key="refresh_data_final_consolidated"):
        st.session_state.all_students_today_data_loaded = False
        st.cache_data.clear(); st.rerun()

    st.title("🧑‍🏫 교사용 대시보드")

    if not st.session_state.all_students_today_data_loaded:
        if students_main_df.empty:
            st.warning("'학생목록' 시트가 비어있거나 접근 불가. 확인 후 '새로고침' 해주세요.")
            st.session_state.all_students_today_data = []
            st.session_state.all_students_today_data_loaded = True # 로드 시도 완료
        else:
            today_str = datetime.today().strftime("%Y-%m-%d")
            st.session_state.all_students_today_data = fetch_all_students_today_data(
                students_main_df, today_str, g_client, EXPECTED_STUDENT_SHEET_HEADER)
            st.session_state.all_students_today_data_loaded = True
            if st.session_state.all_students_today_data: st.success("오늘 자 학생 요약 정보 로드 완료!")
            # 로드 후 자동으로 rerun할 필요 없음, Streamlit이 위젯 변경 시 다시 그림

    summary_data = st.session_state.get("all_students_today_data", [])
    tabs_list = ["오늘의 학급 감정 분포 📊", "학생들이 전달하는 메시지 💌", "학생별 일기 상세 보기 📖"]
    tab1, tab2, tab3 = st.tabs(tabs_list)

    with tab1: # 오늘의 학급 감정 분포
        st.header(tabs_list[0])
        st.markdown(f"**조회 날짜:** {datetime.today().strftime('%Y-%m-%d')}")
        if not summary_data: st.info("요약할 학생 데이터가 없습니다.")
        else:
            cats = {g: [] for g in EMOTION_GROUPS}
            cats.update({"감정 미분류": [], "일기 미제출 또는 오류": []})
            for d in summary_data:
                name = d["name"]
                if d["error"] and d["error"] != "오늘 일기 없음": cats["일기 미제출 또는 오류"].append(f"{name} ({d['error']})")
                elif d["error"] == "오늘 일기 없음" or not d["emotion_today"]: cats["일기 미제출 또는 오류"].append(name)
                elif d["emotion_today"] and isinstance(d["emotion_today"], str) and " - " in d["emotion_today"]:
                    main_emo = d["emotion_today"].split(" - ")[0].strip()
                    if main_emo in EMOTION_GROUPS: cats[main_emo].append(name)
                    else: cats["감정 미분류"].append(f"{name} (감정: {d['emotion_today']})")
                else: cats["일기 미제출 또는 오류"].append(f"{name} (감정 형식 오류: {d['emotion_today']})")
            
            cols_t1 = st.columns(len(EMOTION_GROUPS))
            for i, grp in enumerate(EMOTION_GROUPS):
                with cols_t1[i]:
                    st.subheader(f"{grp} ({len(cats[grp])}명)")
                    if cats[grp]: st.markdown("\n".join([f"- {n}" for n in sorted(cats[grp])]))
                    else: st.info("이 감정을 느낀 학생이 없습니다.")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if cats["일기 미제출 또는 오류"]:
                    with st.expander(f"📝 일기 미제출/오류 ({len(cats['일기 미제출 또는 오류'])}명)"):
                        st.markdown("\n".join([f"- {s}" for s in sorted(cats["일기 미제출 또는 오류"])]))
            with col_exp2:
                if cats["감정 미분류"]:
                    with st.expander(f"🤔 감정 미분류 ({len(cats['감정 미분류'])}명)"):
                        st.markdown("\n".join([f"- {s}" for s in sorted(cats["감정 미분류"])]))
    
    with tab2: # 학생들이 전달하는 메시지
        st.header(tabs_list[1])
        if not summary_data: st.info("요약할 학생 데이터가 없습니다.")
        else:
            neg_msg, other_msg = [], []
            for d in summary_data:
                if d["error"] or not d["emotion_today"] or not d["message_today"] or not d["message_today"].strip(): continue
                emo_full = d["emotion_today"]
                if not isinstance(emo_full, str) or " - " not in emo_full: continue
                item = {"name": d["name"], "emotion": emo_full, "message": d["message_today"].strip()}
                if emo_full.split(" - ")[0].strip() == "😢 부정": neg_msg.append(item)
                elif emo_full.split(" - ")[0].strip() in ["😀 긍정", "😐 보통"]: other_msg.append(item)
            
            if not neg_msg and not other_msg: st.success("오늘 선생님이나 친구들에게 하고 싶은 말을 적은 학생이 없습니다. 😊")
            else:
                st.subheader("😥 부정적 감정 학생들의 메시지")
                if neg_msg:
                    for item in sorted(neg_msg, key=lambda x: x['name']):
                        with st.container(border=True): st.markdown(f"**학생명:** {item['name']} (<span style='color:red;'>{item['emotion']}</span>)\n\n**메시지:**\n> {item['message']}", unsafe_allow_html=True)
                else: st.info("오늘, 부정적인 감정과 함께 메시지를 남긴 학생은 없습니다.")
                st.markdown("---")
                st.subheader("😊 그 외 감정 학생들의 메시지")
                if other_msg:
                    for item in sorted(other_msg, key=lambda x: x['name']):
                        with st.container(border=True): st.markdown(f"**학생명:** {item['name']} ({item['emotion']})\n\n**메시지:**\n> {item['message']}")
                else: st.info("오늘, 긍정적 또는 보통 감정과 함께 메시지를 남긴 학생은 없습니다.")

    with tab3: # 학생별 일기 상세 보기
        st.header(tabs_list[2])
        if students_main_df.empty: st.warning("학생 목록을 불러오세요.")
        else:
            opts_students = [""] + students_main_df["이름"].tolist()
            sel_idx = opts_students.index(st.session_state.detail_view_selected_student) if st.session_state.detail_view_selected_student in opts_students else 0
            st.session_state.detail_view_selected_student = st.selectbox("학생 선택", options=opts_students, index=sel_idx, key="sel_student_tab3_final")
            
            selected_student = st.session_state.detail_view_selected_student
            if selected_student:
                s_info = students_main_df[students_main_df["이름"] == selected_student].iloc[0]
                s_name, s_url = s_info["이름"], s_info["시트URL"]
                
                b_col1, d_col2 = st.columns([1,3])
                with b_col1:
                    if st.button(f"다른 학생 선택", key=f"back_btn_tab3_{s_name}"):
                        st.session_state.detail_view_selected_student = ""; st.rerun()
                with d_col2:
                    sel_date = st.date_input("날짜 선택", value=datetime.today(), key=f"date_pick_tab3_{s_name}")
                sel_date_str = sel_date.strftime("%Y-%m-%d")

                if not s_url or not isinstance(s_url, str) or not s_url.startswith("http"):
                    st.error(f"'{s_name}' 학생의 시트 URL이 올바르지 않습니다.")
                else:
                    try:
                        df_s_all_entries = pd.DataFrame()
                        ws_s_detail = None # 워크시트 객체는 필요할 때만 사용

                        # 학생 전체 기록 로드 (캐싱 안함 - 상세보기에선 항상 최신 데이터)
                        with st.spinner(f"'{s_name}' 학생의 전체 일기 기록 로딩 중..."):
                            ws_s_detail = g_client.open_by_url(s_url).sheet1
                            all_s_entries = get_records_from_row2_header(ws_s_detail, EXPECTED_STUDENT_SHEET_HEADER)
                            df_s_all_entries = pd.DataFrame(all_s_entries)

                        if df_s_all_entries.empty or "날짜" not in df_s_all_entries.columns:
                            st.warning(f"'{s_name}' 학생 시트에 데이터가 없거나 '날짜' 열이 없습니다.")
                        else:
                            entry_df = df_s_all_entries[df_s_all_entries["날짜"] == sel_date_str]
                            if not entry_df.empty: # 선택한 날짜에 일기가 있는 경우
                                diary_e = entry_df.iloc[0]
                                st.subheader(f"📘 {s_name} ({sel_date_str}) 일기"); st.divider()
                                st.write(f"**감정:** {diary_e.get('감정', 'N/A')}")
                                st.write(f"**감사한 일:** {diary_e.get('감사한 일', 'N/A')}")
                                st.write(f"**하고 싶은 말:** {diary_e.get('하고 싶은 말', 'N/A')}")
                                note_val = diary_e.get('선생님 쪽지', '')
                                st.write(f"**선생님 쪽지:** {note_val}")

                                note_in = st.text_area(f"✏️ 쪽지 작성/수정", value=note_val, key=f"note_in_{s_name}_{sel_date_str}")
                                if st.button(f"💾 쪽지 저장", key=f"save_note_btn_{s_name}_{sel_date_str}"):
                                    if not note_in.strip() and not note_val: st.warning("쪽지 비어있음.")
                                    else:
                                        try:
                                            r_idx, hdrs = -1, ws_s_detail.row_values(2)
                                            for i, r in enumerate(all_s_entries):
                                                if r.get("날짜") == sel_date_str: r_idx = i + 3; break
                                            if r_idx != -1:
                                                note_c_idx = hdrs.index("선생님 쪽지") + 1 if "선생님 쪽지" in hdrs else 5
                                                ws_s_detail.update_cell(r_idx, note_c_idx, note_in)
                                                st.success(f"쪽지 저장!"); st.cache_data.clear(); st.rerun() # 상세 데이터는 캐시 안하므로 st.rerun()만으로도 갱신될 것
                                            else: st.error("쪽지 저장 대상 일기 없음.")
                                        except Exception as e: st.error(f"쪽지 저장 오류: {e}")
                            else: 
                                st.info(f"'{s_name}' 학생은 {sel_date_str}에 작성한 일기가 없습니다.")

                        # 학생 전체 기록 기반 분석 (누적 & GPT) - 학생 기록이 있을 때만 버튼 표시
                        if not df_s_all_entries.empty:
                            st.markdown("---"); st.subheader("📊 학생 전체 기록 기반 분석")
                            
                            # 누적 분석 (워드클라우드, 감정통계) 버튼
                            if st.button(f"{s_name} 학생 전체 기록 누적 분석 (워드클라우드 등)", key=f"cumul_analysis_btn_{s_name}"):
                                st.write("##### 학생 전체 감정 대분류 통계")
                                if "감정" in df_s_all_entries.columns and not df_s_all_entries["감정"].empty:
                                    df_s_all_entries['감정 대분류'] = df_s_all_entries['감정'].astype(str).apply(
                                        lambda x: x.split(" - ")[0].strip() if isinstance(x, str) and " - " in x else "미분류")
                                    st.bar_chart(df_s_all_entries['감정 대분류'].value_counts())
                                else: st.info("감정 데이터 부족.")
                                
                                st.write("##### 학생 전체 '감사한 일' & '하고 싶은 말' 단어 분석")
                                wc_txts = []
                                for col in ["감사한 일", "하고 싶은 말"]:
                                    if col in df_s_all_entries.columns: wc_txts.extend(df_s_all_entries[col].dropna().astype(str).tolist())
                                wc_data = " ".join(wc_txts)
                                if wc_data.strip():
                                    try:
                                        wc_img = WordCloud(font_path=FONT_PATH, width=700, height_ratio=0.5, background_color="white").generate(wc_data)
                                        fig, ax = plt.subplots(); ax.imshow(wc_img, interpolation='bilinear'); ax.axis("off"); st.pyplot(fig)
                                    except Exception as e: st.error(f"워드클라우드 오류 (폰트: '{FONT_PATH}'): {e}")
                                else: st.info("워드클라우드용 단어 부족.")
                            
                            # GPT 누적 분석 버튼
                            if st.button(f"{s_name} 학생 전체 기록 GPT 심층 분석 📝", key=f"gpt_cumulative_btn_{s_name}"):
                                if not openai_api_key or not client_openai:
                                    st.error("OpenAI API 키 또는 클라이언트 미설정.")
                                else:
                                    with st.spinner("GPT가 학생의 전체 누적 기록을 심층 분석 중입니다... (시간 소요)"):
                                        try:
                                            # 누적 데이터 준비 (감정, 감사한 일, 하고 싶은 말)
                                            cumulative_emotions = [f"{r.get('날짜','')}: {r.get('감정','')}" for r in all_s_entries if r.get('감정')]
                                            cumulative_gratitude = [f"{r.get('날짜','')}: {r.get('감사한 일','')}" for r in all_s_entries if r.get('감사한 일','').strip()]
                                            cumulative_message = [f"{r.get('날짜','')}: {r.get('하고 싶은 말','')}" for r in all_s_entries if r.get('하고 싶은 말','').strip()]
                                            
                                            # 너무 긴 데이터는 요약 또는 일부만 전달 고려 (여기서는 일단 다 합쳐봄)
                                            # 실제로는 토큰 수 제한 고려 필요
                                            data_for_gpt = (
                                                f"전체 감정 기록:\n" + "\n".join(cumulative_emotions) + "\n\n"
                                                f"전체 감사한 일 기록:\n" + "\n".join(cumulative_gratitude) + "\n\n"
                                                f"전체 하고 싶은 말 기록:\n" + "\n".join(cumulative_message)
                                            )
                                            # 토큰 수 절약을 위해 각 항목별 요약 전달도 가능
                                            # 여기서는 간단히 합쳐서 전달
                                            
                                            # 프롬프트 포맷팅 (데이터 부분만 채움)
                                            user_prompt_content = GPT_CUMULATIVE_SYSTEM_PROMPT.format(
                                                cumulative_diary_data_for_gpt=data_for_gpt
                                            )
                                            
                                            # 시스템 프롬프트와 사용자 프롬프트를 분리하여 전달하는 것이 최신 방식에 더 부합
                                            # GPT_CUMULATIVE_SYSTEM_PROMPT의 첫 부분을 시스템 프롬프트로 사용
                                            prompt_parts = GPT_CUMULATIVE_SYSTEM_PROMPT.split("제공된 누적 일기 내용 형식 (실제 내용은 아래 데이터 형식으로 전달됩니다):")
                                            system_instructions = prompt_parts[0].strip()
                                            user_request_template = "제공된 누적 일기 내용 형식 (실제 내용은 아래 데이터 형식으로 전달됩니다):" + prompt_parts[1]
                                            
                                            formatted_user_request = user_request_template.format(
                                                 cumulative_diary_data_for_gpt=data_for_gpt
                                            )

                                            gpt_resp = client_openai.chat.completions.create(
                                                model="gpt-4o",
                                                messages=[
                                                    {"role": "system", "content": system_instructions},
                                                    {"role": "user", "content": formatted_user_request}
                                                ],
                                                temperature=0.7, max_tokens=3000 # 충분한 길이 확보
                                            )
                                            gpt_analysis = gpt_resp.choices[0].message.content
                                            st.markdown("##### 💡 GPT 누적 기록 심층 분석 리포트:")
                                            with st.expander("분석 결과 보기", expanded=True): st.markdown(gpt_analysis)
                                        except Exception as e: st.error(f"GPT 누적 분석 오류: {e}")
                    except Exception as e: st.error(f"'{s_name}' 학생 데이터 처리 중 오류: {e}")
            else: st.info("상단에서 학생을 선택하여 상세 내용을 확인하세요.")
