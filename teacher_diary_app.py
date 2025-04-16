import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="감정 일기장 (교사용)", page_icon="🧑‍🏫")

# ✅ Streamlit Cloud가 secrets객 GOOGLE_CREDENTIALS 이용
credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)
client = gspread.authorize(creds)

# 시트 로드
student_list_ws = client.open("학생목록").sheet1
teacher_log_ss = client.open("통합기록")
students_df = pd.DataFrame(student_list_ws.get_all_records())

# 상태 초기화
for key in ["page", "is_teacher"]:
    if key not in st.session_state:
        st.session_state[key] = "login" if key == "page" else False

# 로그인 페이지
if st.session_state.page == "login":
    st.title("🧑‍🏫 감정일기 로그인 (교사용)")
    admin_password = st.text_input("관리자 비밀번호를 입력하세요", type="password")
    if st.button("확인"):
        admin_df = pd.DataFrame({"이름": ["관리자"], "비밀번호": ["silverbronz"]})
        if admin_password.strip() == admin_df.iloc[0]["비밀번호"]:
            st.session_state.is_teacher = True
            st.session_state.page = "teacher"
        else:
            st.error("비밀번호가 올바르지 않습니다.")

# 선생님 인터페이스
elif st.session_state.page == "teacher":
    st.title("🧑‍🏫 선생님 감정일기 확인")
    selected_date = st.date_input("날짜 선택")
    date_str = selected_date.strftime("%Y-%m-%d")
    selected_name = st.selectbox("학생 선택", students_df["이름"])
    row = students_df[students_df["이름"] == selected_name].iloc[0:1].squeeze()
    name = row["이름"]
    sheet_url = row["시트URL"]
    try:
        ws = client.open_by_url(sheet_url).sheet1
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        if "날짜" not in df.columns:
            st.warning(f"'{name}' 학생의 일기 데이터가 없습니다. 입력된 내용이 없습니다.")
        else:
            today_row = df[df["날짜"] == date_str]
            if not today_row.empty:
                st.subheader(f"📘 {name}의 일기")
                st.write(f"감정: {today_row.iloc[0]['감정']}")
                st.write(f"감사한 일: {today_row.iloc[0]['감사한 일']}")
                st.write(f"하고 싶은 말: {today_row.iloc[0]['하고 싶은 말']}")

                with st.expander("📊 감정 통계 & 단어 분석"):
                    try:
                        emotion_counts = df["감정"].value_counts()
                        st.bar_chart(emotion_counts)

                        combined_text = []
                        if "감사한 일" in df.columns:
                            combined_text.extend(df["감사한 일"].dropna().astype(str).tolist())
                        if "하고 싶은 말" in df.columns:
                            combined_text.extend(df["하고 싶은 말"].dropna().astype(str).tolist())
                        text_data = " ".join(combined_text)

                        if text_data:
                            wordcloud = WordCloud(font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf", width=800, height=300, background_color="white").generate(text_data)
                            fig, ax = plt.subplots()
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig)
                    except Exception as analysis_error:
                        st.info(f"🔍 분석 데이터를 생성할 수 없어요: {analysis_error}")

                note = st.text_area(f"✏️ {name}에게 쪽지 작성", key=f"note_{name}")
                if st.button(f"💾 {name} 쪽지 저장", key=f"save_{name}"):
                    match_index = df[df["날짜"] == date_str].index[0] + 2
                    ws.update_cell(match_index, 5, note)

                    try:
                        teacher_ws = teacher_log_ss.worksheet(name)
                    except gspread.WorksheetNotFound:
                        teacher_ws = teacher_log_ss.add_worksheet(title=name, rows="100", cols="6")
                        teacher_ws.append_row(["날짜", "감정", "감사한 일", "하고 싶은 말", "선생님 쪽지", "비고"])

                    teacher_data = teacher_ws.get_all_records()
                    dates = [r["날짜"] for r in teacher_data]
                    if date_str in dates:
                        idx = dates.index(date_str) + 2
                        teacher_ws.update_cell(idx, 5, note)
                    else:
                        teacher_ws.append_row([
                            date_str,
                            today_row.iloc[0]['감정'],
                            today_row.iloc[0]['감사한 일'],
                            today_row.iloc[0]['하고 싶은 말'],
                            note,
                            ""
                        ])
                    st.success(f"{name}에게 쪽지를 저장했어요!")
    except Exception as e:
        st.warning(f"{name} 학생의 데이터를 불러오는 중 오류가 발생했어요: {e}")