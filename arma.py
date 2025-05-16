import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from graphviz import Digraph
import re
# تحميل الملف
df_s1 = pd.read_excel("s1.xlsx")
df_s2 = pd.read_excel("s2.xlsx")

st.set_page_config(
    page_title="ميدان العلوم الاقتصادية ", page_icon="🗺️",
    initial_sidebar_state="expanded", layout="wide"
)

def sort_key(major):
    major = major.strip()
    if major.startswith('ثانية'):
        return (0, major)
    elif major.startswith('ثالثة'):
        return (1, major)
    elif major.startswith('ماستر1'):
        return (2, major)
    elif major.startswith('ماستر2'):
        return (3, major)
    else:
        return (4, major)  # أي تخصص غير معروف الترتيب يوضع في النهاية
def draw_specializations_graph(df, section):
    dot = Digraph(comment='خريطة التخصصات حسب المستوى والقسم', format='png')
    dot.attr(rankdir='TB', size='10,10', fontsize='14')
    dot.attr('node', fontname='Arial')

    # الجذر = اسم القسم
    dot.node(section, shape='box', style='filled', color='lightblue')

    # إنشاء فرعين رئيسيين: ثالثة، ماستر
    dot.node(f"{section}_ثالثة", label='سنة ثالثة', shape='ellipse', style='filled', color='yellow')
    dot.edge(section, f"{section}_ثالثة")

    dot.node(f"{section}_ماستر", label='ماستر', shape='ellipse', style='filled', color='lightgreen')
    dot.edge(section, f"{section}_ماستر")

    # تصفية التخصصات حسب القسم
    majors = df[df['القسم'] == section]['التخصص'].dropna().unique()

    third_year_majors = set()
    master_majors = set()

    for major in majors:
        major = major.strip()
        if major.startswith('ثالثة'):
            third_year_majors.add(major.replace('ثالثة', '').strip())
        elif major.startswith('ماستر1') or major.startswith('ماستر2') or major.startswith('ماستر'):
            cleaned = major.replace('ماستر1', '').replace('ماستر2', '').replace('ماستر', '').strip()
            master_majors.add(cleaned)

    # رسم تخصصات سنة ثالثة
    for major in sorted(third_year_majors):
        node_id = f"{section}_ثالثة_{major}"
        dot.node(node_id, label=major)
        dot.edge(f"{section}_ثالثة", node_id)

    # رسم تخصصات ماستر
    for major in sorted(master_majors):
        node_id = f"{section}_ماستر_{major}"
        dot.node(node_id, label=major)
        dot.edge(f"{section}_ماستر", node_id)

    return dot
st.markdown(
    """
    <style>
    /* جعل الاتجاه من اليمين إلى اليسار */
    body {
        direction: rtl;
        text-align: right;
    }

    /* تخصيص جدول البيانات */
    .stDataFrame {
        direction: rtl;
    }

    /* تخصيص عناوين النصوص */
    h1, h2, h3, h4, h5, h6 {
        text-align: right;
    }

    /* تخصيص عناصر الإدخال */
    .stTextInput, .stSelectbox, .stButton {
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# تهيئة CSS لتنسيق الجداول
css = """
<style>
h3 {
    color: #2c3e50;
    margin-top: 2em;
}
table {
    width: 100%;
    border-collapse: collapse;
    direction: rtl;
    margin-bottom: 2em;
}
th, td {
    border: 1px solid #ddd;
    padding: 10px;
    text-align: right;
}
th {
    background-color: #2980b9;
    color: white;
}
tr:nth-child(even) {
    background-color: #f2f2f2;
}
td a {
    color: #2980b9;
    text-decoration: none;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

def mind_module():
    st.title("📚 خريطة التخصصات والمقاييس")

    # اختيار القسم
    sections = df_s1['القسم'].dropna().unique()
    selected_section = st.selectbox("إختر القسم:", sections)
    dot = draw_specializations_graph(df_s1,selected_section)
    st.graphviz_chart(dot)
    if selected_section:
        # تصفية حسب القسم
        df1 = df_s1[df_s1['القسم'] == selected_section]
        df2 = df_s2[df_s2['القسم'] == selected_section]


        # جميع التخصصات في هذا القسم
        majors = df1['التخصص'].dropna().unique()
        majors = sorted(majors, key=sort_key)
        for major in majors:
            st.markdown(f"### 📘 {major}")

            # تصفية التخصص
            major_df1 = df1[df1['التخصص'] == major]
            major_df2 = df2[df2['التخصص'] == major]

            # إنشاء قاموس للمقاييس وربطها حسب الاسم
            s1_subjects = major_df1[['المقياس', 'الرابط']].dropna(subset=['المقياس'])
            s2_subjects = major_df2[['المقياس', 'الرابط']].dropna(subset=['المقياس'])

            max_len = max(len(s1_subjects), len(s2_subjects))

            s1_subjects = s1_subjects.reset_index(drop=True)
            s2_subjects = s2_subjects.reset_index(drop=True)

            # إنشاء جدول HTML
            html = """
            <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                    direction: rtl;
                }
                th, td {
                    border: 1px solid #ccc;
                    padding: 8px;
                    text-align: center;
                    font-family: 'Arial';
                }
                th {
                    background-color: #f0f0f0;
                    color: #333;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
            </style>

            <table>
                <thead>
                    <tr>
                        <th>مقياس السداسي الأول</th>
                        <th>رابط البرنامج</th>
                        <th>مقياس السداسي الثاني</th>
                        <th>رابط البرنامج</th>
                    </tr>
                </thead>
                <tbody>
            """

            for i in range(max_len):
                s1_name = s1_subjects.at[i, 'المقياس'] if i < len(s1_subjects) else ''
                s1_link = s1_subjects.at[i, 'الرابط'] if i < len(s1_subjects) else ''

                s2_name = s2_subjects.at[i, 'المقياس'] if i < len(s2_subjects) else ''
                s2_link = s2_subjects.at[i, 'الرابط'] if i < len(s2_subjects) else ''

                # روابط HTML آمنة
                if s1_link:
                    s1_link_html = f"<a href='{s1_link}' target='_blank'>فتح</a>"
                else:
                    s1_link_html = "—"

                if s2_link:
                    s2_link_html = f"<a href='{s2_link}' target='_blank'>فتح</a>"
                else:
                    s2_link_html = "—"

                row_html = "<tr>"
                row_html += f"<td>{s1_name}</td>"
                row_html += f"<td>{s1_link_html}</td>"
                row_html += f"<td>{s2_name}</td>"
                row_html += f"<td>{s2_link_html}</td>"
                row_html += "</tr>"

                html += row_html

            html += "</tbody></table>"

            st.markdown(html, unsafe_allow_html=True)

def ai_economic():
    st.title("🤖💼 آفاق دمج الذكاء الاصطناعي في العلوم الاقتصادية")

    st.markdown("""
    <div style='background-color: #eaf2f8; padding: 20px; border-radius: 10px; border-left: 8px solid #2980b9;'>
        <h3 style='color: #154360;'>🎯 سؤال استراتيجي</h3>
        <p style='font-size: 18px;'>كيف يمكن إدماج <strong>الذكاء الاصطناعي</strong> في برامج كلية العلوم الاقتصادية بشكل فعّال؟</p>
        <p style='font-size: 16px; color: #566573;'>نقترح ثلاث مستويات تكاملية:</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    # عرض خريطة ذهنية باستخدام graphviz
    dot = Digraph()
    dot.attr(rankdir='TB', size='10,8')
    dot.attr('node', shape='box', fontname='Arial', style='filled', fillcolor='#f0f8ff')

    dot.node("root", "📊 إدماج الذكاء الاصطناعي في العلوم الاقتصادية")

    # الفروع الثلاثة
    dot.node("L1", "1️⃣ فتح تخصصات جديدة (دمج شامل)", fillcolor='#d1f2eb')
    dot.node("L2", "2️⃣ إضافة مواد جديدة (دمج جزئي)", fillcolor='#fcf3cf')
    dot.node("L3", "3️⃣ تحديث مقررات قائمة", fillcolor='#f5c6cb')

    dot.edges([("root", "L1"), ("root", "L2"), ("root", "L3")])

    st.graphviz_chart(dot)

    # استخدام التبويبات لعرض كل مستوى بتفصيل
    tabs = st.tabs(["🔹 دمج شامل (تخصصات جديدة)",
                    "🔸 دمج جزئي (مواد مضافة)",
                    "🔹 تحديث محتوى المقررات"])

    with tabs[0]:
        st.subheader("1️⃣ فتح تخصصات جديدة")
        st.markdown("""
    **أمثلة:**
    - الاقتصاد الرقمي والذكاء الاصطناعي  
    - المالية الذكية والتنبؤ الاقتصادي  
    - تحليل البيانات الاقتصادية  

    **الأهداف:**  
    ✅ خلق هوية أكاديمية جديدة  
    ✅ جذب الطلبة المهتمين بالتكنولوجيا  
    ✅ تعزيز فرص التوظيف العالمية  
    """)

    with tabs[1]:
        st.subheader("2️⃣ إضافة مواد جديدة")
        st.markdown("""
    **أمثلة لمقررات جديدة:**
    - مقدمة في الذكاء الاصطناعي في الاقتصاد  
    - تحليل البيانات باستخدام Python  
    - نظم دعم القرار الاقتصادي  

    **المزايا:**  
    ⚡ مرونة وسرعة التنفيذ  
    ⚡ لا يتطلب إعادة هيكلة البرامج  
    """)

    with tabs[2]:
        st.subheader("3️⃣ تحديث محتوى المقررات")
        st.markdown("""
    **أمثلة لتحديثات داخل المقررات:**
    - الاقتصاد القياسي → تطبيقات تعلم الآلة  
    - التسويق → تحليل سلوك المستهلك آليًا  
    - المحاسبة → استخدام الذكاء الاصطناعي في الكشف عن الغش  

    **المزايا:**  
    🔄 تجديد المحتوى دون تغيير العناوين  
    📘 تدرّج في إدماج الأدوات الرقمية  
    """)

    # خطة إدماج تدريجية
    st.markdown("## 🛠️ خطة الإدماج المقترحة:")

    st.info("""
    - **على المدى القصير:** تحديث محتوى المواد بإدماج محاور رقمية.
    - **على المدى المتوسط:** إدخال مواد مستقلة.
    - **على المدى البعيد:** فتح تخصصات جديدة مدمجة بالكامل.
    """)

pages = {
    "📚 التخصصات والمقاييس": mind_module,
    "🗓️ برنامج العمل": ai_economic,
}
# عرض شريط التنقل للاختيار بين الصفحات

with st.sidebar:
    st.markdown("### 🏛️ ميدان العلوم الاقتصادية والتجارية وعلوم التسيير")
    selection = option_menu(
        ":اختر العملية المناسبة",
        list(pages.keys()),

        menu_icon="cast",
        default_index=0
    )
# selection = st.sidebar.radio("انتقل إلى", list(pages.keys()))

# تنفيذ الصفحة المحددة
pages[selection]()
