import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from graphviz import Digraph
import streamlit.components.v1 as components
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
def program():
    courses = [
        "إعلام آلي 1",
        "إعلام آلي 2",
        "مثال : مقياس تحليل البيانات التسويقية والذكاء الاصطناعي"
    ]

    # اختيار المستخدم من القائمة
    selected_course = st.selectbox("اختر المقرر:", courses)

    if selected_course == "إعلام آلي 1":
        # الهدف العام
        st.markdown("""
            <div style="background-color:#D1E7DD; padding:15px; border-radius:8px; margin-bottom:20px;">
                <h3 style="color:#0F5132;">🎯 الهدف العام</h3>
                <p style="font-size:16px; color:#0F5132;">
                تمكين الطالب من فهم أساسيات البرمجة بلغة Python وتطبيقها في حل المشكلات الاقتصادية البسيطة من خلال تطوير مهارات برمجية أساسية.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # أهداف التعلم
        st.markdown("""
            <div style="background-color:#FFF3CD; padding:15px; border-radius:8px; margin-bottom:30px;">
                <h4 style="color:#664D03;">🎓 أهداف التعلم</h4>
                <ul style="font-size:15px; color:#664D03; line-height:1.6;">
                    <li>فهم مبادئ البرمجة الأساسية ومفاهيم الحوسبة.</li>
                    <li>التعرف على تركيب لغة Python وأنواع البيانات.</li>
                    <li>استخدام جمل التحكم والحلقات البرمجية.</li>
                    <li>تطبيق الدوال والبرمجة الكائنية التوجه.</li>
                    <li>القدرة على التعامل مع مكتبات تحليل البيانات الأساسية.</li>
                    <li>تنفيذ مشاريع برمجية بسيطة لتحليل البيانات الاقتصادية.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        html_code = """
        <style>
          table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Arial', sans-serif;
          }
          th {
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: center;
            font-size: 18px;
          }
          td {
            border: 1px solid #ddd;
            padding: 10px;
            vertical-align: top;
            font-size: 14px;
            color: #333;
          }
          tr:nth-child(even) {
            background-color: #f2f2f2;
          }
          tr:hover {
            background-color: #d1e7dd;
          }
          td:first-child {
            font-weight: bold;
            width: 10%;
            text-align: center;
            color: #0d6efd;
          }
        </style>

        <table>
          <thead>
            <tr>
              <th>الأسبوع</th>
              <th>المحتوى التفصيلي</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>الأسبوع الأول</td><td>مدخل إلى الحاسوب والبرمجة بلغة Python<br>- تعريف علم الحاسوب والبرمجة<br>- أهمية Python في الاقتصاد<br>- التعرف على بيئة العمل (IDE) مثل VS Code وJupyter Notebook<br>- تثبيت Python وتشغيل المحرر</td></tr>
            <tr><td>الأسبوع الثاني</td><td>كتابة أول برنامج بسيط بلغة Python<br>- برنامج "Hello, World!"<br>- أساسيات تركيب الجمل (syntax)<br>- تشغيل البرنامج وتصحيح الأخطاء البسيطة</td></tr>
            <tr><td>الأسبوع الثالث</td><td>المتغيرات وأنواع البيانات<br>- أنواع البيانات (int, float, str, bool)<br>- تعريف المتغيرات وقواعد التسمية<br>- كتابة برامج بسيطة تستخدم المتغيرات وأنواع البيانات</td></tr>
            <tr><td>الأسبوع الرابع</td><td>العمليات الحسابية والدوال الرياضية<br>- العمليات الحسابية (+, -, *, /, %)<br>- مكتبة math والدوال الرياضية<br>- تطبيقات لحل مسائل اقتصادية بسيطة</td></tr>
            <tr><td>الأسبوع الخامس</td><td>التحكم في تدفق البرنامج: جمل الشرط<br>- if، else، elif<br>- العوامل المنطقية (and, or, not)<br>- تمارين تطبيقية على الشروط</td></tr>
            <tr><td>الأسبوع السادس</td><td>الحلقات التكرارية (Loops)<br>- حلقات for و while<br>- التحكم في الحلقات (break, continue)<br>- تطبيقات عملية مثل حساب المتوسطات</td></tr>
            <tr><td>الأسبوع السابع</td><td>الدوال (Functions)<br>- تعريف الدوال واستخدامها<br>- تمرير المعاملات والقيم الراجعة<br>- بناء دوال لحل مشاكل متكررة</td></tr>
            <tr><td>الأسبوع الثامن</td><td>التعامل مع النصوص (Strings)<br>- تعريف النصوص وطرق التلاعب بها<br>- دوال النصوص الأساسية (slice, upper, lower)<br>- تمارين على تحليل بيانات نصية</td></tr>
            <tr><td>الأسبوع التاسع</td><td>القوائم (Lists) والصفوف (Tuples)<br>- تعريف القوائم والصفوف<br>- العمليات على القوائم والتكرار<br>- تطبيقات لتخزين البيانات الاقتصادية</td></tr>
            <tr><td>الأسبوع العاشر</td><td>القواميس (Dictionaries) والمجموعات (Sets)<br>- مفهوم القواميس والمجموعات<br>- إضافة، حذف، بحث في القواميس<br>- استخدام القواميس في تمثيل بيانات مرتبطة</td></tr>
            <tr><td>الأسبوع الحادي عشر</td><td>البرمجة كائنية التوجه (OOP)<br>- الفئات (Classes) والكائنات (Objects)<br>- خصائص وطرق الفئة<br>- مثال تطبيقي: نموذج شركة مالية</td></tr>
            <tr><td>الأسبوع الثاني عشر</td><td>الوراثة والتعددية الشكلية (Inheritance & Polymorphism)<br>- شرح الوراثة وبناء الفئات الفرعية<br>- التعددية الشكلية في البرمجة</td></tr>
            <tr><td>الأسبوع الثالث عشر</td><td>التعامل مع الملفات<br>- فتح، قراءة وكتابة الملفات النصية<br>- تطبيق عملي على حفظ وقراءة بيانات اقتصادية</td></tr>
            <tr><td>الأسبوع الرابع عشر</td><td>مكتبات تحليل البيانات: NumPy و pandas<br>- التعريف بالمكتبات<br>- إجراء العمليات الإحصائية على بيانات اقتصادية<br>- قراءة وتحليل بيانات من ملفات CSV</td></tr>
            <tr><td>الأسبوع الخامس عشر</td><td>مشروع نهائي<br>- دمج مفاهيم البرمجة<br>- كتابة برنامج يحلل بيانات مالية أو اقتصادية<br>- عرض المشروع ومناقشته مع الطلاب</td></tr>
          </tbody>
        </table>
        """
        st.markdown(html_code, unsafe_allow_html=True)
    elif selected_course =="إعلام آلي 2":
        st.markdown("""
        <style>
        .custom-table {
          border-collapse: collapse;
          width: 100%;
          font-family: 'Arial', sans-serif;
          direction: rtl;
        }
        .custom-table th {
          background-color: #4CAF50;
          color: white;
          padding: 10px;
          text-align: center;
          font-size: 18px;
        }
        .custom-table td {
          border: 1px solid #ddd;
          padding: 10px;
          vertical-align: top;
          background-color: #f9f9f9;
        }
        .custom-table tr:nth-child(even) {
          background-color: #f1f1f1;
        }
        </style>

        <h3 style="color:#4CAF50;">🎯 الهدف العام للمقياس</h3>
        <p style="background-color:#e8f5e9; padding: 10px; border-radius: 10px;">
        تزويد الطالب بمهارات تحليل البيانات الاقتصادية باستخدام لغة Python، عبر دمج تقنيات البرمجة، الإحصاء الوصفي، اختبارات الفرضيات، تحليل الارتباط والانحدار، مع التمهيد لمبادئ الذكاء الاصطناعي وتطبيقاته باستخدام الشبكات العصبية البسيطة.
        </p>

        <h3 style="color:#4CAF50;">🎓 أهداف التعلم</h3>
        <ul style="background-color:#e3f2fd; padding: 10px; border-radius: 10px;">
          <li>استخدام لغة Python لتحليل البيانات الاقتصادية.</li>
          <li>تطبيق أدوات الإحصاء الوصفي على بيانات حقيقية.</li>
          <li>إجراء واختبار فرضيات اقتصادية باستخدام نماذج رياضية.</li>
          <li>تحليل العلاقات بين المتغيرات الاقتصادية باستخدام الارتباط والانحدار.</li>
          <li>عرض البيانات باستخدام الرسوم البيانية.</li>
          <li>التعرف على مبادئ الذكاء الاصطناعي وبناء شبكة عصبية بسيطة.</li>
        </ul>

        <h3 style="color:#4CAF50;">📅 البرنامج الأسبوعي التفصيلي</h3>

        <table class="custom-table">
          <tr>
            <th>الأسبوع</th>
            <th>الموضوع وتفاصيله</th>
          </tr>
          <tr>
            <td>الأسبوع 1</td>
            <td><b>مدخل إلى Python وتطبيقاتها في الاقتصاد</b><br>أنواع البيانات، الشروط، التكرار، بيئة Jupyter Notebook.</td>
          </tr>
          <tr>
            <td>الأسبوع 2</td>
            <td><b>الملفات والبيانات الاقتصادية</b><br>قراءة ملفات CSV وتحليلها باستخدام pandas.</td>
          </tr>
          <tr>
            <td>الأسبوع 3</td>
            <td><b>التحليل الإحصائي الوصفي</b><br>المتوسط، الوسيط، الانحراف المعياري، باستخدام pandas وnumpy.</td>
          </tr>
          <tr>
            <td>الأسبوع 4</td>
            <td><b>تصوير البيانات</b><br>استخدام matplotlib وseaborn لعرض البيانات الاقتصادية.</td>
          </tr>
          <tr>
            <td>الأسبوع 5</td>
            <td><b>الاحتمالات والتوزيعات</b><br>Normal, Binomial – تطبيقات باستخدام scipy.stats.</td>
          </tr>
          <tr>
            <td>الأسبوع 6</td>
            <td><b>اختبار الفرضيات</b><br>اختبارات T وZ وتحليل الفرق بين المتوسطات.</td>
          </tr>
          <tr>
            <td>الأسبوع 7</td>
            <td><b>اختبار الاستقلالية (Chi-square)</b><br>تحليل العلاقة بين متغيرين اسميين.</td>
          </tr>
          <tr>
            <td>الأسبوع 8</td>
            <td><b>تحليل الارتباط</b><br>حساب معاملات Pearson وSpearman وتفسير النتائج.</td>
          </tr>
          <tr>
            <td>الأسبوع 9</td>
            <td><b>الانحدار الخطي البسيط</b><br>العلاقة بين متغيرين – تحليل اقتصادي رسومي.</td>
          </tr>
          <tr>
            <td>الأسبوع 10</td>
            <td><b>الانحدار المتعدد</b><br>بناء نموذج يؤثر فيه أكثر من متغير مستقل على متغير تابع.</td>
          </tr>
          <tr>
            <td>الأسبوع 11</td>
            <td><b>مقدمة في الاقتصاد الجزئي باستخدام Python</b><br>نمذجة منحنيات الطلب والعرض والتوازن.</td>
          </tr>
          <tr>
            <td>الأسبوع 12</td>
            <td><b>مقدمة في الذكاء الاصطناعي</b><br>الفرق بين ML وDL وتطبيقات اقتصادية بسيطة.</td>
          </tr>
          <tr>
            <td>الأسبوع 13</td>
            <td><b>الشبكات العصبية البسيطة</b><br>بناء شبكة باستخدام scikit-learn للتصنيف أو التنبؤ الاقتصادي.</td>
          </tr>
          <tr>
            <td>الأسبوع 14</td>
            <td><b>مشروع تطبيقي 1</b><br>تحليل بيانات اقتصادية حقيقية باستخدام ما تم تعلمه.</td>
          </tr>
          <tr>
            <td>الأسبوع 15</td>
            <td><b>مشروع تطبيقي 2 وعرض النتائج</b><br>تقديم النتائج في تقرير بصري وتفسير اقتصادي.</td>
          </tr>
        </table>
        """, unsafe_allow_html=True)


    else:
        # ✅ العنوان الرئيسي
        st.markdown(
            "<h2 style='text-align: center; color: #1f77b4;'>مقياس تحليل البيانات التسويقية و الذكاء الاصطناعي</h2>",
            unsafe_allow_html=True)

        # ✅ الهدف العام
        st.markdown("### 🎯 الهدف العام")
        st.markdown("""
        يهدف هذا المقياس إلى تمكين الطالب من استخدام أدوات وتقنيات الذكاء الاصطناعي لتحليل البيانات التسويقية، وفهم سلوك المستهلك، واتخاذ قرارات تسويقية مبنية على البيانات، بالاعتماد على لغة Python ومكتباتها، بالإضافة إلى أدوات الذكاء الاصطناعي الجاهزة.
        """)

        # ✅ أهداف التعلم
        st.markdown("### 📌 أهداف التعلم")
        st.markdown("""
        بنهاية هذا المقياس سيكون الطالب قادرًا على:
        - فهم أنواع البيانات التسويقية وهيكلتها.
        - استخدام لغة Python ومكتباتها في التحليل الإحصائي والتسويقي.
        - تحليل سلوك المستهلك باستخدام خوارزميات التعلم الآلي.
        - توظيف أدوات الذكاء الاصطناعي في تصميم الحملات التسويقية وتحليل نتائجها.
        - التعامل مع أدوات مثل Google Analytics، Canva AI، ونماذج LLM مثل ChatGPT.
        """)

        # ✅ جدول البرنامج التفصيلي
        html_table = """
        <style>
            table {
                direction: rtl;
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
                font-size: 16px;
            }
            th {
                background-color: #1f77b4;
                color: white;
                padding: 10px;
                text-align: center;
            }
            td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: center;
            }
            tr:nth-child(even) {
                background-color: #f2f9fc;
            }
            tr:hover {
                background-color: #e0f3ff;
            }
        </style>

        <h3 style='color: #1f77b4;'>🗂️ البرنامج التفصيلي</h3>

        <table>
            <thead>
                <tr>
                    <th>الأسبوع</th>
                    <th>محتوى المحاضرة</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>الأسبوع الأول</td><td>مقدمة في تحليل البيانات التسويقية باستخدام Python</td></tr>
                <tr><td>الأسبوع الثاني</td><td>أنواع البيانات التسويقية وهيكلتها (Structured / Unstructured)</td></tr>
                <tr><td>الأسبوع الثالث</td><td>مكتبات Python في تحليل البيانات (pandas، numpy، matplotlib، seaborn)</td></tr>
                <tr><td>الأسبوع الرابع</td><td>تمثيل سلوك المستهلك باستخدام البيانات (Customer Journey)</td></tr>
                <tr><td>الأسبوع الخامس</td><td>التحليل الاستكشافي للبيانات التسويقية (EDA) باستخدام Python</td></tr>
                <tr><td>الأسبوع السادس</td><td>تصنيف العملاء باستخدام K-means (تحليل عنقودي ذكي)</td></tr>
                <tr><td>الأسبوع السابع</td><td>التحليل بالمركبات الرئيسية (PCA) لتقليل الأبعاد التسويقية</td></tr>
                <tr><td>الأسبوع الثامن</td><td>تحليل المشاعر (Sentiment Analysis) للمراجعات التسويقية باستخدام NLP</td></tr>
                <tr><td>الأسبوع التاسع</td><td>التنبؤ بسلوك العملاء باستخدام خوارزميات التعلم الآلي (Decision Trees, Random Forest)</td></tr>
                <tr><td>الأسبوع العاشر</td><td>دراسة حالات: التسويق المستهدف (Targeted Marketing) باستخدام ML</td></tr>
                <tr><td>الأسبوع الحادي عشر</td><td>استخدام نماذج LLMs لتوليد محتوى تسويقي ذكي (مثل ChatGPT, Copy.ai)</td></tr>
                <tr><td>الأسبوع الثاني عشر</td><td>أدوات ذكاء اصطناعي جاهزة في التسويق: Google Analytics, Canva AI, Hubspot AI</td></tr>
                <tr><td>الأسبوع الثالث عشر</td><td>قياس فعالية الحملات التسويقية باستخدام تحليلات البيانات</td></tr>
                <tr><td>الأسبوع الرابع عشر</td><td>مشاريع تطبيقية في تحليل البيانات التسويقية باستخدام أدوات AI</td></tr>
                <tr><td>الأسبوع الخامس عشر</td><td>مراجعة عامة + عرض المشاريع الطلابية</td></tr>
            </tbody>
        </table>
        """

        st.markdown(html_table, unsafe_allow_html=True)


def vition():
    st.markdown("""
    <style>
    .info-box {
        border: 3px solid #1a73e8;
        background-color: #e8f0fe;
        border-radius: 15px;
        padding: 30px;
        margin: 30px 0;
        direction: rtl;
        font-family: 'Cairo', sans-serif;
        color: #0b2545;
        box-shadow: 4px 4px 10px rgba(0,0,0,0.1);
    }

    .info-title {
        font-size: 32px;
        font-weight: bold;
        color: #1a237e;
        margin-bottom: 20px;
        text-align: center;
        text-shadow: 1px 1px 2px #bbb;
    }

    .info-content {
        font-size: 22px;
        line-height: 2;
        color: #0d47a1;
        text-align: justify;
    }
    </style>

    <div class="info-box">
        <div class="info-title">رؤية لدمج الذكاء الاصطناعي وبرمجية بايثون في العلوم الاقتصادية</div>
        <div class="info-content">
            هذه الرؤية تنطلق من تأسيس متين في البرمجة بلغة Python منذ السنة الأولى من خلال مقياس الاعلام الالي 2، 
            وتطبيقها في مجالات الإحصاء والاقتصاد و مقدة في الذكاء الاصطناعي في السنة الثانية من خلال مقياس اعلام الي 2.<br><br>
            ثم تُستخدم هذه القاعدة لدمج الذكاء الاصطناعي  ميدانياً في كل تخصص ضمن الشعب الأربعة،هذا من جهة ، ومن جهة أخرى  
            يتم توظيف أدوات الذكاء الاصطناعي الجاهزة حسب خصوصية كل مجال.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .custom-table {
      border-collapse: collapse;
      width: 100%;
      direction: rtl;
      font-family: "Cairo", sans-serif;
    }
    .custom-table th, .custom-table td {
      border: 1px solid #ddd;
      padding: 10px;
      text-align: right;
    }
    .custom-table th {
      background-color: #004080;
      color: white;
      font-size: 18px;
    }
    .custom-table td {
      background-color: #f9f9f9;
      font-size: 16px;
    }
    .section-title {
      color: #004080;
      font-size: 24px;
      font-weight: bold;
      text-align: center;
      margin-bottom: 20px;
    }
    </style>

    <div class="section-title">الرؤية التفصيلية لدمج الذكاء الاصطناعي في البرامج التعليمية</div>

    <table class="custom-table">
      <thead>
        <tr>
          <th>السنة / المستوى</th>
          <th>المحتوى المقترح</th>
          <th>الهدف</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>السنة الأولى<br>إعلام آلي 1</td>
          <td>تعلم أساسيات البرمجة باستخدام لغة Python</td>
          <td>إكساب الطالب مهارات البرمجة الأساسية وبناء التفكير البرمجي</td>
        </tr>
        <tr>
          <td>السنة الثانية<br>إعلام آلي 2</td>
          <td>تطبيقات على Python في الإحصاء، الاقتصاد، الذكاء الاصطناعي<br>التعرف على مكتبات مثل NumPy، Pandas، Matplotlib، Scikit-learn</td>
          <td>الربط بين البرمجة والمجالات التطبيقية حسب تخصص الطالب</td>
        </tr>
        <tr>
          <td rowspan="2"><b>الدمج الأول<br>(المقاييس الكمية)</b></td>
          <td colspan="2"><b>دمج البرمجة والذكاء الاصطناعي في المقاييس الكمية في ليسانس وماستر</b></td>
        </tr>
        <tr>
          <td>
            - الاقتصاد القياسي<br>
            - الأساليب الكمية في التسويق<br>
            - تحليل البيانات الإدارية<br>
            - برمجيات إحصائية<br>
            - تقنيات الاستقصاء
          </td>
          <td>
            بناء قدرات تحليلية كمية متقدمة باستخدام Python وAI لتخصصات الاقتصاد، التسيير، التجارة، المالية والمحاسبة
          </td>
        </tr>
        <tr>
          <td rowspan="2"><b>الدمج الثاني <br>(أدوات AI الجاهزة)</b></td>
          <td colspan="2"><b>توظيف أدوات الذكاء الاصطناعي الجاهزة في التخصصات الدقيقة</b></td>
        </tr>
        <tr>
          <td>
            <b>العلوم الاقتصادية:</b> Power BI، ChatGPT، توقعات اقتصادية<br>
            <b>التجارة:</b> Google Analytics، Jasper AI، Canva AI<br>
            <b>التسيير:</b> تحليل المشاعر، Tableau، Notion AI<br>
            <b>المالية والمحاسبة:</b> تنبؤ مالي، كشف تزوير، محاكاة ضريبية باستخدام Excel AI
          </td>
          <td>
            تسريع تطبيق الذكاء الاصطناعي العملي وتوجيهه حسب السوق والواقع المهني لكل تخصص
          </td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)
    html_code = """
            <div dir="rtl" style="font-family: 'Arial'; text-align: right; line-height: 2;">
             <h2>أولًا: الخلفية والسياق</h2>
      <p>
        يشهد العالم تحولًا رقميًا متسارعًا أصبحت معه المهارات التقنية ضرورة في جميع التخصصات، خاصة في العلوم الاقتصادية، والتسيير، والعلوم التجارية. 
        وتماشيا مع هذا التوجه، تم إطلاق مبادرة أكاديمية طموحة تهدف إلى:
      </p>
      <ul>
        <li>دمج البرمجيات مفتوحة المصدر (Open Source) في العملية التعليمية.</li>
        <li>توظيف الذكاء الاصطناعي (AI) ومجالاته التطبيقية في تحليل البيانات وصنع القرار.</li>
        <li>تعزيز المهارات الرقمية التطبيقية لدى الطلبة الباحثين في هذه التخصصات.</li>
      </ul>

      <h2>ثانيًا: المرتكزات والمنطلقات</h2>
      <ul>
        <li>تمهيد الطلبة بمقياسي إعلام آلي 1 وإعلام آلي 2، حيث تم التركيز على لغة Python وتطبيقاتها في الإحصاء والاقتصاد والذكاء الاصطناعي.</li>
        <li>توفر مكتبات Python المجانية مثل: pandas، numpy، matplotlib، seaborn، scikit-learn، statsmodels، TensorFlow وغيرها.</li>
        <li>قابلية هذه الأدوات للدمج داخل مشاريع تحليلية واقعية ومحاكاة قرارات اقتصادية حقيقية.</li>
      </ul>

      <h2>ثالثًا: الأهداف الاستراتيجية</h2>
      <ul>
        <li>تحسين جودة التكوين عبر إدخال أدوات تحليل حديثة مفتوحة المصدر.</li>
        <li>ربط النظري بالتطبيقي عبر مشاريع تستخدم بيانات واقعية.</li>
        <li>تعزيز فرص التشغيل الذاتي وتزويد الطلبة بمهارات مطلوبة في سوق العمل.</li>
        <li>تفعيل البعد الرقمي في التكوين الجامعي بما يتماشى مع الجامعة الذكية.</li>
        <li>تنمية التفكير التحليلي والتنبئي لدى الطالب من خلال أدوات الذكاء الاصطناعي.</li>
      </ul>

      <h2>رابعًا: المقاربة البيداغوجية المعتمدة</h2>
      <ul>
        <li>برمجة أساسيات Python في إعلام آلي 1.</li>
        <li>تطبيقات تحليلية اقتصادية في إعلام آلي 2.</li>
        <li>دمج Python وAI في مقاييس تخصصية مثل:
          <ul>
            <li>الاقتصاد القياسي</li>
            <li>تحليل البيانات الاقتصادية</li>
            <li>البرمجيات الإحصائية</li>
            <li>الأساليب الكمية في الإدارة والتسويق</li>
          </ul>
        </li>
        <li>بناء مشاريع تطبيقية في نهاية كل وحدة تعليمية.</li>
        <li>استخدام بيئات عمل مفتوحة مثل:
          <ul>
            <li>Jupyter Notebook</li>
            <li>Google Colab</li>
            <li>Streamlit</li>
            <li>GitHub وKaggle</li>
          </ul>
        </li>
      </ul>

      <h2>خامسًا: أمثلة عن التكامل المقترح</h2>
      <table>
        <thead>
          <tr>
            <th>المقياس</th>
            <th>إدماج Python / AI</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>الاقتصاد القياسي</td>
            <td>بناء نماذج انحدار وتحليل العلاقات الاقتصادية باستخدام statsmodels</td>
          </tr>
          <tr>
            <td>البرمجيات الإحصائية</td>
            <td>استخدام R وPython في التحليل الوصفي، التباين، الارتباط، واختبار الفرضيات</td>
          </tr>
          <tr>
            <td>تحليل البيانات</td>
            <td>دمج أدوات معالجة البيانات وتنظيفها، وتحليلها بصريًا، والتنبؤ</td>
          </tr>
          <tr>
            <td>الأساليب الكمية في التسويق</td>
            <td>استخدام تقنيات التصنيف والتجميع لتوقع سلوك المستهلك</td>
          </tr>
          <tr>
            <td>الأساليب الكمية في الإدارة</td>
            <td>بناء نماذج دعم القرار وتحليل السيناريوهات باستخدام الذكاء الاصطناعي</td>
          </tr>
        </tbody>
      </table>

      <h2>سادسًا: المراحل المقترحة للتنفيذ</h2>
      <ul>
        <li>إعداد برنامج محدث لكل مقياس مع إدماج مبرمج لـ Python وAI.</li>
        <li>تكوين الأساتذة على أدوات الذكاء الاصطناعي والبرمجة.</li>
        <li>إنشاء موارد رقمية داعمة (دفاتر رقمية، كتيبات، مقاطع فيديو، منصات تفاعلية).</li>
        <li>تنظيم مشاريع تطبيقية ومسابقات علمية.</li>
        <li>تقييم دوري للأثر البيداغوجي والتقني لهذه الاستراتيجية.</li>
      </ul>

      <h2>سابعًا: النتائج المنتظرة</h2>
      <ul>
        <li>تحليل البيانات الاقتصادية والمالية بكفاءة.</li>
        <li>فهم واستخدام الذكاء الاصطناعي في قرارات اقتصادية وتسويقية.</li>
        <li>إنجاز دراسات جدوى وتوقعات اقتصادية ذكية.</li>
        <li>تعزيز تصنيف الجامعة في مجالات الابتكار الرقمي والتكوين الذكي.</li>
        <li>تطوير منصة وطنية للبيانات المفتوحة وتطبيقات الذكاء الاصطناعي في الاقتصاد.</li>
      </ul>

      <h2>🔍 توسيع التكامل: إدماج أدوات الذكاء الاصطناعي الجاهزة</h2>
      <p>
        بجانب البرمجيات مفتوحة المصدر، يُقترح إدماج أدوات الذكاء الاصطناعي الجاهزة (No-code/Low-code AI) في المقاييس التالية:
      </p>

      <table>
        <thead>
          <tr>
            <th>المقياس</th>
            <th>أدوات AI جاهزة</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>المنهجية العلمية</td>
            <td>ChatGPT لتحليل الأدبيات، توليد الأسئلة، صياغة الإشكاليات</td>
          </tr>
          <tr>
            <td>بحوث التسويق</td>
            <td>تحليل الآراء باستخدام Sentiment Analysis، وChatGPT لتحليل البيانات النوعية</td>
          </tr>
          <tr>
            <td>التسويق الرقمي</td>
            <td>Canva AI، Copy.ai، أدوات توليد الإعلانات المدعومة بالذكاء الاصطناعي</td>
          </tr>
          <tr>
            <td>إدارة الأعمال</td>
            <td>نماذج اتخاذ القرار باستخدام ChatGPT وAuto-GPT</td>
          </tr>
          <tr>
            <td>تحليل السوق</td>
            <td>Text Mining لاستخلاص الاتجاهات من البيانات النصية</td>
          </tr>
          <tr>
            <td>تسيير الموارد البشرية</td>
            <td>تحليل السير الذاتية، إنشاء إعلانات التوظيف، محاكاة المقابلات</td>
          </tr>
        </tbody>
      </table>

      <h2>✅ الخاتمة</h2>
      <p>
        تمثل هذه الاستراتيجية نقلة نوعية في التكوين الجامعي وتكاملًا حقيقيًا بين المعرفة النظرية والمهارة الرقمية التطبيقية.
        وهي مبادرة قابلة للتوسع نحو دراسات الماستر والدكتوراه مستقبلًا.
      </p>

    </body>
    </html>
            """

    st.markdown(html_code, unsafe_allow_html=True)


pages = {
    "📚 التخصصات والمقاييس": mind_module,
    "🗓️ برنامج العمل": ai_economic,
"💡 الرؤية" : vition,
     "🖥️البرامج": program,
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
