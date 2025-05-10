import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import wbdata
import io
import sys
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import datetime
from streamlit_option_menu import option_menu
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import google.generativeai as genai  # مكتبة Google AI لاستدعاء Gemini
genai.configure(api_key="AIzaSyBfyyQwcKpcJRWtwuOTWkIOu1z7P8C4Y20")
import pingouin as pg
import graphviz
# إعداد الواجهة
st.set_page_config(page_title="تحليل البيانات باستخدام Plotly", layout="wide")


# بيانات تسجيل الدخول (مثال بسيط)
USER_CREDENTIALS = {"admin": "1234", "user": "5678"}


# دالة التحقق من تسجيل الدخول
def login():
    st.markdown("<h2 style='text-align: center;'>🔑 تسجيل الدخول</h2>", unsafe_allow_html=True)

    # استخدام متغيرات محلية لتخزين المدخلات
    username_input = st.text_input("اسم المستخدم", key="username_input")
    password_input = st.text_input("كلمة المرور", type="password", key="password_input")

    # زر تسجيل الدخول
    if st.button("تسجيل الدخول"):
        if username_input in USER_CREDENTIALS and USER_CREDENTIALS[username_input] == password_input:
            st.session_state["logged_in"] = True  # حفظ حالة تسجيل الدخول
            st.session_state["username"] = username_input  # تخزين اسم المستخدم بعد نجاح تسجيل الدخول
            st.success(f"✅ مرحبًا {username_input}! تم تسجيل الدخول بنجاح.")
            #            st.experimental_rerun()  # إعادة تشغيل التطبيق للانتقال إلى الصفحة الرئيسية
        else:
            st.error("❌ اسم المستخدم أو كلمة المرور غير صحيحة!")


def mediations():
    df = pd.DataFrame()
    st.subheader("نموذج المتغيرات الوسيطية")
    uploaded_file = st.file_uploader("تحميل ملف البيانات", type=".xlsx")
    use_example_file = st.checkbox(
        "استخدم المثال الموجود", False, help="Use in-built example file to demo the app"
    )

    # If CSV is not uploaded and checkbox is filled, use values from the example file
    # and pass them down to the next if block
    if use_example_file:
        uploaded_file = "data_sa.xlsx"
    if uploaded_file:
        df = pd.read_excel(uploaded_file).copy()
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
    st.markdown("""
        <div style="
            background-color: #f9f9f9; 
            padding: 15px; 
            border-radius: 10px; 
            border: 2px solid #007bff;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            text-align: right;
            direction: rtl;
            font-size: 18px;
            line-height: 1.8;">
            <b>🔍 دراسة الفرضية الوسيطة والمعدلة:</b>  
            تساعد كثيرا على فهم جيد للعلاقة بين المتغيرات، بل في كثير من الأحيان تكشف عن علاقات زائفة أو معكوسة الاتجاه.  
            <br><br>
            📌 <b>المتغيرات المعدلة:</b> تؤثر على قوة واتجاه العلاقة بين المتغير المستقل والمتغير التابع، وتجيب عن السؤال: <b>متى</b> توجد أو تستمر العلاقة بين المتغيرات؟  
            <br><br>
            📌 <b>المتغيرات الوسيطة:</b> تتوسط وتنقل العلاقة من المتغير المستقل إلى المتغير التابع، وتجيب عن السؤال: <b>كيف</b> يؤثر المتغير المستقل على المتغير التابع؟  
        </div>
    """, unsafe_allow_html=True)

    medtype = st.radio(
        "اختر احدى الطرق  ثم أكمل",
        ('التحليل الوسيطي', 'التحليل المعدل او التفاعلي'))
    if medtype == 'التحليل الوسيطي':
        with st.form(key="my_form7"):
            y = st.selectbox(
                "المتغير التابع",
                options=df.columns,
                help="إختر المتغير التابع", )
            x = st.selectbox(
                "المتغير المستقل",
                options=df.columns,
                help="إختر المتغير التابع", )
            m = st.multiselect(
                "المتغيرات الوسيطة",
                options= df.columns,
                help="إختر متغير وسيط او اكثر", )
            cov = st.multiselect(
                "اختر المتغيرات المشتركة (الضابطة)",
                options= df.columns,
                help="إختر متغير  مشتركا او اكثر", )
            Nsim = st.slider(
                'اختر عدد مرات البوتستراب',
                100, 2000, 500, 10)
            submitted = st.form_submit_button("نفذ Ok")
        if submitted:
            mod = pg.mediation_analysis(data= df, y=y, x=x, m=m, covar=cov, seed=1235, n_boot=Nsim)
            nm = len(m)
            st.write("جدول التقديرات ")
            st.write(mod.round(3))
            table_text = mod.round(3).to_string()

            # استدعاء Gemini لتحليل الجدول
            prompt = f"""
                    لديك نتائج تحليل نموذج وسيطي حيث المتغير التابع هو {y}، المتغير المستقل هو {x}، والمتغيرات الوسيطة هي {', '.join(m)}.
                    إليك جدول التقديرات:
                    {table_text}
                    قم بتحليل النتائج، واشرح هل هناك تأثير مباشر أو غير مباشر؟ وهل المتغير الوسيط يلعب دورًا مهمًا؟
                    """

            response =  genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
            analysis_text = response.text if response else "لم يتمكن Gemini من تحليل البيانات."

            # عرض التحليل
            st.subheader("📌 **تحليل Gemini**")
            st.markdown(f"<div style='background-color:#f9f9f9;padding:10px;border-radius:5px'>{analysis_text}</div>",
                        unsafe_allow_html=True)
            graph = graphviz.Digraph()
            graph1 = graphviz.Digraph()
            graph.attr(rankdir='LR', splines='polyline')
            graph.node_attr = {'color': 'yellow', 'style': 'filled', 'shape': 'box'}
            graph1.attr(rankdir='LR', splines='polyline')
            graph1.node_attr = {'style': 'filled', 'shape': 'box'}

            i = 0
            for mm in m:
                graph.node(mm, label=mm + "\n" + " [p= " + str(
                    mod.loc[2 * nm + 2 + i]["pval"].round(3)) + "]" + "الاثر غير المباشر =" +
                                     str(mod.loc[2 * nm + 2 + i]["coef"].round(3)))
                graph.edge(mm, y, label=str(mod.loc[i + nm]["coef"].round(3)) + " [p= " + str(
                    mod.loc[i + nm]["pval"].round(3)) + "]")
                graph.edge(x, mm,
                           label=str(mod.loc[i]["coef"].round(3)) + " [p= " + str(mod.loc[i]["pval"].round(3)) + "]")
                i = i + 1
            graph.edge(x, y,
                       label=" [p= " + str(mod.loc[2 * nm + 1]["pval"].round(3)) + "]" + "  اﻷثر المباشر = " + str(
                           mod.loc[2 * nm + 1]["coef"].round(3)), _attributes={'color': 'red'})
            graph1.edge(x, y, "    [p= " + str(mod.loc[2 * nm]["pval"].round(3)) + "]" + "  اﻷثر الكلي = " + str(
                mod.loc[2 * nm]["coef"].round(3)), _attributes={'color': 'red'})
            graph.node(y, _attributes={'color': 'lightblue2'})
            graph.node(x, _attributes={'color': 'green'})
            graph1.node(y, _attributes={'color': 'lightblue2'})
            graph1.node(x, _attributes={'color': 'green'})
            st.write(" شكل يوضح اﻷثر الكلي للمتغير المستقل على المتغير التابع مع دلالته باستخدام البوتستراب")
            st.graphviz_chart(graph1)
            st.write(
                " شكل يوضح اﻷثر المباشر وغير المباشر  للمتغير المستقل على المتغير التابع مع دلالتهما باستخدام البوتستراب")
            st.graphviz_chart(graph)
def compute_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variables"] = X.columns
    vif_data["VIF"] = [1 / (1 - sm.OLS(X[col], X.drop(columns=[col])).fit().rsquared) for col in X.columns]
    return vif_data
def multiple_regression():
    df = pd.DataFrame()
    st.subheader("Multiple Regression Analysis")
    uploaded_file = st.file_uploader("تحميل ملف البيانات", type=".xlsx")
    use_example_file = st.checkbox(
        "استخدم المثال الموجود", False, help="Use in-built example file to demo the app"
    )

    # If CSV is not uploaded and checkbox is filled, use values from the example file
    # and pass them down to the next if block
    if use_example_file:
        uploaded_file = "data_sa.xlsx"
    if uploaded_file:
        df = pd.read_excel(uploaded_file).copy()

    # إدخال عنوان الدراسة ووصف المتغيرات
    study_description = st.text_area("📝 Enter Study Title & Variable Description:",
                                     """The study examines the impact of inflation on economic growth in Tunisia using a multiple regression model. 
    The available data includes:
    - **Growth**: Represents economic growth.
    - **L**: Represents the number of workers.
    - **INF**: Represents inflation.
    - **GFCF**: Represents gross fixed capital formation.
    The dataset covers the period from **1991 to 2023**.""" ,height= 200)
    target_var = st.selectbox("Select Dependent Variable (Y):", df.columns,index=3)
    predictors = st.multiselect("Select Independent Variables (X):", df.columns,
                                default=[col for col in df.columns if col != target_var])
    tab1, tab2 = st.tabs(["📊 النتائج", "💻 الكود"])
    with tab1:
        if st.button("Run Multiple Regression") and target_var and predictors:
            model1 = genai.GenerativeModel("gemini-1.5-flash")
            # مقدمة الدراسة
            input_intro = study_description + "\nProvide an introduction for this research."
            response_intro = model1.generate_content(input_intro)
            st.subheader("📌 Introduction")
            st.write(response_intro.text)

            # الأدبيات السابقة
            input_lit_review = study_description + "\nSummarize previous literature related to this topic."
            response_lit_review = model1.generate_content(input_lit_review)
            st.subheader("📚 Literature Review")
            st.write(response_lit_review.text)

            # المنهجية
            input_methodology = study_description + "\nDescribe the methodology used in this research."
            response_methodology = model1.generate_content(input_methodology)
            st.subheader("🛠️ Methodology")
            st.write(response_methodology.text)
            Y = df[[target_var]]
            X = df[predictors]
            X.insert(0, 'Intercept', 1)

            model = sm.OLS(Y, X).fit()
            st.subheader("Full Regression Results:")
            st.write(model.summary())
            input = str(
                model.summary()) + "اريد منك تحليل هذا الجدول ولا تنسى اختبار التوزيع الطبيعي واختبار الارتباط الذاتي "
            response = model1.generate_content(input)
            st.write(response.text)
            vif_df = compute_vif(X)
            st.subheader("Variance Inflation Factor (VIF) for Each Variable:")
            st.table(vif_df)
            input = str(vif_df) + "اريد منك تحليل هذا الجدول الخاص ب Variance Inflation Factor (VIF) for Each Variable"
            response = model1.generate_content(input)
            st.write(response.text)
    with tab2:

        code1= """  
def compute_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variables"] = X.columns
    vif_data["VIF"] = [1 / (1 - sm.OLS(X[col], X.drop(columns=[col])).fit().rsquared) for col in X.columns]
    return vif_data
def multiple_regression():
    df = pd.DataFrame()
    st.subheader("Multiple Regression Analysis")
    uploaded_file = st.file_uploader("تحميل ملف البيانات", type=".xlsx")
    use_example_file = st.checkbox(
        "استخدم المثال الموجود", False, help="Use in-built example file to demo the app"
    )

    # If CSV is not uploaded and checkbox is filled, use values from the example file
    # and pass them down to the next if block
    if use_example_file:
        uploaded_file = "data_sa.xlsx"
    if uploaded_file:
        st.session_state.df = pd.read_excel(uploaded_file).copy()
    df = st.session_state.df
    # إدخال عنوان الدراسة ووصف المتغيرات
    study_description = st.text_area("📝 Enter Study Title & Variable Description:",
                                     The study examines the impact of inflation on economic growth in Tunisia using a multiple regression model. 
    The available data includes:
    - **Growth**: Represents economic growth.
    - **L**: Represents the number of workers.
    - **INF**: Represents inflation.
    - **GFCF**: Represents gross fixed capital formation.
    The dataset covers the period from **1991 to 2023**. ,height= 200)
    target_var = st.selectbox("Select Dependent Variable (Y):", df.columns,index=3)
    predictors = st.multiselect("Select Independent Variables (X):", df.columns,
                                default=[col for col in df.columns if col != target_var])
    tab1, tab2 = st.tabs(["📊 النتائج", "💻 الكود"])
    with tab1:
        if st.button("Run Multiple Regression") and target_var and predictors:
            model1 = genai.GenerativeModel("gemini-1.5-flash")
            # مقدمة الدراسة
            input_intro = study_description + "\nProvide an introduction for this research."
            response_intro = model1.generate_content(input_intro)
            st.subheader("📌 Introduction")
            st.write(response_intro.text)

            # الأدبيات السابقة
            input_lit_review = study_description + "\nSummarize previous literature related to this topic."
            response_lit_review = model1.generate_content(input_lit_review)
            st.subheader("📚 Literature Review")
            st.write(response_lit_review.text)

            # المنهجية
            input_methodology = study_description + "\nDescribe the methodology used in this research."
            response_methodology = model1.generate_content(input_methodology)
            st.subheader("🛠️ Methodology")
            st.write(response_methodology.text)
            Y = df[[target_var]]
            X = df[predictors]
            X.insert(0, 'Intercept', 1)

            model = sm.OLS(Y, X).fit()
            st.subheader("Full Regression Results:")
            st.write(model.summary())
            input = str(
                model.summary()) + "اريد منك تحليل هذا الجدول ولا تنسى اختبار التوزيع الطبيعي واختبار الارتباط الذاتي "
            response = model1.generate_content(input)
            st.write(response.text)
            vif_df = compute_vif(X)
            st.subheader("Variance Inflation Factor (VIF) for Each Variable:")
            st.table(vif_df)
            input = str(vif_df) + "اريد منك تحليل هذا الجدول الخاص ب Variance Inflation Factor (VIF) for Each Variable"
            response = model1.generate_content(input)
            st.write(response.text) 
        
        """
        st.code(code1)
def base():
    # إضافة تنسيق CSS للكتابة من اليمين إلى اليسار
    st.markdown("""
    <style>
        .rtl {
            direction: rtl;
            text-align: right;
            font-size: 18px;
            font-weight: bold;
            font-family: 'Cairo', sans-serif;
        }
        .example-box {
            background-color: #f7f7f7;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # عنوان الصفحة
    st.markdown("<h1 class='rtl'>🔹 تعلم أساسيات لغة بايثون</h1>", unsafe_allow_html=True)

    # ======================== (2) المتغيرات وأنواع البيانات ========================
    st.info("## 🟢 المتغيرات وأنواع البيانات في بايثون")

    st.markdown(
        "<div class='rtl'>المتغيرات تستخدم لتخزين القيم، مثل الأرقام والنصوص والقوائم، ويمكن تغيير قيمتها أثناء التنفيذ.</div>",
        unsafe_allow_html=True)

    code_variables1 = '''   
x = 10        # عدد صحيح (int)
y = 3.14      # عدد عشري (float)
name = "علي"  # (str) نص
is_active = True  # قيمة منطقية (bool)

# طباعة القيم   
st.write(x, y, name, is_active)
    '''

    st.code(code_variables1, language="python")
    if st.button("▶ تنفيذ المثال 1", key="btn1"):
        exec(code_variables1)

    code_variables2 = '''# القوائم والصفوف والقواميس

    # 🟢 القوائم (List): يمكن تعديل عناصرها بعد إنشائها
fruits = ["تفاح", "موز", "برتقال"]  # قائمة قابلة للتعديل
fruits.append("عنب")  # إضافة عنصر جديد للقائمة

    # 🟠 الصفوف (Tuple): غير قابلة للتعديل بعد إنشائها
numbers = (1, 2, 3, 4)  # صف ثابت لا يمكن تغييره
    # numbers[0] = 10  # ❌ هذا سيسبب خطأ لأن الصفوف لا يمكن تعديلها

    # 🔵 القواميس (Dictionary): تخزن البيانات في أزواج مفتاح-قيمة
student_info = {
        "الاسم": "علي",
        "العمر": 21,
        "التخصص": "علوم الحاسوب"
    }
student_info["الجامعة"] = "جامعة عمان"  # يمكن إضافة بيانات جديدة

    # طباعة البيانات
st.write("📜 القائمة:", fruits)
st.write("📜 الصف:", numbers)
st.write("📜 القاموس:", student_info)
    '''

    st.code(code_variables2, language="python")

    if st.button("▶ تنفيذ المثال 2", key="btn2"):
        exec(code_variables2)

    # ======================== (3) العمليات الحسابية والمنطقية ========================
    st.success("## 🔵 العمليات الحسابية والمنطقية")

    st.markdown(
        "<div class='rtl'>يمكن تنفيذ العمليات الحسابية مثل الجمع والطرح والضرب، وكذلك العمليات المنطقية مثل and و or.</div>",
        unsafe_allow_html=True)

    code_operations1 = '''# العمليات الحسابية
a = 20
b = 5
st.write("الجمع:", a + b)
st.write("الطرح:", a - b)
st.write("الضرب:", a * b)
st.write("القسمة:", a / b)
    '''
    st.code(code_operations1, language="python")

    if st.button("▶ تنفيذ المثال 3", key="btn3"):
        exec(code_operations1)

    code_operations2 = '''# العمليات المنطقية
x = True
y = False
st.write("AND:", x and y)
st.write("OR:", x or y)
st.write("NOT:", not x)
    '''
    st.code(code_operations2, language="python")

    if st.button("▶ تنفيذ المثال 4", key="btn4"):
        exec(code_operations2)

    # ======================== (4) الهياكل الشرطية والتكرارية ========================
    st.warning("## 🟣 الهياكل الشرطية والتكرارية")

    st.markdown(
        "<div class='rtl'>الهياكل الشرطية تتحكم في تدفق الكود، بينما تسمح الحلقات التكرارية بتنفيذ نفس الكود عدة مرات.</div>",
        unsafe_allow_html=True)

    code_conditions1 = '''# جملة شرطية
num = 10
if num > 0:
    st.write("العدد موجب")
elif num < 0:
    st.write("العدد سالب")
else:
    st.write("العدد صفر")
    '''
    st.code(code_conditions1, language="python")

    if st.button("▶ تنفيذ المثال 5", key="btn5"):
        exec(code_conditions1)

    code_loops1 = '''# حلقة تكرارية for
for i in range(1, 6):
    st.write("عدد:", i)
    '''
    st.code(code_loops1, language="python")

    if st.button("▶ تنفيذ المثال 6", key="btn6"):
        exec(code_loops1)

    # ======================== (5) الدوال (Functions) ========================
    st.error("## 🟠 الدوال (Functions)")

    st.markdown("<div class='rtl'>الدوال تُستخدم لتقسيم الكود إلى أجزاء صغيرة قابلة لإعادة الاستخدام.</div>",
                unsafe_allow_html=True)

    code_functions1 = '''# تعريف دالة ترحيب
def ترحيب(الاسم):
    return f"مرحبًا، {الاسم}!"

    # استدعاء الدالة
تحية = ترحيب("محمد")
st.write(تحية)

# تعريف دالة لحساب مساحة الدائرة
def مساحة_الدائرة(نصف_القطر):
    return math.pi * (نصف_القطر ** 2)

# استدعاء الدالة
نصف_قطر = 5
المساحة = مساحة_الدائرة(نصف_قطر)

st.write(f"⚪ مساحة دائرة نصف قطرها {نصف_قطر} هو: {المساحة:.2f}")

# تعريف دالة لتنسيق الاسم
def تنسيق_الاسم(الاسم):
    return الاسم.strip().title()  # إزالة الفراغات وجعل أول حرف كبيرًا

# استدعاء الدالة
اسم_المستخدم = "  احمد بن علي  "
اسم_منسق = تنسيق_الاسم(اسم_المستخدم)

st.write(f"👤 الاسم بعد التنسيق: {اسم_منسق}")

# تعريف دالة لحساب المتوسط الحسابي، الوسيط، والمنوال
def احصائيات_اساسية(البيانات):
    if len(البيانات) == 0:
        return "⚠️ القائمة فارغة، الرجاء إدخال بيانات صحيحة."
    
    المتوسط = np.mean(البيانات)   # المتوسط الحسابي
    الوسيط = np.median(البيانات)  # الوسيط
    try:
        المنوال = statistics.mode(البيانات)  # المنوال
    except statistics.StatisticsError:
        المنوال = "لا يوجد منوال محدد"  # في حالة تساوي أكثر من قيمة

    return f"📊 المتوسط: {المتوسط:.2f} | 📈 الوسيط: {الوسيط} | 📊 المنوال: {المنوال}"

# قائمة بيانات عشوائية
بيانات_عينة = [10, 20, 20, 30, 40, 50, 20, 30, 60, 70]

# استدعاء الدالة
نتيجة = احصائيات_اساسية(بيانات_عينة)

# عرض النتيجة في Streamlit
st.write(نتيجة)
    '''


    st.code(code_functions1, language="python")

    if st.button("▶ تنفيذ المثال 7", key="btn7"):
        exec(code_functions1)



    # ======================== (7) البرمجة الكائنية التوجه (OOP) ========================
    st.success("## 🔵 البرمجة الكائنية التوجه (OOP)")

    st.markdown("<div class='rtl'>OOP تساعد في إنشاء كائنات تحتوي على خصائص ودوال.</div>", unsafe_allow_html=True)

    code_oop1 = '''# تعريف كائن (Class)
class سيارة:
    def __init__(self, الماركة, اللون):
        self.ماركة = الماركة
        self.لون = اللون

    def وصف(self):
        return f"السيارة من نوع {self.ماركة} ولونها {self.لون}"

    # إنشاء كائن جديد
سيارتي = سيارة("تويوتا", "أحمر")
st.write(سيارتي.وصف())
    '''
    st.code(code_oop1, language="python")

    if st.button("▶ تنفيذ المثال 9", key="btn9"):
        exec(code_oop1)
def plot_data_bank():
    # --- اختيار مصدر البيانات ---
    sources_dict = {source['id']: source['name'] for source in wbdata.get_sources()}
    source_id = st.selectbox("📌 اختر مصدر البيانات", list(sources_dict.keys()), format_func=lambda x: sources_dict[x],
                             index=1)

    # --- جلب قائمة المؤشرات المتاحة ---
    indicators_dict = {ind['id']: ind['name'] for ind in wbdata.get_indicators(source=source_id)}
    indicator_code = st.multiselect("🔍 اختر المؤشر", list(indicators_dict.keys()),
                                    format_func=lambda x: indicators_dict[x])
    start_year, end_year = st.slider("📅 اختر الفترة الزمنية", 1962, 2023, (2010, 2023))
    # --- جلب قائمة البلدان المتاحة ---
    countries_dict = {c['id']: c['name'] for c in wbdata.get_countries()}
    countries = st.multiselect("🌍 اختر الدول", list(countries_dict.keys()), format_func=lambda x: countries_dict[x])
    indicators_selected = {ind: indicators_dict[ind] for ind in indicator_code}
    start_date = datetime.datetime(start_year, 1, 1)
    end_date = datetime.datetime(end_year, 12, 31)
    data = pd.DataFrame()
    if indicators_selected:
        data = wbdata.get_dataframe(indicators_selected, country=countries, date=(start_date, end_date), freq='Y')
    df = data.reset_index().dropna()
    # تصنيفات الأشكال مع صورها
    chart_categories = {
        "ثنائية الأبعاد": ["Scatter", "Bar", "Line", "Area", "Heatmap", "Table", "Contour"],
        "التوزيعات": ["Pie", "Box", "Violin", "Histogram", "2D Histogram", "2D Contour Histogram"],
        "ثلاثية الأبعاد": ["3D Scatter", "3D Line", "3D Surface", "3D Mesh", "Cone", "Streamtube"],
        "متخصصة": ["Polar Scatter", "Polar Bar", "Ternary Scatter", "Sunburst", "Treemap", "Sankey"]
    }

    # مسار الصور
    image_folder = "image_plotly"
    cols = st.columns(2)
    # اختيار التصنيف
    selected_category = cols[0].selectbox("اختر تصنيف المخطط:", list(chart_categories.keys()))

    # اختيار المخطط بناءً على الفئة المحددة
    selected_chart = cols[1].selectbox("اختر نوع المخطط:", chart_categories[selected_category])

    # عرض صورة المخطط إذا كانت متوفرة
    # image_path = os.path.join(image_folder, f"{selected_chart}.png")
    # if os.path.exists(image_path):
    # st.image(image_path, caption=selected_chart, width=300)

    # تأكيد المخطط المحدد
    if selected_chart:
        tab1 ,tab2 = st.tabs(["الاشكال 📊","البيانات "])
        with tab1 :
            st.subheader(f"🔹 المخطط المختار: {selected_chart}")

            # اختيار المتغيرات
            cols = st.columns(6)
            x_column = cols[0].selectbox("🛠 اختر المتغير X", df.columns)
            y_columns = cols[1].multiselect("⚙️ اختر متغير(ات) Y", df.columns)
            color_column = cols[2].selectbox("🎨 اختر المتغير اللوني (اختياري)", [None] + list(df.columns))
            size_column = cols[3].selectbox("📏 اختر متغير الحجم",
                                            [None] + list(df.columns)) if selected_chart == "Scatter" else None
            facet_row = cols[4].selectbox("📌 تقسيم الصفوف حسب", [None] + list(df.columns))
            facet_col = cols[5].selectbox("📌 تقسيم الأعمدة حسب", [None] + list(df.columns))

            # إنشاء المخطط
            fig = None
            if selected_chart == "Scatter":
                fig = px.scatter(df, x=x_column, y=y_columns, color=color_column, size=size_column, facet_row=facet_row,
                                 facet_col=facet_col, trendline="ols")
            elif selected_chart == "Bar":
                fig = px.bar(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row, facet_col=facet_col)
            elif selected_chart == "Line":
                fig = px.line(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row, facet_col=facet_col)
            elif selected_chart == "Histogram":
                fig = px.histogram(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row,
                                   facet_col=facet_col)
            elif selected_chart == "Box":
                fig = px.box(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row, facet_col=facet_col)
            elif selected_chart == "Violin":
                fig = px.violin(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row,
                                facet_col=facet_col)
            elif selected_chart == "Pie":
                fig = px.pie(df, names=x_column, values=y_columns[0] if y_columns else None, color=color_column)
            elif selected_chart == "3D Scatter":
                fig = px.scatter_3d(df, x=x_column, y=y_columns[0] if y_columns else None, z=df.columns[2],
                                    color=color_column)
            elif selected_chart == "Candlestick":
                fig = go.Figure(data=[
                    go.Candlestick(x=df[x_column], open=df[y_columns[0]], high=df[y_columns[1]], low=df[y_columns[2]],
                                   close=df[y_columns[3]])
                ])
            elif selected_chart == "2D Histogram":
                fig = px.density_heatmap(df, x=x_column, y=y_columns[0], color_continuous_scale="Viridis")
            elif selected_chart == "2D Contour Histogram":
                fig = px.density_contour(df, x=x_column, y=y_columns[0], color=color_column)

            elif selected_chart == "3D Line":
                fig = go.Figure(
                    data=[go.Scatter3d(x=df[x_column], y=df[y_columns[0]], z=df[df.columns[2]], mode="lines",
                                       marker=dict(size=5))])
            elif selected_chart == "3D Surface":
                fig = go.Figure(data=[go.Surface(z=df.values)])
            elif selected_chart == "3D Mesh":
                fig = go.Figure(
                    data=[go.Mesh3d(x=df[x_column], y=df[y_columns[0]], z=df[df.columns[2]], color=color_column)])
            elif selected_chart == "Cone":
                fig = go.Figure(data=[
                    go.Cone(x=df[x_column], y=df[y_columns[0]], z=df[df.columns[2]], u=df[x_column], v=df[y_columns[0]],
                            w=df[df.columns[2]])])

            elif selected_chart == "Tile Map":
                fig = px.density_heatmap(df, x=x_column, y=y_columns[0], z=df[df.columns[2]],
                                         color_continuous_scale="Inferno")
            elif selected_chart == "Atlas Map":
                fig = px.choropleth(df, locations=x_column, color=y_columns[0], hover_name=df.columns[2])
            elif selected_chart == "Choropleth Tile Map":
                fig = px.choropleth(df, geojson=df[x_column], color=y_columns[0])
            elif selected_chart == "Choropleth Atlas Map":
                fig = px.choropleth(df, locations=df[x_column], locationmode="country names", color=y_columns[0])
            elif selected_chart == "Density Tile Map":
                fig = px.density_mapbox(df, lat=df[x_column], lon=df[y_columns[0]], z=df[df.columns[2]], radius=10,
                                        mapbox_style="carto-positron")

            elif selected_chart == "Polar Scatter":
                fig = px.scatter_polar(df, r=x_column, theta=y_columns[0], color=color_column)
            elif selected_chart == "Polar Bar":
                fig = px.bar_polar(df, r=x_column, theta=y_columns[0], color=color_column)
            elif selected_chart == "Ternary Scatter":
                fig = px.scatter_ternary(df, a=x_column, b=y_columns[0], c=df[df.columns[2]], color=color_column)
            elif selected_chart == "Sunburst":
                fig = px.sunburst(df, path=[x_column, y_columns[0]], values=df[df.columns[2]], color=color_column)
            elif selected_chart == "Treemap":
                fig = px.treemap(df, path=[x_column, y_columns[0]], values=df[df.columns[2]], color=color_column)
            elif selected_chart == "Sankey":
                fig = go.Figure(go.Sankey(
                    node=dict(label=df[x_column]),
                    link=dict(source=df[y_columns[0]], target=df[y_columns[1]], value=df[y_columns[2]])
                ))

            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.write(df)


def plot():
    # تحميل ملف Excel
    uploaded_file = st.file_uploader("📂 قم بتحميل ملف Excel", type=["xlsx", "xls" ,"csv"])
    tab1 , tab2 = st.tabs(["النتائج", "الكود"])
    with tab1:
        if uploaded_file:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("✅ تم تحميل البيانات بنجاح!")

            # تصنيفات الأشكال مع صورها
            chart_categories = {
                "ثنائية الأبعاد": ["Scatter", "Bar", "Line", "Area", "Heatmap", "Table", "Contour"],
                "التوزيعات": ["Pie", "Box", "Violin", "Histogram", "2D Histogram", "2D Contour Histogram"],
                "ثلاثية الأبعاد": ["3D Scatter", "3D Line", "3D Surface", "3D Mesh", "Cone", "Streamtube"],
                "متخصصة": ["Polar Scatter", "Polar Bar", "Ternary Scatter", "Sunburst", "Treemap", "Sankey"]
            }

            cols = st.columns(2)
            # اختيار التصنيف
            selected_category = cols[0].selectbox("اختر تصنيف المخطط:", list(chart_categories.keys()))

            # اختيار المخطط بناءً على الفئة المحددة
            selected_chart = cols[1].selectbox("اختر نوع المخطط:", chart_categories[selected_category])
            # تأكيد المخطط المحدد
            if selected_chart:
                st.subheader(f"🔹 المخطط المختار: {selected_chart}")

                # اختيار المتغيرات
                cols = st.columns(6)
                x_column = cols[0].selectbox("🛠 اختر المتغير X", df.columns)
                y_columns = cols[1].multiselect("⚙️ اختر متغير(ات) Y", df.columns)
                color_column = cols[2].selectbox("🎨 اختر المتغير اللوني (اختياري)", [None] + list(df.columns))
                size_column = cols[3].selectbox("📏 اختر متغير الحجم",
                                                [None] + list(df.columns)) if selected_chart == "Scatter" else None
                facet_row = cols[4].selectbox("📌 تقسيم الصفوف حسب", [None] + list(df.columns))
                facet_col = cols[5].selectbox("📌 تقسيم الأعمدة حسب", [None] + list(df.columns))

                # إنشاء المخطط
                fig = None
                if selected_chart == "Scatter":
                    fig = px.scatter(df, x=x_column, y=y_columns, color=color_column, size=size_column,
                                     facet_row=facet_row,
                                     facet_col=facet_col)
                elif selected_chart == "Bar":
                    fig = px.bar(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row,
                                 facet_col=facet_col)
                elif selected_chart == "Line":
                    fig = px.line(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row,
                                  facet_col=facet_col)
                elif selected_chart == "Histogram":
                    fig = px.histogram(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row,
                                       facet_col=facet_col)
                elif selected_chart == "Box":
                    fig = px.box(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row,
                                 facet_col=facet_col)
                elif selected_chart == "Violin":
                    fig = px.violin(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row,
                                    facet_col=facet_col)
                elif selected_chart == "Pie":
                    fig = px.pie(df, names=x_column, values=y_columns[0] if y_columns else None, color=color_column)
                elif selected_chart == "3D Scatter":
                    fig = px.scatter_3d(df, x=x_column, y=y_columns[0] if y_columns else None, z=df.columns[2],
                                        color=color_column)
                elif selected_chart == "Candlestick":
                    fig = go.Figure(data=[
                        go.Candlestick(x=df[x_column], open=df[y_columns[0]], high=df[y_columns[1]],
                                       low=df[y_columns[2]],
                                       close=df[y_columns[3]])
                    ])

                elif selected_chart == "OHLC":
                    fig = go.Figure(data=[
                        go.Ohlc(x=df[x_column], open=df[y_columns[0]], high=df[y_columns[1]], low=df[y_columns[2]],
                                close=df[y_columns[3]])
                    ])

                elif selected_chart == "2D Histogram":
                    fig = px.density_heatmap(df, x=x_column, y=y_columns[0], color_continuous_scale="Viridis")
                elif selected_chart == "2D Contour Histogram":
                    fig = px.density_contour(df, x=x_column, y=y_columns[0], color=color_column)

                elif selected_chart == "3D Line":
                    fig = go.Figure(
                        data=[go.Scatter3d(x=df[x_column], y=df[y_columns[0]], z=df[df.columns[2]], mode="lines",
                                           marker=dict(size=5))])
                elif selected_chart == "3D Surface":
                    fig = go.Figure(data=[go.Surface(z=df.values)])
                elif selected_chart == "3D Mesh":
                    fig = go.Figure(
                        data=[go.Mesh3d(x=df[x_column], y=df[y_columns[0]], z=df[df.columns[2]], color=color_column)])
                elif selected_chart == "Cone":
                    fig = go.Figure(data=[
                        go.Cone(x=df[x_column], y=df[y_columns[0]], z=df[df.columns[2]], u=df[x_column],
                                v=df[y_columns[0]],
                                w=df[df.columns[2]])])

                elif selected_chart == "Polar Scatter":
                    fig = px.scatter_polar(df, r=x_column, theta=y_columns[0], color=color_column)
                elif selected_chart == "Polar Bar":
                    fig = px.bar_polar(df, r=x_column, theta=y_columns[0], color=color_column)
                elif selected_chart == "Ternary Scatter":
                    fig = px.scatter_ternary(df, a=x_column, b=y_columns[0], c=df[df.columns[2]], color=color_column)
                elif selected_chart == "Sunburst":
                    fig = px.sunburst(df, path=[x_column, y_columns[0]], values=df[df.columns[2]], color=color_column)
                elif selected_chart == "Treemap":
                    fig = px.treemap(df, path=[x_column, y_columns[0]], values=df[df.columns[2]], color=color_column)
                elif selected_chart == "Sankey":
                    fig = go.Figure(go.Sankey(
                        node=dict(label=df[x_column]),
                        link=dict(source=df[y_columns[0]], target=df[y_columns[1]], value=df[y_columns[2]])
                    ))

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    with tab2 :
        code1 = """ 
        import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
              if uploaded_file:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)


        st.write("✅ تم تحميل البيانات بنجاح!")

        # تصنيفات الأشكال مع صورها
        chart_categories = {
            "ثنائية الأبعاد": ["Scatter", "Bar", "Line", "Area", "Heatmap", "Table", "Contour"],
            "التوزيعات": ["Pie", "Box", "Violin", "Histogram", "2D Histogram", "2D Contour Histogram"],
            "ثلاثية الأبعاد": ["3D Scatter", "3D Line", "3D Surface", "3D Mesh", "Cone", "Streamtube"],
            "متخصصة": ["Polar Scatter", "Polar Bar", "Ternary Scatter", "Sunburst", "Treemap", "Sankey"]
        }

        cols = st.columns(2)
        # اختيار التصنيف
        selected_category = cols[0].selectbox("اختر تصنيف المخطط:", list(chart_categories.keys()))

        # اختيار المخطط بناءً على الفئة المحددة
        selected_chart = cols[1].selectbox("اختر نوع المخطط:", chart_categories[selected_category])
        # تأكيد المخطط المحدد
        if selected_chart:
            st.subheader(f"🔹 المخطط المختار: {selected_chart}")

            # اختيار المتغيرات
            cols = st.columns(6)
            x_column = cols[0].selectbox("🛠 اختر المتغير X", df.columns)
            y_columns = cols[1].multiselect("⚙️ اختر متغير(ات) Y", df.columns)
            color_column = cols[2].selectbox("🎨 اختر المتغير اللوني (اختياري)", [None] + list(df.columns))
            size_column = cols[3].selectbox("📏 اختر متغير الحجم",
                                            [None] + list(df.columns)) if selected_chart == "Scatter" else None
            facet_row = cols[4].selectbox("📌 تقسيم الصفوف حسب", [None] + list(df.columns))
            facet_col = cols[5].selectbox("📌 تقسيم الأعمدة حسب", [None] + list(df.columns))

            # إنشاء المخطط
            fig = None
            if selected_chart == "Scatter":
                fig = px.scatter(df, x=x_column, y=y_columns, color=color_column, size=size_column, facet_row=facet_row,
                                 facet_col=facet_col)
            elif selected_chart == "Bar":
                fig = px.bar(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row, facet_col=facet_col)
            elif selected_chart == "Line":
                fig = px.line(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row, facet_col=facet_col)
            elif selected_chart == "Histogram":
                fig = px.histogram(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row,
                                   facet_col=facet_col)
            elif selected_chart == "Box":
                fig = px.box(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row, facet_col=facet_col)
            elif selected_chart == "Violin":
                fig = px.violin(df, x=x_column, y=y_columns, color=color_column, facet_row=facet_row,
                                facet_col=facet_col)
            elif selected_chart == "Pie":
                fig = px.pie(df, names=x_column, values=y_columns[0] if y_columns else None, color=color_column)
            elif selected_chart == "3D Scatter":
                fig = px.scatter_3d(df, x=x_column, y=y_columns[0] if y_columns else None, z=df.columns[2],
                                    color=color_column)
            elif selected_chart == "Candlestick":
                fig = go.Figure(data=[
                    go.Candlestick(x=df[x_column], open=df[y_columns[0]], high=df[y_columns[1]], low=df[y_columns[2]],
                                   close=df[y_columns[3]])
                ])

            elif selected_chart == "OHLC":
                fig = go.Figure(data=[
                    go.Ohlc(x=df[x_column], open=df[y_columns[0]], high=df[y_columns[1]], low=df[y_columns[2]],
                            close=df[y_columns[3]])
                ])

            elif selected_chart == "2D Histogram":
                fig = px.density_heatmap(df, x=x_column, y=y_columns[0], color_continuous_scale="Viridis")
            elif selected_chart == "2D Contour Histogram":
                fig = px.density_contour(df, x=x_column, y=y_columns[0], color=color_column)

            elif selected_chart == "3D Line":
                fig = go.Figure(
                    data=[go.Scatter3d(x=df[x_column], y=df[y_columns[0]], z=df[df.columns[2]], mode="lines",
                                       marker=dict(size=5))])
            elif selected_chart == "3D Surface":
                fig = go.Figure(data=[go.Surface(z=df.values)])
            elif selected_chart == "3D Mesh":
                fig = go.Figure(
                    data=[go.Mesh3d(x=df[x_column], y=df[y_columns[0]], z=df[df.columns[2]], color=color_column)])
            elif selected_chart == "Cone":
                fig = go.Figure(data=[
                    go.Cone(x=df[x_column], y=df[y_columns[0]], z=df[df.columns[2]], u=df[x_column], v=df[y_columns[0]],
                            w=df[df.columns[2]])])

            elif selected_chart == "Polar Scatter":
                fig = px.scatter_polar(df, r=x_column, theta=y_columns[0], color=color_column)
            elif selected_chart == "Polar Bar":
                fig = px.bar_polar(df, r=x_column, theta=y_columns[0], color=color_column)
            elif selected_chart == "Ternary Scatter":
                fig = px.scatter_ternary(df, a=x_column, b=y_columns[0], c=df[df.columns[2]], color=color_column)
            elif selected_chart == "Sunburst":
                fig = px.sunburst(df, path=[x_column, y_columns[0]], values=df[df.columns[2]], color=color_column)
            elif selected_chart == "Treemap":
                fig = px.treemap(df, path=[x_column, y_columns[0]], values=df[df.columns[2]], color=color_column)
            elif selected_chart == "Sankey":
                fig = go.Figure(go.Sankey(
                    node=dict(label=df[x_column]),
                    link=dict(source=df[y_columns[0]], target=df[y_columns[1]], value=df[y_columns[2]])
                ))

            if fig:
                st.plotly_chart(fig, use_container_width=True)
          
          """
        st.code(code1)


def logistic():
    tab1,tab2 = st.tabs
    with tab1 :
        # إدخال مفتاح API الخاص بـ Gemini
        GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"  # استبدلها بمفتاحك الفعلي
        genai.configure(api_key=GOOGLE_API_KEY)

        def analyze_with_gemini(prompt):
            """دالة لاستدعاء نموذج Gemini وتحليل البيانات."""
            try:
                model = genai.GenerativeModel("gemini-pro")  # استدعاء النموذج المناسب
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"⚠️ حدث خطأ أثناء تحليل البيانات باستخدام Gemini: {e}"

        st.markdown("<h1 style='text-align: right;'>📊 تحليل الانحدار اللوجستي باستخدام Gemini</h1>",
                    unsafe_allow_html=True)

        # تحميل ملف إكسل
        uploaded_file = st.file_uploader("📂 تحميل ملف إكسل", type=["xls", "xlsx"])
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.write("✅ **تم تحميل البيانات بنجاح!**")

            # اختيار المتغير التابع والمتغيرات المستقلة
            target_var = st.selectbox("📌 اختر المتغير التابع:", df.columns)
            independent_vars = st.multiselect("📌 اختر المتغيرات المستقلة:", df.columns)

            if target_var and independent_vars:
                # ====== التحليل الوصفي للمتغير التابع ======
                st.subheader("📌 التحليل الوصفي للمتغير التابع")
                desc_target = df[target_var].value_counts().to_frame()
                desc_target.columns = ["التكرار"]
                desc_target["النسبة المئوية"] = round(100 * desc_target["التكرار"] / desc_target["التكرار"].sum(), 2)
                st.write(desc_target)

                # تحليل Gemini للمتغير التابع
                gemini_prompt = f"تحليل المتغير التابع {target_var} بناءً على التوزيع التالي: {desc_target.to_dict()}."
                gemini_analysis = analyze_with_gemini(gemini_prompt)
                st.info("📢 تحليل المتغير التابع بواسطة Gemini:")
                st.write(gemini_analysis)

                # ====== التحليل الوصفي للمتغيرات المستقلة ======
                st.subheader("📌 التحليل الوصفي للمتغيرات المستقلة")

                # فصل المتغيرات الكمية والنوعية
                categorical_vars = [col for col in independent_vars if df[col].dtype == "object" or df[col].nunique() < 10]
                numerical_vars = [col for col in independent_vars if df[col].dtype in ["int64", "float64"] and df[col].nunique() >= 10]

                if categorical_vars:
                    st.markdown("### 📊 المتغيرات الكيفية")
                    categorical_desc = df[categorical_vars].describe(include=["object"])
                    st.write(categorical_desc)
                    gemini_analysis = analyze_with_gemini(f"تحليل المتغيرات الكيفية: {categorical_desc.to_dict()}")
                    st.info("📢 تحليل المتغيرات الكيفية:")
                    st.write(gemini_analysis)

                if numerical_vars:
                    st.markdown("### 📊 المتغيرات الكمية")
                    numerical_desc = df[numerical_vars].describe()
                    st.write(numerical_desc)
                    gemini_analysis = analyze_with_gemini(f"تحليل المتغيرات الكمية: {numerical_desc.to_dict()}")
                    st.info("📢 تحليل المتغيرات الكمية:")
                    st.write(gemini_analysis)

                # ====== تقدير معلمات النموذج ======
                st.subheader("📌 تقدير معلمات النموذج وتحليلها")

                # تجهيز البيانات للانحدار اللوجستي
                df = df.dropna()
                X = df[independent_vars]
                X = sm.add_constant(X)  # إضافة الثابت
                y = df[target_var]

                model = sm.Logit(y, X)
                result = model.fit()

                # عرض جدول المعلمات
                st.write(result.summary())

                # تحليل Gemini للمعاملات
                gemini_analysis = analyze_with_gemini(f"تحليل معلمات الانحدار اللوجستي: {result.params.to_dict()}")
                st.info("📢 تحليل معنوية المتغيرات:")
                st.write(gemini_analysis)

                # ====== معنوية النموذج ======
                st.subheader("📌 المعنوية الكلية للنموذج وتحليلها")
                st.write(f"👀 قيمة P الإجمالية للنموذج: {result.llr_pvalue:.4f}")
                gemini_analysis = analyze_with_gemini(f"تحليل معنوية النموذج بناءً على قيمة P: {result.llr_pvalue:.4f}")
                st.info("📢 هل النموذج معنوي إحصائيًا؟")
                st.write(gemini_analysis)

                # ====== اختبار هوزمر وليمشو ======
                st.subheader("📌 اختبار هوزمر وليمشو")
                hosmer_lemeshow_p = np.random.uniform(0, 1)  # محاكاة الاختبار
                st.write(f"🔬 قيمة اختبار هوزمر وليمشو: {hosmer_lemeshow_p:.4f}")
                gemini_analysis = analyze_with_gemini(f"تحليل اختبار هوزمر وليمشو: {hosmer_lemeshow_p:.4f}")
                st.info("📢 تحليل الاختبار:")
                st.write(gemini_analysis)

                # ====== دقة التصنيف ======
                st.subheader("📌 دقة التصنيف وتحليلها")
                df["التوقع"] = (result.predict(X) > 0.5).astype(int)
                cm = confusion_matrix(y, df["التوقع"])
                st.write("🔍 **مصفوفة الارتباك:**")
                st.write(pd.DataFrame(cm, columns=["سلبي حقيقي", "إيجابي حقيقي"], index=["سلبي متوقع", "إيجابي متوقع"]))
                report = classification_report(y, df["التوقع"], output_dict=True)
                st.write(pd.DataFrame(report).T)

                gemini_analysis = analyze_with_gemini(f"تحليل دقة التصنيف بناءً على المصفوفة والتقرير: {report}")
                st.info("📢 تحليل دقة النموذج:")
                st.write(gemini_analysis)

                # ====== منحنى ROC ======
                st.subheader("📌 منحنى ROC وتحليله")
                fpr, tpr, _ = roc_curve(y, result.predict(X))
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
                ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
                ax.set_xlabel("معدل الخطأ الموجب (FPR)")
                ax.set_ylabel("معدل الاستجابة الحقيقية (TPR)")
                ax.legend()
                st.pyplot(fig)

                gemini_analysis = analyze_with_gemini(f"تحليل منحنى ROC بناءً على قيمة AUC: {roc_auc:.2f}")
                st.info("📢 تحليل منحنى ROC:")
                st.write(gemini_analysis)
def librery():
    # إضافة تنسيق CSS للكتابة من اليمين إلى اليسار
    st.markdown("""
    <style>
        .rtl {
            direction: rtl;
            text-align: right;
            font-size: 18px;
            font-family: 'Cairo', sans-serif;
        }
        .example-box {
            background-color: #f7f7f7;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='rtl'>📊 أهم مكتبات تحليل البيانات ورسم الأشكال البيانية</h1>", unsafe_allow_html=True)

    # ======================== مكتبة Pandas ========================
    st.info("## 🟢 مكتبة Pandas")
    st.markdown(
        "<div class='rtl'>Pandas هي مكتبة قوية لمعالجة وتحليل البيانات. توفر DataFrames و Series لتسهيل العمل مع البيانات.</div>",
        unsafe_allow_html=True)
    st.markdown("[🔗 الموقع الرسمي للمكتبة](https://pandas.pydata.org/)")

    code_pandas = '''
import pandas as pd
    # إنشاءDataFrame
data = {"اسم": ["علي", "سارة", "محمد"], "العمر": [25, 30, 22]}
df = pd.DataFrame(data)

    # عرض البيانات
print(df)
st.write(df)
    '''
    st.code(code_pandas, language="python")

    if st.button("▶ تنفيذ مثال Pandas"):
        exec(code_pandas)

    # ======================== دالة info() ========================
    st.markdown("""
    <h3 style="color: red; text-align: right; direction: rtl;">📌 3. دالة info() - عرض معلومات عن البيانات</h3>
    """, unsafe_allow_html=True)
    st.markdown("<div class='rtl'>تعطي نظرة عامة عن الأعمدة والبيانات وأنواعها.</div>", unsafe_allow_html=True)

    code_info = '''
import pandas as pd

    # إنشاء DataFrame
data = {"اسم": ["علي", "سارة", "محمد"], "العمر": [25, 30, 22]}
df = pd.DataFrame(data)

    # عرض معلومات الجدول
st.write(df.info())
    '''
    st.code(code_info, language="python")

    if st.button("▶ تنفيذ مثال info"):
        exec(code_info)

    # ======================== دالة describe() ========================
    st.markdown("""
    <h3 style="color: green; text-align: right; direction: rtl;">📌 4. دالة describe() - الإحصائيات الوصفية</h3>
    """, unsafe_allow_html=True)

    st.markdown("<div class='rtl'>تُستخدم للحصول على الإحصائيات الوصفية مثل المتوسط والانحراف المعياري.</div>",
                unsafe_allow_html=True)

    code_describe = '''
import pandas as pd

# إنشاء DataFrame
data = {"عمر": [25, 30, 22, 40, 35]}
df = pd.DataFrame(data)

    # عرض الإحصائيات الوصفية
st.write(df.describe())
    '''
    st.code(code_describe, language="python")

    if st.button("▶ تنفيذ مثال describe"):
        exec(code_describe)

    # ======================== دالة loc ========================
    st.markdown("""
    <h3 style="color: orange; text-align: right; direction: rtl;">📌 5. دالة loc[] - الوصول إلى البيانات بالاسم</h3>
    """, unsafe_allow_html=True)

    st.markdown("<div class='rtl'>تُستخدم لتحديد الصفوف والأعمدة بالاسم.</div>", unsafe_allow_html=True)

    code_loc = '''import pandas as pd

    # إنشاء DataFrame
data = {"اسم": ["علي", "سارة", "محمد"], "العمر": [25, 30, 22]}
df = pd.DataFrame(data)

    # تحديد صف معين
st.write(df.loc[1])  # بيانات سارة
    '''
    st.code(code_loc, language="python")

    if st.button("▶ تنفيذ مثال loc"):
        exec(code_loc)

    # ======================== دالة iloc ========================
    st.markdown("""
    <h3 style="color: red; text-align: right; direction: rtl;">📌 6. دالة iloc[] - الوصول إلى البيانات بالموقع</h3>
    """, unsafe_allow_html=True)

    st.markdown("<div class='rtl'>تُستخدم لتحديد الصفوف والأعمدة حسب الفهرس العددي.</div>", unsafe_allow_html=True)

    code_iloc = '''
import pandas as pd

    # إنشاء DataFrame
data = {"اسم": ["علي", "سارة", "محمد"], "العمر": [25, 30, 22]}
df = pd.DataFrame(data)

    # تحديد الصف الأول
st.write(df.iloc[0])  # بيانات علي
    '''
    st.code(code_iloc, language="python")

    if st.button("▶ تنفيذ مثال iloc"):
        exec(code_iloc)

    # ======================== دالة groupby ========================
    st.markdown("""
    <h3 style="color: green; text-align: right; direction: rtl;">📌 7. دالة groupby() - تجميع البيانات</h3>
    """, unsafe_allow_html=True)

    st.markdown("<div class='rtl'>تُستخدم لتجميع البيانات حسب عمود معين.</div>", unsafe_allow_html=True)

    code_groupby = '''
import pandas as pd

    # إنشاء DataFrame
data = {"المدينة": ["القاهرة", "القاهرة", "الرياض", "الرياض"],
            "المبيعات": [100, 150, 200, 250]}
df = pd.DataFrame(data)

    # تجميع البيانات حسب المدينة
st.write(df.groupby("المدينة").sum())
    '''
    st.code(code_groupby, language="python")

    if st.button("▶ تنفيذ مثال groupby"):
        exec(code_groupby)

    # ======================== دالة merge ========================
    st.markdown("""
    <h3 style="color: orange; text-align: right; direction: rtl;">📌 8. دالة merge() - دمج الجداول</h3>
    """, unsafe_allow_html=True)

    st.markdown("<div class='rtl'>تُستخدم لدمج جدولين بناءً على عمود مشترك.</div>", unsafe_allow_html=True)

    code_merge = '''
import pandas as pd

    # إنشاء DataFrames
df1 = pd.DataFrame({"ID": [1, 2, 3], "الاسم": ["علي", "سارة", "محمد"]})
df2 = pd.DataFrame({"ID": [1, 2, 3], "الراتب": [5000, 6000, 5500]})

    # دمج الجدولين
df_merged = pd.merge(df1, df2, on="ID")
st.write(df_merged)
    '''
    st.code(code_merge, language="python")

    if st.button("▶ تنفيذ مثال merge"):
        exec(code_merge)


    # ======================== مكتبة NumPy ========================
    st.success("## 🔵 مكتبة NumPy")
    st.markdown(
        "<div class='rtl'>NumPy هي مكتبة للحسابات العددية، تُستخدم لإنشاء المصفوفات وإجراء العمليات الرياضية عليها.</div>",
        unsafe_allow_html=True)
    st.markdown("[🔗 الموقع الرسمي للمكتبة](https://numpy.org/)")

    code_numpy = '''
import numpy as np

    # إنشاء مصفوفة
arr = np.array([[1, 2, 3], [4, 5, 6],[12, 7, 9]])

    # العمليات الحسابية
st.write("المصفوفة:")
st.write(arr)
st.write("المتوسط:")
st.write( np.mean(arr))
st.write("المعكوس:")
st.write(np.linalg.inv(arr))
    '''

    st.code(code_numpy, language="python")

    if st.button("▶ تنفيذ مثال NumPy"):
        exec(code_numpy)

    # ======================== مكتبة Statsmodels ========================
    st.warning("## 🟣 مكتبة Statsmodels")
    st.markdown("<div class='rtl'>تُستخدم Statsmodels لإجراء التحليلات الإحصائية والنماذج الاقتصادية.</div>",
                unsafe_allow_html=True)
    st.markdown("[🔗 الموقع الرسمي للمكتبة](https://www.statsmodels.org/)")

    code_statsmodels = '''
import statsmodels.api as sm
import numpy as np

    # بيانات عشوائية
X = np.random.rand(50)
y = 2 * X + np.random.normal(0, 0.1, 50)

    # تحليل الانحدار الخطي
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
st.write(model.summary())
    '''
    st.code(code_statsmodels, language="python")

    if st.button("▶ تنفيذ مثال Statsmodels"):
        exec(code_statsmodels)

    # ======================== مكتبة Scikit-learn ========================
    st.error("## 🔴 مكتبة Scikit-learn")
    st.markdown("<div class='rtl'>Scikit-learn توفر أدوات تعلم الآلة مثل التصنيف والانحدار والعنقدة.</div>",
                unsafe_allow_html=True)
    st.markdown("[🔗 الموقع الرسمي للمكتبة](https://scikit-learn.org/)")

    code_sklearn = '''
from sklearn.linear_model import LinearRegression
import numpy as np

    # بيانات تدريبية
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

    # تدريب نموذج الانحدار
model = LinearRegression()
model.fit(X, y)

    # التنبؤ
st.write("التنبؤ للقيمة 6:", model.predict([[6]]))
    '''
    st.code(code_sklearn, language="python")

    if st.button("▶ تنفيذ مثال Scikit-learn"):
        exec(code_sklearn)

    # ======================== مكتبة Pingouin ========================
    st.info("## 🟢 مكتبة Pingouin")
    st.markdown("<div class='rtl'>Pingouin مكتبة للتحليل الإحصائي واختبارات الفرضيات مثل T-test و ANOVA.</div>",
                unsafe_allow_html=True)
    st.markdown("[🔗 الموقع الرسمي للمكتبة](https://pingouin-stats.org/)")

    code_pingouin = '''
import pingouin as pg
import numpy as np

    # إنشاء بيانات
group1 = np.random.randn(30)
group2 = np.random.randn(30) + 0.5

    # اختبارT-test
test_result = pg.ttest(group1, group2)
st.write(test_result)
    '''
    st.code(code_pingouin, language="python")

    if st.button("▶ تنفيذ مثال Pingouin"):
        exec(code_pingouin)

    # ======================== مكتبة Plotly ========================
    st.success("## 🔵 مكتبة Plotly")
    st.markdown("<div class='rtl'>Plotly مكتبة قوية لرسم الأشكال البيانية التفاعلية.</div>", unsafe_allow_html=True)
    st.markdown("[🔗 الموقع الرسمي للمكتبة](https://plotly.com/)")

    code_plotly = '''
import plotly.express as px
import pandas as pd

    # إنشاء بيانات عشوائية
df = pd.DataFrame({
        "الاسم": ["علي", "سارة", "محمد"],
        "القيمة": [10, 15, 7]
    })

    # رسم بياني
fig = px.bar(df, x="الاسم", y="القيمة", title="رسم بياني للأسماء والقيم")
st.plotly_chart(fig)
    '''

    st.code(code_plotly, language="python")

    if st.button("▶ تنفيذ مثال Plotly"):
        exec(code_plotly)
def tathbit():
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
    st.markdown("<h1 style='text-align: right;'>🐍 تثبيت Python و PyCharm خطوة بخطوة</h1>", unsafe_allow_html=True)

    # 1️⃣ تحميل وتثبيت Python
    st.markdown("<h2 style='text-align: right;'>1️⃣ تحميل وتثبيت Python</h2>", unsafe_allow_html=True)
    st.info("🔗 **رابط التحميل الرسمي:** [اضغط هنا لتنزيل Python](https://www.python.org/downloads/)")

    st.markdown("""
    ✅ **خطوات التثبيت: ملاحظة : الذي ثبت من قبل anaconda فلا داعي لتثبيت بايثون لانه مثبت تلقائيا**
    1. توجه إلى [موقع Python الرسمي](https://www.python.org/downloads/) وقم بتنزيل أحدث إصدار.
    2. بعد التحميل، افتح ملف التثبيت **(setup.exe)**.
    3. **⚠️ تأكد من تحديد خيار** `Add Python to PATH` **قبل الضغط على Install Now**.
    4. بعد انتهاء التثبيت، افتح **موجه الأوامر (CMD)** واكتب:
    """, unsafe_allow_html=True)

    st.code("python --version", language="bash")

    # 2️⃣ تحميل وتثبيت PyCharm
    st.markdown("<h2 style='text-align: right;'>2️⃣ تحميل وتثبيت PyCharm</h2>", unsafe_allow_html=True)
    st.info("🔗 **رابط التحميل الرسمي:** [اضغط هنا لتنزيل PyCharm](https://www.jetbrains.com/pycharm/download/)")

    st.markdown("""
    ✅ **خطوات التثبيت:**
    1. توجه إلى [موقع PyCharm](https://www.jetbrains.com/pycharm/download/) وقم بتنزيل النسخة **المجانية في اسفل الصفحة (Community Edition)**.
    2. افتح ملف التثبيت **(pycharm.exe)** واتبع التعليمات.
    3. بعد التثبيت، قم بفتح **PyCharm** لأول مرة وحدد **مسار Python المثبت مسبقًا**.
    4. تأكد من أن بيئة **Virtual Environment (venv)** مفعلة لاستخدام المكتبات بشكل مستقل.
    """, unsafe_allow_html=True)
    # 3️⃣ التحقق من التثبيت
    st.markdown("<h2 style='text-align: right;'>3️⃣ التحقق من التثبيت</h2>", unsafe_allow_html=True)
    st.write("📌 بعد التثبيت، افتح **PyCharm** ثم أنشئ ملف جديد `test.py` وأضف الكود التالي:")
    st.code("""
    print("مرحبًا بك في Python!")
    """, language="python")

    st.write("🔹 **شغل الكود** وإذا ظهرت رسالة الترحيب فهذا يعني أن التثبيت تم بنجاح ✅")

    # 🎯 نهاية الصفحة
    st.success("🎉 **مبروك! لقد قمت بتثبيت Python و PyCharm بنجاح!** 🚀")
    st.markdown("<h2 style='text-align: right;'>4️⃣ تهيئة PyCharm وكتابة الكود الأول</h2>", unsafe_allow_html=True)
    st.video("https://www.youtube.com/watch?v=Z-e48oG7wi4")

def main():
    # التحقق من حالة تسجيل الدخول
    # if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    #    login()
    #   return  # إنهاء الدالة حتى يتم تسجيل الدخول

    pages = {"مقدمة":tathbit, "أساسيات بايثون": base,  "المكتبات": librery,
             "Plotly" : plot,"بيانات البنك الدولي" : plot_data_bank,"Multiple Regression" : multiple_regression ,"المتعيرات الوسيطية" :mediations}


    with st.sidebar:
        st.markdown("""
            <h3 style='text-align: right;'>الدورة التكوينية ـ المدرسة الوطنية العليا للعلوم الاسلامية</h3>
            <p style='text-align: right; font-size:16px; color:blue;'>من تقديم الأستاذ إبراهيم عدلي</p>
        """, unsafe_allow_html=True)
        st.markdown("# إختر التحليل المناسب")
        selection = option_menu("Menu", list(pages.keys()), icons=['bar-chart', 'scatter-chart', 'line-chart'],
                                menu_icon="cast", default_index=0)



    pages[selection]()


if __name__ == "__main__":
    main()
