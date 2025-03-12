import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA

import google.generativeai as genai  # مكتبة Google AI لاستدعاء Gemini
genai.configure(api_key="AIzaSyBfyyQwcKpcJRWtwuOTWkIOu1z7P8C4Y20")
st.set_page_config(page_title='arma model',layout="wide")
# تحميل البيانات
uploaded_file = st.file_uploader("تحميل ملف البيانات", type=".xlsx")
df = pd.read_excel(uploaded_file).copy()

# واجهة Streamlit
st.title("📊 تطبيق تحليل السلاسل الزمنية باستخدام ARIMA")

# اختيار السلسلة الزمنية
st.sidebar.header("🔹 الإعدادات")
selected_series = st.sidebar.selectbox("اختر السلسلة الزمنية:", df.columns)



# تحديد معلمات نموذج ARIMA
st.sidebar.subheader("⚙️ إعدادات ARIMA")
p = st.sidebar.number_input("عدد التأخيرات (p)", min_value=0, max_value=10, value=1)
d = st.sidebar.number_input("عدد الفروق (d)", min_value=0, max_value=2, value=1)
q = st.sidebar.number_input("عدد المتوسطات المتحركة (q)", min_value=0, max_value=10, value=1)

# تقدير نموذج ARIMA
if st.button("عرض النتائج"):
    # عرض البيانات
    st.subheader("🔹 عرض البيانات")
    st.line_chart(df[selected_series])

    # اختبار جذر الوحدة (ADF)
    st.subheader("📉 اختبار جذر الوحدة (ADF)")
    adf_test = adfuller(df[selected_series])
    st.write(f"📌 قيمة ADF: {adf_test[0]:.4f}")
    st.write(f"📌 القيمة الاحتمالية (p-value): {adf_test[1]:.4f}")
    st.write("✅ البيانات مستقرة" if adf_test[1] < 0.05 else "⚠️ البيانات غير مستقرة، تحتاج إلى الفرق")

    # تحليل الارتباط الذاتي والارتباط الذاتي الجزئي
    st.subheader("🔄 تحليل الارتباط الذاتي والارتباط الذاتي الجزئي")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sm.graphics.tsa.plot_acf(df[selected_series], ax=ax[0], lags=20)
    ax[0].set_title("Autocorrelation Function (ACF)")
    sm.graphics.tsa.plot_pacf(df[selected_series], ax=ax[1], lags=20)
    ax[1].set_title("Partial Autocorrelation Function (PACF)")
    st.pyplot(fig)

    try:
        model = ARIMA(df[selected_series], order=(p, d, q))
        results = model.fit()
        st.write(results.summary())
        model1 = genai.GenerativeModel("gemini-1.5-flash")
        input_intro = "حلل نتائج نموذج ARIMA"
        response_intro = model1.generate_content(input_intro + str(results.summary))
        st.subheader("📌 analysis")
        st.write(response_intro.text)

        # التنبؤ بالنموذج
        forecast_steps = st.sidebar.slider("عدد الخطوات للتنبؤ", min_value=1, max_value=24, value=12)
        forecast = results.forecast(steps=forecast_steps)

        # عرض التنبؤات
        st.subheader("🔮 التنبؤ بالمستقبل")
        fig, ax = plt.subplots(figsize=(10, 4))
        df[selected_series].plot(ax=ax, label="Actual Data")
        forecast.plot(ax=ax, label="Forecasts", linestyle="--", color="red")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"خطأ في تقدير النموذج: {e}")

    # الاختبارات التشخيصية
    st.subheader("🔍 الاختبارات التشخيصية")
    residuals = results.resid
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(residuals, bins=30, kde=True, ax=ax[0])
    ax[0].set_title("Residuals Distribution")
    sm.graphics.tsa.plot_acf(residuals, ax=ax[1], lags=20)
    ax[1].set_title("ACF of Residuals")
    st.pyplot(fig)

