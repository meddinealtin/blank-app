import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Streamlit Başlığı
st.title("Zaman Serisi Tahmini (Prophet ile)")

# CSV Dosyasını Yükleme
uploaded_file = st.file_uploader("veri", type="csv")
if uploaded_file:
    # Veriyi Yükleme
    df = pd.read_csv(uploaded_file)

    # Tarih ve Değer Sütunlarını Seçme
    st.write("Yüklenen Veri:")
    st.write(df.head())
    date_column = st.selectbox("Tarih sütununu seçin:", df.columns)
    value_column = st.selectbox("Değer sütununu seçin:", [col for col in df.columns if col != date_column])

    # Tarih formatı ve sıralama
    df = df[[date_column, value_column]].rename(columns={date_column: "ds", value_column: "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")  # Tarih formatı kontrolü
    df = df.dropna(subset=["ds", "y"])  # Hatalı verileri kaldır
    df = df.sort_values("ds")  # Tarihe göre sıralama

    # Grafik Gösterimi
    st.subheader("Zaman Serisi Grafiği (Aylık)")
    df_monthly = df.resample('M', on='ds').sum()  # Aylık veri toplama
    st.line_chart(df_monthly["y"])

    # Prophet Modeli Oluşturma
    model = Prophet()
    model.fit(df)

    # Tahmin Periyodu (Ay cinsinden)
    months = st.slider("Kaç ay ileriye tahmin yapılacak?", min_value=1, max_value=12, value=6)
    periods = months * 30  # Ayı gün sayısına dönüştürme (yaklaşık olarak)
    future = model.make_future_dataframe(periods=periods)

    # Tahmin Yapma
    forecast = model.predict(future)

    # Tahminleri Görselleştirme
    st.subheader("Tahmin Sonuçları")
    st.write(forecast.tail())

    st.subheader("Tahmin Grafiği")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("Bileşenler Grafiği")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)


    # Gerçek Veri ve Tahmin Edilen Veriyi Çizgi Grafiği Olarak Gösterme
    st.subheader("Gerçek ve Tahmin Edilen Verilerin Çizgi Grafiği")
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    # Gerçek veriyi çiziyoruz
    ax.plot(df["ds"], df["y"], label="Gerçek Veri", color="blue")
    
    # Tahmin edilen veriyi çiziyoruz
    ax.plot(forecast["ds"], forecast["yhat"], label="Tahmin Edilen Veri", color="red", linestyle="--")
    
    # Başlık ve etiketler
    ax.set_title("Gerçek ve Tahmin Edilen Veriler")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Değer")
    ax.legend()

    # Grafiği gösterme
    st.pyplot(fig3)