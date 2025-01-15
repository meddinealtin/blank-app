import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FCFBFB;  /* Sayfanın arka plan rengi */
    }
    .header-box {
        background-color: #21395E;  /* Görselin arkasındaki kutunun rengi */
        padding: 20px;  /* Kutunun etrafına boşluk eklemek için */
        text-align: center;  /* Görseli kutuda ortalamak için */
    }
    .header-img {
        width: 100%;  /* Görselin genişliğini sayfa ile uyumlu yap */
        max-height: 150px;  /* Görselin yüksekliğini sınırla */
        object-fit: contain;  /* Görselin doğru şekilde görünmesi için */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Görseli başa yerleştirip kutuya yerleştirme
st.markdown('<div class="header-box"><img class="header-img" src="https://goldbergmed.com/wp-content/uploads/Asset-7GOLDBERGMEDLOG-WHITE.png"></div>', unsafe_allow_html=True)

# CSS ile yan taraf not kutusu ekleme
st.markdown(
    """
    <style>
    .sidebar-notes {
        background-color: #FCFBFB;
        border-left: 15px solid #21395E;
        padding: 30px;
        font-size: 20px;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Not Kutusu
st.sidebar.markdown(
    """
    <div class="sidebar-notes">
    <b>Kullanım İpuçları:</b><br><br>
    - Tahmin yapmak için bir <b>CSV dosyası</b> yükleyin.<br><br>
    - Tarih bilgisini içeren sütunu seçin 
      <span style="font-size: 14px; color: gray;">(örneğin: teslim_tarihi)</span>.<br><br>
    - Analiz etmek istediğiniz değeri seçin 
      <span style="font-size: 14px; color: gray;">(örneğin: satış_fiyatı, kar)</span>.<br><br>
    - Tahmin sonuçlarını incelemek için grafikleri görüntüleyin.<br><br>
    - Veri ve tahmin sonuçlarını karşılaştırarak daha iyi kararlar alın.<br><br>

    <hr style="border: none; border-top: 1px solid #ccc; margin: 15px 0;">
    <span style="font-size: 14px; color: gray;">
    *Not: Veriler, teslim tarihi sipariş tarihinin maksimum değerinden büyük olmayacak şekilde filtrelenmiştir.
    </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Streamlit Başlığı
st.title("Zaman Serisi Tahmini")

# CSV Dosyasını Yükleme
uploaded_file = st.file_uploader("CSV dosyanızı yükleyin", type="csv")

if uploaded_file:
    # Veriyi Yükleme
    df = pd.read_csv(uploaded_file)

    # Veri Gözden Geçirme
    st.write("Yüklenen Veri:")
    st.write(df.head())

    # İstenmeyen sütunları kaldırma
    columns_to_keep = ['satis_fiyati', 'kar', 'kar_orani', 'urun_grubu', 'teslim_tarihi', 'siparis_tarihi']
    df = df[columns_to_keep]

    # Tarih ve Değer Sütunlarını Seçme
    col1, col2 = st.columns(2)  # İki sütun oluşturuluyor

    with col1:
        date_column = st.selectbox("Tarih sütununu seçin:", df.columns)

    with col2:
        value_column = st.selectbox("Değer sütununu seçin:", [col for col in df.columns if col != date_column])

    # Ek Tarih Kontrolü (Sipariş ve Teslim Tarihi)
    if "siparis_tarihi" in df.columns and "teslim_tarihi" in df.columns:
        # Tarih sütunlarını datetime formatına çevirme
        df["siparis_tarihi"] = pd.to_datetime(df["siparis_tarihi"])
        df["teslim_tarihi"] = pd.to_datetime(df["teslim_tarihi"])
        
        # Sipariş tarihindeki maksimum değeri bulma
        max_siparis_tarihi = df["siparis_tarihi"].max()

        # Teslim tarihinin bu maksimum değerden büyük olmadığı verileri filtreleme
        df = df[df["teslim_tarihi"] <= max_siparis_tarihi]
        
        # Filtrelenmiş veriyi gösterme
        st.write("Filtrelenmiş Veri (Teslim tarihi, sipariş tarihindeki maksimum değerden büyük değil):")
        st.write(df)

    # Tarih sütununu datetime formatına çevirme
    df[date_column] = pd.to_datetime(df[date_column])

    # Prophet için gerekli sütun adlarını ayarlama
    df = df[[date_column, value_column]].rename(columns={date_column: "ds", value_column: "y"})

    # Aylık Bazda Gruplama (Opsiyonel)
    df = df.groupby(pd.Grouper(key="ds", freq="M")).sum().reset_index()

    # Aylık toplam satışların outlier'larını tespit etme (Sadece üst sınır)
    Q1 = df["y"].quantile(0.25)
    Q3 = df["y"].quantile(0.75)
    IQR = Q3 - Q1

    # Yalnızca üst sınırdaki outlier'ları filtreleme
    upper_limit = Q3 + 3 * IQR

    # Outlier olan satışları yarıya indirme
    df_no_outliers = df.copy()
    df_no_outliers.loc[df_no_outliers["y"] > upper_limit, "y"] = df_no_outliers["y"] / 2

    # Regresör Listesi
    regressors = ['AMELİYAT MASASI', 'HİDROJEN PEROKSİT', 'KARTUŞ', 
                  'OKSİJEN SİSTEMİ', 'OTOKLAV', 'REVERSE OSMOS', 'YIKAMA']

    # Kullanıcıdan Tahmin Süresi Almak için Slider
    periods = st.slider("Tahmin süresini (ay olarak) seçin:", min_value=1, max_value=36, value=12, step=1)

    # Prophet Modeli
    model = Prophet()

    # Regresörleri Modele Eklemek
    for regressor in regressors:
        if regressor in df.columns:
            model.add_regressor(regressor)

    # Modeli Eğitmek (Outlier'lar yarıya indirilmiş veriyle)
    model.fit(df_no_outliers)

    # Tahmin İçin Gelecek Tarihler
    future = model.make_future_dataframe(periods=periods, freq="M")

    # Regresörlerin Gelecek Değerlerini Ekleme
    for regressor in regressors:
        if regressor in df.columns:
            future[regressor] = df[regressor].iloc[-1]  # Son bilinen değerle doldurulabilir

    # Tahmin Yapma
    forecast = model.predict(future)

    # Tahmin Sonuçları
    st.write("Tahmin Sonuçları:")
    st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

    # Orijinal Veri ve Tahminleri Görselleştirme
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Sezonluk Bileşenlerin Görselleştirilmesi
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

