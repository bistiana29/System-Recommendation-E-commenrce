import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from PIL import Image
from streamlit_option_menu import option_menu
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

img_icon = Image.open('onlineshop_icon.png')
st.set_page_config(page_title="UK-OnlineShop", layout="wide", page_icon=img_icon)

st.markdown(
    """
    <style>.top {
        font-size: 35px;
        font-weight: bold;
        margin-top: -50px;
        margin-bottom: 5px;
        text-align: center;
    }
    </style>
    <div class='top'>🏬E-commerce Hybrid Recommendation🏬</div>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    choose = option_menu(
        "Main menu", 
        ["Recommendation", "Admin"], 
        icons=['cart', 'person-gear'],
        menu_icon="cast",
        default_index=0,  
    )

# Load data
df_cleaned = pd.read_csv('df_cleaned.csv')

# pivot table: CustomerNo x ProductNo
pivot = df_cleaned.pivot_table(index='CustomerNo', columns='ProductNo', values='Quantity', aggfunc='sum').fillna(0)
if (pivot % 1 == 0).all().all(): # jika nilai seluruh df bil. bulat maka convert ke int jika tidak maka tetap float
    pivot = pivot.astype(int)

# Inisialisasi model CF
model_cf = NearestNeighbors(metric='cosine', algorithm='brute')
model_cf.fit(pivot)

# function Collaborative Filtering
def recommend_cf(customer_id, k_neighbors=5):
    if customer_id not in pivot.index:
        print(f"Customer {customer_id} tidak ditemukan.")
        return []

    customer_index = pivot.index.get_loc(customer_id)
    query_vector = pivot.iloc[[customer_index]]
    distances, indices = model_cf.kneighbors(query_vector, n_neighbors=k_neighbors+1)

    similar_customers = [pivot.index[i] for i in indices.flatten() if pivot.index[i] != customer_id]
    similarity_scores = 1 - distances.flatten()
    similarity_scores = similarity_scores[1:]  # hilangkan skor untuk diri sendiri

    sim_df = pd.DataFrame({
        'CustomerNo': similar_customers,
        'Similarity': similarity_scores
    })

    # Ambil data transaksi dari customer yang mirip
    similar_data = df_cleaned[df_cleaned['CustomerNo'].isin(similar_customers)].copy()

    cf_recommendations = similar_data.merge(sim_df, on='CustomerNo', how='left')
    cf_recommendations = cf_recommendations.sort_values(by='Similarity', ascending=False)
    cf_recommendations = cf_recommendations[['CustomerNo', 'ProductNo', 'ProductName', 'Quantity', 'Similarity']]
    
    return cf_recommendations

# function content-based filtering
product_info = df_cleaned[['ProductNo', 'ProductName']].drop_duplicates().reset_index(drop=True)
product_info = product_info.reset_index().rename(columns={'index': 'TFIDF_Index'})

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(product_info['ProductName'])

def recommend_cb(product_no, k=9):
    if product_no not in product_info['ProductNo'].values:
        print(f"ProductNo {product_no} tidak ditemukan.")
        return pd.DataFrame()
    
    idx = product_info[product_info['ProductNo'] == product_no].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten() # Count cosine similarity
    similar_indices = cosine_sim.argsort()[-k-1:-1][::-1] # Mengambil ProdukNo mirip (selain dirinya sendiri)
    similarity_scores = cosine_sim[similar_indices]
    
    recommended = product_info.iloc[similar_indices].copy() # Mengambil data produk yang mirip
    recommended['Similarity'] = similarity_scores
    recommended['Rank'] = range(1, len(recommended) + 1)
    
    return recommended[['Rank', 'ProductNo', 'ProductName', 'Similarity']]

# function hybrid filtering (casecade: CF > CB)
def recommend_hybrid(customer_id, k_neighbors=5, top_n=9):
    # Cari customer mirip (Collaborative Filtering)
    cf_results = recommend_cf(customer_id, k_neighbors=k_neighbors)
    if cf_results is None or cf_results.empty:
        return pd.DataFrame()

    # Mengambil produk dari customer mirip, yang belum dibeli customer saat ini
    produk_cf = cf_results['ProductNo'].unique()
    produk_customer = df_cleaned[df_cleaned['CustomerNo'] == customer_id]['ProductNo'].unique()
    produk_kandidat = list(set(produk_cf) - set(produk_customer))
    if not produk_kandidat:
        return pd.DataFrame()

    # Mencari produk mirip dari produk yang sudah dibeli customer saat ini
    rekomendasi = pd.DataFrame()
    for prod in produk_customer:
        rekom = recommend_cb(prod, k=len(produk_kandidat))
        rekom = rekom[rekom['ProductNo'].isin(produk_kandidat)]
        rekomendasi = pd.concat([rekomendasi, rekom], ignore_index=True)

    if rekomendasi.empty:
        print("Tidak menemukan produk mirip.")
        df_if_recom_empty = df_cleaned[['ProductNo', 'ProductName', 'Price']].drop_duplicates().head(top_n)
        return df_if_recom_empty

    # Menggabungkan dan ambil top-N produk mirip
    rekomendasi = (
        rekomendasi.groupby(['ProductNo', 'ProductName'], as_index=False)
        .agg({'Similarity': 'mean'})
        .sort_values(by='Similarity', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    rekomendasi['Rank'] = rekomendasi.index + 1
    harga_produk = (df_cleaned.groupby('ProductNo', as_index=False).agg({'Price': ['min', 'max']}))
    harga_produk.columns = ['ProductNo', 'Price_min', 'Price_max']
    rekomendasi = rekomendasi.merge(harga_produk, on='ProductNo', how='left')
    rekomendasi['Price_Range'] = rekomendasi.apply(
    lambda row: f"£{row['Price_min']:.2f} - £{row['Price_max']:.2f}", axis=1
    )

    return rekomendasi[['Rank', 'ProductNo', 'ProductName', 'Price_Range', 'Similarity']]

# function evaluasi (diversity)
def count_diversity(df_produk):
    if df_produk.empty or len(df_produk) == 1:
        return 0.0

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_produk['ProductName'])

    distances = cosine_distances(tfidf_matrix)
    n = len(df_produk)
    total_dist = distances[np.triu_indices(n, k=1)].sum()
    count = len(np.triu_indices(n, k=1)[0])

    return total_dist / count if count != 0 else 0.0

# Main menu
if choose == "Recommendation":
    st.markdown("""
    <style>
    .card {
        border: 3px solid #00FFDE;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        height: 200px;
        overflow-y: auto;
        transition: all 0.3s ease-in-out;
        background-color: #222831;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
        }

        .card:hover {
            background-color: #003D3B;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
            cursor: pointer;
        }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("Masukkan Customer ID")
    customer_id = st.number_input("Customer ID", min_value=1, step=1, value=None, format="%d")

    if customer_id == None:
        # default: menampilkan 9 produk populer
        st.markdown("### 🛒 Daftar Produk Populer")
        top9_popular_product = pd.read_csv('top9_popular_product.csv')

        cols = st.columns(3)

        for i, row in top9_popular_product.iterrows():
            with cols[i % 3]:
                with st.container():
                    st.markdown(
                        f"""
                        <div class="card">
                            <h3>{row['ProductName']}</h3>
                            🆔 <strong>ProductNo:</strong> {row['ProductNo']}<br>
                            💰 <strong>Harga:</strong> {row['Price_Range']}
                        </div>
                        """, unsafe_allow_html=True
                    )

    # Jika customer_id diisi, jalankan sistem rekomendasi
    else:
        with st.spinner("Mencari produk rekomendasi..."):
            hasil = recommend_hybrid(customer_id)

        if not hasil.empty:
            st.success(f"Rekomendasi untuk Customer ID: {customer_id}")
            st.markdown("### 🎯 Produk Rekomendasi")

            cols = st.columns(3)  
            for i, row in hasil.iterrows():
                with cols[i % 3]: 
                    with st.container():
                        st.markdown(
                        f"""
                        <div class="card">
                            <h3>{row['ProductName']}</h3>
                            🛍️ <strong>ProductNo:</strong> {row['ProductNo']}<br>
                            💸 <strong>Rentang Harga:</strong> {row['Price_Range']}<br>
                            🔍 <strong>Similarity:</strong> {row['Similarity']:.2f}
                        </div>
                        """, unsafe_allow_html=True
                    )
            diversity_score = count_diversity(hasil)
            st.subheader("📊Evaluasi:")
            st.info(f"Diversity score: {diversity_score}")

elif choose == "Admin":
    st.markdown("""
        <style>
        .card {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            height: 100%;
            padding: 15px;
            margin-bottom: 10px;
            min-height: 120px;
            border: 3px solid #00FFDE;
            border-radius: 10px;
            background: #0A0F0F;
            color: white;
            background-color: #222831;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: 0.3s;
        }

        .card:hover {
            background: #003D3B;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            cursor: pointer;
        }

        .card h3 {
            margin: 0;
            color: #00FFDE;
            font-size: 22px;
        }

        .card p {
            margin: 5px 0 0;
            font-size: 28px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    card1, card2, card3, card4, card5 = st.columns([4,3.5,3.5,3.5,3])

    with card1:
        total_penjualan = (df_cleaned['Price']*df_cleaned['Quantity']).sum().round(2)
        st.markdown(f"""<div class="card"><h3>Total Penjualan</h3><p> £{total_penjualan}</p></div>
            """, unsafe_allow_html=True)

    with card2:
        total_transaksi = df_cleaned['TransactionNo'].count()
        st.markdown(f"""<div class="card"><h3>Total Transaksi</h3><p>{total_transaksi}</p></div>
            """, unsafe_allow_html=True)

    with card3:
        total_customer = len(df_cleaned['CustomerNo'].unique())
        st.markdown(f"""<div class="card"><h3>Total Customer</h3><p>{total_customer}</p></div>
            """, unsafe_allow_html=True)

    with card4:
        total_product = len(df_cleaned['ProductName'].unique())
        st.markdown(f"""<div class="card"><h3>Total Produk</h3><p>{total_product}</p></div>
            """, unsafe_allow_html=True)

    with card5:
        total_country = len(df_cleaned['Country'].unique())
        st.markdown(f"""<div class="card"><h3>Total Negara</h3><p>{total_country}</p></div>
            """, unsafe_allow_html=True)

    cart1 = st.columns(1)[0]

    with cart1:
        total = df_cleaned.copy()
        total['Total'] = total['Price'] * total['Quantity']
        total_penjualan_perCountry = total.groupby('Country')['Total'].sum().reset_index()
        total_penjualan_perCountry = total_penjualan_perCountry.sort_values(by='Total', ascending=False)

        fig = px.bar(
            total_penjualan_perCountry,
            x='Country',
            y='Total',
            title='Total Penjualan per Negara',
            labels={'Total': 'Total Penjualan (£)', 'Country': 'Negara'},
            text_auto='.2s',
            color_discrete_sequence=['#00FFDE']
        )

        fig.update_layout(
            title={'text': 'Total Penjualan per-Negara','x': 0.5,'xanchor': 'center'},
            xaxis_tickangle=-45,
            plot_bgcolor='#222831',
            paper_bgcolor='#222831',
            yaxis_title='Total Penjualan (£)',
            xaxis_title='Negara',
        )

        st.plotly_chart(fig, use_container_width=True)
    
    line1 = st.columns(1)[0]

    with line1:
        timeseries = df_cleaned.copy()
        timeseries['Date'] = pd.to_datetime(timeseries['Date'])
        timeseries['Total'] = timeseries['Price'] * timeseries['Quantity']
        timeseries['Month'] = timeseries['Date'].dt.to_period('M').dt.to_timestamp()
        total_bulanan = timeseries.groupby('Month')['Total'].sum().reset_index()

        fig = px.line(
            total_bulanan,
            x='Month',
            y='Total',
            title='Total Penjualan per-Bulan',
            labels={'Month': 'Bulan', 'Total': 'Total Penjualan (£)'},
            markers=True
        )

        fig.update_traces(line=dict(color='#00FFDE'), marker=dict(color='#00FFDE'))
        fig.update_layout(
            plot_bgcolor='#222831',
            paper_bgcolor='#222831',
            font_color='white',
            title={'text': 'Total Penjualan per-Bulan','x': 0.5,'xanchor': 'center'},
            xaxis=dict(tickmode='linear', dtick='M1')
            )

        st.plotly_chart(fig, use_container_width=True)