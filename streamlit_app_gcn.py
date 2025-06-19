import streamlit as st
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolToImage
import networkx as nx
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.data import Data
import pandas as pd # Diperlukan untuk menampilkan DataFrame

# --- GLOBAL VARIABLES DAN FUNGSI PREPROCESSING ---

# Variabel global untuk encoding sekuens protein
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} tidak ada dalam set yang diizinkan {1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Memetakan input yang tidak ada di allowable set ke elemen terakhir."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    # Definisi allowable_set untuk setiap fitur
    allowable_symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    allowable_degrees = [0, 1, 2, 3, 4, 5, 6,7,8,9,10]
    allowable_total_hs = [0, 1, 2, 3, 4, 5, 6,7,8,9,10]
    allowable_implicit_valences = [0, 1, 2, 3, 4, 5, 6,7,8,9,10]

    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), allowable_symbols) +
                    one_of_k_encoding(atom.GetDegree(), allowable_degrees) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), allowable_total_hs) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), allowable_implicit_valences) +
                    [atom.GetIsAromatic()])

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Tidak dapat mengurai SMILES: {smile}")
    
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        if sum(feature) == 0:
            st.error(f"Fitur atom memiliki jumlah nol untuk SMILES: {smile}. Tidak dapat menormalisasi.")
            raise ValueError("Fitur atom tidak dapat dinormalisasi.")
        features.append( feature / sum(feature) ) # Normalisasi fitur
    
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, np.array(features), np.array(edge_index)

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        if ch in seq_dict:
            x[i] = seq_dict[ch]
        else:
            pass # Tangani asam amino yang tidak dikenal (misalnya, beri nilai 0)
    return x

# --- Pembuatan dan Pemuatan Model ---
@st.cache_resource
def load_gcn_model():
    """
    Memuat model GCN yang sudah dilatih (PyTorch) dari file .model.
    Pastikan 'model_GCNNet_davis_gru64.model' berada di direktori yang sama dengan script ini.
    """
    model_path = 'model_GCNNet_davis_gru64.model'

    try:
        # --- DEFINISI KELAS MODEL GCNNet ---
        class GCNNet(torch.nn.Module):
            def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=78, 
                         num_features_xt=25, output_dim=128, gru_hidden_dim=64, dropout=0.2):

                super(GCNNet, self).__init__()

                # SMILES graph branch
                self.n_output = n_output
                self.conv1 = GCNConv(num_features_xd, num_features_xd)
                self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
                self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
                self.conv4 = GCNConv(num_features_xd * 4, num_features_xd * 8)
                self.fc_g1 = torch.nn.Linear(num_features_xd * 8, 1024)
                self.fc_g2 = torch.nn.Linear(1024, output_dim)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)

                # Protein sequence branch (Bi-GRU)
                self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
                self.bi_gru = nn.GRU(input_size=embed_dim, hidden_size=gru_hidden_dim, 
                                     num_layers=1, bidirectional=True, batch_first=True)
                self.fc1_xt = nn.Linear(2 * gru_hidden_dim, output_dim)

                # Combined layers
                self.fc1 = nn.Linear(2 * output_dim, 1024)
                self.fc2 = nn.Linear(1024, 512)
                self.out = nn.Linear(512, self.n_output)

            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch

                x = self.conv1(x, edge_index)
                x = self.relu(x)

                x = self.conv2(x, edge_index)
                x = self.relu(x)

                x = self.conv3(x, edge_index)
                x = self.relu(x)
                
                x = self.conv4(x, edge_index) 
                x = self.relu(x)
                
                x = gmp(x, batch)

                x = self.relu(self.fc_g1(x))
                x = self.dropout(x)
                x = self.fc_g2(x)
                x = self.dropout(x)

                target = data.target
                embedded_xt = self.embedding_xt(target)
                gru_out, _ = self.bi_gru(embedded_xt)
                gru_out = gru_out[:, -1, :]
                xt = self.fc1_xt(gru_out)

                xc = torch.cat((x, xt), 1)

                xc = self.fc1(xc)
                xc = self.relu(xc)
                xc = self.dropout(xc)
                xc = self.fc2(xc)
                xc = self.relu(xc)
                xc = self.dropout(xc)
                out = self.out(xc)

                return out
        # --- AKHIR DARI DEFINISI KELAS MODEL GCNNet ---

        # Inisialisasi model (menggunakan dummy atom untuk menghitung num_features_xd)
        dummy_mol = Chem.MolFromSmiles('C')
        if dummy_mol is None: 
            raise ValueError("Gagal membuat molekul dummy untuk perhitungan fitur atom.")
        dummy_mol.UpdatePropertyCache(strict=False) 
        dummy_atom = dummy_mol.GetAtomWithIdx(0) 

        num_features_xd_calculated = len(atom_features(dummy_atom)) 
        num_features_xt_from_model_def = 25 # Menggunakan nilai dari definisi model GCNNet
        
        model_instance = GCNNet(
            n_output=1,
            num_features_xd=num_features_xd_calculated,
            num_features_xt=num_features_xt_from_model_def,
            n_filters=32,     
            embed_dim=128,    
            output_dim=128,   
            gru_hidden_dim=64,
            dropout=0.2       
        )
        
        model_instance.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model_instance.eval()
        return model_instance
    except Exception as e:
        st.error(f"Error saat memuat model PyTorch dari '{model_path}': {e}. "
                 "Pastikan definisi kelas model GCNNet sudah benar dan file ada.")
        st.stop()

model = load_gcn_model()

# --- Antarmuka Pengguna Streamlit ---
st.title("🧬 Drug-Target Binding Affinity Prediction")
st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .prediction-result {
        font-size: 36px !important;
        color: #2E8B57; /* SeaGreen */
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("Prediksi **afinitas** antara senyawa kimia (diwakili oleh kode SMILES) dan protein (diwakili oleh urutan asam amino) menggunakan **model GCN+BiGRU**.")
st.markdown("---")

st.header("⚙️ Input Data")
col1, col2 = st.columns(2)

with col1:
    smiles_input = st.text_area(
        "**Kode SMILES Senyawa:**",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        help="Masukkan string SMILES untuk senyawa kimia."
    )

with col2:
    protein_input = st.text_area(
        "**Urutan Protein Sequence:**",
        "MVSYWDTGVLLCALLSCLLLTGSSSGSKLKDPELSLKGTQHIMQAGQTLHLQCRGEAAH",
        height=150,
        help="Masukkan urutan asam amino protein."
    )

st.markdown("---")

if st.button("🚀 Prediksi Afinitas"):
    if smiles_input and protein_input:
        with st.spinner("⏳ Memproses input dan memprediksi afinitas..."):
            try:
                mol_for_viz = Chem.MolFromSmiles(smiles_input)
                if mol_for_viz is None:
                    raise ValueError(f"SMILES input tidak valid untuk visualisasi: {smiles_input}")
                
                c_size, features, edge_index = smile_to_graph(smiles_input)

                st.subheader("⚛️ Visualisasi Senyawa")
                mol_img = MolToImage(mol_for_viz) 
                st.image(mol_img, caption="Struktur Molekul dari SMILES")

                st.subheader("🔬 Detail Fitur Senyawa (Molecule Features)")
                with st.expander("Lihat Detail Fitur Molekul"):
                    st.write(f"Jumlah Atom (Nodes): **{c_size}**")
                    
                    # --- Penjelasan Fitur Atom yang Dipersingkat dan Digabungkan ---
                    st.markdown("**Penjelasan Fitur Atom (Node Features) & Contoh:**")
                    st.markdown(
                        """
                        Setiap **atom (node)** diwakili oleh vektor fitur yang dinormalisasi.
                        Vektor ini menggabungkan beberapa properti atom. Berikut detailnya:

                        -   **Simbol Atom:** One-hot encoding dari jenis atom (misal Karbon 'C', Nitrogen 'N', Oksigen 'O', dll.).
                        -   **Derajat Atom:** One-hot encoding dari jumlah ikatan atom (jumlah tetangga yang terhubung langsung).
                        -   **Jumlah Hidrogen Total:** One-hot encoding dari jumlah total atom Hidrogen yang terikat pada atom tersebut (baik yang eksplisit maupun implisit).
                        -   **Valensi Implisit:** One-hot encoding dari valensi implisit atom (ikatan yang "hilang" yang dapat diisi oleh Hidrogen).
                        -   **Aromatisitas:** Nilai biner (1 jika atom adalah bagian dari cincin aromatis, 0 jika tidak).

                        **Contoh:** Untuk atom Karbon (`C`), fitur-fiturnya (sebelum dinormalisasi) akan menandai `1` pada posisi yang sesuai untuk simbol 'C', derajat ikatan atom (misal 4 untuk Karbon jenuh), jumlah H terikat (misal 3 untuk -CH3), dan valensi implisit (biasanya 0 jika sudah terpenuhi), serta 0 atau 1 untuk aromatisitas.

                        **Setelah normalisasi**, semua nilai `1` ini akan dibagi dengan jumlah total `1s` dalam vektor fitur mentah, menghasilkan nilai pecahan (seperti 0.25 atau 0.2) yang Anda lihat di tabel.
                        """
                    )
                    # --- Akhir Penjelasan Fitur Atom yang Digabungkan ---
                    
                    st.markdown("**Tabel Fitur Atom yang Sudah Dinormalisasi:**")
                    features_df = pd.DataFrame(features, columns=[f'F_{i+1}' for i in range(features.shape[1])])
                    st.dataframe(features_df.head())
                    if features.shape[0] > 5: # Menggunakan shape[0] karena ini array 2D
                        st.write(f"... dan {features.shape[0]-5} atom lainnya.")

                    st.markdown("---")
                    st.markdown("**Penjelasan Indeks Edge (Konektivitas Ikatan):**")
                    st.markdown(
                        """
                        **Indeks edge** merepresentasikan ikatan kimia antara atom-atom dalam molekul.
                        Setiap atom diberi **indeks numerik unik**, dimulai dari 0.
                        Setiap baris di tabel ini adalah pasangan `[indeks_atom_sumber, indeks_atom_target]`,
                        yang menunjukkan adanya ikatan (atau panah, karena graf berarah) dari Atom pada `indeks_atom_sumber`
                        ke Atom pada `indeks_atom_target`.
                        """
                    )
                    st.markdown("**Tabel Indeks Edge:**")
                    edge_index_df = pd.DataFrame(edge_index, columns=['Atom_Sumber_Idx', 'Atom_Target_Idx'])
                    st.dataframe(edge_index_df.head())
                    if edge_index.shape[0] > 5: # Menggunakan shape[0] karena ini array 2D
                        st.write(f"... dan {edge_index.shape[0]-5} ikatan lainnya.")

                data_mol = Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0))
                data_mol.batch = torch.zeros(data_mol.num_nodes, dtype=torch.long)

                # --- 2. Preprocessing Protein ---
                processed_protein_np = seq_cat(protein_input)
                
                st.subheader("🧬 Detail Fitur Protein (Protein Features)")
                with st.expander("Lihat Detail Fitur Protein"):
                    st.write(f"Panjang Sekuens Protein yang Diproses: **{len(processed_protein_np)}** (max_seq_len)")
                    
                    st.markdown("**Karakter Asli Protein dan Hasil Encoding (5 Pertama):**")
                    st.write(f"Karakter Asli (Input): `{protein_input[:5]}`")
                    
                    original_chars_from_encoded = []
                    for val in processed_protein_np[:5]:
                        found_char = next((char for char, code in seq_dict.items() if code == val), 'UNKNOWN')
                        original_chars_from_encoded.append(found_char)

                    st.write(f"Hasil Encoding (Numerik): `{np.array2string(processed_protein_np[:5].astype(int), separator=', ', max_line_width=np.inf)}`")

                    if len(protein_input) > 5:
                        st.write("... dan seterusnya.")

                data_mol.target = torch.LongTensor(processed_protein_np).unsqueeze(0)

                # --- 3. Melakukan Prediksi ---
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                data_mol.to(device)

                with torch.no_grad():
                    output = model(data_mol)
                
                affinity_prediction = output.item()

                st.success("✅ Prediksi Selesai!")
                st.markdown(f"<p class='prediction-result'>Nilai Afinitas: {affinity_prediction:.2f}</p>", unsafe_allow_html=True)

            except ValueError as ve:
                st.error(f"❌ Error Input: {ve}")
            except Exception as e:
                st.error(f"🔥 Terjadi kesalahan selama prediksi: {e}")
                st.info("Pesan Error Detail: " + str(e))
                st.info("Pastikan: \n"
                        "1. Definisi kelas `GCNNet` Anda (termasuk semua lapisan dan parameternya) di dalam `load_gcn_model()` adalah **identik** dengan yang digunakan saat melatih `model_GCNNet_davis_gru64.model`. "
                        "2. File model `model_GCNNet_davis_gru64.model` ada di direktori yang sama. "
                        "3. Input SMILES valid dan dapat diurai oleh RDKit.")
    else:
        st.warning("⚠️ Mohon masukkan Kode SMILES dan Urutan Asam Amino Protein untuk mendapatkan prediksi.")

st.markdown("---")
st.markdown("🔬 Dikembangkan oleh Regina &copy; 2025")