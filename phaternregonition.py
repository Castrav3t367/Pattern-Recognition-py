import os
from statistics import mode
import numpy as np
from numpy import linalg as la
import cv2
import matplotlib.pyplot as plt


#Lab 2
def construire_matrice_A(cale_baza, nr_persoane=40, nr_poze_antrenare=8, prefix_folder="s"):
    
    nr_pixeli = 112 * 92  # 10304 
    total_poze_antrenare = nr_persoane * nr_poze_antrenare
    A = np.zeros((nr_pixeli, total_poze_antrenare))
    
    for i in range(1, nr_persoane + 1):
        for j in range(1, nr_poze_antrenare + 1):
            cale_poza = os.path.join(cale_baza, f"{prefix_folder}{i}", f"{j}.pgm")
            poza = cv2.imread(cale_poza, 0) 
            if poza is None:
                raise FileNotFoundError(f"Nu s a putut citi poza: {cale_poza}")
            
            poza_vectorizata = np.array(poza, dtype=np.float64).reshape(-1, 1)
            col_index = nr_poze_antrenare * (i - 1) + (j - 1)
            A[:, col_index] = poza_vectorizata[:, 0]
            
    return A

# NN SI kNN 


def calculeaza_distante(A, pozaCautata, norma):
    nr_coloane = len(A[0])
    z = np.zeros(nr_coloane)
    pozaCautata = pozaCautata.reshape(-1)  # Vector 1D (10304,)
    
    for i in range(nr_coloane):
        x = A[:, i]
        if norma == 1 or norma == '1':
            z[i] = la.norm(x - pozaCautata, 1)
        elif norma == 2 or norma == '2':
            z[i] = la.norm(x - pozaCautata, 2)
        elif norma == np.inf or norma == 'inf' or norma == 'infinit':
            z[i] = la.norm(x - pozaCautata, np.inf)
        elif norma == 'cos' or norma == 'cosinus':
            z[i] = 1 - np.dot(x, pozaCautata) / (la.norm(x, 2) * la.norm(pozaCautata, 2))
        else:
            raise ValueError("Norma invalida! Foloseste: 1, 2, np.inf sau 'cos'.")
    return z

def NN(A, pozaCautata, norma):
    z = calculeaza_distante(A, pozaCautata, norma)
    pozitia = np.argmin(z)
    return pozitia

def kNN(A, pozaCautata, norma, k, nr_poze_antrenare=8):
    z = calculeaza_distante(A, pozaCautata, norma)
    pozitii = np.argsort(z)[:k]
    persoane_vecini = [idx // nr_poze_antrenare + 1 for idx in pozitii]
    persoana_aleasa = mode(persoane_vecini) #care e cel mai smeker
    
    #  Lab 3
    print(f"\nDetalii kNN (k={k}, norma={norma})")
    print("Indicii celor mai apropiate k coloane:", pozitii)
    print("Distantele corespunzatoare:", [round(z[idx], 2) for idx in pozitii])
    print("Persoanele carora le apartin vecinii:", persoane_vecini)
    print("Persoana castigatoare (mode):", persoana_aleasa)

    for idx in pozitii:
        if (idx // nr_poze_antrenare + 1) == persoana_aleasa:
            return idx




#aifsre L
if __name__ == "__main__":
    cale_orl = "./ORL"        
    nr_pers = 40               
    prefix = "s"               
    nr_antr = 8
    A = construire_matrice_A(cale_orl, nr_pers, nr_antr, prefix)
    print("Dimensiunea matricei A:", A.shape) 
    
    #Alegem o poza
    persoana_test =1
    nr_poza_test = 9
    cale_test = os.path.join(cale_orl, f"{prefix}{persoana_test}", f"{nr_poza_test}.pgm")
    poza_citita = cv2.imread(cale_test, 0)
    pozaCautata = np.array(poza_citita, dtype=np.float64).reshape(-1)
    
    
    norma_test = 'cos' #  1, 2, np.inf ,'cos'
    poz_nn = NN(A, pozaCautata, norma_test)
    pers_nn = poz_nn // nr_antr + 1
    print(f"\nNN Coloana gasita: {poz_nn} -> Persoana identificata: {pers_nn}")
    
    
    poz_knn5 = kNN(A, pozaCautata, norma_test, k=5, nr_poze_antrenare=nr_antr)
    pers_knn5 = poz_knn5 // nr_antr + 1
    
    # Afisam
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(pozaCautata.reshape(112, 92), cmap='gray')
    plt.title(f"Poza cautata\n(Persoana {persoana_test}, Poza {nr_poza_test})")
    
    plt.subplot(1, 3, 2)
    plt.imshow(A[:, poz_nn].reshape(112, 92), cmap='gray')
    plt.title(f"Rezultat NN (norma {norma_test})\nPersoana {pers_nn} (col {poz_nn})")
    
    plt.subplot(1, 3, 3)
    plt.imshow(A[:, poz_knn5].reshape(112, 92), cmap='gray')
    plt.title(f"Rezultat kNN (k=5)\nPersoana {pers_knn5} (col {poz_knn5})")
    
    plt.tight_layout()
    plt.show()