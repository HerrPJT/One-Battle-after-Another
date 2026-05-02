#importar as biblotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

###############################################################################################################

# 1. Carregamento 
dfbattles = pd.read_csv("archive/battles.csv") #arquivo principal das batalhas
dfbelligerents=pd.read_csv("archive/belligerents.csv") #arquivo identificador dos atacantes e defensores e números de exércitos e armas
dfterrain=pd.read_csv("archive/terrain.csv") #arquivo identificadores
dfweather=pd.read_csv("archive/weather.csv")
dffront_widths=pd.read_csv("archive/front_widths.csv")
dfCDB90=pd.read_csv("CDB90/CDB90-patched.csv")
dfCDB90.columns = dfCDB90.columns.str.upper() #Padronizar as colunas com letra maíuscula

###############################################################################################################

# 2. Tratamento de Dados

#ordenar os dados pelo ID (isqno)
dfbattles      = dfbattles.sort_values("isqno").reset_index(drop=True)
dfbelligerents = dfbelligerents.sort_values("isqno").reset_index(drop=True)
dfterrain      = dfterrain.sort_values("isqno").reset_index(drop=True)
dfweather      = dfweather.sort_values("isqno").reset_index(drop=True)
dffront_widths = dffront_widths.sort_values("isqno").reset_index(drop=True)

#preencher os dados vazios necessários
patch = dfCDB90[["ISQNO","SURPAA","AEROA","POST1","POST2","POSTYPE","WINA"]].copy()
patch.columns = patch.columns.str.lower()   
patch = patch.set_index("isqno")           # isqno é o índice
dfbattles = dfbattles.set_index("isqno") # Definir o mesmo índice no dfbattles
dfbattles.update(patch, overwrite=False)   # update: Preencher apenas os valores em falta  # overwrite=False = só preenche NaN
dfbattles = dfbattles.reset_index()
  
#unir os pri - objetivo tático antes da batalha
dfbelligerents["pri"]  = dfbelligerents["pri1"].fillna(dfbelligerents["pri2"]).fillna(dfbelligerents["pri3"]).str.upper()

#no weather apenas utilizar aqueles que utilizam wxno=1
dfweather = dfweather[dfweather["wxno"] == 1]

# Separar atacante e defensor, agregar por batalha
att = dfbelligerents[dfbelligerents["attacker"] == 1]
dfd = dfbelligerents[dfbelligerents["attacker"] == 0]
att_agg = att.groupby("isqno")[["str","arty","tank","fly"]].sum(numeric_only=True).add_suffix("_att").reset_index()
dfd_agg = dfd.groupby("isqno")[["str","arty","tank","fly"]].sum(numeric_only=True).add_suffix("_def").reset_index()
fw_agg = dffront_widths.groupby("isqno")[["wofa","wofd"]].mean().reset_index()

# Preparar cada tabela com apenas as colunas necessárias
tabelas = [
    dfbattles[["isqno","surpaa","aeroa","postype","post1","post2","wina"]],
    att_agg,
    dfd_agg,
    dfterrain.groupby("isqno").first()[["terra1","terra2","terra3"]].reset_index(),
    dfweather[["isqno","wx1","wx2","wx3","wx4","wx5"]],
    fw_agg,
    att[["isqno","pri"]].groupby("isqno").first().add_prefix("att_").reset_index(),
    dfd[["isqno","pri"]].groupby("isqno").first().add_prefix("def_").reset_index()
]

# Juntar todas as tabelas de uma vez pelo isqno
df = reduce(lambda esq, dir: esq.merge(dir, on="isqno", how="left"), tabelas)

# Remover linhas sem target
df = df[df["wina"].notna() & (df["wina"] != -9)].reset_index(drop=True)

###############################################################################################################

# 3. Seleção de Features + One-Hot Encoding
all_cols = ["surpaa","aeroa",                        
  "str_att","arty_att","tank_att","fly_att",
  "str_def","arty_def","tank_def","fly_def","wofa","wofd",
  "post1","post2","terra1","terra2","terra3",         
  "wx1","wx2","wx3","wx4","wx5",
  "att_pri","def_pri"]

features = df[all_cols].fillna({"post1":"UNK","post2":"UNK",    
                       "terra1":"UNK","terra2":"UNK","terra3":"UNK",
                       "wx1":"UNK","wx2":"UNK","wx3":"UNK","wx4":"UNK","wx5":"UNK", # preencher NaN categóricos com "UNK"
                       "att_pri":"UNK","def_pri":"UNK",}).fillna(0)  # numéricas com 0

X = pd.get_dummies(features, columns=["post1","post2","terra1","terra2","terra3", #One-Hot Encoding
                             "wx1","wx2","wx3","wx4","wx5",
                             "att_pri","def_pri"])
y = df["wina"]

print(f"Dataset: {X.shape[0]} batalhas × {X.shape[1]} features") #NºBatalhas e NºFeatures
print(f"Distribuicao do target: {y.value_counts().sort_index()}\n")

###############################################################################################################

# 4. Divisão, Scaling e Modelos
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42, stratify=y)

#Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # aprende a escala no treino e aplica
X_test_scaled  = scaler.transform(X_test)        # aplica a mesma escala ao teste

#Decision Trees
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=5,random_state=42) #Decision Trees
clf.fit(X_train, y_train)

#KNN
knn = KNeighborsClassifier(n_neighbors=18, metric="manhattan", weights="uniform")
knn.fit(X_train_scaled, y_train)

#Random Forest
rf = RandomForestClassifier(n_estimators=300,max_depth=15,random_state=42)
rf.fit(X_train, y_train)



###############################################################################################################

# 5. Avaliação
clf_y_train_pred = clf.predict(X_train)
clf_y_test_pred = clf.predict(X_test)

knn_y_train_pred = knn.predict(X_train_scaled)
knn_y_test_pred = knn.predict(X_test_scaled)

rf_y_train_pred = rf.predict(X_train)
rf_y_test_pred = rf.predict(X_test)

print("\nDecison Trees:\n")
print('Train data accuracy: ', accuracy_score(y_train, clf_y_train_pred))
print('Test data accuracy: ', accuracy_score(y_test, clf_y_test_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, clf_y_test_pred,target_names=["Loss (-1)", "Draw (0)", "Win (1)"], zero_division=0))

print("\nKNN:\n")
print('Train data accuracy: ', accuracy_score(y_train, knn_y_train_pred))
print('Test data accuracy: ', accuracy_score(y_test, knn_y_test_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, knn_y_test_pred,target_names=["Loss (-1)", "Draw (0)", "Win (1)"],zero_division=0))

print("\nRandom Forest:\n")
print("Train data accuracy :", accuracy_score(y_train, rf_y_train_pred))
print("Test data accuracy :", accuracy_score(y_test, rf_y_test_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, rf_y_test_pred,target_names=["Loss (-1)", "Draw (0)", "Win (1)"],zero_division=0))

###############################################################################################################

# 6. Gráficos
# Decision Tree
plt.figure(figsize=(40, 20))
tree.plot_tree(clf,max_depth=5,feature_names=X.columns.tolist(),class_names=["Loss", "Draw", "Win"],
                filled=True,rounded=True,fontsize=10,)
plt.savefig("decisiontree.png", dpi=300)
print("O gráfico foi guardado como 'decisiontree.png' na pasta do projeto.")


# Confusion Matrix Confusão dos 3 modelos
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Matrizes de Confusão", fontsize=20)
preds = [clf_y_test_pred, knn_y_test_pred, rf_y_test_pred]
titles = ["Modelo Decision Trees", "Modelo KNN", "Modelo Random Forest"]

for i in range(len(preds)):
    cm = confusion_matrix(y_test, preds[i])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Loss", "Draw", "Win"]
    )
    disp.plot(cmap="Blues", ax=ax[i], values_format="d")
    ax[i].set_title(titles[i])

plt.tight_layout()
plt.savefig("Triple_Confusion_Matrix.png", dpi=300)
print("O gráfico foi guardado como 'Triple_Confusion_Matrix.png' na pasta do projeto.")


# Comparação de K no KNN de Accuracy
k_range = range(1, 31)
scores = []
for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
    # Usando cross-validation para maior robustez
    score = cross_val_score(knn_temp, X_train_scaled, y_train, cv=5).mean()
    scores.append(score)

plt.figure()
plt.plot(k_range, scores, marker="o", linestyle="dashed")
plt.title("KNN accuracy by K value")
plt.xlabel("K Value")
plt.ylabel("Cross-Validated Accuracy")
plt.savefig("knn_optimizationaccuracy.png", dpi=300)
print("O gráfico foi guardado como 'knn_optimizationaccuracy.png' na pasta do projeto.")
plt.close()


# Comparação de K no KNN de Precision,Recall e F1 por Classe
prec_0, rec_0, f1_0 = [], [], []
prec_1, rec_1, f1_1 = [], [], []
prec_2, rec_2, f1_2 = [], [], []

k_range = range(1, 31)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan").fit(
        X_train_scaled, y_train
    )
    y_pred = knn.predict(X_test_scaled)
    p = metrics.precision_score(y_test, y_pred, average=None, zero_division=0)
    r = metrics.recall_score(y_test, y_pred, average=None, zero_division=0)
    f = metrics.f1_score(y_test, y_pred, average=None, zero_division=0)
    prec_0.append(p[0])
    rec_0.append(r[0])
    f1_0.append(f[0])
    prec_1.append(p[1])
    rec_1.append(r[1])
    f1_1.append(f[1])
    prec_2.append(p[2])
    rec_2.append(r[2])
    f1_2.append(f[2])

fig, ax = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle("Desempenho para Classes Diferentes", fontsize=20)

# Gráfico Classe 0 (Derrota)
ax[0].plot(k_range, prec_0, label="Precision")
ax[0].plot(k_range, rec_0, label="Recall")
ax[0].plot(k_range, f1_0, label="F1")
ax[0].set_title("Classe: Derrota")

# Gráfico Classe 1 (Empate)
ax[1].plot(k_range, prec_1, label="Precision")
ax[1].plot(k_range, rec_1, label="Recall")
ax[1].plot(k_range, f1_1, label="F1")
ax[1].set_title("Classe: Empate")

# Gráfico Classe 2 (Vitória)
ax[2].plot(k_range, prec_2, label="Precision")
ax[2].plot(k_range, rec_2, label="Recall")
ax[2].plot(k_range, f1_2, label="F1")
ax[2].set_title("Classe: Vitória")

for a in ax:
    a.legend()
    a.set_xlabel("K")
    a.set_ylabel("Performance")

plt.tight_layout()
plt.savefig("knn_desempenho.png", dpi=300)
print("O gráfico foi guardado como 'knn_desempenho.png' na pasta do projeto.")
plt.close()


## Carateristicas mais importantes do Random Forest
importances = rf.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.Series(importances, index=feature_names).sort_values(
    ascending=False
)
top_10_features = feature_importance_df.head(10)
plt.figure(figsize=(10, 6))
top_10_features.plot(kind="barh", color="green").invert_yaxis()  # Inverter para a maior ficar no topo
plt.title("Top 10 Características mais Importantes - Random Forest")
plt.xlabel("Importância")
plt.ylabel("Características")
plt.tight_layout()
plt.savefig("RandomForestFeatures.png", dpi=300)
print("O gráfico foi guardado como 'RandomForestFeatures.png' na pasta do projeto.")
plt.close()