#Les bibiliotheques necessaires
import pandas
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Création du Dataframe à partir du fichier "market_basket.txt"
D = pandas.read_table("market_basket.txt",delimiter="\t",header=0)

#Affichage de 10 premiers lignes du Dataframe
print(D.head(10))

#Affichage des dimensions du Dataframe
print("Taille du Dataframe: ",D.shape)

#Construction de la table binaire
TB= pandas.crosstab(D.ID,D.Product)

#Affichage de 30 premieres transactions et 3 premiéres colonnes de la table binaire
print(TB.iloc[:30,:3])

#Execution de la fonction a_priori
freq_itemsets = apriori(TB,min_support=0.025,max_len=4,use_colnames=True)

#affichage des 15 premiers itemsets
print(freq_itemsets.head(15))

#fonction is_inclus
def is_inclus(x,items):
    return items.issubset(x)

#itemsets contenant Aspirin
print(freq_itemsets[freq_itemsets['itemsets'].ge({'Aspirin'})])

#itemsets contenant Aspirin et Eggs
print(freq_itemsets[freq_itemsets['itemsets'].ge({'Aspirin','Eggs'})])

#génération des règles à partir des itemsets fréquents
regles = association_rules(freq_itemsets,metric="confidence",min_threshold=0.75)

#affichage des 5 premières règles
print(regles.iloc[:5,:])

#affichage des règles avec un LIFT supérieur ou égal à 7
print(regles[regles['lift'].ge(7.0)])

#filtrer les règles menant au conséquent {‘2pct_milk’}
print(regles[regles['consequents'].eq({'2pct_Milk'})])