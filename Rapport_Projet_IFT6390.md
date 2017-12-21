---
title: Thomas Bayes aurait-il dû croquer le fruit défendu?
author: |
  Guillaume Poirier-Morency \
  Département d'informatique et de recherche opérationelle \
  Université de Montréal \
  Montréal \
  \texttt{guillaume.poirier-morency@umontreal.ca}
  \And
  Gabriel Lemyre \
  Département de mathématiques et statistiques \
  Université de Montréal \
  Montréal \
  \texttt{gabriell@dms.umontreal.ca}
header-includes:
 - \usepackage[final]{nips_2017}
 - \usepackage{wrapfig}
 - \usepackage{graphicx}
lang: fr
---

\begin{abstract}
Il existe différents types de données qui exhibent des relations particulières
entre les dimensions. Nous comparons ici l'efficacité de différents modèles
d'apprentissage sur deux ensembles de données: les caractères manuscrits de
l'échantillon MNIST et les données de prédiction de salaires. En particulier,
nous nous intéressons au classifieur de Bayes, aux arbres de décisions et au
perceptron multi-couche. Nous explorerons différents pré-traitement pour
mesurer les gains possibles lorsque combinés avec un modèle traditionnel de
classification. Notre intuition nous porte à croire que les méthodes de types
arbres de décision et classifieur de Bayes seront plus efficaces sur les
prédictions de salaire que sur MNIST. Cette intuition est justifiée par le fait
que les attributs pour les salaires sont plus clairement scindés. Pour
contourner le fait que certains des attributs de l'échantillon de prédiction de
salaire sont de type catégorielle, nous prévoyons effectué une transformation
de type \textit{one-hot} et ainsi considérer des vecteurs numériques plutôt que des
classes.
\end{abstract}

# Analyses préliminaires
Les attributs de l'échantillons de prévision de salaires étaient au nombre de treize (13) et étaient séparables en attributs catégoriels et attributs continus :

**Continus**

- Age
- Financial weighted
- Capital gains
- Capital loss
- Hours per week

**Catégoriels**

- Work class
- Education
- Education code
- Marital status
- Occupation
- Relationship
- Race
- Sex
- Native country

Nous avons choisis de considérer que l'entrée *Education* et de laisser tomber
*Education code*, puisque les deux font référence à la même chose. Nous
traiterons donc douze (12) attributs.

Pour traiter ces attributs, il est d'abord important de prendre une décision en
ce qui a trait à l'imputation données manquantes. Puisque les algorithmes de
type arbres de décisions et classifieur de Bayes ne prennent pas de données
manquantes, nous avons choisis de faire les remplacements suivants :

\begin{itemize}
  \item[Attributs \textit{continus}] - \textbf{Moyenne} empirique sur $D$ des valeurs de cet attribut
  \item[Attributs \textit{catégoriels}] - \textbf{Mode} empirique sur $D$ des valeurs de cet attribut
\end{itemize}

Pour traiter les données catégorielles, il était important de les transformer,
tel que discuté dans le résumé, en vecteur *one-hot*. Nous obtenons donc un
total de 99 attributs binaires, cinq (5) attributs continues et une (1) valeur
binaire pour la sortie (ou cible).

Les graphiques suivants montrent que la tâches de classification des données de salaires n'est pas triviale. En effet, pour les attributs continus (figure \ref{Analyse par paires d'attributs continus}), une analyse par paires d'attributs ne permet pas d'entrevoir la possibilité d'une séparabilité linéaire des données. La même hypothèse est faites en observant les histogrammes correspondants aux attributs catégoriels (figure \ref{Histogramme des données catégorielles en fonction de la cible associée}) puisqu'aucun attribut ne permet de séparer parfaitement les entrées. Quelques valeurs des attributs catégoriels semblent permettre de trancher, ceci laisse croire que l'utilisation d'arbre de décision est justifiée.

\begin{figure}[h!]
\centerline{\includegraphics[width=1.0\paperwidth]{figures/salary-count-plot.png}}
  \caption{Histogramme des données catégorielles en fonction de la cible associée}
  \label{Histogramme des données catégorielles en fonction de la cible associée}
\end{figure}

\begin{figure}[h!]
\centerline{\includegraphics[width=0.9\paperwidth]{figures/salary-pair-plot.png}}
  \caption{Analyse par paires d'attributs continus}
  \label{Analyse par paires d'attributs continus}
\end{figure}



# Méthodologie

Les données de salaire ont été imputés avec le mode pour les caractéristiques
catégoriques et la moyenne pour celles continues.

Toutes les opérations ont été effectuées à l'aide d'un
`Pipeline`[^sklearn.pipeline.Pipeline] qui permet d'assembler les opérations
(i.e. imputation, codage *one-hot*) dans une chaîne de montage.

[^sklearn.pipeline.Pipeline]: http://scikit-learn.org/0.18/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline

La recherche par grille fournie par la classe `GridSearchCV`[^sklearn.model_selection.GridSearchCV]
a été utilisée pour déterminer les meilleurs hyper-paramètres. Nous avons
également utilisé un super calculateur pour paralléliser ce procédé.

[^sklearn.model_selection.GridSearchCV]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

Les réseaux de neurones ont été hyper-paramétré à la main et seulement l'époque
optimale a été déterminée par validation croisée.

Nous avons utilisé l'algorithme Adadelta [@DBLP:journals/corr/abs-1212-5701]
pour faire l'optimisation qui utilise une moyenne exponentielle des gradients
des étapes précédents.

Nous avons également favorisé l'approche Dropout (Srivastava et al. 2014) au
lieu des régularisations L1/L2 puisqu'elle semblait bien fonctionner. Cette
méthode consiste à rendre inneffective une proportion aléatoire d'unités dans
une couche de neurones.

Les courbes d'apprentissages affichent l'erreur de classification en fonction
de la valeur d'un hyper-paramètre en considérant les valeurs des autres
hyper-paramètres qui minimisent l'erreur de validation. Cela fait en sorte que
le minimum observé correspond au vrai minimum.

# Classifieurs de Bayes

Nous avons expérimenté trois variantes du classifieurs de Bayes:

- Noyau Gaussien,
- Bernoulli,
- Variante mixte.

La variante mixte combine les log-probabilité à postériori de chacun des
modèles de la manière suivante:

\begin{align}
\log \Pr[c|X, X] = \lambda (\log \Pr[c|X]) + (1 - \lambda) (\log \Pr[c|X])
\end{align}

\begin{wrapfigure}{r}{0.5\textwidth}
\includegraphics[width=0.48\textwidth]{figures/mixed-naive-bayes-salary-learning-curve-lambda.png}
\caption{Courbe d'apprentissage du classifieur Bayésien mixte pour le paramètre $\lambda$ sur les données de salaire}
\end{wrapfigure}

Nous remarquons que la variante mixte performe particulièrement bien sur les
données de salaires. Ce qui est remarquable est qu'elle est significativement
meilleure que les deux modèles pures (i.e. chaque extrémités du graphe).

Une hypothèse pouvant expliquer cette performance supérieure est que chaque méthode
comporte des avantages sur certains types d'attributs. Le noyau gaussien est
fort probablement avantageux sur les attributs continus et le noyau de Bernoulli
n'est actif que lorsque les attributs sont catégoriels. Ainsi plus d'information
implique plus de performance.

La mauvaise performance du noyau Bernoulli est potentiellement explicable par le
fait que les entrées du vecteur *onehot* sont mutuellement-exclusive et donc
corrélées.

L'autre aspect intéressant est que sa courbe d'apprentissage est identique pour
l'entraînement et la validation, ce qui est consistant avec le fait que les
modèles Bayésiens on une très faible capacité.

\newpage

\begin{wrapfigure}{r}{0.5\textwidth}
\includegraphics[width=0.48\textwidth]{figures/mixed-naive-bayes-salary-learning-curve-alpha.png}
\caption{Courbe d'apprentissage du classifieur Bayésien mixte pour le lissage Laplacien sur les données de salaire}
\end{wrapfigure}

Puisque le classifieur de bayes que nous utilisons est naïf, si dans un sous-ensemble
au moins une valeur possible d'un des attributs n'est pas observée, le calcul du
produit des densités sur chaque dimensions de l'entrée retournera zéro (0) et
l'entré en question perdra tout sont poids dans le calcul. Pour contourner ce
problème, nous utilisons le lissage laplacien. Cette méthode ajoute simplement
une constante $\Delta$ à tous les comptes lors du calcul de fréquence de chaque
classe. Il est très important que cet ajout ne soit pas significatif par rapport
à la valeur de compte la plus faible observée avant le lissage.

\begin{wrapfigure}{r}{0.5\textwidth}
\includegraphics[width=0.48\textwidth]{figures/bernoulli-naive-bayes-mnist-learning-curve-alpha.png}
\caption{Courbe d'apprentissage du classifieur Bayésien à noyau Bernoulli sur MNIST}
\end{wrapfigure}

Pour les données de MNIST, nous avons testé le classifieur de Bayes à noyau
Gaussien ainsi que celui de Bernoulli en arrondissant les degrés de gris à des
valeurs binaires. Le taux de bonnes classifications est particulièrement fort:
83.18%!

Nous sommes portés à croire que puisque les images sont centrée et normalisée,
les similarités entre les exemplaires d'une même classe se traduisent par un
ensemble de pixels communs assez stable. Cette signature est facilement
reconnue par une distribution conjointe de succès.

# Arbres de décisions

\begin{wrapfigure}{r}{0.5\textwidth}
\includegraphics[width=0.48\textwidth]{figures/decision-tree-salary-learning-curve-max-depth.png}
\caption{Courbe d'apprentissage des arbres de décisions sur les données de salaire}
\end{wrapfigure}

\begin{wrapfigure}{r}{0.5\textwidth}
\includegraphics[width=0.48\textwidth]{figures/decision-tree-salary-learning-curve-min-samples-leaf.png}
\end{wrapfigure}

\begin{wrapfigure}{r}{0.5\textwidth}
\includegraphics[width=0.48\textwidth]{figures/decision-tree-mnist-learning-curve-max-depth.png}
\end{wrapfigure}

\begin{wrapfigure}{r}{0.5\textwidth}
\includegraphics[width=0.48\textwidth]{figures/decision-tree-mnist-learning-curve-min-samples-leaf.png}
\end{wrapfigure}

Les arbres de décisions sont des modèles à très forte capacité et la profondeur
maximale est définitivement l'hyper-paramètre contrôlant le mieux la capacité
du modèle.

# Perceptron multi-couche

Le perceptron multi-couche ne convergait pas sur les données de salaire.

\begin{wrapfigure}{r}{0.5\textwidth}
\includegraphics[width=0.48\textwidth]{figures/multilayer-perceptron-salary-learning-curve-epoch.png}
\end{wrapfigure}

\begin{wrapfigure}{r}{0.5\textwidth}
\includegraphics[width=0.48\textwidth]{figures/multilayer-perceptron-mnist-learning-curve-epoch.png}
\end{wrapfigure}

\begin{wrapfigure}{r}{0.5\textwidth}
\includegraphics[width=0.48\textwidth]{figures/convolutional-neural-network-mnist-learning-curve-epoch.png}
\end{wrapfigure}

# Résultats finaux

Les tableaux suivants corresponent aux valeurs de précision sur les ensembles
de tests des meilleurs modèle déterminés par le processus de validation.

Salary Validation Test
-----  ---------- ----
MNB    83.81%     73.01%
DT     85.57%     75.82%
NN     75.45%     76.37%

En générale, on remarque que les modèles utilisés n'ont pas très bien
fonctionné sur la classification de données de salaire puisque la précision est
très proche de la répartition des classes.

MNIST Validation Test
----- ---------- ----
NB    56.28%     55.77%
BNB   83.18%     83.36%
DT    86.19%     87.37%
NN    96.93%     96.86%
CNN   98.77%     98.87%

# Répartition

Guillaume s'est occupé de la programmation et Gabriel de la rédaction de la
présentation du projet et du rapport.

Nous avons fait l'analyse exploratoire des données ensemble.

# Références
