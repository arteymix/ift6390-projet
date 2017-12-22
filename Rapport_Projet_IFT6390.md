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
prédictions de salaire que sur MNIST. Nous expérimentons aussi avec un modèle
de Bayes mixte. Cette intuition est justifiée par le fait que les attributs
pour les salaires sont plus clairement scindés. Pour contourner le fait que
certains des attributs de l'échantillon de prédiction de salaire sont de type
catégorielle, nous prévoyons effectué une transformation de type
\textit{one-hot} et ainsi considérer des vecteurs numériques plutôt que des
classes.
\end{abstract}

# Analyses préliminaires

Les attributs de l'échantillons de prévision de salaires étaient au nombre de
treize (13) et étaient séparables en attributs catégoriels et attributs
continus :

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
tel que discuté dans le résumé, en vecteur *"one-hot"*. Nous obtenons donc un
total de 99 attributs binaires, cinq (5) attributs continues et une (1) valeur
binaire pour la sortie (ou cible).

Les graphiques suivants montrent que la tâches de classification des données de
salaires n'est pas triviale. En effet, pour les attributs continus (figure
\ref{Analyse par paires d'attributs continus}), une analyse par paires
d'attributs ne permet pas d'entrevoir la possibilité d'une séparabilité
linéaire des données. La même hypothèse est faites en observant les
histogrammes correspondants aux attributs catégoriels (figure \ref{Histogramme
des données catégorielles en fonction de la cible associée}) puisqu'aucun
attribut ne permet de séparer parfaitement les entrées. Quelques valeurs des
attributs catégoriels semblent permettre de trancher, ceci laisse croire que
l'utilisation d'arbre de décision est justifiée.

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

Nous avons utilisé les bibliothèques scikit-learn [@scikit-learn] et Keras
[@chollet2015keras].

Toutes les opérations ont été effectuées à l'aide d'un `Pipeline`[^sklearn.pipeline.Pipeline]
qui permet d'assembler les opérations (i.e. imputation, codage *"one-hot"*)
dans une chaîne de montage.

[^sklearn.pipeline.Pipeline]: http://scikit-learn.org/0.18/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline

La recherche par grille fournie par la classe `GridSearchCV`[^sklearn.model_selection.GridSearchCV]
a été utilisée pour déterminer les meilleurs hyper-paramètres. Nous avons
également utilisé un super calculateur pour paralléliser ce procédé.

[^sklearn.model_selection.GridSearchCV]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

Les réseaux de neurones ont été hyper-paramétré manuellement et seulement
l'époque optimale a été déterminée par validation croisée.

Nous avons utilisé l'algorithme Adadelta [@DBLP:journals/corr/abs-1212-5701]
pour faire l'optimisation qui utilise une moyenne exponentielle des gradients
des étapes précédentes.

Nous avons également favorisé l'approche Dropout (Srivastava et al. 2014) au
lieu des régularisations L1/L2 puisqu'elle semblait bien fonctionner
empiriquement. Cette méthode consiste à rendre inneffective une proportion
aléatoire d'unités dans une couche de neurones.

Les courbes d'apprentissages affichent l'erreur de classification en fonction
de la valeur d'un hyper-paramètre en considérant les valeurs des autres
hyper-paramètres qui minimisent l'erreur de validation. Cela fait en sorte que
le minimum observé correspond au vrai minimum.

# Classifieurs de Bayes

Nous avons expérimenté trois variantes du classifieurs de Bayes:

- noyau Gaussien;
- noyeau de Bernoulli;
- variante mixte.

La variante mixte combine les log-probabilité à postériori de chacun des
modèles de la manière suivante:

\begin{align*}
\log \Pr[c|X_{cat}, X_{cont}] = \lambda (\log \Pr[c|X_{cat}]) + (1 - \lambda) (\log \Pr[c|X_{cont}])
\end{align*}

\begin{wrapfigure}{r}{0.45\textwidth}
\vspace{-20pt}
\includegraphics[width=0.43\textwidth]{figures/mixed-naive-bayes-salary-learning-curve-lambda.png}
\caption{Courbe d'apprentissage du classifieur Bayésien mixte pour le paramètre $\lambda$ sur les données de salaire}
\vspace{-20pt}
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
fait que les entrées du vecteur *"one-hot"* sont mutuellement exclusive et donc
corrélées ce qui pose problème avec notre utilisation du classifieur naïf.

L'autre aspect intéressant est que sa courbe d'apprentissage est identique pour
l'entraînement et la validation, ce qui est consistant avec le fait que les
modèles Bayésiens on une très faible capacité.

\begin{figure}
\centering
\includegraphics[width=0.43\textwidth]{figures/mixed-naive-bayes-salary-learning-curve-alpha.png}
\caption{Courbe d'apprentissage du classifieur Bayésien mixte pour le lissage Laplacien sur les données de salaire}
\end{figure}

Puisque le classifieur de Bayes naïf fait l'hypothèse d'indépendance, si dans
un sous-ensemble au moins une valeur possible d'un des attributs n'est pas
observée, le calcul du produit des densités sur chaque dimensions de l'entrée
retournera zéro (0) et l'entré en question perdra tout sont poids dans le
calcul. Pour contourner ce problème, nous utilisons le lissage Laplacien. Cette
méthode ajoute simplement une constante $\Delta$ à tous les comptes lors du
calcul de fréquence de chaque classe. Il est très important que cet ajout ne
soit pas significatif par rapport à la valeur de compte la plus faible observée
avant le lissage.

\begin{wrapfigure}{r}{0.45\textwidth}
\vspace{-20pt}
\centering
\includegraphics[width=0.43\textwidth]{figures/bernoulli-naive-bayes-mnist-learning-curve-alpha.png}
\caption{Courbe d'apprentissage du classifieur Bayésien à noyau Bernoulli sur MNIST}
\vspace{-20pt}
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

L'utilisation des arbres de décision est habituellement appropriée dans le cas
de données dont les attributs présentent des caractéristiques de haut niveau.
Les données de prédiction de salaires présentent de tels attributs, mais les
points de MNIST (en niveaux de gris) n'ont pas une représentation de haut
niveau ce qui explique peut-être la faible performance de cet algorithme pour
la classification des images.

\begin{figure}
\centering
\includegraphics[width=0.48\textwidth]{figures/decision-tree-salary-learning-curve-max-depth.png}
\caption{Courbe d'apprentissage des arbres de décisions sur les données de salaire HP: Profondeur de l'arbre}
\label{Courbe d'apprentissage des arbres de décisions sur les données de salaire HP: Profondeur de l'arbre}
\end{figure}

Modifier la profondeur maximale permet de contrôler la capacité et la
propension de l'algorithme au sur-apprentissage. Ceci est observable sur les
figures \ref{Courbe d'apprentissage des arbres de décisions sur les données de
salaire HP: Profondeur de l'arbre} et \ref{Courbe d'apprentissage des arbres de
décisions sur MNIST HP: Profondeur de l'arbre}. On remarque en fait qu'à partir
d'une certaine profondeur, l'erreur d'entraînement diminue alors que celle de
validation stagne, indiquant un sur-apprentissage.

\begin{figure}
\centering
\includegraphics[width=0.48\textwidth]{figures/decision-tree-salary-learning-curve-min-samples-leaf.png}
\caption{Courbe d'apprentissage des arbres de décisions sur les données de salaire HP: Nombre d'observation minimale par feuille}
\label{Courbe d'apprentissage des arbres de décisions sur les données de salaire HP: Nombre d'observation minimale par feuille}
\end{figure}

\begin{figure}
\centering
\includegraphics[width=0.48\textwidth]{figures/decision-tree-mnist-learning-curve-max-depth.png}
\caption{Courbe d'apprentissage des arbres de décisions sur MNIST HP: Profondeur de l'arbre}
\label{Courbe d'apprentissage des arbres de décisions sur MNIST HP: Profondeur de l'arbre}
\end{figure}

Une augmentation du nombre minimal d'observation par feuille implique une
diminution de la capacité du modèle. À l'extrême, sans limiter la profondeur
maximale de l'arbre, permettre qu'il n'y ait qu'un exemplaire par feuille
donnerait un arbre où il existe un chemin menant à chaque exemplaire de
l'ensemble d'entrainement. Ce cas constiturait un exemple type de
sur-apprentissage que l'on peut observer aux figures \ref{Courbe
d'apprentissage des arbres de décisions sur les données de salaire HP: Nombre
d'observation minimale par feuille} et \ref{Courbe d'apprentissage des arbres
de décisions sur MNIST HP: Nombre d'observation minimale par feuille}.

\begin{figure}
\centering
\includegraphics[width=0.48\textwidth]{figures/decision-tree-mnist-learning-curve-min-samples-leaf.png}
\caption{Courbe d'apprentissage des arbres de décisions sur MNIST HP: Nombre d'observation minimale par feuille}
\label{Courbe d'apprentissage des arbres de décisions sur MNIST HP: Nombre d'observation minimale par feuille}
\end{figure}

Les arbres de décisions sont des modèles à très forte capacité et la profondeur
maximale est définitivement l'hyper-paramètre contrôlant le mieux la capacité
du modèle.


# Perceptron multi-couche

Le perceptron multi-couche ne convergait pas sur les données de salaire. Nous
avons tout de même essayé plusieurs architectures, régularisations et même
l'ajout de couches cachées.

\begin{figure}
\centering
\includegraphics[width=0.48\textwidth]{figures/multilayer-perceptron-salary-learning-curve-epoch.png}
\caption{Courbe d'apprentissage du perceptron multi-couches sur les données de salaire}
\end{figure}

La descente en escalier semble être liée à la régularisation Dropout: à chaque
fois qu'un neurone est neutralisé (i.e. la perte stagne), la rétro-propagation
tente de trouver une nouvelle façon de faire baisser la perte.

\begin{figure}
\centering
\includegraphics[width=0.48\textwidth]{figures/multilayer-perceptron-mnist-learning-curve-epoch.png}
\caption{Courbe d'apprentissage du perceptron multi-couches sur MNIST}
\end{figure}

\begin{wrapfigure}{r}{0.45\textwidth}
\vspace{-20pt}
\centering
\includegraphics[width=0.48\textwidth]{figures/convolutional-neural-network-mnist-learning-curve-epoch.png}
\caption{Courbe d'apprentissage du réseau de neurone convolutif sur MNIST}
\vspace{-10pt}
\end{wrapfigure}

Tel qu'anticipé, le réseau de neurones convolutif est très performant et
convergent rapidement. L'approche est très intéressante: $k$ noyeaux de
convolution sont appliqués sur chaque région de l'image pour former un ensemble
de représentation intermédiaire en ensuite une mise en commun (*pooling*) est
effectué pour écraser ces features dans une représentation compacte. Ensuite,
la représentation en 2 dimensions est ramené à un vecteur sur lequel on
applique un réseau de neurones traditionnel.

Un des avantages de ce type de modèles est qu'il réutilise les même poids pour
chaque noyau de convolutions, ce qui réduit considérablement le temps
d'entraînement et il peut exploiter les caractéristiques de localité de
l'image.

# Résultats finaux

Les tableaux suivants corresponent aux valeurs de précision sur les ensembles
de validation et de test.

Salary                  Validation Test
-----                   ---------- ----
Bayes naïf mixte        83.81%     73.01%
Arbres de décisions     85.57%     75.82%
Perceptron multi-couche 75.45%     76.37%

En générale, on remarque que les modèles utilisés n'ont pas très bien performé
en test, ce qui est très surprenant pour le modèle de Bayes qui ne devrait pas
avoir souffert de sur-apprentissage.

MNIST                         Validation Test
-----                         ---------- ----
Bayes naïf gaussien           56.28%     55.77%
Bayes naïf de Bernoulli       83.18%     83.36%
Arbres de décisions           86.19%     87.37%
Perceptron multi-couche       96.93%     96.86%
Réseau de neurones convolutif 98.77%     98.87%

En général, les modèle ont bien performé sur l'ensemble MNIST. Nous sommes
étonnés de voir l'erreur de test diminuer pour les arbres de décision, quoique
le taux de classification n'est peut-être pas suffisament élevé pour que ce
soit significatif.

Les modèles de réseaux de neurones ont très bien conservé leur performance
assez élevée sur cet ensemble de données.

# Répartition

Guillaume s'est occupé de la programmation et Gabriel de la rédaction de la
présentation du projet et du rapport.

Nous avons fait l'analyse exploratoire des données ensemble.

# Références
