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
 - \usepackage{caption}
 - \usepackage{multicol}
 - \usepackage{subcaption}
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

\begin{figure}
    \centering
    \begin{subfigure}[t]{0.5\textwidth}
      \textbf{Continus}
        \begin{itemize}
          \item Age
          \item Financial weighted
          \item Capital gains
          \item Capital loss
          \item Hours per week
        \end{itemize}
    \end{subfigure}%
    ~
    \begin{subfigure}[t]{0.5\textwidth}
      \textbf{Catégoriels}
        \begin{itemize}
          \item Work class
          \item Education
          \item Education code
          \item Marital status
          \item Occupation
          \item Relationship
          \item Race
          \item Sex
          \item Native country
        \end{itemize}
      \end{subfigure}
\end{figure}




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

Les figures en annexes montrent que la tâches de classification des données de
salaires n'est pas triviale. En effet, pour les attributs continus (figure
\ref{Analyse par paires d'attributs continus}), une analyse par paires
d'attributs ne permet pas d'entrevoir la possibilité d'une séparabilité
linéaire des données. La même hypothèse est faites en observant les
histogrammes correspondants aux attributs catégoriels (figure \ref{Histogramme
des données catégorielles en fonction de la cible associée}) puisqu'aucun
attribut ne permet de séparer parfaitement les entrées. Quelques valeurs des
attributs catégoriels semblent permettre de trancher, ceci laisse croire que
l'utilisation d'arbre de décision est justifiée.


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
- noyau de Bernoulli;
- variante mixte.

La variante mixte combine les log-probabilité à postériori de chacun des
modèles de la manière suivante:

\begin{align*}
\log \Pr[c|X_{cat}, X_{cont}] = \lambda (\log \Pr[c|X_{cat}]) + (1 - \lambda) (\log \Pr[c|X_{cont}])
\end{align*}

\begin{wrapfigure}{r}{0.45\textwidth}
\vspace{-10pt}
\includegraphics[width=0.43\textwidth]{figures/mixed-naive-bayes-salary-learning-curve-lambda.png}
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
fait que les entrées du vecteur *"one-hot"* sont mutuellement exclusive et donc
corrélées ce qui pose problème avec notre utilisation du classifieur naïf.

L'autre aspect intéressant est que sa courbe d'apprentissage est identique pour
l'entraînement et la validation, ce qui est cohérent avec le fait que les
modèles Bayésiens on une très faible capacité.

\begin{wrapfigure}{r}{0.45\textwidth}
\vspace{-10pt}
\centering
\includegraphics[width=0.43\textwidth]{figures/mixed-naive-bayes-salary-learning-curve-alpha.png}
\caption{Courbe d'apprentissage du classifieur Bayésien mixte pour le lissage Laplacien sur les données de salaire}
\end{wrapfigure}

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
\vspace{-40pt}
\centering
\includegraphics[width=0.43\textwidth]{figures/bernoulli-naive-bayes-mnist-learning-curve-alpha.png}
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

\begin{figure}
    \centering
    \begin{subfigure}[t]{0.5\textwidth}
      \centering
      \includegraphics[width=1.0\textwidth]{figures/decision-tree-salary-learning-curve-max-depth.png}
      \captionof{figure}{HP: Profondeur de l'arbre}
      \label{SALAIRE HP: Profondeur de l'arbre}
    \end{subfigure}%
    ~
    \begin{subfigure}[t]{0.5\textwidth}
      \centering
      \includegraphics[width=1.0\textwidth]{figures/decision-tree-salary-learning-curve-min-samples-leaf.png}
      \captionof{figure}{HP: Nombre d'observation minimale par feuille}
      \label{SALAIRE HP: Nombre d'observation minimale par feuille}
      \end{subfigure}
      \caption{Courbe d'apprentissage pour Arbres de décision sur la prédiction de salaires}
\end{figure}

\begin{figure}
    \centering
    \begin{subfigure}[t]{0.5\textwidth}
      \centering
      \includegraphics[width=1.0\textwidth]{figures/decision-tree-mnist-learning-curve-max-depth.png}
      \caption{MNIST HP: Profondeur de l'arbre}
      \label{MNIST HP: Profondeur de l'arbre}
    \end{subfigure}%
    ~
    \begin{subfigure}[t]{0.5\textwidth}
      \centering
      \includegraphics[width=1.0\textwidth]{figures/decision-tree-mnist-learning-curve-min-samples-leaf.png}
      \caption{MNIST HP: Nombre d'observation minimale par feuille}
      \label{MNIST HP: Nombre d'observation minimale par feuille}
      \end{subfigure}
      \caption{Courbe d'apprentissage pour Arbres de décision sur MNIST}
\end{figure}

L'utilisation des arbres de décision est habituellement appropriée dans le cas
de données dont les attributs présentent des caractéristiques de haut niveau.
Les données de prédiction de salaires présentent de tels attributs, mais les
points de MNIST (en niveaux de gris) n'ont pas une représentation de haut
niveau ce qui explique peut-être la faible performance de cet algorithme pour
la classification des images.

Modifier la profondeur maximale permet de contrôler la capacité et la
propension de l'algorithme au sur-apprentissage. Ceci est observable sur les
figures \ref{SALAIRE HP: Profondeur de l'arbre} et \ref{MNIST HP: Profondeur de l'arbre}.
On remarque en fait qu'à partir
d'une certaine profondeur, l'erreur d'entraînement diminue alors que celle de
validation stagne, indiquant un sur-apprentissage.

Une augmentation du nombre minimal d'observation par feuille implique une
diminution de la capacité du modèle. À l'extrême, sans limiter la profondeur
maximale de l'arbre, permettre qu'il n'y ait qu'un exemplaire par feuille
donnerait un arbre où il existe un chemin menant à chaque exemplaire de
l'ensemble d'entrainement. Ce cas constiturait un exemple type de
sur-apprentissage que l'on peut observer aux figures \ref{SALAIRE HP: Nombre
d'observation minimale par feuille} et \ref{MNIST HP: Nombre d'observation minimale par feuille}.

Les arbres de décisions sont des modèles à très forte capacité et la profondeur
maximale est définitivement l'hyper-paramètre contrôlant le mieux la capacité
du modèle.

# Perceptron multi-couche

\begin{wrapfigure}{r}{0.45\textwidth}
\includegraphics[width=0.48\textwidth]{figures/multilayer-perceptron-salary-learning-curve-epoch.png}
\caption{Courbe d'apprentissage du perceptron multi-couches sur les données de salaire}
\label{Courbe d'apprentissage du perceptron multi-couches sur les données de salaire}
\end{wrapfigure}

Le perceptron multi-couche ne convergait pas sur les données de salaire. Nous
avons tout de même essayé plusieurs architectures, régularisations et même
l'ajout de couches cachées. Les résultats de la figure \ref{Courbe d'apprentissage du perceptron multi-couches sur les données de salaire}
illustre bien que le réseau ne s'améliore pas vraiment au fur et à mesure des époques.

La descente en escalier de la figure \ref{perceptron multi-couches} semble être liée à la régularisation Dropout: à chaque
fois qu'un neurone est neutralisé (i.e. la perte stagne), la rétro-propagation
tente de trouver une nouvelle façon de faire baisser la perte.


Tel qu'anticipé, le réseau de neurones convolutif est très performant et
convergent rapidement. L'approche est très intéressante: $k$ noyaux de
convolution sont appliqués sur chaque région de l'image pour former un ensemble
de représentation intermédiaire en ensuite une mise en commun (*pooling*) est
effectué pour écraser ces features dans une représentation compacte. Ensuite,
la représentation en 2 dimensions est transformée en un vecteur sur lequel on
applique un réseau de neurones traditionnel.

\begin{figure}[h]
    \centering
    \begin{subfigure}[t]{0.5\textwidth}
      \centering
      \includegraphics[width=1.0\textwidth]{figures/multilayer-perceptron-mnist-learning-curve-epoch.png}
      \caption{\textbf{perceptron multi-couches}}
      \label{perceptron multi-couches}
    \end{subfigure}%
    ~
    \begin{subfigure}[t]{0.5\textwidth}
      \centering
      \includegraphics[width=1.0\textwidth]{figures/convolutional-neural-network-mnist-learning-curve-epoch.png}
      \caption{Réseau de neurone \textbf{convolutif}}
      \end{subfigure}
      \caption{Courbe d'apprentissage des réseaux de neurones sur MNIST}
\end{figure}

Un des avantages de ce type de modèles est qu'il réutilise les même poids pour
chaque noyau de convolutions, ce qui réduit considérablement le temps
d'entraînement et il peut exploiter les caractéristiques de localité de
l'image.

# Résultats finaux et discussion

Les tableaux suivants corresponent aux valeurs de précision sur les ensembles
de validation et de test.

Salary                  Validation Test
-----                   ---------- ----
Bayes naïf mixte        83.81%     73.01%
Arbres de décisions     85.57%     75.82%
Perceptron multi-couche 75.45%     76.37%

Table: (*Résultats finaux Prédiction de salaire*) \label{Résultats finaux Prédiction de salaire}

En générale, on remarque que les modèles utilisés n'ont pas très bien performé
en test. Ceci est très surprenant pour le modèle de Bayes qui ne devrait pas
avoir souffert de sur-apprentissage.

Nous observons que l'échantillon complet recuilli est déséquilibré,
présentant environ 75% de cibles $<=50K$ et 25% de cibles $>50K$. Nous avons
construit les sous-ensembles d'entrainement, de validation et de test selon ces
mêmes proportions. En analysant les classifications finales sur l'ensemble de
validation pour les arbres de décision, nous obtenons les valeurs du tableau \ref{RappClassValid}.

Cible      precision  recall     f1-score   support
-----      ---------  ------     --------   -------
$<=50K$    0.88       **0.95**   0.91       24720
$>50K$     0.79       **0.59**   0.67       7841
avg/total  0.86       0.86       0.86       32561

Table:  (*Rapport de classification DT*, **Validation**) \label{RappClassValid}

La colonne "*recall*" fait référence au nombre de points de cette classe qui
furent bien classés. Il est évident que l'algorithme est nettement meilleur pour
identifier les exemplaire ayant pour cible $<=50K$. En analysant le rapport de
classification sur l'ensemble de test, nous observons que ce déséquilibre est
encore plus fort.

Cible      precision  recall     f1-score   support
-----      ---------  ------     --------   -------
$<=50K$    0.81       **0.89**   0.85       12435
$>50K$     0.48       **0.33**   0.39       3846
avg/total  0.73       0.76       0.74       16281

Table:  (*Rapport de classification DT*, **Test**) \label{RappClassTest}

Puisque les critères utilisés pour la selection des tests aux noeuds de l'arbre
suppose souvent une distribution uniforme des exemplaires y étant traités, un
déséquilibre entre la fréquence de chaque classe pourrait entrainer une
préférence des tests pour une classe en particulier. C'est peut-être ce que nous
observons avec les données de prédiction de salaire. L'utilisation de l'enthropie
décentrée [@ArbreDeseq] comme mesure de désordre serait une solution potentielle à étudier
dans une analyse subséquente de cet échantillon puisque cette méthode pondère
en fonction de l'importance de chaque classe aux noeuds.

Il serait aussi intéressant de discrétiser les attributs continus pour tester
nos classifieur de Bayes et arbres de décision sur des attributs discrets
seulement.

Il était attendu que les réseaux de neurones ne soient pas optimaux sur les
prédictions de salaires à cause de la quantité élevée d'attributs catégoriels
mélangés aux attributs continus. La performance sur l'ensemble de validation et
de test est cependant très comparable ce qui confirme que les réseaux de
neurones ne sont pas très sensible aux problèmes de sur-apprentissage. En
analysant la figure \ref{Courbe d'apprentissage du perceptron multi-couches sur les données de salaire},
il ne semble pas que d'augmenter le nombre d'époque est le moindre effet sur
la précision du réseau. Les résultats de la table \ref{Résultats finaux Prédiction de salaire}
mette en évidence le même type de problème conformément à la répartition des
classes dans l'échantillon initial.

Pour ce qui est des résultats sur MNIST, la performance obtenue est relativement
conforme à notre intuition.

MNIST                         Validation Test
-----                         ---------- ----
Bayes naïf gaussien           56.28%     55.77%
Bayes naïf de Bernoulli       83.18%     83.36%
Arbres de décisions           86.19%     87.37%
Perceptron multi-couche       96.93%     96.86%
Réseau de neurones convolutif 98.77%     98.87%

Table: (*Résultats finaux MNIST*) \label{Résultats finaux MNIST}

En général, les modèles ont bien performé sur l'ensemble MNIST. Nous sommes
étonnés de voir l'erreur de test diminuer pour les arbres de décision, quoique
le taux de classification n'est peut-être pas suffisament élevé pour que ce
soit significatif.

Le positionnement des images au centre et leurs dimensions sensiblement stables a
beaucoup contribué à la performance des classifieur de Bayes naïf. Ceux-ci ayant
pour objectif de trouver une "distribution" pour analyser les pixels auraient
probablement très mal performé sur des images de caractères non transformées.
Les applications de ce type d'algorithme dans la monde "réel" serait donc très
limitées sans l'ajout d'une transformation a priori sur les images pour normaliser
la position et l'échelle des caractères.

Les modèles de réseaux de neurones ont très bien conservé leur performance
assez élevée sur cet ensemble de données.

La position et l'échelle n'est pas aussi significatif pour les réseaux
convolutifs qui analyse de petites fenêtres de l'image. Ceux-ci pourraient fort probablement
déceler les subtilités des caractères mêmes si leur taille ou positionnement
n'est pas constant. Leur faible sensibilité aux variations de ces deux facteurs
explique en partie leur performance plus élevée que les réseaux de neuronnes MLP.

Nous avions prévu tester la performance d'arbres de décision dont les noeuds prennent
une décision en fonction d'une combinaison linéaire des attributs obtenue par
une PCA. Le temps nous ayant malheureusement manqué, il serait très intéressant
de tester ce type d'algorithme dans un projet subséquent. Nous espérons ainsi
augmenter la performance de ces arbres sur des échantillons tels la prédiction
de salaire où ces algorithmes semblent souffir de sur-apprentissage.

# Répartition

Guillaume s'est occupé de la programmation et Gabriel de la rédaction de la
présentation du projet et du rapport.

Nous avons fait l'analyse exploratoire des données ensemble.

# Annexe
\label{annexe}

\centerline{\includegraphics[width=0.9\paperwidth]{figures/salary-pair-plot.png}}
  \captionof{figure}{Analyse par paires d'attributs continus}
  \label{Analyse par paires d'attributs continus}
  \centerline{\includegraphics[width=1.0\paperwidth]{figures/salary-count-plot.png}}
    \captionof{figure}{Histogramme des données catégorielles en fonction de la cible associée}
    \label{Histogramme des données catégorielles en fonction de la cible associée}

# Références
