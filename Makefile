all: Proposition_Projet_IFT6390.pdf Rapport_Projet_IFT6390.pdf

Proposition_Projet_IFT6390.pdf: Proposition_Projet_IFT6390.tex
	xelatex $<

Rapport_Projet_IFT6390.pdf: Rapport_Projet_IFT6390.md
	pandoc -o $@ $<
