# Machine-learning-in-simulation-of-pedestrian-flow-problem
**Soubory:**
learning.py - objekty, funkce a proměnné potřebné pro strojové učení
model_base.py - funkce obsahující kód potřebný pro spuštění simulace
visualisator.py - obsahuje funkce pro vytváření obrazů simulace
world.py - obsahuje objekty a funkce potřebné pro simulaci mapy
worlditems.py - obsahuje objekty potřebné při vytváření objektu ve world.py

modely - obsahují kód, který tvoří oblast a konstanty upravující chování, předáno do funkce z model_base:
 - Learn_model - tvoří křížovou oblast s východy v levo, v pravo, nahoře a dole, překážky jsou před každým východem, chodci začínají uprostřed
 - R_model - ulička s východem v pravo a překážkou před východem, chodci začínají vlevo
 - T_model - mapa ve tvaru T, východy jsou nahoře v pravo a levo, chodci začínají dole