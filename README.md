# Machine-learning-in-simulation-of-pedestrian-flow-problem
**Soubory:**

python kód:
-   learning.py - objekty, funkce a proměnné potřebné pro strojové učení
-   model_base.py - funkce obsahující kód potřebný pro spuštění simulace
-   visualisator.py - obsahuje funkce pro vytváření obrazů simulace
-   world.py - obsahuje objekty a funkce potřebné pro simulaci mapy
-   worlditems.py - obsahuje objekty potřebné při vytváření objektu ve world.py

modely - obsahují kód, který tvoří oblast a konstanty upravující chování, předáno do funkce z model_base:
 - Learn_model - tvoří křížovou oblast s východy v levo, v pravo, nahoře a dole, překážky jsou před každým východem, chodci začínají uprostřed
 - R_model - ulička s východem v pravo a překážkou před východem, chodci začínají vlevo
 - T_model - mapa ve tvaru T, východy jsou nahoře v pravo a levo, chodci začínají dole

 **konstanty** - nutné nastavit u každého modelu, mění chování modelu:
 -  VERSION - verze simulace - mění název složky, do které se budou ukládat obrazy, aby se nepřepisovali vygenerované obrazy, při změně konstant (celé číslo)
 -  EPISODE - jaká epizoda uloženého modelu má být nahrána ze složky models, při ukládání obrazu bude zaznamenáno, jaká epizoda byla využita, ve složce se stejnou verzí mohou být obrazy využívající jiných epizod uložených modelů (celé číslo)
 -  FILE_NAME - kam se mají uložit vygenerované obrazy simulace (cesta)
 -  IS_LEARNING - má se simulovaný model dále učit? True: ano(při simulaci bude aktualizovat hodnoty neuronové sítě), False: ne(bude využívat stále stejný model při vypočítání pohybu)
 -  SAVE_MODEL - má naučený model ukládat do složky models/ ? True: ano, False: ne
 -  LOADED_VERSION - jaká verze uloženého modelu má být nahrána ze složky models (celé číslo)
 -  VISION_RANGE - jak daleko chodci vidí při simulaci (celé číslo)
 -  KNOW_EXIT - znají chodci směr k nejbližšímu východu? True: ano(při generování prostoru, který chodec vidí, je měněna hodnota daného chodce podle směru k nejbližšímu východu), False: ne(při generování prostoru, který chodec vidí, má chodec hodnotu jako všichni ostatní chodci)
 -  REAL_W - šířka simulované mapy v metrech
 -  REAL_H - výška simulované mapy v metrech
 -  cell_size - velikost jedné buňky a chodce - standard je 0.4
 -  im_width - šířka vygenerovaného obrazu
 -  hw_factor - kolikrát bude výška vygenerované obrazu menší než šířka (při hodnotě 1 je výška stejná jako šířka) -> šířka obrazu = im_width//hw_factor
