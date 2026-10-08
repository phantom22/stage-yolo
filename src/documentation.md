# Progetto di stage di Maksym Naumenko, matricola 899645.

### Dove si trovano le immagini acquisite?
La cartella `src/dataset/images` contiene tutte le immagini del dataset. Mentre `src/dataset/log/annotation_manifest.json` contiene la descrizione di quali e quanti oggetti sono contenuti all'interno di ciascuna immagine. Gli oggetti sono nominati con codici a due lettere che sono definiti in `src/dataset/__init__.py`. E' presente anche una versione di questo log scritta in formato umano e si trova in `src/dataset/log/annotation_log.txt`

### Dove si trovano le strategie di addestramento?
La cartella `src/yolo/data/strategies` che contiene tutte le strategie utilizzabili per l'addestramento di un modello

### Dove si trovano le annotazioni del dataset?
La cartella `src/yolo/data/labels` contiene tutte le etichette che possono essere referenziate dalle strategie contenute in `src/yolo/data/strategies`, in particolare sono state utilizzate solo le label `generic_labels` e `specific_labels`.

### Dove si trovano tutti i modelli addestrati?
La cartella `src/yolo/runs/segment` contiene tutti i modelli addestrati

### src/yolo/data/images
Cartella in cui, durante l'addestramento, viene creato un dynamic link (col nome delle labels utilizzate) alla cartella che contiene tutte le immagini, ovvero `src/dataset/images`.

### src/yolo/train.py: per addestrare un modello
```bash
python src/yolo/train.py NOME_MODELLO
```
Così verrà utilizzata la prima strategia, in ordine lessicografico, della cartella `src/yolo/data/strategies`. Il modello addestrato si troverà in `src/yolo/runs/segment/NOME_MODELLO` (i pesi effettivi si trovano in `src/yolo/runs/segment/NOME_MODELLO/weights/best.pt`, mentre i dati completi in `src/yolo/runs/segment/NOME_MODELLO/full_data.zip`)

```bash
python src/yolo/train.py NOME_MODELLO NOME_STRATEGIA
```
NOME_STRATEGIA deve riferirsi ad un file yaml situato nella cartella `src/yolo/data/strategies`, omettendo l'estensione `.yaml`, quindi la strategia chiamata `2_chinotto.yaml` verrà indicata come `2_chinotto`

```bash
python src/yolo/train.py NOME_MODELLO NOME_STRATEGIA MODELLO_UTILIZZATO_COME_BASE
```
Il MODELLO_UTILIZZATO_COME_BASE, se termina con `.pt` (eg. `yolo11m-seg.pt`, `yolo26m-seg.pt`), allora si assume l'utilizzo di un modello pre-addestrato pubblicato da Ultralytics, altrimenti deve rifersi al nome di una cartella valida( quindi contenente un modello addestrato) situata in `src/yolo/runs/segment`

```bash
python src/yolo/train.py help
```
Ricorda quali argomenti bisogna passare allo script.

### Per estrarre/zippare tutti i dati dell'addestramento di un modello
```bash
python src/yolo/util.py zip NOME_MODELLO
```

```bash
python src/yolo/util.py unzip NOME_MODELLO
```

### Per far inferire ad un modello addestrato
```bash
python src/yolo/run.py NOME_MODELLO
```
NOME_MODELLO deve riferirsi al nome di una cartella valida( quindi contenente un modello addestrato) situata in `src/yolo/runs/segment`. Nella finestra che si apre:
- si andare alla foto precedente o con la `freccia a sinistra` o con `a`
- si può andare alla foto successiva o con la `freccia a destra` o con `d`
- si può terminare l'inferenza con `q`, `ESC` oppure chiundendo la finestra manualmente
- le immagini inferite vengono salvate in `src/yolo/cache` e verranno sovrascritte al prossimo utilizzo dello script

```bash
python src/yolo/run.py NOME_MODELLO CONF_THRESHOLD
```
CONF_THRESHOLD valore nel range [0,1], indica la soglia minima di confidence sotto la quale YOLO scarterà le sue predizioni

```bash
python src/yolo/run.py NOME_MODELLO CONF_THRESHOLD CON_TRAIN
```
se CON_TRAIN è uguale a `true` allora il modello fara inferenza su tutte le immagini del dataset, anche quelle utilizzate durante l'addestramento

```bash
python src/yolo/run.py help
```
Ricorda quali argomenti bisogna passare allo script.

### Per fare la classifica dei modelli addestrati
```bash
python src/yolo/compare.py NOME_MODELLO_A NOME_MODELLO_B
```
Entrambi `NOME_MODELLO_A` e `NOME_MODELLO_B` devono rifersi al nome di una cartella valida (quindi contenente un modello addestrato) situata in `src/yolo/runs/segment`, confronta i punteggi mAP della Bounding Box e della Segmentation Mask tra i due modelli

```bash
python src/yolo/compare.py rank
```
Mostra la classifica dei modelli situati in `src/yolo/runs/segment` in base al punteggi mAP della Segmentation Mask

```bash
python src/yolo/compare.py rank SUFFISSO
```
Mostra la classifica dei modelli situati in `src/yolo/runs/segment`, aventi come suffisso del nome `SUFFISSO`, in base al punteggi mAP della Segmentation Mask.


### Per creare i grafici del dataset
`src/main.py` contiene vari esempi di codice che sono stati utilizzati per generare i grafici relativi all'analisi visiva del dataset

### Altri script presenti nel progetto

```bash
python src/yolo/data/strategies/visualize_labels.py NOME_STRATEGIA
```
dove `NOME_STRATEGIA` deve riferirsi ad un file yaml situato nella cartella `src/yolo/data/strategies`, omettendo l'estensione `.yaml`; offre la visualizzazione delle maschere con le corrispettive label, utilizzato ai fini di debug.

```bash
python src/yolo/data/labels/fix_labels.py CARTELLA_INPUT CARTELLA_OUTPUT MODALITA
```
dove sia `CARTELLA_INPUT` sia `CARTELLA_OUTPUT` sono nomi di cartelle situate all'interno di `src/yolo/data/labels` mentre MODALITA può essere o `specific` o `generic`; serve per convertire il dataset annotato su CVAT nei due dataset analizzati.

```bash
python src/yolo/graph.py
```
decommentando le varie sezioni, genera i grafici utilizzati sia per la classifica dei modelli addestrati nel primo e nel terzo scenario, ma anche per la classifica finale.

```bash
python src/yolo/fix_conf_mat.py
```
serve specificamente per correggere la confusion matrix del modello `src/yolo/runs/segment/3b_experiment2.4_Y11_1_0`, ovvero il migliore modello del terzo scenario, in quanto questo modello erroneamente è stato addestrato con più etichette di quelle che erano presenti nel dataset annotato, infatti la confusion matrix nelle ultime righe e colonne presenta uno spazio vuoto.

```bash
python src/yolo/can_dataset.py CLASSE_LATTINA
```
dove `CLASSE_LATTINA` può assumere un valore tra `DD_COCA_COLA`,`DD_CHINOTTO`,`DD_MONSTER`,`DD_FOCO_MARACUYA`,`DD_THE_PESCA`,`DD_SPRITE`. Questo script è stato utilizzato per generare gli insiemi di addestramento e di training per le strategie multi-istante.

### src/dataset/backups
Cartella che contiene tutti i backup relativi alle immagini e al progetto CVAT