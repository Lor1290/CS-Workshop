# 🖐✨ Hand Draw

> _Disegna nell'aria. Dipingi con le dita. Lascia il mouse sul tavolo._

```
  ☝️  indice su          →  disegna
  ✌️  indice + medio      →  cancella
  👍  solo pollice        →  pulisci tutto
  🎨  indice sulla barra  →  cambia colore
```

---

## 🌿 Cos'è

**Hand Draw** è un piccolo programma Python che trasforma la tua webcam in una tela.  
Usa la tua mano come pennello — nessun tocco, nessun click, solo gesture.

Costruito con amore (e un po' di debug) per il corso **CS Workshop** 🎓

---

## 🛠️ Setup

### 1 · Installa le dipendenze

```bash
pip install opencv-python mediapipe numpy
```

### 2 · Scarica il modello *(solo la prima volta)*

```bash
wget "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
  -O hand_landmarker.task
```

> 💡 Il file pesa ~9 MB e va messo nella stessa cartella dello script.

### 3 · Avvia

```bash
python hand_draw.py
```

---

## 🎮 Controlli

| Gesture | Azione |
|---|---|
| ☝️ Solo indice | **Disegna** con il colore selezionato |
| ✌️ Indice + medio | **Gomma** (cerchio grande) |
| 👍 Solo pollice | **Pulisci** tutto il canvas (tieni ~1 s) |
| ☝️ Indice sulla barra in alto | **Seleziona** un colore |
| `Q` | Esci |

---

## 🎨 Palette

La barra in cima alla finestra ha **9 colori** selezionabili al volo:

`🔴 rosso` · `🟠 arancio` · `🟡 giallo` · `🟢 verde` · `🩵 ciano` · `🔵 blu` · `🟣 viola` · `⚪ bianco` · `⚫ nero`

---

## 📦 Struttura

```
CS-Workshop/
├── hand_draw.py          ← il programma principale
├── hand_landmarker.task  ← modello MediaPipe (scaricato a parte)
└── README.md             ← sei qui 👋
```

---

## 🔧 Tecnologie

- **[OpenCV](https://opencv.org/)** — cattura webcam e rendering
- **[MediaPipe](https://developers.google.com/mediapipe)** — rilevamento landmarks della mano (Tasks API ≥ 0.10)
- **[NumPy](https://numpy.org/)** — canvas e operazioni sui pixel

---

## 💛 Note

- Funziona meglio con **buona illuminazione** e sfondo non troppo caotico
- Una mano per volta (per ora 👀)
- Tested su **mediapipe 0.10.33** — versioni precedenti usano una API diversa

---

<div align="center">
  <sub>fatto con ☕ e tanta pazienza · CS Workshop 2025</sub>
</div>
