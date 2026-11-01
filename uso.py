# uso.py — exemplo simples de utilização do modelo treinado (Atividade Aula 4)
# Uso: python uso.py imagem1.jpg [imagem2.jpg ...]
import sys
from ultralytics import YOLO

model = YOLO("best.pt")                       # modelo treinado
imagens = sys.argv[1:] or ["minha_imagem.jpg"]

# conf=0.25: só mostra detecções com confiança >= 25%; save=True grava as imagens com as caixas
results = model.predict(imagens, conf=0.25, save=True)

for r in results:
    print(f"\n{r.path}: {len(r.boxes)} objeto(s) — resultado salvo em {r.save_dir}")
    for box in r.boxes:
        print(f"  {model.names[int(box.cls)]}: {float(box.conf):.2f}")
