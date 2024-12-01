import sys
import threading
import cv2
import numpy as np

from PIL import Image, ImageOps
from network.faster_rcnn.faster_rcnn_predictor import Predictor

# Variabili globali per condividere i risultati tra i thread
scores = None
boxes = None
frame = None
stop = False

# Lock per sincronizzare l'accesso alle variabili condivise
lock = threading.Lock()

def start_prediction(predictor):
    """
    Funzione che gestisce la predizione utilizzando il modello.
    Viene eseguita in un thread separato.
    """
    global scores, boxes, frame, stop

    while stop == False:
        with lock:
            local_frame = frame.copy() if frame is not None else None

        if local_frame is not None:
            try:
                temp_scores, temp_boxes = predictor.start(local_frame)
                with lock:
                    scores = temp_scores
                    boxes = temp_boxes
            except Exception as e:
                print(f"Errore durante la predizione: {e}")

def get_webcam():
    """
    Trova la webcam disponibile.
    """
    for i in range(5):  # Prova i primi 5 dispositivi
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Webcam trovata su indice {i}")
            return cap
    print("Nessuna webcam trovata.")
    return None

def preprocess(image):
    final_size = (600, 600)
    h, w = image.shape[:2]
    
    scale = min(final_size[0] / w, final_size[1] / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4) 
    background = np.full((final_size[1], final_size[0], 3), 255, dtype=np.uint8)
    offset = ((final_size[0] - new_w) // 2, (final_size[1] - new_h) // 2)
    background[offset[1]:offset[1]+new_h, offset[0]:offset[0]+new_w] = resized_image
    
    return background

def start_camera_loop(predictor):
    """
    Funzione principale per gestire il loop della webcam e la visualizzazione.
    """
    global scores, boxes, frame, stop

    # Avvia il thread per la predizione
    prediction_thread = threading.Thread(target=start_prediction, args=(predictor,))
    prediction_thread.start()

    # Ottieni la webcam
    cap = get_webcam()
    if cap is None:
        return

    print("Premi 'q' per uscire.")
    while True:
        _, raw_frame = cap.read()
        if raw_frame is None:
            print("Errore nel catturare il frame, esco...")
            break
        
        raw_frame = preprocess(raw_frame)
    
        # Aggiorna il frame globale
        with lock:
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

        # Disegna i risultati delle predizioni sul frame
        with lock:
            if scores is not None and boxes is not None and len(scores) > 0 and len(boxes) > 0:
                print(f'Trovate {len(boxes)} box')
                for j, box in enumerate(boxes):
                    class_name = scores[j]

                    cv2.rectangle(raw_frame,
                                  (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])),
                                  (0, 0, 255), 2)

                    cv2.putText(raw_frame, class_name,
                                (int(box[0]), int(box[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                2, lineType=cv2.LINE_AA)

        # Mostra il frame
        cv2.imshow('Webcam', raw_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia risorse
    cap.release()
    cv2.destroyAllWindows()
    
    stop = True
    prediction_thread.join()

def main():
    """
    Funzione principale per gestire le modalità di esecuzione.
    """
    if len(sys.argv) < 2:
        print("Specifica una modalità: 'train' o 'predict'")
        return

    mode = sys.argv[1]

    if mode == 'train':
        from network.faster_rcnn.faster_rcnn_trainer import Trainer
        from helper.config import DEVICE

        print(f"Utilizzo del dispositivo: {DEVICE}")
        trainer = Trainer('dataset/AI.FasterRCNN.RotationDetector-4')
        trainer.start()

    elif mode == 'predict':
        print("Modalità predizione avviata...")
        predictor = Predictor('network/faster_rcnn/output/best_model.pth')
        start_camera_loop(predictor)

if __name__ == '__main__':
    main()
