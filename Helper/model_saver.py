import torch
import os
from torch.utils.mobile_optimizer import optimize_for_mobile

class ModelSaver:
    def __init__(self, output_dir = "Output"):
        self.history = []
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def add_model(self, train_loss, val_loss, model, optimizer):
        """
        Aggiunge un modello al salvataggio, confrontandolo con il migliore precedente.
        Se è il migliore, lo salva immediatamente.
        """
        current_model = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        self.history.append(current_model)
        
        # Confronta e salva se il nuovo modello è migliore
        if len(self.history) == 1 or val_loss < min(hist['val_loss'] for hist in self.history[:-1]):
            self.save_model(current_model, os.path.join(self.output_dir, "best_model.pth"))
            print(f"New best model saved with Validation Loss: {val_loss}")
            
            
    def save_model(self, model_data, file_path):
        """
        Salva il modello e lo stato dell'ottimizzatore nel percorso specificato.
        """
        torch.save({
            'model_state_dict': model_data['model_state_dict'],
            'optimizer_state_dict': model_data['optimizer_state_dict']
        }, file_path)
        
    def save_best_model(self):
        """
        Salva il miglior modello basandosi sulla validation loss.
        """
        if not self.history:
            print("No models in history to save.")
            return
        
        min_history = min(self.history, key=lambda x: x['val_loss'])
        self.save_model(min_history, os.path.join(self.output_dir, "best_model.pth"))
        print('Best model saved in Output/best_model.pth \n'
              f'Validation Loss: {min_history["val_loss"]}')