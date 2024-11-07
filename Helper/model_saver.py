import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

class ModelSaver:
    def __init__(self):
        self.history = []
    
    def add_model(self, loss, model, optimizer):
        self.history.append({
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        })
        
    def save_best_model(self):
        min_history = None
        
        min_loss = float('inf')
        for hist in self.history:
            if hist['loss'] < min_loss:
                min_history = hist
                min_loss = hist['loss']
                
        torch.save({
            'model_state_dict': min_history['model_state_dict'],
            'optimizer_state_dict': min_history['optimizer_state_dict']
        }, 'Output/best_model.pth')
        
        print('Best model saved in Output/best_model.pth \n'
              f'Loss: {min_loss}')
        
    def save_for_mobile(self, file_path):
        traced_script_module = torch.jit.script(self.model)
        traced_script_module.eval()
        optimized = optimize_for_mobile(traced_script_module)
        optimized._save_for_lite_interpreter(file_path)