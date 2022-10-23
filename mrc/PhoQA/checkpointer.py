import os
import torch

class Checkpointer:
    def __init__(self, model, optimizer, save_dir='./output'):
        self.save_dir = save_dir
        self.model = model
        self.optimizer = optimizer

    def save(self, name):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(self.save_dir, name + '.pth'))

    def load(self, model_path):
        checkpoint = torch.load(os.path.join(self.save_dir, model_path))
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])