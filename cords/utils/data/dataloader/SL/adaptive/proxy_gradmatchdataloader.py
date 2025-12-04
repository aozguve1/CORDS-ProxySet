from .gradmatchdataloader import GradMatchDataLoader
import time, copy
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

class ProxyGradMatchDataLoader(GradMatchDataLoader):
    """
    Proxy-Set Strategy (Robust Implementation):
    Uses Loss-Based Selection on a Proxy Subset.
    This is much faster and more stable than full GradMatch.
    """
    def __init__(self, train_loader, val_loader, dss_args, logger, *args, **kwargs):
        super(ProxyGradMatchDataLoader, self).__init__(train_loader, val_loader, dss_args, logger, *args, **kwargs)
        
        # Proxy Ratio (Default 30%)
        self.proxy_ratio = dss_args.get('proxy_ratio', 0.30)
        self.device = dss_args.device
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none') # Per-sample loss
        self.logger.info(f"Proxy-Set Loader initialized. Ratio: {self.proxy_ratio}")

    def _resample_subset_indices(self):
        """
        Manual Proxy Selection:
        1. Select a random Proxy subset.
        2. Calculate loss values for this subset.
        3. Select samples with highest loss (hardest examples).
        """
        start = time.time()
        self.logger.info(f'--- Proxy Selection (Loss-Based) Start (Epoch {self.cur_epoch}) ---')

        # STEP 1: Proxy Pool
        N = self.len_full
        proxy_size = int(N * self.proxy_ratio)
        
        all_indices = np.arange(N)
        proxy_indices = np.random.choice(all_indices, size=proxy_size, replace=False)
        
        self.logger.info(f"Proxy Pool Created: {proxy_size} samples.")

        # STEP 2: Temporary Loader for Proxy
        original_dataset = self.strategy.trainloader.dataset
        proxy_dataset = Subset(original_dataset, proxy_indices)
        
        proxy_loader = DataLoader(
            proxy_dataset, 
            batch_size=self.strategy.trainloader.batch_size * 2, # Increase batch size for faster inference
            shuffle=False,
            num_workers=self.strategy.trainloader.num_workers,
            pin_memory=self.strategy.trainloader.pin_memory
        )

        # STEP 3: Loss Calculation (Inference)
        self.train_model.eval()
        all_losses = []
        
        with torch.no_grad():
            for inputs, targets in proxy_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.train_model(inputs)
                losses = self.criterion(outputs, targets)
                all_losses.extend(losses.cpu().numpy())
        
        self.train_model.train() # Set model back to train mode

        # STEP 4: Select Data with Highest Loss
        # We will select as many samples as our budget allows
        budget = self.budget
        if budget > len(all_losses):
            budget = len(all_losses)
            
        # Sort by loss (descending) and take top 'budget' indices
        # argsort sorts ascending, so we reverse and take first 'budget' elements
        sorted_local_indices = np.argsort(all_losses)[::-1][:budget]
        
        # STEP 5: Convert to Global Indices
        final_global_indices = proxy_indices[sorted_local_indices]
        
        # Weights (set all to 1 for simplicity)
        subset_weights = torch.ones(len(final_global_indices))

        end = time.time()
        duration = end - start
        self.logger.info(f'Epoch: {self.cur_epoch}, PROXY selection finished. Duration: {duration:.4f}s.')
        
        return final_global_indices, subset_weights