import numpy as np
from pyhessian import hessian 
import torch

class Simplicity():
    def __init__(self, model, criterion, train_data, cuda=True):
        self.model = model
        self.criterion = criterion 
        self.train_data = train_data
        self.cuda = cuda
        self.device = 'cuda:0' if cuda else 'cpu'
    
    def hessian_based_metrics(self):
        """
        We use the Hessian to compute some simplicity metrics w.r.t. the loss landscape. 
        In particular, we compute approximations to the determinant and trace of the Hessian,
        as well as the spectral norm (greatest eigenvalue).
        """
        hessian_comp = hessian(self.model, self.criterion, dataloader=[self.train_data], cuda=self.cuda)
        # Determinant of Hessian (equivalent to product of eigenvalues)
        eigenvalues, _ = hessian_comp.eigenvalues(top_n = 100) # this takes ~2 minutes
        det_h_ish = np.product(eigenvalues)
        trace_h_ish = np.sum(eigenvalues)
        spectral_norm_h = eigenvalues[0]
        return {
            "Product of first 100 eigenvalues": det_h_ish, 
            "Sum of fist 100 eigenvalues": trace_h_ish, 
            "Greatest Eigenvalue": spectral_norm_h
        }
        
    def random_noise_metric(self):
        """
        We measure the effect of adding random noise to the training data, via how much
        the model's output changes w.r.t. the input.
        """
        currently_training = self.model.training
        if currently_training:
            self.model.eval()
        with torch.no_grad():
            noise_outputs = self.model(self.train_data[0], embedding_noise=0.1)
            no_noise_outputs = self.model(self.train_data[0])
            diff = noise_outputs - no_noise_outputs
            # get the norm of the difference
            norms = torch.norm(diff, dim=1)
            # get the mean of the norm
            mean_norm = torch.mean(norms)
            # get the standard deviation of the norm
            std_norm = torch.std(norms)
        if currently_training:
            self.model.train()
        return {
            "Simplicity/Avg Norm of Output Change after Random Noise (0.1)": mean_norm,
            "Simplicity/Std Dev of Norm of Output Change after Random Noise (0.1)": std_norm
        }
        
        

# def calc_top_ev_of_hessian(model, criterion, data, cuda=True):
#     """
#     Given a model, criterion, and data, calculates (an approximation of) the 
#     top eigenvalue of the Hessian of the criterion w.r.t. model parameters.
#     """
#     hessian_comp = hessian(model, criterion, dataloader=[data], cuda=cuda)
#     top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
#     print("The top Hessian eigenvalue of this model is %.4f"%top_eigenvalues[-1])   
#     return top_eigenvalues[-1]

# def get_eigenvalue_density_of_hessian(model, criterion, data, cuda=True):
#     import ipdb; ipdb.set_trace()
#     hessian_comp = hessian(model, criterion, dataloader=[data], cuda=cuda)
#     density_eigen, density_weight = hessian_comp.density()
#     p = get_esd_plot(density_eigen, density_weight)
#     wandb.log({"Hessian Eigenvalue Density Figure": p,
#                "Hessian Trace": hessian_comp.trace()}) 

# def get_esd_plot(eigenvalues, weights):
#     density, grids = density_generate(eigenvalues, weights)
#     plt.semilogy(grids, density + 1.0e-7)
#     plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
#     plt.xlabel('Eigenvalue', fontsize=14, labelpad=10)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
#     plt.tight_layout()
#     return plt


# def density_generate(eigenvalues,
#                      weights,
#                      num_bins=10000,
#                      sigma_squared=1e-5,
#                      overhead=0.01):

#     eigenvalues = np.array(eigenvalues)
#     weights = np.array(weights)

#     lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
#     lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

#     grids = np.linspace(lambda_min, lambda_max, num=num_bins)
#     sigma = sigma_squared * max(1, (lambda_max - lambda_min))

#     num_runs = eigenvalues.shape[0]
#     density_output = np.zeros((num_runs, num_bins))

#     for i in range(num_runs):
#         for j in range(num_bins):
#             x = grids[j]
#             tmp_result = gaussian(eigenvalues[i, :], x, sigma)
#             density_output[i, j] = np.sum(tmp_result * weights[i, :])
#     density = np.mean(density_output, axis=0)
#     normalization = np.sum(density) * (grids[1] - grids[0])
#     density = density / normalization
#     return density, grids


# def gaussian(x, x0, sigma_squared):
#     return np.exp(-(x0 - x)**2 /
#                   (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)