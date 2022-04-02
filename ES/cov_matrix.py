import torch
import numpy
class CovMatrix(): # TODO: Verify convergence with diag
  #
    def __init__(self, centroid: torch.Tensor, sigma, noise_multiplier,diag_cov=False,**kwargs):
        self.diag_cov = diag_cov
        policy_dim = centroid.size()[0]
        
        if self.diag_cov:
          self.cov = sigma *torch.ones(policy_dim)
          self.noise = torch.ones(policy_dim) * sigma
        else:
          self.noise = torch.diag(torch.ones(policy_dim) * sigma)
          self.cov = torch.diag(torch.ones(policy_dim) * torch.var(centroid)) + self.noise
        self.noise_multiplier = noise_multiplier

    def update_noise(self) -> None:
        self.noise = self.noise * self.noise_multiplier

    def generate_weights(self, centroid, pop_size):

      if self.diag_cov:
        param_noise = torch.randn(pop_size, centroid.nelement())
        weights = centroid + param_noise * torch.sqrt(self.cov)
      else:
        dist = torch.distributions.MultivariateNormal(centroid, covariance_matrix=self.cov)
        weights = [dist.sample() for _ in range(pop_size)]
      return weights

    def update_covariance(self, elite_weights) -> None:

      if self.diag_cov:
        self.cov = torch.var(elite_weights,dim=0) + self.noise 
      else : 
        self.cov = torch.cov(elite_weights.T) + self.noise 

      
class CEM():
  def __init__(self,mu_init,elites_nb,**kwargs) -> None:
      self.centroid = mu_init
      self.cov_matrix = CovMatrix(**{**kwargs,'centroid':self.centroid})
      self.elites_nb = elites_nb

  def tell(self,params,fitness):
    elites_idxs= torch.argsort(fitness)[-self.elites_nb :]
    elites_weights = [params[k] for k in elites_idxs]
    elites_weights = torch.cat([torch.tensor(w).unsqueeze(0) for w in elites_weights],dim=0)
    self.centroid = elites_weights.mean(0)
    self.cov_matrix.update_noise()
    self.cov_matrix.update_covariance(elites_weights)

  def ask(self,pop_size):
    return self.cov_matrix.generate_weights(self.centroid,pop_size)