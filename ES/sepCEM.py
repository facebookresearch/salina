import torch 

class SepCEM:

    """
    Cross-entropy methods.
    """

    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=1e-3,
                 pop_size=256,
                 damp=1e-3,
                 damp_limit=1e-5,
                 parents=None,
                 elitism=False,
                 antithetic=False,**kwargs):

        # misc
        self.num_params = num_params

        # distribution parameters
        if mu_init is None:
            self.mu = torch.zeros(self.num_params)
        else:
            self.mu = mu_init
        self.sigma = sigma_init
        self.damp = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
        self.cov = self.sigma * torch.ones(self.num_params)

        # elite stuff
        self.elitism = elitism
        self.elite = torch.sqrt(torch.tensor(self.sigma)) * torch.rand(self.num_params)
        self.elite_score = None

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic

        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        if parents is None or parents <= 0:
            self.parents = pop_size // 2
        else:
            self.parents = parents
        self.weights = torch.tensor([torch.log(torch.tensor((self.parents + 1) / i))
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        if self.antithetic and not pop_size % 2:
            epsilon_half = torch.random.randn(pop_size // 2, self.num_params)
            epsilon = torch.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = torch.randn(pop_size, self.num_params)

        inds = self.mu + epsilon * torch.sqrt(self.cov)
        if self.elitism:
            inds[-1] = self.elite

        return inds

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        scores = torch.tensor(scores)
        scores *= -1
        idx_sorted = torch.argsort(scores)

        old_mu = self.mu
        self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        z = (solutions[idx_sorted[:self.parents]] - old_mu)
        self.cov = 1 / self.parents * self.weights @ (
            z * z) + self.damp * torch.ones(self.num_params)

        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]
        #print(self.cov)

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return torch.copy(self.mu), torch.copy(self.cov)
