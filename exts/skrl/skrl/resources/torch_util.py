import math

import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.transforms import TanhTransform


class SymLogTransform():

    def __init__(self):
        pass

    def __call__(self,x, inverse=False, *args, **kwargs):
        if inverse:
            return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)
        else:
            return torch.sign(x) * torch.log(1 + torch.abs(x))
    


@torch.no_grad()
def soft_update(target_net, current_net, tau: float):
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
        tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

"""A Gaussian distribution with tanh transformation."""
import torch
from torch.distributions import Normal
from torch.distributions.independent import Independent


class TanhNormal(torch.distributions.Distribution):
    r"""A distribution induced by applying a tanh transformation to a Gaussian random variable.

    Algorithms like SAC and Pearl use this transformed distribution.
    It can be thought of as a distribution of X where
        :math:`Y ~ \mathcal{N}(\mu, \sigma)`
        :math:`X = tanh(Y)`

    Args:
        loc (torch.Tensor): The mean of this distribution.
        scale (torch.Tensor): The stdev of this distribution.

    """ # noqa: 501

    def __init__(self, loc, scale):
        self._normal = Independent(Normal(loc, scale), 1)
        super().__init__()

    def log_prob(self, value, pre_tanh_value=None, epsilon=1e-6):
        """The log likelihood of a sample on the this Tanh Distribution.

        Args:
            value (torch.Tensor): The sample whose loglikelihood is being
                computed.
            pre_tanh_value (torch.Tensor): The value prior to having the tanh
                function applied to it but after it has been sampled from the
                normal distribution.
            epsilon (float): Regularization constant. Making this value larger
                makes the computation more stable but less precise.

        Note:
              when pre_tanh_value is None, an estimate is made of what the
              value is. This leads to a worse estimation of the log_prob.
              If the value being used is collected from functions like
              `sample` and `rsample`, one can instead use functions like
              `sample_return_pre_tanh_value` or
              `rsample_return_pre_tanh_value`


        Returns:
            torch.Tensor: The log likelihood of value on the distribution.

        """
        # pylint: disable=arguments-differ
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1 + epsilon + value) / (1 + epsilon - value)) / 2
        norm_lp = self._normal.log_prob(pre_tanh_value)
        ret = (norm_lp - torch.sum(
            torch.log(self._clip_but_pass_gradient((1. - value**2)) + epsilon),
            axis=-1))
        return ret

    def sample(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal Distribution.

        Args:
            sample_shape (list): Shape of the returned value.

        Note:
            Gradients `do not` pass through this operation.

        Returns:
            torch.Tensor: Sample from this TanhNormal distribution.

        """
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal Distribution.

        Args:
            sample_shape (list): Shape of the returned value.

        Note:
            Gradients pass through this operation.

        Returns:
            torch.Tensor: Sample from this TanhNormal distribution.

        """
        z = self._normal.rsample(sample_shape)
        return torch.tanh(z)

    def rsample_with_pre_tanh_value(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal distribution.

        Returns the sampled value before the tanh transform is applied and the
        sampled value with the tanh transform applied to it.

        Args:
            sample_shape (list): shape of the return.

        Note:
            Gradients pass through this operation.

        Returns:
            torch.Tensor: Samples from this distribution.
            torch.Tensor: Samples from the underlying
                :obj:`torch.distributions.Normal` distribution, prior to being
                transformed with `tanh`.

        """
        z = self._normal.rsample(sample_shape)
        return z, torch.tanh(z)

    def cdf(self, value):
        """Returns the CDF at the value.

        Returns the cumulative density/mass function evaluated at
        `value` on the underlying normal distribution.

        Args:
            value (torch.Tensor): The element where the cdf is being evaluated
                at.

        Returns:
            torch.Tensor: the result of the cdf being computed.

        """
        return self._normal.cdf(value)

    def icdf(self, value):
        """Returns the icdf function evaluated at `value`.

        Returns the icdf function evaluated at `value` on the underlying
        normal distribution.

        Args:
            value (torch.Tensor): The element where the cdf is being evaluated
                at.

        Returns:
            torch.Tensor: the result of the cdf being computed.

        """
        return self._normal.icdf(value)

    @classmethod
    def _from_distribution(cls, new_normal):
        """Construct a new TanhNormal distribution from a normal distribution.

        Args:
            new_normal (Independent(Normal)): underlying normal dist for
                the new TanhNormal distribution.

        Returns:
            TanhNormal: A new distribution whose underlying normal dist
                is new_normal.

        """
        # pylint: disable=protected-access
        new = cls(torch.zeros(1), torch.zeros(1))
        new._normal = new_normal
        return new

    def expand(self, batch_shape, _instance=None):
        """Returns a new TanhNormal distribution.

        (or populates an existing instance provided by a derived class) with
        batch dimensions expanded to `batch_shape`. This method calls
        :class:`~torch.Tensor.expand` on the distribution's parameters. As
        such, this does not allocate new memory for the expanded distribution
        instance. Additionally, this does not repeat any args checking or
        parameter broadcasting in `__init__.py`, when an instance is first
        created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance(instance): new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            Instance: New distribution instance with batch dimensions expanded
            to `batch_size`.

        """
        new_normal = self._normal.expand(batch_shape, _instance)
        new = self._from_distribution(new_normal)
        return new

    def enumerate_support(self, expand=True):
        """Returns tensor containing all values supported by a discrete dist.

        The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Note:
            Calls the enumerate_support function of the underlying normal
            distribution.

        Returns:
            torch.Tensor: Tensor iterating over dimension 0.

        """
        return self._normal.enumerate_support(expand)

    @property
    def mean(self):
        """torch.Tensor: mean of the distribution."""
        return torch.tanh(self._normal.mean)

    @property
    def variance(self):
        """torch.Tensor: variance of the underlying normal distribution."""
        return self._normal.variance

    def entropy(self):
        """Returns entropy of the underlying normal distribution.

        Returns:
            torch.Tensor: entropy of the underlying normal distribution.

        """
        return self._normal.entropy()

    @staticmethod
    def _clip_but_pass_gradient(x, lower=0., upper=1.):
        """Clipping function that allows for gradients to flow through.

        Args:
            x (torch.Tensor): value to be clipped
            lower (float): lower bound of clipping
            upper (float): upper bound of clipping

        Returns:
            torch.Tensor: x clipped between lower and upper.

        """
        clip_up = (x > upper).float()
        clip_low = (x < lower).float()
        with torch.no_grad():
            clip = ((upper - x) * clip_up + (lower - x) * clip_low)
        return x + clip

    def __repr__(self):
        """Returns the parameterization of the distribution.

        Returns:
            str: The parameterization of the distribution and underlying
                distribution.

        """
        return self.__class__.__name__
    
class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        try:
            self.base_dist = pyd.Normal(loc, scale)
        except (AssertionError, ValueError) as e:
            print(torch.where(torch.isnan(loc)))
        transforms = [TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        return self.base_dist.entropy()


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), device='cuda'):
        self.device = device
        self.mean = torch.zeros(shape, device=self.device)
        self.var = torch.ones(shape, device=self.device)
        self.epsilon = epsilon
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x):
        out = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return out

    def unnormalize(self, x):
        out = x * torch.sqrt(self.var + self.epsilon) + self.mean
        return out

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def get_states(self, device=None):
        if device is not None:
            return self.mean.to(device), self.var.to(device), self.epsilon
        else:
            return self.mean, self.var, self.epsilon

    def load_state_dict(self, info):
        self.mean = info[0]
        self.var = info[1]
        self.count = info[2]
