import os
import pickle
from functools import partial

import causaldag as cd
import igraph as ig
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import random, lax, jit
from jax import vmap
from jax.tree_util import tree_map
from scipy.special import logsumexp
from tqdm import tqdm

from .dibs.eval.target import make_graph_model
from .dibs.inference import MarginalDiBS, JointDiBS
from .dibs.kernel import (
    FrobeniusSquaredExponentialKernel,
    JointAdditiveFrobeniusSEKernel,
)
from .dibs.models.linearGaussian import LinearGaussianJAX
from .dibs.models.linearGaussianEquivalent import BGeJAX
from .dibs.models.nonlinearGaussian import DenseNonlinearGaussianJAX
from .dibs.utils.func import (
    particle_marginal_empirical,
    particle_joint_empirical,
    particle_joint_mixture,
    particle_marginal_mixture,
)
from .dibs.utils.graph import elwise_acyclic_constr_nograd
from .dibs.utils.tree import tree_select, tree_index
from .posterior_model import PosteriorModel
from ..utils import utils


class DiBS_BGe(PosteriorModel):
    def __init__(self, args, precision_matrix=None):
        self.key = random.PRNGKey(args.seed)
        self.num_nodes = args.num_nodes
        self.precision_matrix = precision_matrix
        self.ensemble = False
        self.reset_after_each_update = not args.warm_start

        graph_model = make_graph_model(
            n_vars=args.num_nodes,
            graph_prior_str=args.dibs_graph_prior,
            edges_per_node=args.exp_edges,
        )

        inference_model = BGeJAX(
            mean_obs=jnp.zeros(args.num_nodes),
            alpha_mu=1.0,
            alpha_lambd=args.num_nodes + 2,
        )

        def log_prior(single_w_prob):
            """log p(G) using edge probabilities as G"""
            return graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

        def log_likelihood(single_w, x, interv_targets):
            log_lik = inference_model.log_marginal_likelihood_given_g(
                w=single_w, data=x, interv_targets=interv_targets
            )
            return log_lik

        self.eltwise_log_prob = vmap(
            lambda g, x, interv_targets: log_likelihood(g, x, interv_targets),
            (0, None, None),
            0,
        )

        # SVGD + DiBS hyperparams
        self.n_particles = args.n_particles
        self.n_steps = lambda t: 3000 #int(100*t/15)

        # initialize kernel and algorithm
        kernel = FrobeniusSquaredExponentialKernel(h=args.h_latent)

        self.model = MarginalDiBS(
            kernel=kernel,
            target_log_prior=log_prior,
            target_log_marginal_prob=log_likelihood,
            alpha_linear=args.alpha_linear,
        )

        self.key, subk = random.split(self.key)
        self.particles_z = self.model.sample_initial_random_particles(
            key=subk, n_particles=self.n_particles, n_vars=args.num_nodes
        )

    def update(self, data):
        data_samples = jnp.array(data.samples)
        interv_targets = jnp.zeros((data_samples.shape[0], self.num_nodes)).astype(bool)
        int_idx = np.argwhere(data.nodes >= 0)
        interv_targets = interv_targets.at[int_idx, data.nodes[int_idx]].set(True)

        if self.reset_after_each_update:
            self.key, subk = random.split(self.key)
            self.particles_z = self.model.sample_initial_random_particles(
                key=subk, n_particles=self.n_particles, n_vars=self.num_nodes
            )

        self.key, subk = random.split(self.key)
        self.particles_z = self.model.sample_particles(
            key=subk,
            n_steps=self.n_steps(data_samples.shape[0]),
            init_particles_z=self.particles_z,
            data=data_samples,
            interv_targets=interv_targets,
        )
        self.update_dist(data_samples, interv_targets)

    def sample_interventions(self, nodes, value_samplers, nsamples):

        # Collect interventional samples
        # Bootstraps x Interventions x Samples x Nodes
        cov_mat = np.linalg.inv(self.precision_matrix)
        dags = [
            utils.cov2dag(cov_mat, cd.DAG.from_amat(dag)) for dag in self.posterior[0]
        ]
        datapoints = np.array(
            [
                [
                    dag.sample_interventional({node: sampler}, nsamples=nsamples)
                    for node, sampler in zip(nodes, value_samplers)
                ]
                for dag in dags
            ]
        )

        return datapoints

    def update_dist(self, data, interv_targets):
        particles_g = self.model.particle_to_g_lim(self.particles_z)
        # self.posterior = particle_marginal_empirical(particles_g)
        self.posterior = particle_marginal_mixture(
           particles_g, self.eltwise_log_prob, data, interv_targets
        )
        # self.is_dag = elwise_acyclic_constr_nograd(self.posterior[0], self.num_nodes) == 0
        # print(f'{self.posterior[0]}\n{self.posterior[1]}\n{particles_g.shape}\n{self.particles_z.shape}')


    def sample(self, num_samples):
        self.key, subk = random.split(self.key)
        sampled_particles = random.categorical(
            key=subk, logits=self.posterior[1], shape=[num_samples]
        )
        return self.model.particle_to_g_lim(self.particles_z)[sampled_particles]

    def log_prob_single(self, graph):
        particles, log_prob = self.posterior
        equal = jnp.sum(not jnp.equal(particles, graph), axis=0) == 0
        index = jnp.nonzero(equal, size=1, fill_value=-1)

        if index == -1:
            return jnp.inf
        else:
            return log_prob[index]

    def log_prob(self, graphs):
        return vmap(self.log_prob_single, 0, 0)(graphs)

    def interventional_likelihood(self, graph, data, interventions):
        graph = jnp.array(graph)
        data = jnp.array(data)
        interv_targets = jnp.zeros(data.shape[-1]).astype(bool)
        nodes = interventions.keys(0)
        interv_targets[nodes] = True
        return self.eltwise_log_prob(graph, x, interv_targets)


class DiBS_Linear(PosteriorModel):
    def __init__(self, args, precision_matrix = None):
        self.key = random.PRNGKey(args.seed)
        self.num_nodes = args.num_nodes
        self.precision_matrix = precision_matrix
        self.ensemble = False
        self.reset_after_each_update = not args.warm_start

        graph_model = make_graph_model(
            n_vars=args.num_nodes,
            graph_prior_str=args.dibs_graph_prior,
            edges_per_node=args.exp_edges,
        )

        self.inference_model = LinearGaussianJAX(
            obs_noise=args.noise_sigma,
            mean_edge=0.0,
            sig_edge=2.0,
        )

        def log_prior(single_w_prob):
            """log p(G) using edge probabilities as G"""
            return graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

        def log_likelihood(single_w, single_theta, single_sigma, x, interv_targets, rng):
            log_prob_theta = self.inference_model.log_prob_parameters(theta=single_theta, w=single_w)
            log_lik = self.inference_model.log_likelihood(
                w=single_w, theta=single_theta, sigma=single_sigma, data=x, interv_targets=interv_targets
            )
            return log_prob_theta + log_lik

        self.eltwise_log_prob = vmap(
            lambda g, theta, sigma, x, interv_targets: log_likelihood(
                g, theta, sigma, x, interv_targets, None
            ),
            (0, 0, 0, None, None),
            0,
        )

        # SVGD + DiBS hyperparams
        self.n_particles = args.n_particles
        self.n_steps = lambda t: 3000 #int(100*t/15)

        # initialize kernel and algorithm
        kernel = JointAdditiveFrobeniusSEKernel(
            h_latent=args.h_latent, h_theta=args.h_theta, h_sigma=args.h_sigma
        )

        self.model = JointDiBS(
            kernel=kernel,
            target_log_prior=log_prior,
            target_log_joint_prob=log_likelihood,
            alpha_linear=args.alpha_linear,
        )

        if self.reset_after_each_update or (not hasattr(self, 'particles_z')):
            self.key, subk = random.split(self.key)
            self.particles_z, self.particles_w, self.particles_sigma = self.model.sample_initial_random_particles(
                key=subk, n_particles=self.n_particles, model = self.inference_model, n_vars=self.num_nodes
            )

        self.args = args

    def update(self, data):
        data_samples = jnp.array(data.samples)
        interv_targets = jnp.zeros((data_samples.shape[0], self.num_nodes)).astype(bool)
        int_idx = np.argwhere(data.nodes >= 0)
        interv_targets = interv_targets.at[int_idx, data.nodes[int_idx]].set(True)

        if self.reset_after_each_update:
            self.__init__(self.args)
            # self.key, subk = random.split(self.key)
            # self.particles_z, self.particles_w, self.particles_sigma = self.model.sample_initial_random_particles(
            # key=subk, n_particles=self.n_particles, model = self.inference_model, n_vars=self.num_nodes
            # )

        self.key, subk = random.split(self.key)

        self.particles_z, self.particles_w, self.particles_sigma = self.model.sample_particles(
            key=subk,
            n_steps=self.n_steps(data_samples.shape[0]),
            init_particles_z=self.particles_z,
            init_particles_theta=self.particles_w,
            init_particles_sigma=self.particles_sigma,
            data=data_samples,
            interv_targets=interv_targets,
        )
        self.update_dist(data_samples, interv_targets)

    def update_dist(self, data, interv_targets):
        particles_g = self.model.particle_to_g_lim(self.particles_z)
        # self.dibs_empirical = particle_marginal_empirical(particles_g)
        self.posterior = particle_joint_mixture(
            particles_g, self.particles_w, self.particles_sigma, self.eltwise_log_prob, data, interv_targets
        )
        is_dag = elwise_acyclic_constr_nograd(self.posterior[0], self.num_nodes) == 0

        self.dags = self.posterior[0][is_dag, :, :]
        self.thetas = self.posterior[1][is_dag, :, :]
        self.sigmas = self.posterior[2][is_dag, :]
        self.logits = self.posterior[3][is_dag]
        self.is_dag = is_dag

    def sample(self, num_samples):
        self.key, subk = random.split(self.key)
        sampled_particles = random.categorical(
            key=subk, logits=self.logits, shape=[num_samples]
        )
        graphs, thetas, sigmas = self.model.particle_to_g_lim(self.particles_z[self.is_dag])[sampled_particles], self.particles_w[sampled_particles], self.particles_sigma[sampled_particles]
        return graphs, thetas, sigmas

    def log_prob_single(self, graph):
        particles, log_prob = self.posterior
        equal = jnp.sum(not jnp.equal(particles, graph), axis=0) == 0
        index = jnp.nonzero(equal, size=1, fill_value=-1)

        if index == -1:
            return jnp.inf
        else:
            return log_prob[index]

    def log_prob(self, graphs):
        return vmap(self.log_prob_single, 0, 0)(graphs)

class DiBS_NonLinear(PosteriorModel):
    def __init__(self, args):
        self.key = random.PRNGKey(args.seed)
        self.num_nodes = args.num_nodes
        self.ensemble = True
        self.reset_after_each_update = args.warm_start

        graph_model = make_graph_model(
            n_vars=args.num_nodes,
            graph_prior_str=args.dibs_graph_prior,
            edges_per_node=args.exp_edges,
        )

        self.inference_model = DenseNonlinearGaussianJAX(obs_noise = args.noise_sigma,
            sig_param = 1.0, hidden_layers=[5,]
        )

        def log_prior(single_w_prob):
            """log p(G) using edge probabilities as G"""
            return graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

        def log_likelihood(single_w, single_theta, single_sigma, x, interv_targets, rng):
            log_prob_theta = self.inference_model.log_prob_parameters(theta=single_theta, w=single_w)
            log_lik = self.inference_model.log_likelihood(
                w=single_w, theta=single_theta, sigma=single_sigma, data=x, interv_targets=interv_targets
            )
            return log_lik + log_prob_theta


        self.eltwise_log_prob_single = vmap(
            lambda g, theta, sigma, x, interv_targets: self.inference_model.log_likelihood_single(
                w=g, theta=theta, sigma=sigma, data=x, interv_targets=interv_targets
            ),
            (0, 0, 0, None, None),
            0,
        )

        # SVGD + DiBS hyperparams
        self.n_particles = args.n_particles
        self.n_steps = args.dibs_steps

        # initialize kernel and algorithm
        kernel = JointAdditiveFrobeniusSEKernel(
            h_latent=args.h_latent, h_theta=args.h_theta, h_sigma=args.h_sigma
        )

        self.model = JointDiBS(
            kernel=kernel,
            target_log_prior=log_prior,
            target_log_joint_prob=log_likelihood,
            alpha_linear=args.alpha_linear,
        )

        self.key, subk = random.split(self.key)
        self.particles_z, self.particles_w, self.particles_sigma = self.model.sample_initial_random_particles(
            key=subk, n_particles=self.n_particles, model = self.inference_model, n_vars=self.num_nodes
        )

    def update(self, data):
        data_samples = jnp.array(data.samples)

        interv_targets = jnp.zeros((data_samples.shape[0], self.num_nodes)).astype(bool)
        int_idx = np.argwhere(data.nodes >= 0)
        interv_targets = interv_targets.at[int_idx, data.nodes[int_idx]].set(True)

        if self.reset_after_each_update:
            self.key, subk = random.split(self.key)
            self.particles_z, self.particles_w, self.particles_sigma = self.model.sample_initial_random_particles(
            key=subk, n_particles=self.n_particles, model = self.inference_model, n_vars=self.num_nodes
            )

        self.key, subk = random.split(self.key)

        self.particles_z, self.particles_w, self.particles_sigma = self.model.sample_particles(
            key=subk,
            n_steps=self.n_steps,
            init_particles_z=self.particles_z,
            init_particles_theta=self.particles_w,
            init_particles_sigma=self.particles_sigma,
            data=data_samples,
            interv_targets=interv_targets,
        )
        self.update_dist()

    def update_dist(self):
        particles_g = self.model.particle_to_g_lim(self.particles_z)
        _posterior = particle_joint_empirical(particles_g, self.particles_w, self.particles_sigma)

        is_dag = elwise_acyclic_constr_nograd(_posterior[0], self.num_nodes) == 0
        self.all_graphs = _posterior[0]
        self.dags = _posterior[0][is_dag, :, :]
        self.sigma = _posterior[2][is_dag, :]

        self.posterior = self.dags, tree_select(_posterior[1], is_dag), tree_select(_posterior[2], is_dag), _posterior[3][is_dag] - logsumexp(_posterior[3][is_dag])
        self.full_posterior = _posterior

    def sample(self, num_samples):
        self.key, subk = random.split(self.key)
        sampled_particles = random.categorical(
            key=subk, logits=self.posterior[3], shape=[num_samples]
        )
        return self.model.particle_to_g_lim(self.particles_z)[sampled_particles], self.particles_w[sampled_particles], self.particles_sigma[sampled_particles]

    def sample_interventions(self, nodes, value_samplers, nsamples):

        # Collect interventional samples
        # Bootstraps x Interventions x Samples x Nodes
        thetas = self.posterior[1]
        #self.key, subk = random.split(self.key)
        all_dags = []
        for i, dag in enumerate(self.dags):
            theta = tree_index(thetas, i)
            all_interventions = []
            for node, sampler in zip(nodes, value_samplers):
                self.key, subk = random.split(self.key)
                all_interventions.append(self.inference_model.sample_obs(key = subk, n_samples=nsamples, g = ig.Graph.Weighted_Adjacency(dag.tolist()), theta = theta, node = node, value_sampler = sampler))
            all_dags.append(all_interventions)
        return np.array(all_dags)

    def log_prob_single(self, graph):
        particles, particles_w, log_prob = self.posterior
        equal = jnp.sum(not jnp.equal(particles, graph), axis=0) == 0
        index = jnp.nonzero(equal, size=1, fill_value=-1)

        if index == -1:
            return jnp.inf
        else:
            return log_prob[index]

    def log_prob(self, graphs):
        return vmap(self.log_prob_single, 0, 0)(graphs)

    def interventional_likelihood(
        self, graph_ix, data, interventions, all_graphs=False, onehot=False
    ):
        if all_graphs:
            posterior = self.full_posterior
        else:
            posterior = self.posterior

        graph = posterior[0][graph_ix] # inner
        theta = tree_index(posterior[1], graph_ix)
        sigma = tree_index(posterior[2], graph_ix)

        data = jnp.array(data)
        interv_targets = jnp.zeros(data.shape[-1]).astype(bool)
        if interventions is not None:
            if type(interventions) == dict:
                nodes = list(interventions.keys())[0]
            else:
                nodes = interventions

            if onehot:
                interv_targets = nodes
            else:
                interv_targets = interv_targets.at[jnp.int32(nodes)].set(True)

        # print(f'data shape: {data.shape} interv_targets: {interv_targets.shape}')

        # we nest the vmaps because we want to
        # broadcast over 0 and 1 axes of data.
        fn = lambda graph, theta, sigma, data, targets: self.eltwise_log_prob_single(graph, theta, sigma, data, targets)

        res =  vmap(
                    vmap(fn, (None, None, None, 0, None)
                ),
                (None, None, None, 1, None)
        )(graph, theta, sigma, data, interv_targets)

        # print(f'res shape: {res.shape}')

        # nodes x samples x outer x inner x 1
        return res

    def _update_likelihood(self, nodes, nsamples, value_samplers, datapoints):
        matrix = np.stack([
                    self.interventional_likelihood(
                        graph_ix=jnp.arange(len(self.dags)),
                        data=datapoints[:, intv_ix].reshape(-1, len(nodes)),
                        interventions={nodes[intv_ix]: intervention}
                    ).reshape(len(self.dags), len(self.dags), nsamples)
            for intv_ix, intervention in tqdm(enumerate(value_samplers), total=len(value_samplers))])
        logpdfs = xr.DataArray(
            matrix,
            dims=['intervention_ix', 'inner_dag', 'outer_dag', 'datapoint'],
            coords={
                'intervention_ix': list(range(len(nodes))),
                'inner_dag': list(range(len(self.dags))),
                'outer_dag': list(range(len(self.dags))),
                'datapoint': list(range(nsamples)),
            })

        return logpdfs

    def save(self, path):
        with open(os.path.join(path, "particles_z.pkl"), "wb") as b:
            pickle.dump(self.particles_z, b)
            b.close()
        with open(os.path.join(path, "particles_w.pkl"), "wb") as f:
            pickle.dump(self.particles_w, f)
            f.close()

    def load(self, path):
        with open(os.path.join(path, "particles_z.pkl"), "rb") as b:
            self.particles_z = pickle.load(b)
            b.close()
        self.particles_z = jnp.array(self.particles_z)
        with open(os.path.join(path, "particles_w.pkl"), "rb") as f:
            self.particles_w = pickle.load(f)
            f.close()
        self.particles_w = tree_map(lambda arr: jnp.array(arr), self.particles_w)
        self.update_dist()

    @partial(jit, static_argnames=('self', 'n_samples', 'deterministic', 'onehot'))
    def _batch_interventional_samples(self, nodes, values, g_mats, thetas, toporders, subks, n_samples,
                                      deterministic=False, onehot=False):

        """
		This is a jitted function which samples interventional data through ancestral sampling by iterating through
		ensemble (particles, indexed by j//T) and designs in a batch (indexed by j%T).
		Args:
		designs [opt_batch_size, batch_size, 2]: Trainable parameter of designs
		g_mats  [num_particles, d, d]: set of all adjacency matrices of all particles
		toporders [num_particles, opt_batch_size, d]: Topological ordering corresponding to g_mats, repeated over axis 1 (opt_batch_size)
		n_samples: number of samples to take per dag, design pair
		Output:
		datapoints [num_particles, batch_size, opt_batch_size, n_samples, d]: Interventioanl samples
		"""

        if not onehot:
            print("Warning: There won't be any gradients with respect to nodes.")
            nodes = jnp.int32(nodes)

        B, T = values.shape

        datapoints = lax.fori_loop(
            0,
            T * len(self.dags),
            lambda j, arr: arr.at[j // T, j % T].set(
                self.inference_model.new_sample_obs(
                    key=subks[j % T, j // T],
                    g_mat=g_mats[j // T],
                    toporder=toporders[j // T],
                    theta=tree_index(thetas, j // T),
                    n_samples=n_samples,
                    nodes=nodes[:, j % T],
                    values=values[:, j % T],
                    deterministic=deterministic,
                    onehot=onehot
                )
            )
            ,
            jnp.zeros((len(self.dags), T, B, n_samples, self.num_nodes))
        )

        # Dags x T x B x N x D
        return datapoints

    def batch_interventional_samples(self, nodes, values, n_samples, deterministic=False, onehot=False):
        # Collect interventional samples
        # Bootstraps x Interventions x Samples x Nodes
        _thetas = self.posterior[1]

        B, T = values.shape

        g_mats = []
        toporders = []
        thetas = []
        for i, dag in enumerate(self.dags):
            g = ig.Graph.Weighted_Adjacency(dag.tolist())
            g_mat = jnp.array(g.get_adjacency().data)
            g_mats.append(g_mat)
            thetas.append(tree_index(_thetas, i))
            toporders.append(g.topological_sorting())
        toporders = jnp.array(toporders)
        g_mats = jnp.array(g_mats)

        toporders = toporders[:, None].repeat(B, 1)

        # bulk fetch keys
        keys = random.split(self.key, T * len(self.dags) + 1)
        self.key = keys[0]
        subks = keys[1:].reshape(T, len(self.dags), 2)

        return self._batch_interventional_samples(nodes,
                                                  values,
                                                  g_mats,
                                                  _thetas,
                                                  toporders,
                                                  subks,
                                                  n_samples,
                                                  deterministic=deterministic,
                                                  onehot=onehot)

    @partial(jit, static_argnames=('self', 'roll', 'onehot'))
    def batch_likelihood(self, nodes, datapoints, roll=False, onehot=False):
        # dags x designs x batch x samples x dims
        like_fn = lambda nodes, data: self.likelihood(nodes, data, onehot=onehot)
        logprobs = vmap(
            like_fn, (0, 2)
        )(nodes, datapoints)

        return logprobs

    def likelihood(self, nodes, datapoints, onehot=False):
        # (20, 3, 100, 50)
        # datapoints: dags x interventions x samples x nodes
        like_fn = lambda graph_idx, data, nodes: self.interventional_likelihood(
                        graph_ix=graph_idx, data=data, interventions=nodes,
                    )

        logprobs = vmap(
            like_fn, (None, 1, 0)
        )(jnp.arange(len(self.dags)), datapoints, nodes)
        # print(f'logprobs shape: {logprobs.shape}')

        return logprobs[..., 0].transpose(0, 2, 3, 1)