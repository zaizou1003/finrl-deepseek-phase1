#!/usr/bin/env python
# coding: utf-8
# Windows / single-process / GPU-friendly Phase 1 training script

from datasets import load_dataset
import os
import time
import argparse
import scipy.signal
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from gymnasium.spaces import Box, Discrete

from spinup.utils.logx import EpochLogger

from env_stocktrading_llm_risk_phase1 import StockTradingEnv


# ---------------------------------------------------------------------
# Local config fallback to avoid finrl dependency issues
# ---------------------------------------------------------------------
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

TRAINED_MODEL_DIR = "trained_models"
RESULTS_DIR = "results"


def check_and_make_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------------------------------------------------
# Single-process fallbacks (replace MPI utilities)
# ---------------------------------------------------------------------
def setup_pytorch_for_mpi():
    pass


def sync_params(module):
    pass


def mpi_avg_grads(module):
    pass


def mpi_avg(x):
    return x


def proc_id():
    return 0


def num_procs():
    return 1


def mpi_statistics_scalar(x, with_min_and_max=False):
    x = np.asarray(x, dtype=np.float32)
    mean = float(np.mean(x)) if x.size else 0.0
    std = float(np.std(x)) if x.size else 1.0

    if with_min_and_max:
        min_val = float(np.min(x)) if x.size else 0.0
        max_val = float(np.max(x)) if x.size else 0.0
        return mean, std, min_val, max_val

    return mean, std


def setup_logger_kwargs(exp_name, seed):
    output_dir = os.path.join(RESULTS_DIR, f"{exp_name}_s{seed}")
    os.makedirs(output_dir, exist_ok=True)
    return dict(output_dir=output_dir, exp_name=exp_name)


# ---------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------
def load_phase1_train_data():
    dataset = load_dataset(
        "benstaf/nasdaq_2013_2023",
        data_files="train_data_deepseek_risk_2013_2018.csv"
    )

    train = pd.DataFrame(dataset["train"])

    if "Unnamed: 0" in train.columns:
        train = train.drop("Unnamed: 0", axis=1)

    unique_dates = train["date"].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    train["new_idx"] = train["date"].map(date_to_idx)
    train = train.set_index("new_idx")

    if "llm_sentiment" not in train.columns:
        raise ValueError("Column 'llm_sentiment' not found in training data.")
    if "llm_risk" not in train.columns:
        raise ValueError("Column 'llm_risk' not found in training data.")

    # Phase 1 neutral fill
    train["llm_sentiment"] = train["llm_sentiment"].fillna(3)
    train["llm_risk"] = train["llm_risk"].fillna(3)

    return train


train = load_phase1_train_data()
stock_dimension = len(train.tic.unique())

# cash + prices + shares + indicators + sentiment + risk
state_space = 1 + 2 * stock_dimension + (2 + len(INDICATORS)) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1_000_000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()


# ---------------------------------------------------------------------
# PPO / CPPO helpers
# ---------------------------------------------------------------------
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum(np.prod(p.shape) for p in module.parameters())


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
        else:
            raise TypeError("Unsupported action space type.")

        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.detach().cpu().numpy(), v.detach().cpu().numpy(), logp_a.detach().cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]


class CPPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.valupdate_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, valupdate, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew.item() if hasattr(rew, "item") else rew
        self.val_buf[self.ptr] = val.item() if hasattr(val, "item") else val
        self.valupdate_buf[self.ptr] = valupdate.item() if hasattr(valupdate, "item") else valupdate
        self.logp_buf[self.ptr] = logp.item() if hasattr(logp, "item") else logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        self.adv_buf[path_slice] = self.adv_buf[path_slice] - self.valupdate_buf[path_slice]
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_std = adv_std if adv_std > 1e-8 else 1.0
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def cppo(
    env_fn,
    actor_critic=MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[256, 128], activation=torch.nn.ReLU),
    seed=42,
    steps_per_epoch=10000,   # smoke test
    epochs=200,               # smoke test
    gamma=0.995,
    clip_ratio=0.5,
    pi_lr=3e-5,
    vf_lr=1e-4,
    train_pi_iters=80,
    train_v_iters=100,
    lam=0.95,
    max_ep_len=3000,
    target_kl=0.2,
    logger_kwargs=dict(),
    save_freq=10,
    alpha=0.85,
    beta=3000.0,
    nu_lr=1e-4,
    lam_lr=5e-5,
    nu_start=0.1,
    lam_start=0.01,
    nu_delay=0.5,
    lam_low_bound=0.001,
    delay=1.0,
    cvar_clip_ratio=0.05,
    pretrained_path=None,
):
    setup_pytorch_for_mpi()

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    if pretrained_path is not None:
        print(f"Loading pretrained weights from: {pretrained_path}", flush=True)

        if pretrained_path.endswith(".pth"):
            state_dict = torch.load(pretrained_path, map_location=device)
            ac.load_state_dict(state_dict, strict=True)

        elif pretrained_path.endswith(".pt"):
            loaded_obj = torch.load(pretrained_path, map_location=device)

            # SpinUp pyt_save/model.pt often stores the whole actor-critic object
            if isinstance(loaded_obj, nn.Module):
                ac.load_state_dict(loaded_obj.state_dict(), strict=True)
            elif isinstance(loaded_obj, dict):
                ac.load_state_dict(loaded_obj, strict=True)
            else:
                raise TypeError(f"Unsupported checkpoint content in {pretrained_path}: {type(loaded_obj)}")

        else:
            raise ValueError("Unsupported pretrained file format. Use .pth or .pt")

        print("Pretrained weights loaded successfully.", flush=True)
    sync_params(ac)

    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = CPPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    nu = nu_start
    cvarlam = lam_start

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        return ((ac.v(obs) - ret) ** 2).mean()

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    logger.setup_pytorch_saver(ac)

    def update():
        data = {k: v.to(device) for k, v in buf.get().items()}

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info["kl"])
            if kl > 1.5 * target_kl:
                logger.log("Early stopping at step %d due to reaching max kl." % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)
            pi_optimizer.step()

        logger.store(StopIter=i)

        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)
            vf_optimizer.step()

        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(loss_pi.item() - pi_l_old),
            DeltaLossV=(loss_v.item() - v_l_old),
        )

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    for epoch in range(epochs):
        trajectory_num = 0
        bad_trajectory_num = 0
        cvarlam = cvarlam + lam_lr * (beta - nu)
        lam_delta = 0
        nu_delta = 0
        update_num = 0

        for t in range(local_steps_per_epoch):
            if t % 5000 == 0 or t == local_steps_per_epoch - 1:
                print(f"Epoch {epoch + 1}/{epochs} | Step {t + 1}/{local_steps_per_epoch}", flush=True)
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            llm_risks = np.array(next_o[0, -stock_dimension:])
            risk_to_weight = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}
            llm_risks_weights = np.vectorize(risk_to_weight.get)(llm_risks)

            prices = np.array(next_o[0, 1:stock_dimension + 1])
            shares = np.array(next_o[0, stock_dimension + 1:stock_dimension * 2 + 1])

            stock_values = prices * shares
            total_value = np.sum(stock_values)

            if total_value == 0:
                llm_risk_factor = 1
            else:
                stock_weights = stock_values / total_value
                llm_risk_factor = np.dot(stock_weights, llm_risks_weights)

            adjusted_D_pi = llm_risk_factor * (ep_ret + v - r)

            trajectory_num += 1
            nu_delta += adjusted_D_pi
            updates = np.float32(0.0)

            if adjusted_D_pi < nu:
                bad_trajectory_num += 1
                lam_delta += adjusted_D_pi
                updates = delay * cvarlam / (1 - alpha) * (nu - adjusted_D_pi)
                if updates > abs(v) * cvar_clip_ratio:
                    updates = abs(v) * cvar_clip_ratio
                    update_num += 1
                updates = np.float32(updates)

            buf.store(o, a, r, v, updates, logp)
            logger.store(VVals=v)

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print(f"Warning: trajectory cut off by epoch at {ep_len} steps.", flush=True)

                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))
                else:
                    v = 0

                buf.finish_path(v)

                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)

                o, ep_ret, ep_len = env.reset(), 0, 0

        if bad_trajectory_num > 0:
            lam_delta = lam_delta / bad_trajectory_num
        if trajectory_num > 0:
            nu_delta = nu_delta / trajectory_num
        nu = nu_delta * nu_delay

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)

        update()

        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("VVals", with_min_and_max=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossV", average_only=True)
        logger.log_tabular("DeltaLossPi", average_only=True)
        logger.log_tabular("DeltaLossV", average_only=True)
        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("ClipFrac", average_only=True)
        logger.log_tabular("StopIter", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()

        print("-" * 37)
        print("bad_trajectory_num:", bad_trajectory_num)
        print("update num:", update_num)
        print("nu:", nu)
        print("lam:", cvarlam)
        print("-" * 37, flush=True)

    return ac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hid", type=int, default=512)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="cppo_phase1")
    parser.add_argument("-f", "--file", type=str, help="Kernel connection file")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    parser.add_argument("--pretrained_path", type=str, default=None,
                    help="Path to pretrained .pth or SpinUp pyt_save/model.pt")

    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    trained_cppo = cppo(
        lambda: env_train,
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        seed=args.seed,
        logger_kwargs=logger_kwargs,
        pretrained_path=args.pretrained_path,
    )

    model_path = os.path.join(TRAINED_MODEL_DIR, "agent_cppo_deepseek_phase1.pth")
    torch.save(trained_cppo.state_dict(), model_path)
    print("Training finished and saved in " + model_path)


if __name__ == "__main__":
    main()