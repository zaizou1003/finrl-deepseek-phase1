from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

from config_phase1 import Phase1Config
from signal_utils_phase1 import sent_z, risk_z, modulate_action

matplotlib.use("Agg")

CFG = Phase1Config()


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym - Phase 1 upgraded signal integration."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        llm_sentiment_col="llm_sentiment",
        llm_risk_col="llm_risk",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )

        # keep a local cleaned copy so env always sees neutral values instead of NaN
        self.df = self.df.copy()
        if llm_sentiment_col in self.df.columns:
            self.df[llm_sentiment_col] = self.df[llm_sentiment_col].fillna(CFG.FILL_SENTIMENT)
        if llm_risk_col in self.df.columns:
            self.df[llm_risk_col] = self.df[llm_risk_col].fillna(CFG.FILL_RISK)

        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.llm_sentiment_col = llm_sentiment_col
        self.llm_risk_col = llm_risk_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        self.state = self._initiate_state()

        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0

        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        self.seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index + 2 * self.stock_dim + 1] != True:
                if self.state[index + self.stock_dim + 1] > 0:
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    self.state[0] += sell_amount
                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    if self.state[index + self.stock_dim + 1] > 0:
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if self.state[index + 2 * self.stock_dim + 1] != True:
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )

                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount
                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def _get_current_turbulence_value(self):
        if self.risk_indicator_col not in self.data.index and self.risk_indicator_col not in getattr(self.data, "columns", []):
            return 0.0

        try:
            if len(self.df.tic.unique()) == 1:
                turb = self.data[self.risk_indicator_col]
                if isinstance(turb, (pd.Series, np.ndarray, list)):
                    return float(np.asarray(turb).flatten()[0])
                return float(turb)
            else:
                return float(self.data[self.risk_indicator_col].values[0])
        except Exception:
            return 0.0

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            if self.make_plots:
                self._make_plot()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )

            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)

            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]

            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    f"results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv"
                )
                df_total_value.to_csv(
                    f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                df_rewards.to_csv(
                    f"results/account_rewards_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.png"
                )
                plt.close()

            return self.state, self.reward, self.terminal, False, {}

        else:
            # -------- Phase 1 signal integration --------
            raw_sentiments = np.asarray(
                self.data[self.llm_sentiment_col].values, dtype=np.float32
            )
            raw_risks = np.asarray(
                self.data[self.llm_risk_col].values, dtype=np.float32
            )

            # fill again defensively in case something slips through
            raw_sentiments = np.where(np.isnan(raw_sentiments), CFG.FILL_SENTIMENT, raw_sentiments)
            raw_risks = np.where(np.isnan(raw_risks), CFG.FILL_RISK, raw_risks)

            s_z = sent_z(raw_sentiments)
            r_z = risk_z(raw_risks)

            current_turbulence = self._get_current_turbulence_value()
            self.turbulence = current_turbulence

            actions = modulate_action(
                base_action=actions,
                sent_z_value=s_z,
                risk_z_value=r_z,
                turbulence_value=current_turbulence,
                turbulence_threshold=getattr(self, "turbulence_threshold", 70),
                gamma=CFG.GAMMA,
                lambda_reg=CFG.LAMBDA_REG,
                alpha_pos=CFG.ALPHA_POS,
                alpha_neg=CFG.ALPHA_NEG,
            )
            # -------------------------------------------

            actions = actions * self.hmax
            actions = actions.astype(int)

            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)

            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            self.day += 1
            self.data = self.df.loc[self.day, :]

            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]

            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(self.state)

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            if len(self.df.tic.unique()) > 1:
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                        [],
                    )
                    + self.data[self.llm_sentiment_col].fillna(CFG.FILL_SENTIMENT).values.tolist()
                    + self.data[self.llm_risk_col].fillna(CFG.FILL_RISK).values.tolist()
                )
            else:
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                    + [self.data[self.llm_sentiment_col] if pd.notna(self.data[self.llm_sentiment_col]) else CFG.FILL_SENTIMENT]
                    + [self.data[self.llm_risk_col] if pd.notna(self.data[self.llm_risk_col]) else CFG.FILL_RISK]
                )
        else:
            if len(self.df.tic.unique()) > 1:
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    + sum(
                        (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                        [],
                    )
                    + self.data[self.llm_sentiment_col].fillna(CFG.FILL_SENTIMENT).values.tolist()
                    + self.data[self.llm_risk_col].fillna(CFG.FILL_RISK).values.tolist()
                )
            else:
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                    + [self.data[self.llm_sentiment_col] if pd.notna(self.data[self.llm_sentiment_col]) else CFG.FILL_SENTIMENT]
                    + [self.data[self.llm_risk_col] if pd.notna(self.data[self.llm_risk_col]) else CFG.FILL_RISK]
                )

        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                    [],
                )
                + self.data[self.llm_sentiment_col].fillna(CFG.FILL_SENTIMENT).values.tolist()
                + self.data[self.llm_risk_col].fillna(CFG.FILL_RISK).values.tolist()
            )
        else:
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                + [self.data[self.llm_sentiment_col] if pd.notna(self.data[self.llm_sentiment_col]) else CFG.FILL_SENTIMENT]
                + [self.data[self.llm_risk_col] if pd.notna(self.data[self.llm_risk_col]) else CFG.FILL_RISK]
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs