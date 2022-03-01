import copy
import logging
from abc import abstractmethod
from bandits.bandit_helper_v1 import Reward

import numpy

import constants


class C3UCBBaseBandit:

    def __init__(self, context_size, hyper_alpha, hyper_lambda, oracle, cluster_id=None):
        self.arms = []
        self.alpha_original = hyper_alpha
        self.hyper_alpha = hyper_alpha
        self.hyper_lambda = hyper_lambda        # lambda in C2CUB
        self.v = hyper_lambda * numpy.identity(context_size)    # identity matrix of n*n
        self.b = numpy.zeros((context_size, 1))  # [0, 0, ..., 0]T (column matrix) size = number of arms
        self.oracle = oracle
        self.context_vectors = []
        self.upper_bounds = []
        self.context_size = context_size
        self.cluster_id = cluster_id

    @abstractmethod
    def select_arm(self, context_vectors, current_round):
        pass

    @abstractmethod
    def update(self, played_arms, arm_rewards, useless, mv_size_weight, index_size_weight):
        pass


class C3UCB(C3UCBBaseBandit):

    def select_arm(self, context_vectors, current_round):
        """
        This method is responsible for returning the super arm

        :param context_vectors: context vector for this round
        :param current_round: current round number
        :return: selected set of arms
        """
        v_inverse = numpy.linalg.inv(self.v)
        weight_vector = v_inverse @ self.b
        logging.info(f"================================\n{weight_vector.transpose().tolist()[0]}")
        self.context_vectors = context_vectors

        # find the upper bound for every arm
        for i in range(len(self.arms)):
            creation_cost = weight_vector[1] * self.context_vectors[i][1]
            average_reward = numpy.asscalar(weight_vector.transpose() @ self.context_vectors[i]) - creation_cost
            temp_upper_bound = average_reward + self.hyper_alpha * numpy.sqrt(
                numpy.asscalar(self.context_vectors[i].transpose() @ v_inverse @ self.context_vectors[i]))
            temp_upper_bound = temp_upper_bound + (creation_cost / constants.CREATION_COST_REDUCTION_FACTOR)
            self.upper_bounds.append(temp_upper_bound)

        logging.debug(self.upper_bounds)
        return self.oracle.get_super_arm(self.upper_bounds, self.context_vectors, self.arms)

    def select_super_arm_v2(self, context_vectors):
        """
        This method is responsible for returning the super arm

        :param context_vectors: context vector for this round
        :return: selected set of arms
        """
        v_inverse = numpy.linalg.inv(self.v)
        weight_vector = v_inverse @ self.b
        logging.info(f"================================\n{weight_vector.transpose().tolist()[0]}")
        self.context_vectors = context_vectors

        # find the upper bound for every arm
        for i in range(len(self.arms)):
            creation_cost = (weight_vector[1] * self.context_vectors[i][1]) + (weight_vector[0] * self.context_vectors[i][0])
            average_reward = numpy.asscalar(weight_vector.transpose() @ self.context_vectors[i]) - creation_cost
            temp_upper_bound = average_reward + self.hyper_alpha * numpy.sqrt(
                numpy.asscalar(self.context_vectors[i].transpose() @ v_inverse @ self.context_vectors[i]))
            temp_upper_bound = temp_upper_bound + (creation_cost / constants.CREATION_COST_REDUCTION_FACTOR)
            self.upper_bounds.append(temp_upper_bound)

        self.hyper_alpha = self.hyper_alpha / constants.ALPHA_REDUCTION_RATE
        return self.oracle.get_super_arm(self.upper_bounds, self.context_vectors, self.arms), weight_vector[0], weight_vector[1]

    @staticmethod
    def convert_reward(arm_reward, arm_memory):
        mod_arm_reward = copy.copy(arm_reward)
        arm_mem = (arm_memory // 1024) + 1
        mod_arm_reward.execution = mod_arm_reward.execution / arm_mem
        mod_arm_reward.maintenance = mod_arm_reward.maintenance / arm_mem
        mod_arm_reward.creation = mod_arm_reward.creation / arm_mem
        mod_arm_reward.offset = mod_arm_reward.offset / arm_mem
        return mod_arm_reward

    def update(self, played_arms, arm_rewards, useless, mv_size_weight, index_size_weight):
        """
        This method can be used to update the reward after each play (improvements required)

        :param played_arms: list of played arms (super arm)
        :param arm_rewards: tuple (gains, creation cost) reward got form playing each arm
        """
        for i in played_arms:
            is_useless = self.arms[i].index_name in useless
            if self.arms[i].index_name in arm_rewards:
                arm_reward_original = arm_rewards[self.arms[i].index_name]
                arm_reward_modified = self.convert_reward(arm_rewards[self.arms[i].index_name], self.arms[i].memory)
                arm_reward = arm_reward_modified
            else:
                arm_reward = Reward()
                arm_reward_original = arm_reward
                arm_reward_modified = arm_reward
            logging.info(f"[Useless?{is_useless}]Reward for {self.arms[i].index_name}, {arm_reward.queries} "
                         f"is: (e, m, c, o) - "
                         f"({round(arm_reward_original.execution, 2)}, {round(arm_reward_original.maintenance, 2)}, {round(arm_reward_original.creation, 2)}, {round(arm_reward_original.offset, 2)})"
                         f"({round(arm_reward_modified.execution, 2)}, {round(arm_reward_modified.maintenance, 2)}, {round(arm_reward_modified.creation, 2)}, {round(arm_reward_modified.offset, 2)})")
            execution_reward = arm_reward.execution + arm_reward.maintenance
            if constants.UNCLAIMED_REWARD_DISTRIBUTION:
                execution_reward += arm_reward.offset
            self.arms[i].index_usage_last_batch = (self.arms[i].index_usage_last_batch + execution_reward) / 2

            temp_context = numpy.zeros(self.context_vectors[i].shape)
            temp_context[1] = self.context_vectors[i][1]
            self.context_vectors[i][1] = 0

            self.v = self.v + (self.context_vectors[i] @ self.context_vectors[i].transpose())
            self.b = self.b + self.context_vectors[i] * execution_reward
            if not is_useless:
                self.v = self.v + (temp_context @ temp_context.transpose())
                self.b = self.b + temp_context * arm_reward.creation
            else:
                if type(self.arms[i]).__name__ == 'BanditArmMV' and mv_size_weight < 10 and index_size_weight != 0 :
                    self.v = self.v + (temp_context @ temp_context.transpose())
                    self.b = self.b + temp_context * (temp_context[1] * index_size_weight)

        if played_arms:
            self.hyper_alpha = self.hyper_alpha / constants.ALPHA_REDUCTION_RATE

        self.context_vectors = []
        self.upper_bounds = []

    def update_super_v3(self, played_arms, arm_rewards, useless, mv_size_weight, index_size_weight):
        """
        This method can be used to update the reward after each play (improvements required)

        :param played_arms: list of played arms (super arm)
        :param arm_rewards: tuple (gains, creation cost) reward got form playing each arm
        """
        for i in played_arms:
            is_useless = self.arms[i].index_name in useless
            if self.arms[i].index_name in arm_rewards:
                arm_reward_original = arm_rewards[self.arms[i].index_name]
                arm_reward_modified = self.convert_reward(arm_rewards[self.arms[i].index_name], self.arms[i].memory)
                arm_reward = arm_reward_modified
            else:
                arm_reward = Reward()
                arm_reward_original = arm_reward
                arm_reward_modified = arm_reward
            logging.info(f"[Useless?{is_useless}] Super Reward for {self.arms[i].index_name}, {arm_reward.queries} "
                         f"is: (e, m, c, o) - "
                         f"({round(arm_reward_original.execution, 2)}, {round(arm_reward_original.maintenance, 2)}, {round(arm_reward_original.creation, 2)}, {round(arm_reward_original.offset, 2)})"
                         f"({round(arm_reward_modified.execution, 2)}, {round(arm_reward_modified.maintenance, 2)}, {round(arm_reward_modified.creation, 2)}, {round(arm_reward_modified.offset, 2)})")
            execution_reward = arm_reward.execution + arm_reward.maintenance
            if constants.UNCLAIMED_REWARD_DISTRIBUTION:
                execution_reward += arm_reward.offset
            self.arms[i].index_usage_last_batch = (self.arms[i].index_usage_last_batch + execution_reward) / 2

            if type(self.arms[i]).__name__ == 'BanditArmMV':
                size_context = 0
            else:
                size_context = 1

            temp_context = numpy.zeros(self.context_vectors[i].shape)
            temp_context[size_context] = self.context_vectors[i][size_context]
            self.context_vectors[i][size_context] = 0

            self.v = self.v + (self.context_vectors[i] @ self.context_vectors[i].transpose())
            self.b = self.b + self.context_vectors[i] * execution_reward
            if not is_useless:
                self.v = self.v + (temp_context @ temp_context.transpose())
                self.b = self.b + temp_context * arm_reward.creation
            else:
                if type(self.arms[i]).__name__ == 'BanditArmMV' and mv_size_weight < 10 and index_size_weight != 0:
                    self.v = self.v + (temp_context @ temp_context.transpose())
                    self.b = self.b + temp_context * (temp_context[size_context] * index_size_weight)

        self.context_vectors = []
        self.upper_bounds = []

    def set_arms(self, bandit_arms):
        """
        This can be used to initially set the bandit arms in the algorithm

        :param bandit_arms: initial set of bandit arms
        :return:
        """
        for arm in bandit_arms:
            arm.bandit_cluster = self.cluster_id
        self.arms = bandit_arms

    def hard_reset(self):
        """
        Resets the bandit
        """
        self.hyper_alpha = self.alpha_original
        self.v = self.hyper_lambda * numpy.identity(self.context_size)  # identity matrix of n*n
        self.b = numpy.zeros((self.context_size, 1))  # [0, 0, ..., 0]T (column matrix) size = number of arms

    def workload_change_trigger(self, workload_change):
        """
        This forgets history based on the workload change

        :param workload_change: Percentage of new query templates added (0-1) 0: no workload change, 1: 100% shift
        """
        logging.info("Workload change identified " + str(workload_change))
        if workload_change > 0.5:
            self.hard_reset()
        elif workload_change > 0.05:
            forget_factor = 1 - workload_change * 2
            if workload_change > 0.1:
                self.hyper_alpha = self.alpha_original
            self.v = self.hyper_lambda * numpy.identity(self.context_size) + forget_factor * self.v
            self.b = forget_factor * self.b
