import logging
import operator
from abc import abstractmethod


class BaseOracle:

    def __init__(self, max_memory):
        self.max_memory = max_memory

    @abstractmethod
    def get_super_arm(self, upper_bounds, context_vectors, bandit_arms):
        pass

    @staticmethod
    def removed_covered_tables(arm_ucb_dict, chosen_id, bandit_arms):
        """

        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            if not (set(bandit_arms[arm_id].table_names) == set(bandit_arms[chosen_id].table_names) and
                    set(bandit_arms[arm_id].group_by_columns) == set(bandit_arms[chosen_id].group_by_columns) and
                    bandit_arms[arm_id].filter_by == bandit_arms[chosen_id].filter_by):
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict

    @staticmethod
    def removed_covered_queries(arm_ucb_dict, chosen_id, bandit_arms):
        """

        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            if not (bandit_arms[arm_id].query_ids == bandit_arms[chosen_id].query_ids):
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict

    @staticmethod
    def removed_low_expected_rewards(arm_ucb_dict, threshold):
        """
        It make sense to remove arms with low expected reward
        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param threshold: expected reward threshold
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            if arm_ucb_dict[arm_id] > threshold:
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict


class OracleV1(BaseOracle):

    def get_super_arm(self, upper_bounds, context_vectors, bandit_arms):
        used_memory = 0
        chosen_arms = []
        arm_ucb_dict = {}

        for i in range(len(bandit_arms)):
            arm_ucb_dict[i] = upper_bounds[i]

        max_ucb = max(arm_ucb_dict.values()) if arm_ucb_dict.values() else 1
        filter_val = min(max_ucb/10, 1)
        arm_ucb_dict = self.removed_low_expected_rewards(arm_ucb_dict, filter_val)

        while len(arm_ucb_dict) > 0:
            max_ucb_arm_id = max(arm_ucb_dict.items(), key=operator.itemgetter(1))[0]
            if bandit_arms[max_ucb_arm_id].memory < self.max_memory - used_memory:
                logging.debug("Selected: " + str(bandit_arms[max_ucb_arm_id].index_name) + ' - ' + str(arm_ucb_dict[max_ucb_arm_id]))
                chosen_arms.append((max_ucb_arm_id, arm_ucb_dict[max_ucb_arm_id]))
                used_memory += bandit_arms[max_ucb_arm_id].memory
                arm_ucb_dict = self.removed_covered_tables(arm_ucb_dict, max_ucb_arm_id, bandit_arms)
                arm_ucb_dict = self.removed_covered_queries(arm_ucb_dict, max_ucb_arm_id, bandit_arms)
            else:
                arm_ucb_dict.pop(max_ucb_arm_id)

        return
