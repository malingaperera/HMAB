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

    @staticmethod
    def removed_covered_tables(arm_ucb_dict, chosen_id, bandit_arms, table_count):
        """

        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :param table_count: count of indexes already chosen for each table
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            if not (bandit_arms[arm_id].table_name == bandit_arms[chosen_id].table_name and table_count[
                   bandit_arms[arm_id].table_name] >= 6):
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict


class OracleV1(BaseOracle):

    def get_super_arm(self, upper_bounds, context_vectors, bandit_arms):
        used_memory = 0
        chosen_arms = []
        arm_ucb_dict = {}
        table_count = {}

        for i in range(len(bandit_arms)):
            if bandit_arms[i].memory > 0:
                arm_ucb_dict[i] = upper_bounds[i]

        logging.debug({bandit_arms[k].index_name: v for k, v in sorted(arm_ucb_dict.items(), key=lambda item: item[1], reverse=True)})

        # arm_ucb_dict = self.removed_low_expected_rewards(arm_ucb_dict, 0)

        while len(arm_ucb_dict) > 0:
            max_ucb_arm_id = max(arm_ucb_dict.items(), key=operator.itemgetter(1))[0]
            if bandit_arms[max_ucb_arm_id].memory < self.max_memory - used_memory:
                logging.debug("Selected: " + str(bandit_arms[max_ucb_arm_id].index_name) + ' - ' + str(arm_ucb_dict[max_ucb_arm_id]))
                chosen_arms.append(max_ucb_arm_id)
                used_memory += bandit_arms[max_ucb_arm_id].memory
                if bandit_arms[max_ucb_arm_id].table_name in table_count:
                    table_count[bandit_arms[max_ucb_arm_id].table_name] += 1
                else:
                    table_count[bandit_arms[max_ucb_arm_id].table_name] = 1
                arm_ucb_dict = self.removed_covered_tables(arm_ucb_dict, max_ucb_arm_id, bandit_arms, table_count)
            if max_ucb_arm_id in arm_ucb_dict:
                arm_ucb_dict.pop(max_ucb_arm_id)

        return chosen_arms
