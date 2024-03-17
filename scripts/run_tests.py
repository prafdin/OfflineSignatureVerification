from itertools import product

import numpy as np
import yaml

class ConfigurationMatrix:
    def __init__(self, axis, variants, excludes):
        self.axis_idx = list(range(len(axis)))
        self.axis_names = axis

        self.variants = {}
        for variant in variants:
            axe_name = list(variant.keys())[0]
            if axe_name not in self.axis_names:
                print(f"[WARNING] There is not '{axe_name}' axe in axis list")
                continue
            axe_id = self.axis_names.index(axe_name)
            self.variants[axe_id] = variant[axe_name]

        self.matrix_components = []
        for axe_id in self.axis_idx:
            self.matrix_components.append(self.variants[axe_id])

        self.excludes = []
        for exclude in excludes:
            exclude_by_idx = {}
            for k in exclude.keys():
                exclude_by_idx[self.axis_names.index(k)] = exclude[k]
            exclude_idx = []
            for axe_idx in self.axis_idx:
                if axe_idx in exclude_by_idx:
                    exclude_idx.append([i for i, val in enumerate(self.matrix_components[axe_idx]) if val in exclude_by_idx[axe_idx]])
                else:
                    exclude_idx.append(list(range(len(self.matrix_components[axe_idx]))))
            self.excludes.extend(list(product(*exclude_idx)))

        self.matrix = np.array(np.meshgrid(*[list(range(len(matrix_component))) for matrix_component in self.matrix_components], indexing='ij'))



    def print(self):
        dimension, *length_per_dimension = self.matrix.shape
        indexes_for_all_elements = product(*[list(range(length)) for length in length_per_dimension])
        for idx in indexes_for_all_elements:
            print(self.get_element(idx))


    def get_element(self, idx):
        return [self.matrix_components[m][self.matrix[m][idx]] for m in self.axis_idx]


    @staticmethod
    def get_configuration_matrix(configuration):
        axis = configuration['axis']
        variants = configuration['variants']
        excludes =  configuration['excludes']
        return ConfigurationMatrix(axis, variants, excludes)


def read_test(test):
    test_name = list(test.keys())[0]
    configuration_matrix = ConfigurationMatrix.get_configuration_matrix(test[test_name])

    configuration_matrix.print()


if __name__ == '__main__':
    with open("tests.yaml") as f:
        y = yaml.safe_load(f)


    first_test = y['tests'][0]
    read_test(first_test)