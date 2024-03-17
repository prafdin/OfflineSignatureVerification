from itertools import product

import numpy as np
import yaml
from abc import ABC, abstractmethod

class ConfigurationMatrixInterface(ABC):
    @abstractmethod
    def get_configuration(self, idx):
        pass
    @abstractmethod
    def get_shape(self):
        pass
    @abstractmethod
    def axis_components(self, axis_name):
        pass

class ConfigurationMatrix(ConfigurationMatrixInterface):
    def __init__(self, axis, variants):
        self._axis_idx = list(range(len(axis)))
        self._axis_names = axis

        variants_by_idx = {}
        for variant in variants:
            axe_name = list(variant.keys())[0]
            if axe_name not in self._axis_names:
                print(f"[WARNING] There is not '{axe_name}' axe in axis list")
                continue
            axe_id = self._axis_names.index(axe_name)
            variants_by_idx[axe_id] = variant[axe_name]

        self._matrix_components = []
        for axe_id in self._axis_idx:
            self._matrix_components.append(variants_by_idx[axe_id])

        self._matrix = np.array(
            np.meshgrid(*[list(range(len(matrix_component))) for matrix_component in self._matrix_components],
                        indexing='ij')
        )
        self._shape = self._matrix.shape[1:]

    def get_configuration(self, idx):
        return [self._matrix_components[m][self._matrix[m][idx]] for m in self._axis_idx]

    def axis_components(self, axis_name):
        return self._matrix_components[self._axis_names.index(axis_name)]

    def get_shape(self):
        return self._shape

class ConfigurationMatrixWithExcludes(ConfigurationMatrixInterface):
    def __init__(self, configuration_matrix: ConfigurationMatrixInterface, axes, excludes):
        self._configuration_matrix = configuration_matrix
        self._axis_names = axes

        self._excludes = []
        for exclude in excludes:
            exclude_idx = []
            for axis_name in self._axis_names:
                if axis_name in exclude:
                    exclude_idx.append(
                        [i for i, val in enumerate(self._configuration_matrix.axis_components(axis_name)) if val in exclude[axis_name]])
                else:
                    exclude_idx.append(list(range(len(self._configuration_matrix.axis_components(axis_name)))))
            self._excludes.extend(list(product(*exclude_idx)))

    def get_configuration(self, idx):
        if idx in self._excludes:
            return None
        return self._configuration_matrix.get_configuration(idx)

    def axis_components(self, axis_name):
        return self._configuration_matrix.axis_components(axis_name)

    def get_shape(self):
        return self._configuration_matrix.get_shape()

class PrintableConfigurationMatrix:
    def __init__(self, configuration_matrix: ConfigurationMatrixInterface):
        self.configuration_matrix = configuration_matrix

    def _get_all_configuration(self):
        length_per_dimension = self.configuration_matrix.get_shape()
        indexes_for_all_elements = product(*[list(range(length)) for length in length_per_dimension])
        return [self.configuration_matrix.get_configuration(idx) for idx in indexes_for_all_elements]

    def print(self):
        for configuration in self._get_all_configuration():
            print(configuration)


def read_test(test):
    test_name = list(test.keys())[0]
    configuration = test[test_name]
    configuration_matrix = ConfigurationMatrix(
        configuration['axis'],
        configuration['variants']
    )
    configuration_matrix = ConfigurationMatrixWithExcludes(configuration_matrix, configuration['axis'], configuration['excludes'])

    printable_configuration_matrix = PrintableConfigurationMatrix(configuration_matrix)
    printable_configuration_matrix.print()


if __name__ == '__main__':
    with open("tests.yaml") as f:
        y = yaml.safe_load(f)

    first_test = y['tests'][0]
    read_test(first_test)
