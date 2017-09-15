# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import pandas as pd
import numpy as np
from collections import OrderedDict
import os
import logging

from doppelganger import marginals

FILE_PATTERN = 'state_{}_puma_{}_{}'
logging.basicConfig(filename='logs', filemode='a', level=logging.INFO)


class Accuracy(object):
    def __init__(self, person_pums, household_pums, marginals,
                 generated_persons, generated_households):
        self.person_pums = person_pums
        self.household_pums = household_pums
        self.marginals = marginals
        self.generated_persons = generated_persons
        self.generated_households = generated_households

    @staticmethod
    def from_data_dir(state, puma, data_dir):
        '''Helper method for loading datafiles with same format output by download_allocate_generate
        run script

        Args:
            state: state id
            puma: puma id
            data_dir: directory with stored csv files

        Return: an initialized Accuracy object
        '''
        return Accuracy.from_csvs(
                state, puma,
                data_dir + os.path.sep + FILE_PATTERN.format(state, puma, 'persons_pums.csv'),
                data_dir + os.path.sep + FILE_PATTERN.format(state, puma, 'households_pums.csv'),
                data_dir + os.path.sep + FILE_PATTERN.format(state, puma, 'marginals.csv'),
                data_dir + os.path.sep + FILE_PATTERN.format(state, puma, 'people.csv'),
                data_dir + os.path.sep + FILE_PATTERN.format(state, puma, 'households.csv')
            )

    @staticmethod
    def from_csvs(
                state, puma,
                person_pums_filepath,
                household_pums_filepath,
                marginals_filepath,
                generated_persons_filepath,
                generated_households_filepath,
            ):
        '''Load csv files for use in accuracy calcs'''

        msg = '''Accuracy's from_data_dir Assumes files of the form:
            state_{state_id}_puma_{puma_id}_X
        Where X is contained in the set:
            {persons_pums.csv, households_pums.csv, marginals.csv, people.csv, households.csv}
        '''
        try:
            df_person = pd.read_csv(person_pums_filepath)
            df_household = pd.read_csv(household_pums_filepath)
            df_marginal = pd.read_csv(marginals_filepath)
            df_gen_persons = pd.read_csv(generated_persons_filepath)
            df_gen_households = pd.read_csv(generated_households_filepath)
        except IOError as e:
            logging.exception('{}\n{}'.format(msg, str(e)))
        return Accuracy(df_person, df_household, df_marginal, df_gen_persons, df_gen_households)

    def calc_accuracy(self, verbose=False):
        '''
        Args:
            verbose (boolean): toggle individual variable printing
        '''
        # Marginal Variable Subset
        variables = dict()
        # variables['age'] = ['0-17', '18-34', '65+', '35-64']
        # variables['num_people'] = ['1', '3', '2', '4+']

        for control, bin_dict in marginals.CONTROLS.items():
            variables[control] = bin_dict.keys()

        variables['num_people'].remove('count')  # count should be its own marginals variable
        d = OrderedDict()
        for variable, bins in variables.items():
            for bin in bins:
                d[(variable, bin)] = list()
                if variable == 'age':
                    d[(variable, bin)].append(
                            self.person_pums[self.person_pums[variable] == bin].person_weight.sum())
                    d[(variable, bin)].append(
                            self.generated_persons[self.generated_persons[variable] == bin]
                            .count()[0])
                elif variable == 'num_people':
                    d[(variable, bin)].append(self.household_pums[
                        self.household_pums[variable] == bin].household_weight.sum())
                    d[(variable, bin)].append(
                            self.generated_households[self.generated_households[variable] == bin]
                            .count()[0]
                        )
                elif variable == 'num_vehicles':
                    d[(variable, bin)].append(self.household_pums[
                        self.household_pums[variable] == bin].household_weight.sum())
                    d[(variable, bin)].append(
                            self.generated_households[self.generated_households[variable] == bin]
                            .count()[0]
                        )
                d[(variable, bin)].append(self.marginals[variable+'_'+bin].sum())
            # end bin
        # end variable

        df = pd.DataFrame(d.values(), index=d.keys(), columns=['pums', 'gen1', 'marginal'])

        baseline = np.abs(df.pums - df.marginal)/((df.pums + df.marginal)/2)
        doppel1 = np.abs(df.gen1 - df.marginal)/((df.gen1 + df.marginal)/2)

        df = pd.DataFrame([baseline, doppel1]).transpose()
        df.columns = ['marginal-pums', 'marginal-doppelganger']
        if verbose:
            print(df)
            logging.info(df)
        return np.mean(baseline), np.mean(doppel1)
