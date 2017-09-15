# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import pandas as pd
from collections import OrderedDict
import logging

from doppelganger import Accuracy

FILE_PATTERN = 'state_{}_puma_{}_{}'
logging.basicConfig(filename='logs', filemode='a', level=logging.INFO)


def main():
    data_dir = 'data'
    print('Current Accuracy (in Mean Absolute %Diff)')
    state_puma = dict()
    '''
    state_puma['29'] = ['00901', '00902', '01001', '01002', '01003',
                        '01004', '01005', '01100', '00800']
    state_puma['20'] = ['00500', '00602', '00604']
    '''
    state_puma['29'] = ['00901', '00902']

    d_result = OrderedDict()
    for state, pumas in state_puma.items():
        for puma in pumas:
            print(state, puma)
            accuracy = Accuracy.from_data_dir(state, puma, data_dir)
            d_result[(state, puma)] = accuracy.calc_accuracy(True)

    print(pd.DataFrame(d_result.values(), index=d_result.keys(),
                       columns=['marginal-pums', 'marginal-doppelganger']))
    logging.info(pd.DataFrame(d_result.values(), index=d_result.keys(),
                              columns=['marginal-pums', 'marginal-doppelganger']))


if __name__ == '__main__':
    main()
