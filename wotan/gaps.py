from __future__ import print_function, division
from numpy import array, float32, isnan, diff, where, add, concatenate, append


def get_gaps_indexes(time, break_tolerance):
    """Array indexes where ``time`` has gaps longer than ``break_tolerance``"""
    gaps = diff(time)
    gaps_indexes = where(gaps > break_tolerance)
    gaps_indexes = add(gaps_indexes, 1) # Off by one :-)
    gaps_indexes = concatenate(gaps_indexes).ravel()  # Flatten
    gaps_indexes = append(array([0]), gaps_indexes)  # Start
    gaps_indexes = append(gaps_indexes, array([len(time)+1]))  # End point
    return gaps_indexes
