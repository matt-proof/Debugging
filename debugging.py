import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import logging


def display_var(name, val, add_space = False):
    """Displays a variable with its name"""

    assert isinstance(name, str)

    if isinstance(val, pd.DataFrame):
        print(name)
        display(val)
    else:
        print(f'{name} = {val}')
        if add_space:
            print()


def debug(*expressions, add_space = False):
    """Prints variables and names for debugging"""

    frame = sys._getframe(1)

    for expression in expressions:
        val = eval(expression, frame.f_globals, frame.f_locals)
        display_var(expression, val, add_space=add_space)
    print()

def log_var(*expressions):
    """logs variables with names in expressions"""

    frame = sys._getframe(1)

    for expression in expressions:
        assert isinstance(expression,str)
        val = eval(expression, frame.f_globals, frame.f_locals)
        logging.debug(f'{expression} = {val}')

def show(locals_dict, *names, add_space = False):
    """Older version of debug

    Always set the first argument to be locals(), and set any remaining arguments to be string versions of any variable names you
    would like to know more about.
    """

    assert isinstance(locals_dict, dict)

    for name in names:
        display_var(name,locals_dict[name])
    print()

def show_all(exclude = set()):
    """Shows all local variables
    """

    frame = sys._getframe(1)
    locals_dict = eval('locals()', frame.f_globals, frame.f_locals)

    print(f'Function name: {inspect.stack()[1].function}\n')
    vars_to_include = set(locals_dict.keys()).difference(exclude)
    show(locals_dict, *list(vars_to_include))

class IndexEqualityError(Exception):
    pass

def assert_Indexes_same(index_0,index_1):
    """Checks to see if pandas Index objects are the same, with helpful error messages"""

    if index_0.equals(index_1):
        pass
    else:
        msg = 'Index objects are not equal!\n'

        set_0 = set(index_0)
        set_1 = set(index_1)

        if set_0 != set_1:
            set_0_not_1 = set_0 - set_1
            set_1_not_0 = set_1 - set_0

            def update_msg(s,label):

                nonlocal msg

                if s:
                    msg += f'Elements unique to {label} Index: {s}\n'
                else:
                    msg += f'The {label} Index does not have unique elements\n'

            update_msg(set_0_not_1,'first')
            update_msg(set_1_not_0,'second')
            raise IndexEqualityError(msg)
        elif list(sorted(index_0)) == list(sorted(index_1)):
            msg += 'Index elements are the same, but out of order. Sort objects to avoid this error'
            raise IndexEqualityError(msg)
        elif index_0.size == index_1.size:
            joined_DataFrame = pd.DataFrame({'First Index': pd.Series(index_0),'Second Index': pd.Series(index_1)})
            joined_DataFrame.index.name = 'Integer loc'
            errors = (joined_DataFrame['First Index'] != joined_DataFrame['Second Index'])
            display(joined_DataFrame[errors])
            msg += f'Element sets of Index objects are the same. Discrepancy points displayed above.'
            raise IndexEqualityError(msg)
        else:
            msg += 'Indexes have the same set of elements, but are different sizes\n'
            msg += f'First Index size is {index_0.size}\n'
            msg += f'Second Index size is {index_1.size}\n'
            msg += f'Common element set is {set(index_0)}'
            raise IndexEqualityError(msg)


def display_nonempty_set(s, name):
    if s:
        print(f'{name} is nonempty: {s}')


def compare_sets(set0, set1, name_0, name_1):
    set0_not_1 = set0 - set1
    set1_not_0 = set1 - set0
    display_nonempty_set(set0_not_1, f'The set of elements in {name_0} and not {name_1}')
    display_nonempty_set(set1_not_0, f'The set of elements in {name_1} and not {name_0}')

    return not (set0_not_1 or set1_not_0)


def compare_dicts(dict0, dict1):
    keys0 = set(dict0.keys())
    keys1 = set(dict1.keys())

    keys_are_same = compare_sets(keys0, keys1, 'keys0', 'keys1')

    if keys_are_same:
        for key in dict0:
            if dict0[key] != dict1[key]:
                print(f'Disagreement on key {key}')
                print(f'dict0: {dict0[key]}')
                print(f'dict1: {dict1[key]}')


def compare_dfs(df_0_orig, df_1_orig, show_dfs='if_false', restrict_cols=True, sorting=False,
                printing=True, rtol=1e-05, atol=1e-08, almost=True, remove_timestamp=True,
                display_hist=False, bins=100):
    """Compares dataframes, shows problematic columns if not equal

    @show_dfs prints out dataframes only when outputs do now match if set to 'if_false'.
    Other options are 'always' and 'never'. The @almost variable determines whether you
    have error tolerance, whose parameters are controlled by @atol and @rtol (see
    https://numpy.org/doc/stable/reference/generated/numpy.allclose.html).

    @display_hist shows the histogram of differences if set True, @almost == True,
    and a field is not close in the two dataframes.
    """

    df_0 = df_0_orig.copy()
    df_1 = df_1_orig.copy()

    show_options = ['if_false', 'always', 'never']

    assert show_dfs in show_options, f'show_dfs = {show_dfs} must be in in {show_options}'

    is_numeric = lambda s: pd.to_numeric(s, errors='coerce').notnull().all()

    if restrict_cols:
        remove_cols = ['TICK_TYPE', 'SYMBOL_NAME']
        df_0.drop(columns=remove_cols, errors='ignore', inplace=True)
        df_1.drop(columns=remove_cols, errors='ignore', inplace=True)

    if remove_timestamp:
        df_0.drop(columns=[TS, T], errors='ignore', inplace=True)
        df_1.drop(columns=[TS, T], errors='ignore', inplace=True)

    df_0_is_numeric = df_0.apply(is_numeric).all()
    df_1_is_numeric = df_1.apply(is_numeric).all()
    dfs_are_numeric = df_0_is_numeric and df_1_is_numeric

    if sorting:
        def sort_df(df):
            index_is_default = default_index(df)
            if not index_is_default:
                df.reset_index(inplace=True)
            df = df.reindex(sorted(df.columns), axis=1)
            # sorting by integer columns first
            types = df.dtypes
            is_int = types == 'int64'
            int_cols = list(types[is_int].index)
            non_int_cols = list(types[~is_int].index)
            sorting_cols = list(sorted(int_cols)) + list(sorted(non_int_cols))
            # debug('int_cols','non_int_cols','sorting_cols')

            df.sort_values(by=sorting_cols, inplace=True)
            df.reset_index(inplace=True, drop=True)
            return df

        df_0 = sort_df(df_0)
        df_1 = sort_df(df_1)


    if df_0.equals(df_1) or (
            almost and dfs_are_numeric and df_0.shape == df_1.shape and np.allclose(df_0, df_1, rtol=rtol, atol=atol)):
        if show_dfs == 'always':
            show(locals(), 'df_0')
        if printing:
            print('Correct output!')
        else:
            return True
    else:
        correct = False
        if set(df_0.columns) != set(df_1.columns):
            print('Columns do not match')
            cols_0 = set(df_0.columns)
            cols_1 = set(df_1.columns)
            cols_0_not_1 = cols_0 - cols_1
            cols_1_not_0 = cols_1 - cols_0
            show(locals(), 'cols_0', 'cols_1', 'cols_1_not_0', 'cols_0_not_1')
        elif not df_0.index.equals(df_1.index):
            print('Indexes do not match')
            index_0 = df_0.index
            index_1 = df_1.index
            show(locals(), 'index_0', 'index_1')
        else:
            correct = True
            for col in df_0.columns:
                equals = df_0[col].equals(df_1[col])
                if almost:
                    cols_are_numeric = is_numeric(df_0[col]) and is_numeric(df_1[col])
                    numerical_close = (cols_are_numeric and np.allclose(df_0[col], df_1[col], rtol=rtol, atol=atol))
                    equals = equals or numerical_close
                if not equals:
                    correct = False
                    print(f'{col} column does not match')

                    if show_dfs in ['always', 'if_false']:
                        where = df_0[col] != df_1[col]
                        discrepancy_0 = df_0[where]
                        discrepancy_1 = df_1[where]
                        show(locals(), 'discrepancy_0', 'discrepancy_1')
                        if almost and cols_are_numeric:
                            error_series = (df_0[col] - df_1[col]).abs().reset_index(drop = True)
                            mean_error = error_series.mean()
                            median_error = error_series.median()
                            error_tolerance = (atol + rtol * df_1[col].abs()).reset_index(drop = True)
                            breaking_index = np.argmin(error_tolerance - error_series)
                            worst_val_0 = df_0[col].iloc[breaking_index]
                            worst_val_1 = df_1[col].iloc[breaking_index]
                            error_val = error_series.iloc[breaking_index]
                            absolute_tolerance = atol
                            relative_tolerance = rtol * abs(worst_val_1)
                            total_tolerance = error_tolerance[breaking_index]
                            tolerance_broken_by = error_val - total_tolerance
                            show(locals(), 'worst_val_0', 'worst_val_1', 'error_val', 'absolute_tolerance',
                                 'relative_tolerance', 'total_tolerance', 'tolerance_broken_by', 'breaking_index',
                                 'mean_error', 'median_error')
                            if display_hist:
                                plt.clf()
                                plt.hist(df_0[col] - df_1[col], bins=bins)
                                plt.title('0-1 Differences')
                                plt.show()
            show_dfs = 'never'
        if show_dfs in ['always', 'if_false']:
            show(locals(), 'df_0', 'df_1')
        if correct:
            if printing:
                print('Correct Output!')
            else:
                return True
        else:
            if printing:
                print('Incorrect Output')
            else:
                return False


def compare_df_list(df_list,name_list,*args,**kwargs):
    """Compares all the dataframe in df_list via compare_dfs"""

    if len(df_list) == 0:
        print('No dataframes to compare')
        return
    if len(df_list) == 1:
        print('Only one dataframe to compare')
        return

    all_correct = True
    for i, (result_0, result_1) in enumerate(zip(df_list[:-1], df_list[1:])):
        if not compare_dfs(result_0, result_1, *args, printing = False, **kwargs):
            all_correct = False
            print(f'Outputs on {name_list[i]} and {name_list[i + 1]} do not match')
    if all_correct:
        print('All dataframes match!')


def get_unique_tups(df,grouping):
    """Gets a list of all unique tuples in the grouping,tup_name columns in df

    Adds TUP column to df"""

    get_tuple = lambda row: tuple(row[eq] for eq in grouping)

    df[TUP] = df.apply(get_tuple, axis=1)

    return set(sorted(df[TUP].unique())), df


def compare_dfs_by_group(df_dict, display_min, display_max, grouping, show_dfs='if_false'):
    '''Compares dataframes in df_dict side-by-side on their groups with keys in grouping'''

    unique_tups = set()
    for name, df in df_dict.items():
        df, tups = get_unique_tups(df,grouping)
        df_dict[name] = df
        unique_tups = unique_tups.union(tups)

    unique_tups = sorted(unique_tups)

    for i, tup in enumerate(unique_tups):
        if display_min <= i < display_max:
            show(locals(), 'tup')
            df_tuples = list(df_dict.items())
            for (name_0, df_0), (name_1, df_1) in zip(df_tuples[:-1], df_tuples[1:]):
                compare_dfs(df_0[df_0[TUP] == tup], df_1[df_1[TUP] == tup], sorting=True, show_dfs=show_dfs)


def check_monotonicity(series, atol=1e-16):
    """Returns monotonicity and where monotonicity in @series fails"""

    series = series + np.arange(series.size) * atol

    if series.is_monotonic:
        return True, None
    else:
        diffs = series.iloc[1:].reset_index(drop=True) - series.iloc[:-1].reset_index(drop=True)
        errors = (diffs < 0).to_numpy()
        error_seqs = np.concatenate((np.array([False]), errors,)) | np.concatenate((errors, np.array([False]),))
        return False, error_seqs


def check_monotonicity_df(df, cols, atol=1e-16):
    """Checks monotonicities for all columns in @cols for @df"""

    correct = True

    for col in cols:
        is_monotonic, error_seqs = check_monotonicity(df[col], atol=atol)
        if not is_monotonic:
            print(f'Monotonicity error on column {col}:')
            display(df[error_seqs])
            correct = False

    return correct