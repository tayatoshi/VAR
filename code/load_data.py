import pandas as pd

def get_data():
    data = pd.read_csv('../input/dow.csv',index_col = 0, parse_dates = True).dropna(axis=0)
    data = data.pct_change().dropna(axis=0)
    print('data-shape : ({},{})'.format(data.shape[0],data.shape[1]))
    print('')
    return data

if __name__ == '__main__':
    print(get_data().tail())
