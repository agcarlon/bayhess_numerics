from logreg import load_data
from sgd import sgd
from sgd_mice import sgd_mice
from sgd_mice_bay import sgd_mice_bay
from svrg import svrg
from svrg_bay import svrg_bay
from sarah import sarah
from sarah_bay import sarah_bay

from make_plots import make_plots
from plot_newton_cg import plot_newton_cg
from os.path import exists

args = {'dataset': 'HIGGS',
        'data_size': 11000000,
        'n_features': 28,
        'reg_param': 1e-5,
        'seed': 1,
        'epochs': 15,
        'clean_data': False,
        'label_format': '0 1'}
file_name = f"{ args['dataset'] }/{ args['dataset'] }"

if not exists(file_name):
    print("Warning! Downloading a large file, 1.4GB.")
    from urllib.request import urlretrieve
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    file_name = f"{ args['dataset'] }/{ args['dataset'] }"
    urlretrieve(url, file_name+".csv.gz")
    import gzip
    import shutil
    with gzip.open(file_name+".csv.gz", 'rb') as f_in:
        with open(file_name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

mice_params = {'eps': 0.8,
           'min_batch': 10,
           'drop_param': 2,
           'max_hierarchy_size': 1000,
           'adpt': True}

bay_params = {'penal': 1e-2,
              'reg_param': 1e-2,
              'yk_err_tol': 0.3,
              'pairs_to_use': 100}

bay_update_when = 1

svrg_step = 0.1
svrg_batch_size = 5

# sgd(**args)
# sgd_mice(**args, mice_params=mice_params)
sgd_mice_bay(**args, update_when=bay_update_when, mice_params=mice_params,
             bay_params=bay_params)

# svrg(**args, step_param=svrg_step, batch_size=svrg_batch_size)
# svrg_bay(**args, step_param=svrg_step, batch_size=svrg_batch_size,
#          update_when=bay_update_when, bay_params=bay_params)
# sarah(**args, step_param=svrg_step, batch_size=svrg_batch_size)
# sarah_bay(**args, step_param=svrg_step, batch_size=svrg_batch_size,
#           update_when=bay_update_when, bay_params=bay_params)

make_plots('HIGGS', 11000000)
plot_newton_cg('HIGGS')
plot_newton_cg('HIGGS', 'svrg_bay')