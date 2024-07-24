import copy
import hashlib
import inspect
import numpy as np
import os

from datetime import timedelta
from flufl.lock import Lock, TimeOutError
from jsonc_parser.parser import JsoncParser
from time import sleep

from src.utils.composer import compose, propagate
from src.utils.logger import GLogger


class Context(object):
    __create_key = object()
    __global = None

    def __init__(self, create_key, config_file):
        ###################################################
        self.factories = {
            "datasets": None,
            "embedders": None,
            "oracles": None,
            "explainers": None,
            "metrics": None,
            "plotters": None
        }
        self.lock_release_tout : None
        self.lock_timeout = None
        ###################################################
        assert(create_key == Context.__create_key), \
            "Context objects must be created using Context.get_context"
              
        # Check that the path to the config file exists
        if not os.path.exists(config_file):
            raise ValueError(f'''The provided config file does not exist. PATH: {config_file}''')

        self.config_file = config_file
        # Read the config dictionary inside the config path with the composer
        '''with open(self.config_file, 'r') as config_reader:
            self.conf = propagate(compose(jsonpickle.decode(config_reader.read()))) #First read config, then apply the compose and finally it propagate some homogeneous config params
        '''
        self.conf = propagate(compose(JsoncParser.parse_file(self.config_file)))

        self._scope = self.conf['experiment'].get('scope','default_scope')
        self.conf['experiment']['parameters']=self.conf['experiment'].get('parameters',{})
        self.conf['experiment']['parameters']['lock_release_tout']=self.conf['experiment']['parameters'].get('lock_release_tout',24*5) #Expressed in hours
        self.lock_release_tout = self.conf['experiment']['parameters']['lock_release_tout']
        self.conf['experiment']['parameters']['lock_timeout']=self.conf['experiment']['parameters'].get('lock_timeout',5) #Expressed in seconds
        self.lock_timeout = self.conf['experiment']['parameters']['lock_timeout']

        self.raw_conf = copy.deepcopy(self.conf) #TODO: I think it is will be useless remove that in the future.

        self.__create_storages()
        
    @property
    def logger(self):
        GLogger._path = os.path.join(self.log_store_path, self._scope)
        return GLogger.getLogger()
        
    @classmethod
    def get_context(cls,config_file=None):
        if(Context.__global == None):
            if config_file is None:
                raise ValueError(f'''The configuration file must be passed to the method as PATH the first time. Now you did not pass as parameter.''')
            Context.__global = Context(Context.__create_key,config_file)            
        return Context.__global
    
    @classmethod
    def get_by_pkvs(cls, conf, parent, key, value, son):
        for obj in conf[parent]:
            if(obj[key] == value):
                return obj[son]
    
    def get_path(self, obj):
        fullname = self.get_fullname(obj).split('.')
        qualifier = fullname[1] + '_store_path'
        store_dir = 'data'if not self._get_store_path(qualifier) else self._get_store_path(qualifier)

        # change this path when the dataset factories are finished
        if 'dataset' in obj.__dict__.keys():
            directory = os.path.join(store_dir, str(obj.dataset))
        else:
            directory = store_dir
        lock_file = f'{directory}.lck'
        lock_lifetime = timedelta(hours=self.lock_release_tout)
        lock_timeout = timedelta(seconds=self.lock_timeout)
        while True:
            try:
                with Lock(lock_file, lock_lifetime, default_timeout=lock_timeout):
                    os.makedirs(directory, exist_ok=True)
                    return os.path.join(directory, obj.name)
            except TimeOutError:
                sleep(1)
                continue

    @classmethod
    def get_fullname(cls, o):
        klass = o.__class__
        module = klass.__module__
        if module == 'builtins':
            return klass.__qualname__ # avoid outputs like 'builtins.str'
        return module + '.' + klass.__qualname__
        

    @classmethod
    def get_class_fullname(cls, kls):
        """
        This method is designed to get the full name of a class (kls) from the class itself and not from an instance of it
        """
        module = kls.__module__
        if module == 'builtins':
            return kls.__qualname__ # avoid outputs like 'builtins.str'
        return module + '.' + kls.__qualname__


    def get_name(self, inst, dictionary=None, alias=None):        
        cls = inst.__class__.__name__ if not alias else alias
        dictionary= clean_cfg(inst.local_config) if dictionary is None else clean_cfg(dictionary)
        md5_hash = hashlib.md5()       

        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f'{parent_key}{sep}{k}' if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        if dictionary is not None:
            payload = f'{cls}_' + '_'.join([f'{key}={value}' for key, value in flatten_dict(dictionary).items()])
            md5_hash.update(payload.encode('utf-8'))           
            return cls+'-'+md5_hash.hexdigest()
        else:
            return cls
        
    def _get_store_path(self,value):
        return Context.get_by_pkvs(self.conf, "store_paths", "name",value,"address")
            
    def __create_storages(self):
        for store_path in self.conf['store_paths']:
            os.makedirs(store_path['address'], exist_ok=True)

    @property
    def dataset_store_path(self):
        return self._get_store_path(inspect.stack()[0][3])
    
    @property
    def embedder_store_path(self):
        return self._get_store_path(inspect.stack()[0][3])
    
    @property
    def oracle_store_path(self):
        return self._get_store_path(inspect.stack()[0][3])
    
    @property
    def explainer_store_path(self):
        return self._get_store_path(inspect.stack()[0][3])
    
    @property
    def output_store_path(self):
        return self._get_store_path(inspect.stack()[0][3])
    
    @property
    def log_store_path(self):
        return self._get_store_path(inspect.stack()[0][3])
    
    @property
    def working_store_path(self):
        dflt = './data/working/'
        usr = self._get_store_path(inspect.stack()[0][3])
        if usr == None:
            os.makedirs(dflt, exist_ok=True)
            return dflt
        return usr
    
#context = Context.get_context()
#context = Context.get_context("config/test/temp.json")
#print(context)
#print(context._scope)
#print(context.dataset_store_path)

def clean_cfg(cfg):
    if isinstance(cfg,dict):
        new_cfg = {}
        for k in cfg.keys():
            if hasattr(cfg[k],"local_config"):#k == 'oracle' or k == 'dataset':
                new_cfg[k] = clean_cfg(cfg[k].local_config)
            elif isinstance(cfg[k], (list,dict, np.ndarray)):
                new_cfg[k] = clean_cfg(cfg[k])
            else:
                new_cfg[k] = cfg[k]
    elif isinstance(cfg, (list, np.ndarray)):
        new_cfg = []
        for k in cfg:
            new_cfg.append(clean_cfg(k))
    else:
        new_cfg = cfg

    return new_cfg

            