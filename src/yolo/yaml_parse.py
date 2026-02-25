from yolo_config import *
import dataset
import yaml

def yaml_error(strategy, key_prefix, key, v, expected_type):
    full_key = f"{key_prefix}.{key}" if key_prefix != "" else key

    if expected_type == int:
        print_error(f"{strategy}.yaml: '{full_key}': expected [type=int,value>=0], got [type={expected_type.__name__},value={v}]")
    elif expected_type == float:
        print_error(f"{strategy}.yaml: '{full_key}': expected [type=float,value>=0.0], got [type={expected_type.__name__},value={v}]")
    elif expected_type == str:
        print_error(f"{strategy}.yaml: '{full_key}': expected [type=str], got [type={expected_type.__name__},value={v}]")
    
    exit(1)

def yaml_get(dest, src, key, default_value):
    v = src.get(key)
    
    if key == "from":
        print(f"yaml_get yaml[{key}]='{v}', conf[{key}]='{dest.get(key)}', default_value='{default_value}'")

    T = type(default_value)

    if v is None:
        if dest.get(key) is None:
            dest[key] = default_value
        return False

    dest[key] = v

    if key == "from":
        print(f"conf[{key}] = {dest[key]}")

    if T == int:
        return v is not None and (not isinstance(v,int) or v < 0)
    elif T == float:
        return v is not None and (not isinstance(v,float) or v < 0.0)
    elif T == str:
        return v is not None and not isinstance(v,str)

def yaml_get_indices_arr(dest, src, key):
    v = src.get(key)

    if v is None and dest.get(key) is None:
        dest[key] = None
        return False

    dest[key] = v

    return v is not None and (not isinstance(v,list) or any(not isinstance(x,int) or x < 0 or x > NUM_IMAGES for x in v))

def yaml_get_e(strategy, dest, src, key, key_prefix, default_value):
    if yaml_get(dest, src, key, default_value):
        yaml_error(strategy, key_prefix, key, dest[key], type(default_value))

def yaml_convert_dataset_query(strategy, yaml):
    o = {}
    for entry in yaml:
        k,v = entry
        code = getattr(dataset, k)
        if code is None:
            print_error(f"{strategy}.yaml: the code '{code}' is not a valid global constant from the module dataset")
            exit(1)
        o[code] = v
    return o

def yaml_get_dataset_query(strategy, yaml):
    gt = yaml_convert_dataset_query(strategy, yaml["gt"]) if "gt" in yaml else None
    lt = yaml_convert_dataset_query(strategy, yaml["lt"]) if "lt" in yaml else None
    return gt, lt

def yaml_parse_args(strategy, key_prefix, yaml, args={}):
    yaml_get_e(strategy, args, yaml, "batch", key_prefix, 16)
    yaml_get_e(strategy, args, yaml, "epochs", key_prefix, 100)

    yaml_get_e(strategy, args, yaml, "freeze", key_prefix, 0)
    yaml_get_e(strategy, args, yaml, "copy_paste", key_prefix, 0.0)
    yaml_get_e(strategy, args, yaml, "mosaic", key_prefix, 1.0)
    yaml_get_e(strategy, args, yaml, "mixup", key_prefix, 0.0)

    yaml_get_e(strategy, args, yaml, "lr0", key_prefix, 0.01)
    yaml_get_e(strategy, args, yaml, "lrf", key_prefix, 0.01)
    yaml_get_e(strategy, args, yaml, "momentum", key_prefix, 0.937)
    yaml_get_e(strategy, args, yaml, "weight_decay", key_prefix, 0.0005)
    yaml_get_e(strategy, args, yaml, "warmup_epochs", key_prefix, 3.0)

def yaml_parse_index_selection(strategy, key_prefix, yaml, ind={}, args={}):
    yaml_parse_args(strategy, key_prefix, yaml, args)

    if "train" in yaml or "val" in yaml:
        yaml_get_indices_arr(ind, yaml, "train")
        yaml_get_indices_arr(ind, yaml, "val")

        if ind["train"] is None or ind["val"] is None:
            print_error(f"{strategy}.yaml: 'train' and 'val': if defined, must be both defined")
            exit(1)

        return
    
    if "indices" in yaml:
        yaml_get_indices_arr(ind, yaml, "indices")
        return

    yaml_get_e(strategy, ind, yaml, "from", key_prefix, "")

# raimondo schettini davide mazzini, comparison of SIFT based - google scholar

def yaml_parse_strategy(config, yaml):
    strategy = config["strategy"]
    print(f"strategy: '{strategy}'")

    instant = config["instant"]

    config["args"] = {
        "batch": 16,
        "epochs": 100,

        # Training Hyperparameters
        "freeze": 0,
        "copy_paste": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,

        # Optimization
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0
    }
    config["ind"] = {
        "split_ratio": YOLO_TRAIN_TEST_SPLIT_RATIO,
        "indices": None,
        "train": None,
        "val": None,
        "from": None
    }

    yaml_get_e(strategy, config["ind"], yaml, "split_ratio", "", YOLO_TRAIN_TEST_SPLIT_RATIO)
    
    yaml_parse_index_selection(strategy, "", yaml, config["ind"], config["args"])

    if "instants" not in yaml:
        if instant > 0:
            print_warning(f"{strategy}.yaml: the current instant is {instant} but 'instants' is missing: no indices restriction applied")
        return config

    yaml_parse_index_selection(strategy, "instants", yaml["instants"], config["ind"], config["args"])

    if "index_selection" not in yaml["instants"]:
        print_warning(f"{strategy}.yaml: 'instants' is defined, but 'instants.index_selection' is missing: no indices restriction applied")
        return config

    index_selection = yaml["instants"]["index_selection"]
    ninstants = len(index_selection)

    if ninstants < instant:
        print_warning(f"{strategy}.yaml: len('instants.index_selection')={ninstants}: current instant={instant}, no indices restriction applied")
        return config

    yaml_parse_index_selection(strategy, f"instants.index_selection[{instant}]", index_selection[instant], config["ind"], config["args"])
    
    if config["args"]["warmup_epochs"] > config["args"]["epochs"]:
        print_error(f"{strategy}.yaml: 'warmup_epochs': [value={config['args']['warmup_epochs']}] exceeds the value of 'epochs' [value={config['args']['epochs']}]")
        exit(1)

    gt,lt = yaml_get_dataset_query(strategy, yaml)

    if gt is not None or lt is not None:
        _from = _from = config["ind"]["from"]

        if _from not in ["dataset","detailed_dataset"]:
            print_error(f"{strategy}.yaml: 'from': must be defined for 'gt' and 'lt' queries, accepted values ['dataset','detailed_dataset']")
            exit(1)

        ind["indices"] = dataset.get_dataset_manifest().query(gt,lt) \
                            if _from == "dataset" \
                            else dataset.get_detailed_dataset_manifest().query(gt,lt)

    return config

def yaml_get_run_details(run_name, yaml):
    det = {}

    yaml_get_e(strategy, det, yaml, "strategy", "", "")
    if det["strategy"] == "":
        print_error(f"{run_name}/data.yaml: 'strategy' is missing")
        exit(1)
    if det["strategy"] not in YOLO_AVAILABLE_TRAIN_STRATEGIES:
        print_error(f"{run_name}/data.yaml: 'strategy' is missing")
        exit(1)

    yaml_get_e(strategy, det, yaml, "instant", "", -1)
    if det["instant"] == -1:
        print_error(f"{run_name}/data.yaml: 'instant' is missing")
        exit(1)

    yaml_get_e(strategy, det, yaml, "labels", "", "")
    if det["labels"] == "":
        print_error(f"{run_name}/data.yaml: 'labels': property is missing")
        exit(1)
    if det["labels"] not in YOLO_AVAILABLE_TRAIN_LABELS:
        print_error(f"{run_name}/data.yaml: 'strategy': 'labels/{det['labels']}' does not exist")
        exit(1)

    return det

def yaml_get_strategy_details(strategy, yaml):
    det = {
        "strategy": strategy,
        "instant": 0
    }

    yaml_get_e(strategy, det, yaml, "labels", "", "")
    if det["labels"] == "":
        print_error(f"{strategy}.yaml: 'labels': property is missing")
        exit(1)
    if det["labels"] not in YOLO_AVAILABLE_TRAIN_LABELS:
        print_error(f"{strategy}.yaml: 'labels': 'labels/{det['labels']}' does not exist")
        exit(1)

    return det