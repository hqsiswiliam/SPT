import yaml
from dotmap import DotMap


def extend_dict(extend_me, extend_by):
    if isinstance(extend_me, dict):
        for k, v in extend_by.iteritems():
            if k in extend_me:
                extend_dict(extend_me[k], v)
            else:
                extend_me[k] = v
    else:
        if isinstance(extend_me, list):
            extend_list(extend_me, extend_by)
        else:
            if extend_by is not None:
                extend_me += extend_by


def extend_list(extend_me, extend_by):
    missing = []
    for item1 in extend_me:
        if not isinstance(item1, dict):
            continue

        for item2 in extend_by:
            if not isinstance(item2, dict) or item2 in missing:
                continue
            extend_dict(item1, item2)


def extend_compatibility_for_gated_transformer(configuration):
    dict_config = configuration.toDict()
    return configuration


def get_config(path):
    with open(path, 'r') as file:
        configuration = yaml.load(file, Loader=yaml.FullLoader)
    with open('config/default.yml', 'r') as file:
        base_configuration = yaml.load(file, Loader=yaml.FullLoader)
    configuration = DotMap(configuration)
    base_configuration = DotMap(base_configuration)
    extend_dict(configuration, base_configuration)
    configuration = extend_compatibility_for_gated_transformer(configuration)
    return configuration


if __name__ == '__main__':
    config = get_config('config/bert-base.yml')
