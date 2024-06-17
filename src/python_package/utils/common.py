import re
from datetime import datetime, timedelta
from uuid import uuid4


def serialize_datetime(dictionary):
    for attr, value in dictionary.items():
        if isinstance(value, datetime):
            dictionary[attr] = value.strftime("%Y-%m-%dT%H:%M:%SZ")
        if isinstance(value, timedelta):
            dictionary[attr] = value.seconds / 3600
    return dictionary


def camel_to_underscore(name):
    camel_pat = re.compile(r"([A-Z])")
    return camel_pat.sub(lambda x: "_" + x.group(1).lower(), name)


def underscore_to_camel(name):
    under_pat = re.compile(r"_([a-z])")
    return under_pat.sub(lambda x: x.group(1).upper(), name)


def case_type_converter(dictionary, convert):
    new = {}
    for k, v in dictionary.items():
        new_v = v
        if isinstance(v, dict):
            new_v = case_type_converter(v, convert)
        elif isinstance(v, list):
            new_v = list()
            for x in v:
                if isinstance(x, dict):
                    new_v.append(case_type_converter(x, convert))
                if isinstance(x, str):
                    # new_v.append(convert(x))
                    new_v.append(x)
        new[convert(k)] = new_v
    return new


def get_uuid():
    return str(uuid4())[:8]
