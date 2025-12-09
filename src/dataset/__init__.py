import os
from helpers import parse_json_file

id2txt = {
    "br":"broccoli",
    "cf":"cauliflower",
    "ps":"peas",
    "ct":"carrot",
    #
    "eg":"egg",
    #
    "st":"steak",
    "sn":"salmon",
    "sg":"sausage",
    #
    "ra":"red apple",
    "ga":"green apple",
    "og":"orange",
    "bn":"banana",
    "pc":"peach",
    #
    "bd":"bread",
    #
    "cr":"croissant",
    "ck":"cake",
    #
    "cc":"crackers",
    "ml":"milka",
    "kg":"kellogs",
    "bt":"breadsticks",
    #
    "lc":"licorice",
    #
    "mf":"metal fork",
    "bf":"black fork",
    "tf":"transparent fork",
    #
    "ms":"metal spoon",
    "bs":"black spoon",
    "ts":"teaspoon",
    #
    "bk":"black knife",
    "rk":"red knife",
    #
    "wn":"white napkin",
    "xn":"bordeaux napkin",
    "rn":"red napkin",
    #
    "wt":"water",
    "ga":"gatorade",
    "hv":"estathe verde",
    #
    "cl":"coca-cola",
    "cz":"coca-cola zero",
    "mr":"monster",
    "sp":"sprite",
    #
    "tb":"thermal bottle",
    #
    "pj":"pear juice",
    "j1":"peach juice 1",
    "j2":"peach juice 2",
    #
    "nn":"tennent's",
    #
    "yt":"yogurt",
    #
    "wc":"white cup",
    "rc":"red cup",
    "tc":"transparent cup",
    #
    "mg":"mug",
    #
    "oc":"oil cruet",
    "ss":"tissues",
    "ks":"keys",
    "ll":"wallet",
    "sh":"salt",
    "ph":"telephone",
    "eb":"ear buds",
    "lg":"lighter",
    "lb":"lip balm",
    #
    "r":"red tray",
    "g":"gray tray",
    "p":"parchment paper"
}

CL_FOOD = 0
CL_SNACK = 1
CL_SWEETS = 2
CL_FORK = 3
CL_SPOON = 4
CL_KNIFE = 5
CL_NAPKIN = 6
CL_PLASTIC_BOTTLE = 7
CL_ALLUMINUM_CAN = 8
CL_THERMAL_BOTTLE = 9
CL_TETRAPAK = 10
CL_GLASS_BOTTLE = 11
CL_YOGURT = 12
CL_PLASTIC_CUP = 13
CL_MUG = 14
CL_OIL_CRUET = 15
CL_TISSUES = 16
CL_KEYS = 17
CL_WALLET = 18
CL_SALT_SHAKER = 19
CL_PHONE = 20
CL_EAR_BUDS = 21
CL_LIGHTER = 22
CL_LIP_BALM = 23

classes = range(CL_LIP_BALM+1)

id2cl = {
    "br":CL_FOOD,
    "cf":CL_FOOD,
    "ps":CL_FOOD,
    "ct":CL_FOOD,
    "eg":CL_FOOD,
    "st":CL_FOOD,
    "sn":CL_FOOD,
    "sg":CL_FOOD,
    "ra":CL_FOOD,
    "ga":CL_FOOD,
    "og":CL_FOOD,
    "bn":CL_FOOD,
    "pc":CL_FOOD,
    "bd":CL_FOOD,
    "cr":CL_FOOD,
    "ck":CL_FOOD,
    #
    "cc":CL_SNACK,
    "ml":CL_SNACK,
    "kg":CL_SNACK,
    "bt":CL_SNACK,
    #
    "lc":CL_SWEETS,
    #
    "mf":CL_FORK,
    "bf":CL_FORK,
    "tf":CL_FORK,
    #
    "ms":CL_SPOON,
    "bs":CL_SPOON,
    "ts":CL_SPOON,
    #
    "bk":CL_KNIFE,
    "rk":CL_KNIFE,
    #
    "wn":CL_NAPKIN,
    "xn":CL_NAPKIN,
    "rn":CL_NAPKIN,
    #
    "wt":CL_PLASTIC_BOTTLE,
    "ga":CL_PLASTIC_BOTTLE,
    "hv":CL_PLASTIC_BOTTLE,
    #
    "cl":CL_ALLUMINUM_CAN,
    "cz":CL_ALLUMINUM_CAN,
    "mr":CL_ALLUMINUM_CAN,
    "sp":CL_ALLUMINUM_CAN,
    #
    "tb":CL_THERMAL_BOTTLE,
    #
    "pj":CL_TETRAPAK,
    "j1":CL_TETRAPAK,
    "j2":CL_TETRAPAK,
    #
    "nn":CL_GLASS_BOTTLE,
    #
    "yt":CL_YOGURT,
    #
    "wc":CL_PLASTIC_CUP,
    "rc":CL_PLASTIC_CUP,
    "tc":CL_PLASTIC_CUP,
    #
    "mg":CL_MUG,
    #
    "oc":CL_OIL_CRUET,
    "ss":CL_TISSUES,
    "ks":CL_KEYS,
    "ll":CL_WALLET,
    "sh":CL_SALT_SHAKER,
    "ph":CL_PHONE,
    "eb":CL_EAR_BUDS,
    "lg":CL_LIGHTER,
    "lb":CL_LIP_BALM,
    #
    "r":None,
    "g":None,
    "p":None
}

ids = sorted(id2txt)

current_dir = os.path.dirname(__file__)
json_path = os.path.join(current_dir, "dataset.json")

obj = parse_json_file(json_path)

# None on sequence gaps
dataset = [obj.get(str(i)) for i in range(len(obj))]

__all__ = [
    'id2txt',
    'id2cl',
    'ids',
    'classes'
    'dataset',
    'CL_FOOD',
    'CL_SNACK',
    'CL_SWEETS',
    'CL_FORK',
    'CL_SPOON',
    'CL_KNIFE',
    'CL_NAPKIN',
    'CL_PLASTIC_BOTTLE',
    'CL_ALLUMINUM_CAN',
    'CL_THERMAL_BOTTLE',
    'CL_TETRAPAK',
    'CL_GLASS_BOTTLE',
    'CL_YOGURT',
    'CL_PLASTIC_CUP',
    'CL_MUG',
    'CL_OIL_CRUET',
    'CL_TISSUES',
    'CL_KEYS',
    'CL_WALLET',
    'CL_SALT_SHAKER',
    'CL_PHONE',
    'CL_EAR_BUDS',
    'CL_LIGHTER',
    'CL_LIP_BALM'
]