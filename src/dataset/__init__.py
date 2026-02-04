from helpers import parse_json_file
from .log.dataset_manifest import DatasetManifest
import pandas as pd

var2txt = {"br":"broccoli","cf":"cauliflower","ps":"peas","ct":"carrot","eg":"egg","st":"steak","sn":"salmon","sg":"sausage","ra":"red apple","ga":"green apple","og":"orange","bn":"banana","pc":"peach","bd":"bread","cr":"croissant","ck":"cake","cc":"crackers","ml":"milka","kg":"kellogs","lc":"licorice","mf":"metal fork","bf":"black fork","tf":"transparent fork","ms":"metal spoon","bs":"black spoon","ts":"teaspoon","bk":"black knife","rk":"red knife","wn":"white napkin","xn":"bordeaux napkin","rn":"red napkin","wt":"water","ga":"gatorade","hv":"estathe verde","cl":"coca-cola","cz":"coca-cola zero","mr":"monster","sp":"sprite","rz":"red thermal bottle","lz":"blue thermal bottle","bz":"brown thermal bottle","az":"black thermal bottle","dz":"dark thermal bottle","gz":"green thermal bottle","pj":"pear juice","j1":"peach juice 1","j2":"peach juice 2","nn":"tennent's","yt":"yogurt","wc":"white cup","rc":"red cup","tc":"transparent cup","sc":"small cup","mg":"mug","oc":"oil cruet","ss":"tissues","ks":"keys","ll":"wallet","sh":"salt","ph":"phone","eb":"ear buds","lg":"lighter","lb":"lip balm","r":"red tray","g":"gray tray","p":"parchment paper","ci":"cigarettes","dc":"disposable cutlery","cb":"crushed plastic bottle","ch":"chinotto","tp":"the pesca","fm":"foco maracuya"}
variables = sorted(var2txt, key=lambda k: var2txt[k])

class2txt = {"fd":"food","sn":"snack","sw":"sweets","fo":"fork","sp":"spoon","kn":"knife","na":"napkin","pb":"plastic bottle","ac":"alluminum can","tb":"thermal bottle","tt":"tetrapak","gb":"glass bottle","yo":"yogurt","pc":"plastic cup","mu":"mug","oc":"oil cruet","ti":"tissues","ke":"keys","wa":"wallet","ss":"salt shaker","ph":"phone","ea":"earbuds","li":"lighter","lb":"lip balm","ci":"cigarettes","dc":"disposable cutlery","cb":"crushed plastic bottle"}
classes = sorted(class2txt, key=lambda k: class2txt[k])

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
CL_EARBUDS = 21
CL_LIGHTER = 22
CL_LIP_BALM = 23
CL_CIGARETTES = 24
CL_DISPOSABLE_CUTLERY = 25
CL_CRUSHED_BOTTLE = 26

DD_BROCCOLI="br"
DD_CAULIFLOWER="cf"
DD_PEAS="ps"
DD_CARROT="ct"
DD_EGG="eg"
DD_STEAK="st"
DD_SALMON="sn"
DD_SAUSAGE="sg"
DD_RED_APPLE="ra"
DD_GREEN_APPLE="ga"
DD_ORANGE="og"
DD_BANANA="bn"
DD_PEACH="pc"
DD_BREAD="bd"
DD_CROISSANT="cr"
DD_CAKE="ck"
DD_CRACKERS="cc"
DD_MILKA="ml"
DD_KELLOGS="kg"
DD_LICORICE="lc"
DD_METAL_FORK="mf"
DD_BLACK_FORK="bf"
DD_TRANSPARENT_FORK="tf"
DD_METAL_SPOON="ms"
DD_BLACK_SPOON="bs"
DD_TEASPOON="ts"
DD_BLACK_KNIFE="bk"
DD_RED_KNIFE="rk"
DD_WHITE_NAPKIN="wn"
DD_BORDEAUX_NAPKIN="xn"
DD_RED_NAPKIN="rn"
DD_WATER="wt"
DD_GATORADE="ga"
DD_ESTATHE_VERDE="hv"
DD_COCA_COLA="cl"
DD_COCA_COLA_ZERO="cz"
DD_MONSTER="mr"
DD_SPRITE="sp"
DD_RED_THERMAL_BOTTLE="rz"
DD_BLUE_THERMAL_BOTTLE="lz"
DD_BROWN_THERMAL_BOTTLE="bz"
DD_BLACK_THERMAL_BOTTLE="az"
DD_DARK_THERMAL_BOTTLE="dz"
DD_GREEN_THERMAL_BOTTLE="gz"
DD_PEAR_JUICE="pj"
DD_PEACH_JUICE_1="j1"
DD_PEACH_JUICE_2="j2"
DD_TENNENTS="nn"
DD_YOGURT="yt"
DD_WHITE_CUP="wc"
DD_RED_CUP="rc"
DD_TRANSPARENT_CUP="tc"
DD_SMALL_CUP="sc"
DD_MUG="mg"
DD_OIL_CRUET="oc"
DD_TISSUES="ss"
DD_KEYS="ks"
DD_WALLET="ll"
DD_SALT_SHAKER="sh"
DD_PHONE="ph"
DD_EAR_BUDS="eb"
DD_LIGHTER="lg"
DD_LIP_BALM="lb"
DD_CIGARETTES="ci"
DD_DISPOSABLE_CUTLERY="dc"
DD_CRUSHED_PLASTIC_BOTTLE="cb"
DD_CHINOTTO="ch"
DD_THE_PESCA="tp"
DD_FOCO_MARACUYA="fm"
DD_RED_TRAY="r"
DD_GRAY_TRAY="g"
DD_PARCHMENT_PAPER="p"

D_FOOD="fd"
D_SNACK="sn"
D_SWEETS="sw"
D_FORK="fo"
D_SPOON="sp"
D_KNIFE="kn"
D_NAPKIN="na"
D_PLASTIC_BOTTLE="pb"
D_ALLUMINUM_CAN="ac"
D_THERMAL_BOTTLE="tb"
D_TETRAPAK="tt"
D_GLASS_BOTTLE="gb"
D_YOGURT="yo"
D_PLASTIC_CUP="pc"
D_MUG="mu"
D_OIL_CRUET="oc"
D_TISSUES="ti"
D_KEYS="ke"
D_WALLET="wa"
D_SALT_SHAKER="ss"
D_PHONE="ph"
D_EAR_BUDS="ea"
D_LIGHTER="li"
D_LIP_BALM="lb"
D_CIGARETTES="ci"
D_DISPOSABLE_CUTLERY="dc"
D_CRUSHED_PLASTIC_BOTTLE="cb"

class2classid = {"fd":CL_FOOD,"sn":CL_SNACK,"sw":CL_SWEETS,"fo":CL_FORK,"sp":CL_SPOON,"kn":CL_KNIFE,"na":CL_NAPKIN,"pb":CL_PLASTIC_BOTTLE,"ac":CL_ALLUMINUM_CAN,"tb":CL_THERMAL_BOTTLE,"tt":CL_TETRAPAK,"gb":CL_GLASS_BOTTLE,"yo":CL_YOGURT,"pc":CL_PLASTIC_CUP,"mu":CL_MUG,"oc":CL_OIL_CRUET,"ti":CL_TISSUES,"ke":CL_KEYS,"wa":CL_WALLET,"ss":CL_SALT_SHAKER,"ph":CL_PHONE,"ea":CL_EARBUDS,"li":CL_LIGHTER,"lb":CL_LIP_BALM,"ci":CL_CIGARETTES,"dc":CL_DISPOSABLE_CUTLERY,"cb":CL_CRUSHED_BOTTLE}
classid2class = {value: key for key, value in class2classid.items()}

var2classid = {"br":CL_FOOD,"cf":CL_FOOD,"ps":CL_FOOD,"ct":CL_FOOD,"eg":CL_FOOD,"st":CL_FOOD,"sn":CL_FOOD,"sg":CL_FOOD,"ra":CL_FOOD,"ga":CL_FOOD,"og":CL_FOOD,"bn":CL_FOOD,"pc":CL_FOOD,"bd":CL_FOOD,"cr":CL_FOOD,"ck":CL_FOOD,"cc":CL_SNACK,"ml":CL_SNACK,"kg":CL_SNACK,"lc":CL_SWEETS,"mf":CL_FORK,"bf":CL_FORK,"tf":CL_FORK,"ms":CL_SPOON,"bs":CL_SPOON,"ts":CL_SPOON,"bk":CL_KNIFE,"rk":CL_KNIFE,"wn":CL_NAPKIN,"xn":CL_NAPKIN,"rn":CL_NAPKIN,"wt":CL_PLASTIC_BOTTLE,"ga":CL_PLASTIC_BOTTLE,"hv":CL_PLASTIC_BOTTLE,"cl":CL_ALLUMINUM_CAN,"cz":CL_ALLUMINUM_CAN,"mr":CL_ALLUMINUM_CAN,"sp":CL_ALLUMINUM_CAN,"rz":CL_THERMAL_BOTTLE,"lz":CL_THERMAL_BOTTLE,"bz":CL_THERMAL_BOTTLE,"az":CL_THERMAL_BOTTLE,"dz":CL_THERMAL_BOTTLE,"gz":CL_THERMAL_BOTTLE,"pj":CL_TETRAPAK,"j1":CL_TETRAPAK,"j2":CL_TETRAPAK,"nn":CL_GLASS_BOTTLE,"yt":CL_YOGURT,"wc":CL_PLASTIC_CUP,"rc":CL_PLASTIC_CUP,"tc":CL_PLASTIC_CUP,"sc":CL_PLASTIC_CUP,"mg":CL_MUG,"oc":CL_OIL_CRUET,"ss":CL_TISSUES,"ks":CL_KEYS,"ll":CL_WALLET,"sh":CL_SALT_SHAKER,"ph":CL_PHONE,"eb":CL_EARBUDS,"lg":CL_LIGHTER,"lb":CL_LIP_BALM,"r":None,"g":None,"p":None,"ci":CL_CIGARETTES,"dc":CL_DISPOSABLE_CUTLERY,"cb":CL_CRUSHED_BOTTLE,"ch":CL_ALLUMINUM_CAN,"tp":CL_ALLUMINUM_CAN,"fm":CL_ALLUMINUM_CAN}

obj_ann_man = parse_json_file(__file__, "log/annotation_manifest.json")
vec_detailed_man = []

man = []
vec_man = []

for i in obj_ann_man:
    remapped_entry = {}
    full_remapped = {}
    entry = obj_ann_man[i]
    full_entry = {}
    for var in variables:
        clid = var2classid[var]
        # meta descriptors: remapped_entry, g, p
        if clid is None:
            full_entry[var] = entry.get(var,0)
            continue

        cl = classid2class[clid]
        # initialize empty row columns
        if entry.get(var) is None:
            if full_entry.get(var) is None:
                full_entry[var] = 0
            if full_remapped.get(cl) is None:
                full_remapped[cl] = 0
            continue
        else:
            full_entry[var] = entry[var]

        # initialize missing keys
        if remapped_entry.get(cl) is None:
            remapped_entry[cl] = 0
            full_remapped[cl] = 0

        remapped_entry[cl] += entry[var]
        full_remapped[cl] = remapped_entry[cl]

    vec_detailed_man.append(full_entry)
    man.append(remapped_entry)
    vec_man.append(full_remapped)


detailed_man = [obj_ann_man.get(str(i)) for i in range(len(obj_ann_man))]

detailed_dataset_column_ids = variables
dataset_column_ids = classes

dd_df = pd.DataFrame(vec_detailed_man, columns=variables)
detailed_dataset_df = dd_df.T
dd_bin_df = (dd_df > 0).astype(int)
detailed_dataset_bin_df = dd_bin_df.T
detailed_dataset_instance_counts = dd_df.sum(axis=0)
detailed_dataset_occurrence_counts = dd_bin_df.sum(axis=0)
detailed_dataset_ccm = detailed_dataset_df.dot(dd_df)
detailed_dataset_bin_ccm = detailed_dataset_bin_df.dot(dd_bin_df)

d_df = pd.DataFrame(vec_man, columns=classes)
dataset_df = d_df.T
d_bin_df = (d_df > 0).astype(int)
dataset_bin_df = d_bin_df.T
dataset_instance_counts = d_df.sum(axis=0)
dataset_occurrence_counts = d_bin_df.sum(axis=0)
dataset_ccm = dataset_df.dot(d_df)
dataset_bin_ccm = dataset_bin_df.dot(d_bin_df)

def get_detailed_dataset_manifest(**kw):
    return DatasetManifest("annotation_manifest", 
                           detailed_dataset_df, detailed_dataset_bin_df, 
                           detailed_dataset_ccm, detailed_dataset_bin_ccm,
                           detailed_dataset_instance_counts, detailed_dataset_occurrence_counts,
                           variables, var2txt, **kw)

def get_dataset_manifest(**kw):
    return DatasetManifest("stripped_annotation_manifest",
                           dataset_df, dataset_bin_df, 
                           dataset_ccm, dataset_bin_ccm,
                           dataset_instance_counts, dataset_occurrence_counts,
                           classes, class2txt, **kw)

__all__ = [
    'var2txt',
    'class2txt',
    'classes',
    'class2classid',
    'classid2class',
    'var2classid',
    'variables',

    'detailed_dataset_column_ids',
    'dataset_column_ids',

    'detailed_dataset_df',
    'detailed_dataset_bin_df',
    'detailed_dataset_ccm',
    'detailed_dataset_bin_ccm',
    'detailed_dataset_instance_counts',
    'detailed_dataset_occurrence_counts',

    'dataset_df',
    'dataset_bin_df',
    'dataset_ccm',
    'dataset_bin_ccm',
    'dataset_instance_counts',
    'dataset_occurrence_counts',

    'get_detailed_dataset_manifest',
    'get_dataset_manifest',

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
    'CL_EARBUDS',
    'CL_LIGHTER',
    'CL_LIP_BALM',

    'DD_BROCCOLI',
    'DD_CAULIFLOWER',
    'DD_PEAS',
    'DD_CARROT',
    'DD_EGG',
    'DD_STEAK',
    'DD_SALMON',
    'DD_SAUSAGE',
    'DD_RED_APPLE',
    'DD_GREEN_APPLE',
    'DD_ORANGE',
    'DD_BANANA',
    'DD_PEACH',
    'DD_BREAD',
    'DD_CROISSANT',
    'DD_CAKE',
    'DD_CRACKERS',
    'DD_MILKA',
    'DD_KELLOGS',
    'DD_LICORICE',
    'DD_METAL_FORK',
    'DD_BLACK_FORK',
    'DD_TRANSPARENT_FORK',
    'DD_METAL_SPOON',
    'DD_BLACK_SPOON',
    'DD_TEASPOON',
    'DD_BLACK_KNIFE',
    'DD_RED_KNIFE',
    'DD_WHITE_NAPKIN',
    'DD_BORDEAUX_NAPKIN',
    'DD_RED_NAPKIN',
    'DD_WATER',
    'DD_GATORADE',
    'DD_ESTATHE_VERDE',
    'DD_COCA_COLA',
    'DD_COCA_COLA_ZERO',
    'DD_MONSTER',
    'DD_SPRITE',
    'DD_RED_THERMAL_BOTTLE',
    'DD_BLUE_THERMAL_BOTTLE',
    'DD_BROWN_THERMAL_BOTTLE',
    'DD_BLACK_THERMAL_BOTTLE',
    'DD_DARK_THERMAL_BOTTLE',
    'DD_GREEN_THERMAL_BOTTLE',
    'DD_PEAR_JUICE',
    'DD_PEACH_JUICE_1',
    'DD_PEACH_JUICE_2',
    'DD_TENNENTS',
    'DD_YOGURT',
    'DD_WHITE_CUP',
    'DD_RED_CUP',
    'DD_TRANSPARENT_CUP',
    'DD_SMALL_CUP',
    'DD_MUG',
    'DD_OIL_CRUET',
    'DD_TISSUES',
    'DD_KEYS',
    'DD_WALLET',
    'DD_SALT_SHAKER',
    'DD_PHONE',
    'DD_EAR_BUDS',
    'DD_LIGHTER',
    'DD_LIP_BALM',
    'DD_CIGARETTES',
    'DD_DISPOSABLE_CUTLERY',
    'DD_CRUSHED_PLASTIC_BOTTLE',
    'DD_CHINOTTO',
    'DD_THE_PESCA',
    'DD_FOCO_MARACUYA',
    'DD_RED_TRAY',
    'DD_GRAY_TRAY',
    'DD_PARCHMENT_PAPER',
    
    'D_FOOD',
    'D_SNACK',
    'D_SWEETS',
    'D_FORK',
    'D_SPOON',
    'D_KNIFE',
    'D_NAPKIN',
    'D_PLASTIC_BOTTLE',
    'D_ALLUMINUM_CAN',
    'D_THERMAL_BOTTLE',
    'D_TETRAPAK',
    'D_GLASS_BOTTLE',
    'D_YOGURT',
    'D_PLASTIC_CUP',
    'D_MUG',
    'D_OIL_CRUET',
    'D_TISSUES',
    'D_KEYS',
    'D_WALLET',
    'D_SALT_SHAKER',
    'D_PHONE',
    'D_EAR_BUDS',
    'D_LIGHTER',
    'D_LIP_BALM',
    'D_CIGARETTES',
    'D_DISPOSABLE_CUTLERY',
    'D_CRUSHED_PLASTIC_BOTTLE',
]