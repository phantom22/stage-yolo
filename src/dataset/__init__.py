import os
from helpers import parse_json_file

id2txt = {
    "br":"broccoli",
    "ca":"cauliflower",
    "pi":"peas",
    "ct":"carrot",
    #
    "uo":"egg",
    #
    "st":"steak",
    "sa":"salmon",
    "sg":"sausage",
    #
    "mr":"red apple",
    "mv":"green apple",
    "ar":"orange",
    "ba":"banana",
    "pe":"peach",
    #
    "pa":"bread",
    #
    "cr":"croissant",
    "to":"cake",
    #
    "ck":"crackers",
    "mi":"milka",
    "ke":"kellogs",
    "gr":"breadsticks",
    #
    "lq":"licorice",
    #
    "fm":"metal fork",
    "fn":"black fork",
    "ft":"plastic fork",
    #
    "cm":"metal spoon",
    "cn":"black spoon",
    #
    #"lm":"coltello metallo",
    "ln":"black knife",
    "lr":"red knife",
    #
    "cc":"teaspoon",
    #
    "tb":"white napkin",
    "td":"bordeaux napkin",
    "tr":"red napkin",
    #
    "aq":"water",
    "cl":"coca-cola",
    "cz":"coca-cola zero",
    "sp":"sprite",
    "bo":"borraccia",
    "qp":"pear juice",
    "qc":"peach juice",
    "tn":"tennent's",
    "yo":"yogurt",
    #
    "bi":"white cup",
    "bs":"red cup",
    "bt":"transparent cup",
    "tz":"mug",
    #
    "ol":"oil cruet",
    "fz":"fazzoletti",
    "ch":"keys",
    "pf":"wallet",
    "sl":"salt",
    "tl":"telephone",
    #
    "r":"red tray",
    "g":"gray tray",
    "c":"parchment paper"
}

ids = sorted(id2txt)

current_dir = os.path.dirname(__file__)
json_path = os.path.join(current_dir, "dataset.json")

obj = parse_json_file(json_path)

# None on sequence gaps
dataset = [obj.get(str(i)) for i in range(len(obj))]

__all__ = ['id2txt','ids','dataset']