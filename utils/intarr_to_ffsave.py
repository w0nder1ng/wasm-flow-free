from ast import literal_eval
from pathlib import Path

val = bytes(literal_eval(input("convert: ")))
fname = input("save to where? ")
with Path(fname).open("wb") as f:
    f.write(val)
