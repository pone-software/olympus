from argparse import ArgumentParser
import pickle
import re

parser = ArgumentParser()
parser.add_argument("-i", nargs="+", dest="infiles", required=True)
parser.add_argument("-o", dest="outfile", required=True)

args = parser.parse_args()


data = []
for f in args.infiles:

    # Hotfix
    pmts = re.match(".*_([0-9]*)_", f).groups()[0]
    d = pickle.load(open(f, "rb"))
    d["pmts"] = int(pmts)
    data.append(d)

pickle.dump(data, open(args.outfile, "wb"))
