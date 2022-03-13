from argparse import ArgumentParser
import pickle
import re
from glob import glob

parser = ArgumentParser()
parser.add_argument("-i", dest="infiles", required=True)
parser.add_argument("-o", dest="outfile", required=True)

args = parser.parse_args()


data = []
for f in glob(args.infiles):

    # Hotfix
    # pmts = re.match(".*_([0-9]*)_", f).groups()[0]
    d = pickle.load(open(f, "rb"))
    # d["pmts"] = int(pmts)
    data.append(d)

pickle.dump(data, open(args.outfile, "wb"))
