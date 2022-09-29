from argparse import ArgumentParser
import pickle
from glob import glob

parser = ArgumentParser()
parser.add_argument("-i", dest="infiles", required=True)
parser.add_argument("-o", dest="outfile", required=True)

args = parser.parse_args()


data = []
for f in glob(args.infiles):

    d = pickle.load(open(f, "rb"))
    data += d

pickle.dump(data, open(args.outfile, "wb"))
