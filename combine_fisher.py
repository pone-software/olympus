from argparse import ArgumentParser
import pickle

parser = ArgumentParser()
parser.add_argument("-i", nargs="+", dest="infiles", required=True)
parser.add_argument("-o", dest="outfile", required=True)

args = parser.parse_args()


data = []
for f in args.infiles:
    data.append(pickle.load(open(f, "rb")))

pickle.dump(data, open(args.outfile, "wb"))
