from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()
parser.add_argument("-i", nargs="+", dest="infiles", required=True)
parser.add_argument("-o", dest="outfile", required=True)

args = parser.parse_args()

spacings = []
energies = []
fishers = []
for f in args.infiles:
    arr = np.load(f)
    spacings.append(arr["spacing"])
    energies.append(arr["energy"])
    fishers.append(arr["fisher"])

spacings = np.asarray(spacings)
energies = np.asarray(energies)
fishers = np.asarray(fishers)

np.savez(args.outfile, spacing=spacings, energy=energies, fisher=fishers)
