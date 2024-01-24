import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Example script.")

# Add the argument
# The '+' sign means "one or more". 
# The parser will expect at least one value, and all values will be gathered into a list.
parser.add_argument("--mylist", nargs='+', type=int, help="A list of integer values", default=[1, 2, 3])
parser.add_argument("--mylist2", type=int, help="A list of integer values", default=2)

# Parse the arguments
args = parser.parse_args()

# Access the argument values (which will be a list)
print(args.mylist)

