import argparse

"""
    This was just a temp file to print the args from argeparse/ .sh scripts, instead of having 
"""


def init_argparse() -> argparse.ArgumentParser:
    """
        Argeparse is weird for working with bools - see this thread here:
            https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse#:~:text=The%20boolean%20value%20is%20always,something%22)%20args%20%3D%20parser.

        This fix should work for now.
    """

    parser = argparse.ArgumentParser(description='Train BohsNet on the Bohs dataset!')

    parser.add_argument('--aws', default=False, type=lambda x: (str(x).lower() == 'true'), help="True or False")
    parser.add_argument('--aws_testing', default=False, type=lambda x: (str(x).lower() == 'true'), help='aws_testing')
    parser.add_argument('--run_validation', default=False, type=lambda x: (str(x).lower() == 'true'), help='run_validation; True or False')
    parser.add_argument('--save_weights_when_testing', default=False, type=lambda x: (str(x).lower() == 'true'), help='True or False')

    # Print all the args
    args_ = parser.parse_args()
    print(args_)

    # Print the type of aws
    print(f"aws is of type {type(args_.aws)})")

    return parser


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    print("Here are the args:)")
    print(args)
