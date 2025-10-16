import sys

MINIMUM_VERSION = (3, 8)


def main():
    if sys.version_info < MINIMUM_VERSION:
        required = ".".join(map(str, MINIMUM_VERSION))
        raise TypeError(
            "This project requires Python {} or higher. Found: Python {}".format(
                required, sys.version))
    print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()
