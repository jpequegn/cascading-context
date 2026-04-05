import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="ctx",
        description="Cascading context for AI agents",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    parser.parse_args()


if __name__ == "__main__":
    main()
