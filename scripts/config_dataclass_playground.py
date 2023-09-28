from config import Config


def intialize_config():
    config = Config(ran_script=True, aws=False, aws_testing=False)
    print(config)


def main():
    intialize_config()


if __name__ == '__main__':
    main()
