from config import AppConfig


def main():
    c = AppConfig()
    print(c.get_args())


if __name__ == '__main__':
    main()
